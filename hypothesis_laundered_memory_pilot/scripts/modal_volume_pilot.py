from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_DIR = "/root/hlm"
RESULTS_DIR = Path("/results")
HF_CACHE_DIR = Path("/cache/huggingface")


def _ignore(path: Path) -> bool:
    parts = set(path.parts)
    return "__pycache__" in parts or ".git" in parts or "outputs" in parts or path.name.endswith(".pyc")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HOME": str(HF_CACHE_DIR), "TRANSFORMERS_CACHE": str(HF_CACHE_DIR / "transformers")})
    .pip_install(
        "openai>=1.40.0",
        "transformers>=4.45.0",
        "torch",
        "accelerate",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(".", remote_path=APP_DIR, ignore=_ignore)
)

app = modal.App("hlm-scientific-pilot-volume")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("hlm-results", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")


def _safe_name(model_id: str, n: int, suffix: str = "") -> str:
    safe = model_id.lower().replace("/", "_").replace("-", "_").replace(".", "_")
    name = f"modal_{safe}_v2_n{n}"
    if suffix:
        clean_suffix = suffix.lower().replace("/", "_").replace("-", "_").replace(".", "_")
        name = f"{name}_{clean_suffix}"
    return name


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 12,
    volumes={"/cache": hf_cache, "/results": results_volume},
    secrets=[hf_secret],
)
def run_model_to_volume(model_id: str, n: int = 80, max_new_tokens: int = 80, run_suffix: str = "") -> dict[str, str]:
    if os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    run_name = _safe_name(model_id, n, run_suffix)
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        "run_pilot.py",
        "--backend",
        "transformers",
        "--hf-model",
        model_id,
        "--model",
        model_id,
        "--benchmark",
        "data/benchmark_v2.json",
        "--n",
        str(n),
        "--out",
        str(out_dir),
        "--temperature",
        "0",
        "--max-new-tokens",
        str(max_new_tokens),
        "--audit-sample-size",
        "40",
        "--allow-download",
        "true",
    ]
    log_path = out_dir / "modal_run.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.write("HF_TOKEN_PRESENT: " + str(bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))) + "\n\n")
        log.flush()
        result = subprocess.run(cmd, cwd=APP_DIR, text=True, stdout=log, stderr=subprocess.STDOUT)
    payload = {"run_name": run_name, "model_id": model_id, "returncode": result.returncode}
    (out_dir / "modal_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    results_volume.commit()
    hf_cache.commit()
    return {k: str(v) for k, v in payload.items()}


@app.function(image=image, volumes={"/results": results_volume})
def list_runs() -> list[str]:
    return sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())


@app.function(image=image, volumes={"/results": results_volume})
def fetch_run(run_name: str) -> dict[str, str]:
    root = RESULTS_DIR / run_name
    payload: dict[str, str] = {"_run_name": run_name}
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if path.stat().st_size <= 5_000_000:
                payload[rel] = path.read_text(encoding="utf-8", errors="replace")
    return payload


@app.function(image=image, volumes={"/results": results_volume})
def status_run(run_name: str) -> dict[str, object]:
    root = RESULTS_DIR / run_name
    if not root.exists():
        return {"run_name": run_name, "exists": False}
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size})
    cache_path = root / "llm_cache.jsonl"
    log_path = root / "modal_run.log"
    status_path = root / "modal_status.json"
    return {
        "run_name": run_name,
        "exists": True,
        "files": files,
        "cache_lines": sum(1 for _ in cache_path.open("r", encoding="utf-8")) if cache_path.exists() else 0,
        "has_summary": (root / "summary.csv").exists() and (root / "summary.md").exists(),
        "has_case_scores": (root / "case_scores.jsonl").exists(),
        "has_modal_status": status_path.exists(),
        "modal_status": status_path.read_text(encoding="utf-8", errors="replace") if status_path.exists() else "",
        "log_tail": "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]) if log_path.exists() else "",
    }


@app.local_entrypoint()
def main(
    action: str = "spawn",
    model: str = "Qwen/Qwen3-8B-Instruct",
    n: int = 80,
    max_new_tokens: int = 80,
    run_name: str = "",
    run_suffix: str = "",
) -> None:
    if action == "submit":
        print(json.dumps(run_model_to_volume.remote(model, n=n, max_new_tokens=max_new_tokens, run_suffix=run_suffix), indent=2))
        return
    if action == "spawn":
        call = run_model_to_volume.spawn(model, n=n, max_new_tokens=max_new_tokens, run_suffix=run_suffix)
        print(
            json.dumps(
                {
                    "run_name": _safe_name(model, n, run_suffix),
                    "model": model,
                    "function_call_id": call.object_id,
                    "status": "spawned",
                },
                indent=2,
            )
        )
        return
    if action == "list":
        print(json.dumps(list_runs.remote(), indent=2))
        return
    if action == "fetch":
        if not run_name:
            run_name = _safe_name(model, n, run_suffix)
        payload = fetch_run.remote(run_name)
        out = Path("outputs") / run_name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        for rel, text in payload.items():
            if rel.startswith("_"):
                continue
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        print(json.dumps({"run_name": run_name, "files": sorted(k for k in payload if not k.startswith("_"))}, indent=2))
        return
    if action == "status":
        if not run_name:
            run_name = _safe_name(model, n, run_suffix)
        print(json.dumps(status_run.remote(run_name), indent=2, sort_keys=True))
        return
    raise ValueError(f"Unknown action: {action}")
