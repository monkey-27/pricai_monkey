from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


APP_DIR = "/root/hlm"


def _ignore(path: Path) -> bool:
    parts = set(path.parts)
    return (
        "__pycache__" in parts
        or ".git" in parts
        or "outputs" in parts
        or path.name.endswith(".pyc")
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HOME": "/cache/huggingface", "TRANSFORMERS_CACHE": "/cache/huggingface/transformers"})
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

app = modal.App("hlm-scientific-pilot")


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 8, volumes={"/cache": modal.Volume.from_name("hf-cache", create_if_missing=True)})
def run_hlm_model(model_id: str, n: int = 80, max_new_tokens: int = 300) -> dict[str, str]:
    safe = model_id.lower().replace("/", "_").replace("-", "_").replace(".", "_")
    out_dir = f"outputs/modal_{safe}_v2_n{n}"
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
        out_dir,
        "--temperature",
        "0",
        "--max-new-tokens",
        str(max_new_tokens),
        "--audit-sample-size",
        "40",
        "--allow-download",
        "true",
    ]
    result = subprocess.run(cmd, cwd=APP_DIR, text=True)
    payload = {
        "_run_name": out_dir.split("/", 1)[1],
        "_model_id": model_id,
        "_returncode": str(result.returncode),
        "_stdout": "",
        "_stderr": "",
    }
    root = Path(APP_DIR) / out_dir
    for name in [
        "summary.csv",
        "summary.md",
        "run_metadata.json",
        "manual_audit_sample.csv",
        "manual_audit_instructions.md",
        "case_scores.jsonl",
    ]:
        path = root / name
        if path.exists():
            payload[name] = path.read_text(encoding="utf-8")
    if result.returncode != 0:
        metadata = json.loads(payload.get("run_metadata.json", "{}") or "{}")
        metadata.update(
            {
                "run_role": "failed_run",
                "scientific_evidence": False,
                "research_verdict": "NO_SCIENTIFIC_RUNS",
                "classification_reason": f"Run failed before producing complete outputs; returncode={result.returncode}.",
            }
        )
        payload["run_metadata.json"] = json.dumps(metadata, indent=2, sort_keys=True)
    return payload


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", n: int = 80, max_new_tokens: int = 300) -> None:
    payload = run_hlm_model.remote(model, n=n, max_new_tokens=max_new_tokens)
    run_name = payload["_run_name"]
    out = Path("outputs") / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "modal_payload.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        (out / key).write_text(value, encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["_run_name", "_model_id", "_returncode"]}, indent=2))
