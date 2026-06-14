from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .experiment_index import build_index
from .model_config import load_model_config, select_models


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    skipped_path = OUTPUTS / "skipped_runs.jsonl"
    if skipped_path.exists():
        skipped_path.unlink()
    run_cmd(
        "mock_full_deployable",
        [
            sys.executable,
            "run_pilot.py",
            "--mock",
            "--n",
            "80",
            "--out",
            "outputs/mock_full_deployable",
            "--audit-sample-size",
            "20",
        ],
    )
    run_cmd(
        "smoke_tiny_gpt2_deployable",
        [
            sys.executable,
            "run_pilot.py",
            "--backend",
            "transformers",
            "--hf-model",
            "sshleifer/tiny-gpt2",
            "--model",
            "sshleifer/tiny-gpt2",
            "--n",
            "5",
            "--out",
            "outputs/smoke_tiny_gpt2_deployable",
            "--temperature",
            "0",
            "--max-new-tokens",
            "100",
            "--audit-sample-size",
            "5",
            "--allow-download",
            "false",
        ],
        allow_fail=True,
    )
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    if endpoint_available(base_url):
        run_cmd(
            "local_openai_compatible_n80",
            [
                sys.executable,
                "run_pilot.py",
                "--backend",
                "openai_compatible",
                "--base-url",
                base_url,
                "--api-key",
                os.environ.get("OPENAI_API_KEY", "dummy"),
                "--model",
                os.environ.get("MODEL_NAME", "local-open-model"),
                "--n",
                "80",
                "--out",
                "outputs/local_openai_compatible_n80",
                "--temperature",
                "0",
            ],
            allow_fail=True,
        )
    else:
        record_skip("local_openai_compatible_n80", os.environ.get("MODEL_NAME", "local-open-model"), f"No server reachable at {base_url}")
    config = load_model_config(ROOT / "configs" / "open_models.yaml")
    for model in select_models(config, "recommended_small") + select_models(config, "recommended_main"):
        safe = safe_name(model)
        if hf_cache_exists(model):
            run_cmd(
                f"real_{safe}_n80",
                [
                    sys.executable,
                    "run_pilot.py",
                    "--backend",
                    "transformers",
                    "--hf-model",
                    model,
                    "--model",
                    model,
                    "--n",
                    "80",
                    "--out",
                    f"outputs/real_{safe}_n80",
                    "--temperature",
                    "0",
                    "--max-new-tokens",
                    "800",
                    "--audit-sample-size",
                    "20",
                    "--allow-download",
                    "false",
                ],
                allow_fail=True,
            )
            break
        record_skip(f"real_{safe}_n80", model, "Model not present in local Hugging Face cache and downloads are disabled by runner.")
    build_index(OUTPUTS)


def run_cmd(run_name: str, cmd: list[str], allow_fail: bool = False) -> None:
    print(f"Running {run_name}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode and not allow_fail:
        raise SystemExit(result.returncode)
    if result.returncode:
        record_skip(run_name, cmd[cmd.index("--model") + 1] if "--model" in cmd else "", f"Command failed with exit code {result.returncode}")


def endpoint_available(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def hf_cache_exists(model: str) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model.replace("/", "--"))
    return cache_dir.exists()


def record_skip(run_name: str, model: str, reason: str) -> None:
    with (OUTPUTS / "skipped_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_name": run_name, "model": model, "reason": reason}, sort_keys=True) + "\n")


def safe_name(model: str) -> str:
    return model.lower().replace("/", "_").replace("-", "_").replace(".", "_")


if __name__ == "__main__":
    main()
