from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .aggregate import aggregate
from .experiment_index import build_index
from .model_config import load_model_config, select_models
from .verdict import write_verdict


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full HLM scientific pilot as far as this machine allows.")
    parser.add_argument("--allow-download", default="false")
    parser.add_argument("--target", default="minimum", choices=["minimum", "strong", "smoke-real"])
    args = parser.parse_args()
    allow_download = args.allow_download.lower() == "true"
    OUTPUTS.mkdir(exist_ok=True)
    skipped = OUTPUTS / "skipped_runs.jsonl"
    if skipped.exists():
        skipped.unlink()
    run(
        "mock_full_deployable",
        [
            sys.executable,
            "run_pilot.py",
            "--mock",
            "--benchmark",
            "data/benchmark_v2.json",
            "--n",
            "120",
            "--out",
            "outputs/mock_full_deployable",
            "--audit-sample-size",
            "20",
        ],
    )
    run(
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
            "--benchmark",
            "data/benchmark_v2.json",
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
            str(allow_download).lower(),
        ],
        allow_fail=True,
    )
    completed_scientific = 0
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    if endpoint_available(base_url):
        if run(
            "local_openai_compatible_v2_n120",
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
                os.environ.get("MODEL_NAME", "local-open-instruct"),
                "--benchmark",
                "data/benchmark_v2.json",
                "--n",
                "120",
                "--out",
                "outputs/local_openai_compatible_v2_n120",
                "--temperature",
                "0",
                "--audit-sample-size",
                "40",
            ],
            allow_fail=True,
        ):
            completed_scientific += 1
    else:
        record_skip("local_openai_compatible_v2_n120", os.environ.get("MODEL_NAME", "local-open-instruct"), f"No server reachable at {base_url}")
    target_count = {"smoke-real": 1, "minimum": 2, "strong": 4}[args.target]
    config = load_model_config(ROOT / "configs" / "open_models.yaml")
    if args.target == "strong":
        model_order = (
            select_models(config, "recommended_main")
            + select_models(config, "optional_stronger")
            + select_models(config, "recommended_small")
        )
    else:
        model_order = (
            select_models(config, "recommended_small")
            + select_models(config, "recommended_main")
            + select_models(config, "optional_stronger")
        )
    for model in dict.fromkeys(model_order):
        if completed_scientific >= target_count:
            break
        safe = safe_name(model)
        if not allow_download and not hf_cache_exists(model):
            record_skip(f"real_{safe}_v2_n120", model, "Model not present in local Hugging Face cache and downloads are disabled.")
            continue
        n = "120" if args.target != "smoke-real" else "40"
        run_name = f"real_{safe}_v2_n{n}"
        ok = run(
            run_name,
            [
                sys.executable,
                "run_pilot.py",
                "--backend",
                "transformers",
                "--hf-model",
                model,
                "--model",
                model,
                "--benchmark",
                "data/benchmark_v2.json",
                "--n",
                n,
                "--out",
                f"outputs/{run_name}",
                "--temperature",
                "0",
                "--max-new-tokens",
                "800",
                "--audit-sample-size",
                "40",
                "--allow-download",
                str(allow_download).lower(),
            ],
            allow_fail=True,
        )
        if ok:
            completed_scientific += 1
        else:
            record_skip(run_name, model, "Command failed; inspect terminal logs for download, memory, or backend error.")
    aggregate(OUTPUTS)
    build_index(OUTPUTS)
    write_verdict(OUTPUTS)
    if completed_scientific < target_count:
        print(f"Scientific target not met: completed {completed_scientific}/{target_count} real instruct runs.")
        return
    print(f"Scientific target met: completed {completed_scientific}/{target_count} real instruct runs.")


def run(run_name: str, cmd: list[str], allow_fail: bool = False) -> bool:
    print(f"Running {run_name}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        if allow_fail:
            return False
        raise SystemExit(result.returncode)
    return True


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
    return (Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model.replace("/", "--"))).exists()


def record_skip(run_name: str, model: str, reason: str) -> None:
    with (OUTPUTS / "skipped_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_name": run_name, "model": model, "reason": reason}, sort_keys=True) + "\n")


def safe_name(model: str) -> str:
    return model.lower().replace("/", "_").replace("-", "_").replace(".", "_")


if __name__ == "__main__":
    main()
