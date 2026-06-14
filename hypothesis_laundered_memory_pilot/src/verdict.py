from __future__ import annotations

import csv
import json
from pathlib import Path

from .run_classifier import classify_run


def write_verdict(outputs_dir: str | Path = "outputs") -> Path:
    outputs = Path(outputs_dir)
    runs = []
    for meta_path in sorted(outputs.glob("*/run_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(classify_run(meta))
        complete = (meta_path.parent / "summary.csv").exists() and (meta_path.parent / "case_scores.jsonl").exists()
        meta["completed_outputs"] = complete
        if not complete and not meta.get("mock"):
            meta.update(
                {
                    "run_role": "incomplete_run" if meta.get("run_role") != "failed_run" else "failed_run",
                    "scientific_evidence": False,
                    "research_verdict": "NO_SCIENTIFIC_RUNS",
                }
            )
        runs.append((meta_path.parent.name, meta))
    skipped = []
    skipped_path = outputs / "skipped_runs.jsonl"
    if skipped_path.exists():
        skipped = [json.loads(line) for line in skipped_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scientific = [(name, m) for name, m in runs if m.get("scientific_evidence") and m.get("completed_outputs")]
    scientific_families = {_model_family(str(m.get("hf_model") or m.get("model") or "")) for _, m in scientific}
    minimum = len(scientific) >= 2 and len(scientific_families) >= 2
    strong = len(scientific) >= 4 and len(scientific_families) >= 3
    aggregate = _read_aggregate(outputs / "aggregate_results.csv")
    verdict = "NO_SCIENTIFIC_RUNS"
    if scientific and not minimum:
        verdict = "FAILED_MINIMUM"
    elif minimum:
        verdict = "CONTINUE_WEAK"
    if strong:
        verdict = "CONTINUE_STRONG"
    lines = [
        "# Scientific Pilot Verdict",
        "",
        f"Final verdict: `{verdict}`",
        "",
        "## Models Actually Run",
        "",
    ]
    for name, meta in runs:
        lines.append(
            f"- `{name}`: model=`{meta.get('hf_model') or meta.get('model')}`, backend=`{meta.get('backend')}`, n=`{meta.get('n_items')}`, role=`{meta.get('run_role')}`, scientific=`{meta.get('scientific_evidence')}`"
        )
    lines.extend(["", "## Models Skipped", ""])
    if skipped:
        for row in skipped:
            lines.append(f"- `{row.get('model')}` for `{row.get('run_name')}`: {row.get('reason')}")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            f"- Minimum pilot achieved: `{str(minimum).lower()}`",
            f"- Strong pilot achieved: `{str(strong).lower()}`",
            f"- Scientific model count: `{len(scientific)}`",
            f"- Scientific model families: `{', '.join(sorted(scientific_families)) if scientific_families else 'none'}`",
            "",
            "## Main Aggregate Numbers",
            "",
        ]
    )
    if aggregate:
        for row in aggregate[:20]:
            lines.append(f"- `{row['method']}` `{row['metric']}` mean=`{row['mean']}` n_models=`{row['n_models']}`")
    else:
        lines.append("- No scientific aggregate rows because no qualifying real instruct-model runs completed.")
    strongest = _baseline_threat(outputs)
    lines.extend(
        [
            "",
            "## Strongest Baseline Threat",
            "",
            strongest,
            "",
            "## Manual Audit Status",
            "",
            "Manual audit samples were generated, but no completed human audit file was found. Manual audit is still required before paper claims.",
            "",
            "## Exact Next Step",
            "",
            "Run at least two real instruct/chat models with `n >= 80` using `scripts/run_scientific_pilot.sh --allow-download true --target minimum`, or provide a reachable local OpenAI-compatible endpoint serving such models.",
            "",
        ]
    )
    path = outputs / "scientific_pilot_verdict.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _read_aggregate(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def _baseline_threat(outputs: Path) -> str:
    rows = _read_aggregate(outputs / "aggregate_results.csv")
    if not rows:
        return "No scientific aggregate exists. Baseline threat cannot be evaluated yet."
    by = {(r["method"], r["metric"]): float(r["mean"]) for r in rows}
    quote = by.get(("quote_required_plus_self_check", "downstream_contamination_rate"))
    enforced = by.get(("evidence_labeled_enforced", "downstream_contamination_rate"))
    if quote is not None and enforced is not None and quote <= enforced:
        return "quote_required_plus_self_check matches or beats evidence_labeled_enforced; novelty is threatened."
    return "No simple baseline is shown to match evidence_labeled_enforced in current aggregate rows."


def _model_family(model_name: str) -> str:
    compact = model_name.lower()
    if "qwen" in compact:
        return "qwen"
    if "gemma" in compact or "google/" in compact:
        return "gemma"
    if "mistral" in compact or "magistral" in compact:
        return "mistral"
    if "llama" in compact:
        return "llama"
    if "/" in compact:
        return compact.split("/", 1)[0]
    return compact or "unknown"


def main() -> None:
    write_verdict()


if __name__ == "__main__":
    main()
