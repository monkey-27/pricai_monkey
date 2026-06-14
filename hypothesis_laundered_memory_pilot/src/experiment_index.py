from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .run_classifier import classify_run


METRIC_FIELDS = [
    "false_evidence_promotion_rate",
    "downstream_contamination_rate",
    "useful_memory_retention",
    "confirmed_hypothesis_promotion_rate",
    "tentative_overblocking_rate",
]


def build_index(outputs_dir: str | Path = "outputs", out_path: str | Path | None = None) -> Path:
    outputs = Path(outputs_dir)
    out = Path(out_path) if out_path else outputs / "experiment_index.md"
    rows = []
    for meta_path in sorted(outputs.glob("*/run_metadata.json")):
        run_dir = meta_path.parent
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.update(classify_run(metadata))
        summary = _read_summary(run_dir / "summary.csv")
        complete = bool(summary) and (run_dir / "case_scores.jsonl").exists()
        if not complete and not metadata.get("mock"):
            metadata.update(
                {
                    "run_role": "incomplete_run",
                    "scientific_evidence": False,
                    "research_verdict": "NO_SCIENTIFIC_RUNS",
                }
            )
        rows.append((run_dir.name, metadata, summary, run_dir / "summary.md"))
    skipped = _read_skipped(outputs / "skipped_runs.jsonl")
    lines = [
        "# Experiment Index",
        "",
        "| run_name | backend | model | n_items | mock | run_role | scientific_evidence | main_metrics | verdict | summary |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for run_name, metadata, summary, summary_path in rows:
        model = metadata.get("hf_model") or metadata.get("model")
        metrics = _main_metrics(summary)
        summary_link = summary_path.relative_to(outputs.parent).as_posix()
        lines.append(
            f"| {run_name} | {metadata.get('backend')} | {model} | {metadata.get('n_items')} | {metadata.get('mock')} | "
            f"{metadata.get('run_role')} | {metadata.get('scientific_evidence')} | {metrics} | {metadata.get('research_verdict')} | "
            f"[summary.md]({summary_link}) |"
        )
    if skipped:
        lines.extend(["", "## Skipped Runs", ""])
        for row in skipped:
            lines.append(f"- `{row.get('run_name')}` `{row.get('model')}` skipped: {row.get('reason')}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_skipped(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _main_metrics(rows: list[dict[str, str]]) -> str:
    target = next((row for row in rows if row.get("method") == "evidence_labeled_enforced"), None)
    if not target:
        return "NA"
    parts = [f"{field}={target.get(field, 'NA')}" for field in METRIC_FIELDS]
    return "<br>".join(parts)


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
