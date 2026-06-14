from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .run_classifier import classify_run


METRICS = [
    "false_evidence_promotion_rate",
    "downstream_contamination_rate",
    "trap_task_accuracy",
    "verified_control_accuracy",
    "useful_memory_retention",
    "confirmed_hypothesis_promotion_rate",
    "tentative_overblocking_rate",
    "mixed_rejected_rate",
    "mixed_endorsed_rate",
    "uncertain_rate",
    "net_utility_trap",
    "net_utility_control",
    "coding_contamination_rate",
    "data_analysis_contamination_rate",
    "research_assistant_contamination_rate",
]


def aggregate(outputs_dir: str | Path = "outputs") -> tuple[Path, Path, Path]:
    outputs = Path(outputs_dir)
    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, str]] = []
    for meta_path in sorted(outputs.glob("*/run_metadata.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(classify_run(meta))
        summary = run_dir / "summary.csv"
        complete = summary.exists() and (run_dir / "case_scores.jsonl").exists()
        if complete:
            for row in csv.DictReader(summary.open("r", encoding="utf-8")):
                row.update(
                    {
                        "run_name": run_dir.name,
                        "backend": meta.get("backend", ""),
                        "model": meta.get("hf_model") or meta.get("model", ""),
                        "run_role": meta.get("run_role", ""),
                        "scientific_evidence": str(meta.get("scientific_evidence", False)).lower(),
                    }
                )
                records.append(row)
        audit = run_dir / "manual_audit_sample.csv"
        if audit.exists():
            for row in csv.DictReader(audit.open("r", encoding="utf-8")):
                row["run_name"] = run_dir.name
                audit_rows.append(row)
    agg_path = outputs / "aggregate_results.csv"
    report_path = outputs / "aggregate_report.md"
    audit_path = outputs / "manual_audit_combined.csv"
    _write_aggregate_csv(agg_path, records)
    _write_aggregate_report(report_path, records)
    _write_audit(audit_path, audit_rows)
    return agg_path, report_path, audit_path


def _write_aggregate_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["method", "metric", "mean", "stddev", "ci95_low", "ci95_high", "n_models", "per_model_values"]
    scientific = [r for r in records if r.get("scientific_evidence") == "true"]
    grouped: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in scientific:
        for metric in METRICS:
            if row.get(metric, "") != "":
                grouped[row["method"]][metric].append((row["model"], float(row[metric])))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, by_metric in sorted(grouped.items()):
            for metric, values in sorted(by_metric.items()):
                nums = [v for _, v in values]
                mean = sum(nums) / len(nums) if nums else 0.0
                std = _stddev(nums)
                margin = 1.96 * std / math.sqrt(len(nums)) if len(nums) > 1 else 0.0
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "mean": round(mean, 4),
                        "stddev": round(std, 4),
                        "ci95_low": round(mean - margin, 4),
                        "ci95_high": round(mean + margin, 4),
                        "n_models": len(nums),
                        "per_model_values": json.dumps(dict(values), sort_keys=True),
                    }
                )


def _write_aggregate_report(path: Path, records: list[dict[str, Any]]) -> None:
    scientific = [r for r in records if r.get("scientific_evidence") == "true"]
    real_models = sorted({r.get("model", "") for r in scientific})
    lines = [
        "# Aggregate HLM Pilot Report",
        "",
        f"- scientific models: `{len(real_models)}`",
        f"- models: `{real_models}`",
        "",
    ]
    if not real_models:
        lines.append("Verdict: NO_SCIENTIFIC_RUNS. No completed real instruct-model run produced scientific evidence.")
    elif len(real_models) < 2:
        lines.append("Verdict: FAILED_MINIMUM. Fewer than two real instruct models completed n >= 80.")
    else:
        lines.append("Verdict: aggregate criteria should be interpreted with manual audit and per-model reports.")
    lines.append("")
    lines.append("See `aggregate_results.csv` for per-method means, standard deviations, and approximate 95% confidence intervals.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = ["run_name", *[f for f in rows[0].keys() if f != "run_name"]]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def main() -> None:
    aggregate()


if __name__ == "__main__":
    main()
