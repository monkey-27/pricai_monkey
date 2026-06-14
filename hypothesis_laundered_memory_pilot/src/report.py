from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .data_model import BenchmarkItem, ScoreRecord


def write_summary(out_dir: str | Path, scores: list[ScoreRecord], items: list[BenchmarkItem], methods: list[str]) -> list[dict[str, object]]:
    out = Path(out_dir)
    rows = summarize(scores, methods)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.md").write_text(render_markdown(rows, scores, items), encoding="utf-8")
    return rows


def summarize(scores: list[ScoreRecord], methods: list[str]) -> list[dict[str, object]]:
    by_method: dict[str, list[ScoreRecord]] = defaultdict(list)
    for score in scores:
        by_method[score.method].append(score)
    baseline = by_method.get("no_memory", [])
    base_trap = rate([s.trap_task_correct for s in baseline])
    base_control = rate([s.control_task_correct for s in baseline])
    rows: list[dict[str, object]] = []
    for method in methods:
        group = by_method.get(method, [])
        rows.append(
            {
                "method": method,
                "n_items": len(group),
                "false_evidence_promotion_rate": round(rate([s.false_evidence_promotion for s in group]), 4),
                "downstream_contamination_rate": round(rate([s.downstream_contamination for s in group]), 4),
                "trap_task_accuracy": round(rate([s.trap_task_correct for s in group]), 4),
                "verified_control_accuracy": round(rate([s.control_task_correct for s in group]), 4),
                "useful_memory_retention": round(rate([s.useful_memory_retention for s in group]), 4),
                "mixed_rate": round(rate([s.mixed for s in group]), 4),
                "net_utility_trap": round(rate([s.trap_task_correct for s in group]) - base_trap, 4),
                "net_utility_control": round(rate([s.control_task_correct for s in group]) - base_control, 4),
            }
        )
    return rows


def rate(values: list[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else 0.0


def render_markdown(rows: list[dict[str, object]], scores: list[ScoreRecord], items: list[BenchmarkItem]) -> str:
    item_map = {item.id: item for item in items}
    lines = [
        "# Hypothesis-Laundered Memory Pilot Report",
        "",
        "## Experiment setup",
        "",
        "This pilot tests whether memory extraction methods turn an assistant's unverified intermediate hypotheses into reusable long-term facts, then measures whether those memories contaminate later tasks when current evidence contradicts them.",
        "",
        f"Benchmark size: {len(items)} items across coding/debugging and data-analysis domains.",
        "",
        "## Methods",
        "",
        "- `no_memory`: downstream task receives no long-term memory.",
        "- `naive`: generic long-term memory summary; extracted memories are treated as stable by default.",
        "- `reflection`: reusable lesson extraction; extracted lessons are treated as stable by default.",
        "- `evidence_labeled`: memories are labeled by evidence type and split into Stable versus Tentative.",
        "",
        "## Results",
        "",
        "| method | n | false promotion | contamination | trap accuracy | control accuracy | useful retention | mixed | net trap | net control |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['n_items']} | {row['false_evidence_promotion_rate']} | {row['downstream_contamination_rate']} | {row['trap_task_accuracy']} | {row['verified_control_accuracy']} | {row['useful_memory_retention']} | {row['mixed_rate']} | {row['net_utility_trap']} | {row['net_utility_control']} |"
        )
    lines.extend(["", "## Representative failure cases", ""])
    failures = [s for s in scores if s.downstream_contamination][:5]
    for score in failures:
        item = item_map[score.item_id]
        lines.append(f"- `{score.item_id}` `{score.method}`: trap `{item.trap_hypothesis}` contaminated the answer despite ground truth `{item.ground_truth}`.")
    if not failures:
        lines.append("- No contaminated cases were found.")
    lines.extend(["", "## Evidence-labeled helped", ""])
    helped = [
        s
        for s in scores
        if s.method == "evidence_labeled" and not s.false_evidence_promotion and s.control_task_correct and s.trap_task_correct
    ][:5]
    for score in helped:
        item = item_map[score.item_id]
        lines.append(f"- `{score.item_id}`: kept trap tentative while retaining verified memory `{item.verified_memory_control.memory}`.")
    if not helped:
        lines.append("- No clear evidence-labeled wins were found.")
    lines.extend(["", "## Pilot decision criteria", ""])
    criteria = evaluate_criteria(rows)
    for label, passed in criteria:
        mark = "PASS" if passed else "FAIL"
        lines.append(f"- {mark}: {label}")
    if all(passed for _, passed in criteria):
        lines.append("")
        lines.append("Decision: continue. The pilot shows enough false promotion, downstream harm, and mitigation from evidence labeling to justify scaling.")
    else:
        failed = [label for label, passed in criteria if not passed]
        lines.append("")
        lines.append("Decision: do not scale yet. Failed criteria: " + "; ".join(failed) + ".")
    lines.append("")
    return "\n".join(lines)


def evaluate_criteria(rows: list[dict[str, object]]) -> list[tuple[str, bool]]:
    by = {str(row["method"]): row for row in rows}
    naive = by.get("naive", {})
    reflection = by.get("reflection", {})
    evidence = by.get("evidence_labeled", {})
    naive_ret = float(naive.get("useful_memory_retention", 0.0))
    refl_ret = float(reflection.get("useful_memory_retention", 0.0))
    ref_false = float(reflection.get("false_evidence_promotion_rate", 0.0))
    ev_false = float(evidence.get("false_evidence_promotion_rate", 0.0))
    best_ret = max(naive_ret, refl_ret)
    return [
        ("Naive false evidence promotion >= 40%", float(naive.get("false_evidence_promotion_rate", 0.0)) >= 0.40),
        ("Reflection false evidence promotion >= 50%", ref_false >= 0.50),
        ("Naive downstream contamination >= 25%", float(naive.get("downstream_contamination_rate", 0.0)) >= 0.25),
        ("Evidence-labeled memory reduces false promotion by >= 50% relative to reflection", ref_false > 0 and ev_false <= ref_false * 0.50),
        ("Evidence-labeled memory loses <= 15 percentage points useful memory retention compared to naive/reflection", best_ret - float(evidence.get("useful_memory_retention", 0.0)) <= 0.15),
    ]
