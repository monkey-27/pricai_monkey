from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data_model import BenchmarkItem, ScoreRecord


def write_summary(
    out_dir: str | Path,
    scores: list[ScoreRecord],
    items: list[BenchmarkItem],
    methods: list[str],
    metadata: dict[str, Any],
) -> list[dict[str, object]]:
    out = Path(out_dir)
    rows = summarize(scores, methods)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.md").write_text(render_markdown(rows, scores, items, metadata), encoding="utf-8")
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
        verified = [s for s in group if s.case_subtype == "verified_hypothesis"]
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
                "confirmed_hypothesis_promotion_rate": round(rate([s.confirmed_hypothesis_promoted for s in verified]), 4),
                "tentative_overblocking_rate": round(rate([s.tentative_overblocked for s in verified]), 4),
                "net_utility_trap": round(rate([s.trap_task_correct for s in group]) - base_trap, 4),
                "net_utility_control": round(rate([s.control_task_correct for s in group]) - base_control, 4),
            }
        )
    return rows

def rate(values: list[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else 0.0


def render_markdown(
    rows: list[dict[str, object]],
    scores: list[ScoreRecord],
    items: list[BenchmarkItem],
    metadata: dict[str, Any],
) -> str:
    item_map = {item.id: item for item in items}
    domain_counts = Counter(item.domain for item in items)
    subtype_counts = Counter(item.case_subtype for item in items)
    mock = bool(metadata.get("mock"))
    lines = [
        "# Hypothesis-Laundered Memory Pilot Report",
        "",
    ]
    if mock:
        lines.extend([
            "> WARNING: This run used mock mode. Mock outputs are programmed to follow the expected pattern and are not scientific evidence.",
            "",
        ])
    lines.extend(
        [
            "## Run Type",
            "",
            f"- mock: `{str(mock).lower()}`",
            f"- scientific_evidence: `{str(bool(metadata.get('scientific_evidence'))).lower()}`",
            f"- run_role: `{metadata.get('run_role')}`",
            f"- classification_reason: `{metadata.get('classification_reason')}`",
            f"- backend: `{metadata.get('backend')}`",
            f"- model: `{metadata.get('model')}`",
            f"- hf_model: `{metadata.get('hf_model')}`",
            f"- base_url: `{metadata.get('base_url')}`",
            "",
            "## Benchmark",
            "",
            f"- total items: `{len(items)}`",
            f"- domain breakdown: `{dict(domain_counts)}`",
            f"- case subtype breakdown: `{dict(subtype_counts)}`",
            "",
            "## Methods",
            "",
            "- `no_memory`: no long-term memory.",
            "- `naive`: generic memory summary.",
            "- `reflection`: reusable lesson extraction.",
            "- `source_aware`: stores only source-supported memories.",
            "- `quote_required`: stable memories require direct support.",
            "- `evidence_labeled_no_enforcement`: labels memories but gives all of them downstream.",
            "- `evidence_labeled_stable_only`: withholds tentative memories.",
            "- `evidence_labeled_enforced`: stable memories are facts; tentative memories cannot override current evidence.",
            "",
            "## Main Metric Table",
            "",
            _table(rows, ["method", "n_items", "false_evidence_promotion_rate", "downstream_contamination_rate", "trap_task_accuracy", "verified_control_accuracy", "useful_memory_retention"]),
            "",
            "## False-Promotion Table",
            "",
            _table(rows, ["method", "false_evidence_promotion_rate", "confirmed_hypothesis_promotion_rate", "tentative_overblocking_rate"]),
            "",
            "## Contamination Table",
            "",
            _table(rows, ["method", "downstream_contamination_rate", "mixed_rate", "net_utility_trap"]),
            "",
            "## Verified-Memory Retention Table",
            "",
            _table(rows, ["method", "verified_control_accuracy", "useful_memory_retention", "net_utility_control"]),
            "",
            "## Confirmed-Hypothesis Promotion Table",
            "",
            _table(rows, ["method", "confirmed_hypothesis_promotion_rate", "tentative_overblocking_rate"]),
            "",
            "## Baseline Comparison",
            "",
            _baseline_comparison(rows),
            "",
            "## Representative Failures",
            "",
        ]
    )
    failures = [s for s in scores if s.downstream_contamination][:5]
    for score in failures:
        item = item_map[score.item_id]
        lines.append(f"- `{score.item_id}` `{score.method}` `{score.downstream_label}`: answer reused `{item.trap_hypothesis}` while current evidence supported `{item.ground_truth}`.")
    if not failures:
        lines.append("- No contaminated cases were found.")
    lines.extend(["", "## Representative Successes", ""])
    successes = [
        s
        for s in scores
        if s.method == "evidence_labeled_enforced" and not s.false_evidence_promotion and s.control_task_correct and s.trap_task_correct
    ][:5]
    for score in successes:
        item = item_map[score.item_id]
        lines.append(f"- `{score.item_id}`: avoided unstable memory while retaining `{item.verified_memory_control.memory}`.")
    if not successes:
        lines.append("- No clear evidence-labeled successes were found.")
    lines.extend(["", "## Evidence-Labeled Overblocked Useful Memory", ""])
    overblocked = [s for s in scores if s.method == "evidence_labeled_enforced" and s.tentative_overblocked][:5]
    for score in overblocked:
        item = item_map[score.item_id]
        lines.append(f"- `{score.item_id}`: verified hypothesis was not promoted despite `{item.verification_evidence}`.")
    if not overblocked:
        lines.append("- No overblocking cases found in this run.")
    lines.extend(["", "## Pilot Decision Criteria", ""])
    criteria = evaluate_criteria(rows)
    for label, passed in criteria:
        mark = "PASS" if passed else "FAIL"
        lines.append(f"- {mark}: {label}")
    lines.extend(["", "## Research Verdict", "", _verdict(rows, metadata), "", "## Scientific Interpretation", ""])
    run_role = str(metadata.get("run_role", ""))
    if mock:
        lines.append("This run validates code paths only. It must not be cited as evidence for the research claim.")
    elif run_role == "plumbing_smoke":
        lines.append("This run used a tiny, non-instruct, or too-small model setup. It validates model/backend plumbing only. It is not evidence for or against the research claim.")
    else:
        lines.append("This run is preliminary evidence, but it still requires manual audit and additional models before paper-level claims.")
    lines.append("")
    return "\n".join(lines)


def _table(rows: list[dict[str, object]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] + ["---:"] * (len(fields) - 1)) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _baseline_comparison(rows: list[dict[str, object]]) -> str:
    by = {str(row["method"]): row for row in rows}
    reflection = float(by.get("reflection", {}).get("downstream_contamination_rate", 0.0))
    enforced = float(by.get("evidence_labeled_enforced", {}).get("downstream_contamination_rate", 0.0))
    source = float(by.get("source_aware", {}).get("downstream_contamination_rate", 0.0))
    quote = float(by.get("quote_required", {}).get("downstream_contamination_rate", 0.0))
    if reflection and source <= enforced * 1.1 and quote <= enforced * 1.1:
        return "Source-aware and quote-required baselines are close to evidence-labeled enforcement; novelty is weak unless richer audits show a qualitative advantage."
    return f"Relative to reflection contamination `{reflection}`, evidence-labeled enforcement is `{enforced}`, source-aware is `{source}`, and quote-required is `{quote}`."


def evaluate_criteria(rows: list[dict[str, object]]) -> list[tuple[str, bool]]:
    by = {str(row["method"]): row for row in rows}
    naive = by.get("naive", {})
    reflection = by.get("reflection", {})
    enforced = by.get("evidence_labeled_enforced", {})
    source = by.get("source_aware", {})
    quote = by.get("quote_required", {})
    ref_cont = float(reflection.get("downstream_contamination_rate", 0.0))
    enf_cont = float(enforced.get("downstream_contamination_rate", 0.0))
    source_solves = float(source.get("downstream_contamination_rate", 1.0)) <= enf_cont * 1.1 and float(source.get("useful_memory_retention", 0.0)) >= 0.70
    quote_solves = float(quote.get("downstream_contamination_rate", 1.0)) <= enf_cont * 1.1 and float(quote.get("useful_memory_retention", 0.0)) >= 0.70
    return [
        ("naive false_evidence_promotion_rate >= 0.30", float(naive.get("false_evidence_promotion_rate", 0.0)) >= 0.30),
        ("reflection false_evidence_promotion_rate >= 0.35", float(reflection.get("false_evidence_promotion_rate", 0.0)) >= 0.35),
        ("naive downstream_contamination_rate >= 0.15", float(naive.get("downstream_contamination_rate", 0.0)) >= 0.15),
        ("reflection downstream_contamination_rate >= 0.15", ref_cont >= 0.15),
        ("evidence_labeled_enforced reduces contamination by >= 40% relative to reflection", ref_cont > 0 and enf_cont <= ref_cont * 0.60),
        ("evidence_labeled_enforced useful_memory_retention >= 0.70", float(enforced.get("useful_memory_retention", 0.0)) >= 0.70),
        ("evidence_labeled_enforced confirmed_hypothesis_promotion_rate >= 0.50", float(enforced.get("confirmed_hypothesis_promotion_rate", 0.0)) >= 0.50),
        ("evidence_labeled_enforced overblocking_rate <= 0.30", float(enforced.get("tentative_overblocking_rate", 1.0)) <= 0.30),
        ("source_aware and quote_required do not already solve the problem", not (source_solves or quote_solves)),
    ]


def _verdict(rows: list[dict[str, object]], metadata: dict[str, Any]) -> str:
    base = str(metadata.get("research_verdict", "REDESIGN"))
    if base in {"MOCK_ONLY", "PLUMBING_ONLY"}:
        return f"{base}: {metadata.get('classification_reason')}"
    criteria = evaluate_criteria(rows)
    if all(passed for _, passed in criteria):
        if base == "PAPER_CANDIDATE":
            return "PAPER_CANDIDATE: criteria pass, but claims still need paper-quality audit text."
        return "CONTINUE_STRONG: criteria pass, but replicate on additional open models before paper claims."
    failed = "; ".join(label for label, passed in criteria if not passed)
    return f"REDESIGN: failed criteria: {failed}."
