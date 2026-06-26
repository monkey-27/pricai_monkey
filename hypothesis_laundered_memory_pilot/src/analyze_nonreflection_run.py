from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HEADLINE_METHODS = [
    "no_memory",
    "naive",
    "source_aware",
    "quote_required",
    "evidence_labeled_no_enforcement",
    "evidence_labeled_stable_only",
    "evidence_labeled_enforced",
    "current_evidence_self_check",
    "quote_required_plus_self_check",
]

POOR_LABELS = {"contaminated", "mixed_endorsed_trap", "irrelevant", "unparseable", "incorrect"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-analyze an HLM run without reflection in headline metrics.")
    parser.add_argument("--run", required=True, help="Completed run directory containing summary/results artifacts.")
    parser.add_argument("--out", required=True, help="Output directory for non-reflection analysis.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(run_dir / "summary.csv")
    results = read_jsonl(run_dir / "results_raw.jsonl")
    case_scores = read_jsonl(run_dir / "case_scores.jsonl")
    memory_outputs = read_jsonl(run_dir / "memory_outputs.jsonl")
    downstream_outputs = read_jsonl(run_dir / "downstream_outputs.jsonl")
    metadata = read_json(run_dir / "run_metadata.json")
    benchmark = load_benchmark(metadata, run_dir)

    present_methods = [m for m in HEADLINE_METHODS if any(row.get("method") == m for row in results)]
    filtered = [row for row in results if row.get("method") in present_methods]
    summary = summarize(filtered, present_methods)

    write_csv(out_dir / "summary_nonreflection.csv", summary)
    poor_cases = build_poor_cases(filtered, case_scores, memory_outputs, downstream_outputs, benchmark)
    write_csv(out_dir / "poor_cases_nonreflection.csv", poor_cases)
    (out_dir / "poor_cases_by_method.md").write_text(render_poor_cases_by_method(poor_cases), encoding="utf-8")
    (out_dir / "report_nonreflection.md").write_text(
        render_report(summary, filtered, poor_cases, summary_rows, metadata, benchmark, run_dir),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_benchmark(metadata: dict[str, Any], run_dir: Path) -> dict[str, dict[str, Any]]:
    candidates: list[Path] = []
    benchmark = metadata.get("benchmark")
    if benchmark:
        candidates.append(Path(str(benchmark)))
        candidates.append(Path.cwd() / str(benchmark))
        candidates.append(run_dir.parents[1] / str(benchmark) if len(run_dir.parents) > 1 else Path(str(benchmark)))
    candidates.append(Path.cwd() / "data" / "benchmark_v2.json")
    candidates.append(Path(__file__).resolve().parents[1] / "data" / "benchmark_v2.json")
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                items = data.get("items", data.get("benchmark", []))
            else:
                items = data
            return {str(item.get("id")): item for item in items if isinstance(item, dict)}
    return {}


def summarize(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method"))].append(row)
    baseline = by_method.get("no_memory", [])
    base_trap = rate(boolish(row.get("trap_task_correct")) for row in baseline)
    base_control = rate(boolish(row.get("control_task_correct")) for row in baseline)
    base_contam = rate(boolish(row.get("downstream_contamination")) for row in baseline)
    output: list[dict[str, Any]] = []
    for method in methods:
        group = by_method.get(method, [])
        verified = [row for row in group if row.get("case_subtype") == "verified_hypothesis"]
        contamination = rate(boolish(row.get("downstream_contamination")) for row in group)
        trap_accuracy = rate(boolish(row.get("trap_task_correct")) for row in group)
        control_accuracy = rate(boolish(row.get("control_task_correct")) for row in group)
        retention = rate(boolish(row.get("useful_memory_retention")) for row in group)
        output.append(
            {
                "method": method,
                "n_items": len(group),
                "false_evidence_promotion_rate": round(rate(boolish(row.get("false_evidence_promotion")) for row in group), 4),
                "downstream_contamination_rate": round(contamination, 4),
                "contamination_delta_vs_no_memory": round(contamination - base_contam, 4),
                "trap_task_accuracy": round(trap_accuracy, 4),
                "trap_accuracy_delta_vs_no_memory": round(trap_accuracy - base_trap, 4),
                "verified_control_accuracy": round(control_accuracy, 4),
                "useful_memory_retention": round(retention, 4),
                "retention_delta_vs_no_memory": round(retention - base_control, 4),
                "mixed_rate": round(rate(boolish(row.get("mixed")) for row in group), 4),
                "mixed_rejected_rate": round(rate(row.get("downstream_label") == "mixed_rejected_trap" for row in group), 4),
                "mixed_endorsed_rate": round(rate(row.get("downstream_label") == "mixed_endorsed_trap" for row in group), 4),
                "uncertain_rate": round(rate(row.get("downstream_label") == "uncertain" for row in group), 4),
                "coding_contamination_rate": round(rate(boolish(row.get("downstream_contamination")) for row in group if row.get("domain") == "coding"), 4),
                "data_analysis_contamination_rate": round(rate(boolish(row.get("downstream_contamination")) for row in group if row.get("domain") == "data_analysis"), 4),
                "research_assistant_contamination_rate": round(rate(boolish(row.get("downstream_contamination")) for row in group if row.get("domain") == "research_assistant"), 4),
                "confirmed_hypothesis_promotion_rate": round(rate(boolish(row.get("confirmed_hypothesis_promoted")) for row in verified), 4),
                "tentative_overblocking_rate": round(rate(boolish(row.get("tentative_overblocked")) for row in verified), 4),
            }
        )
    return output


def build_poor_cases(
    results: list[dict[str, Any]],
    case_scores: list[dict[str, Any]],
    memory_outputs: list[dict[str, Any]],
    downstream_outputs: list[dict[str, Any]],
    benchmark: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    case_map = {(row.get("item_id"), row.get("method")): row for row in case_scores}
    memory_map = {(row.get("item_id"), row.get("method")): row for row in memory_outputs}
    downstream_map = {(row.get("item_id"), row.get("method")): row for row in downstream_outputs}
    poor: list[dict[str, Any]] = []
    for row in results:
        label = str(row.get("downstream_label") or row.get("auto_label") or "")
        is_poor = (
            label in POOR_LABELS
            or boolish(row.get("downstream_contamination"))
            or not boolish(row.get("trap_task_correct"))
            or not boolish(row.get("useful_memory_retention"))
        )
        if not is_poor:
            continue
        key = (row.get("item_id"), row.get("method"))
        item = benchmark.get(str(row.get("item_id")), {})
        future_task = item.get("future_task", {}) if isinstance(item.get("future_task"), dict) else {}
        case_row = case_map.get(key, {})
        memory_row = memory_map.get(key, {})
        downstream_row = downstream_map.get(key, {})
        memories = row.get("memories") or memory_row.get("memories") or case_row.get("memories") or []
        answer = row.get("trap_answer") or case_row.get("answer") or downstream_row.get("trap_answer") or ""
        poor.append(
            {
                "item_id": row.get("item_id", ""),
                "domain": row.get("domain", ""),
                "case_subtype": row.get("case_subtype", ""),
                "method": row.get("method", ""),
                "auto_label": label,
                "contaminated": boolish(row.get("downstream_contamination")),
                "false_promotion": boolish(row.get("false_evidence_promotion")),
                "trap_task_accuracy": boolish(row.get("trap_task_correct")),
                "verified_control_accuracy": boolish(row.get("control_task_correct")),
                "memory_text": clean_text(compact_memory(memories)),
                "answer": clean_text(str(answer)),
                "scoring_rationale": clean_text(str(row.get("scoring_rationale") or case_row.get("scoring_rationale", ""))),
                "current_evidence_summary": clean_text(str(future_task.get("current_evidence", ""))),
                "false_hypothesis": clean_text(str(item.get("trap_hypothesis", ""))),
                "true_answer": clean_text(str(future_task.get("correct_answer") or item.get("ground_truth", ""))),
            }
        )
    return poor


def render_report(
    summary: list[dict[str, Any]],
    filtered_results: list[dict[str, Any]],
    poor_cases: list[dict[str, Any]],
    original_summary_rows: list[dict[str, str]],
    metadata: dict[str, Any],
    benchmark: dict[str, dict[str, Any]],
    run_dir: Path,
) -> str:
    by = {row["method"]: row for row in summary}
    no_memory = by.get("no_memory", {})
    naive = by.get("naive", {})
    enforced = by.get("evidence_labeled_enforced", {})
    stable_only = by.get("evidence_labeled_stable_only", {})
    self_check = by.get("current_evidence_self_check", {})
    quote = by.get("quote_required", {})
    source = by.get("source_aware", {})
    item_rows = {str(row.get("item_id")): row for row in filtered_results if row.get("item_id")}
    best_contam = min(summary, key=lambda row: (float(row["downstream_contamination_rate"]), -float(row["useful_memory_retention"])))
    best_memory = min(
        [row for row in summary if row["method"] != "no_memory"],
        key=lambda row: (-(float(row["useful_memory_retention"])), float(row["downstream_contamination_rate"])),
    )
    best_balanced = min(
        [row for row in summary if row["method"] != "no_memory" and float(row["useful_memory_retention"]) >= 0.90],
        key=lambda row: (float(row["downstream_contamination_rate"]), -float(row["useful_memory_retention"])),
    )
    increased = [
        row
        for row in summary
        if row["method"] != "no_memory"
        and float(row["downstream_contamination_rate"]) > float(no_memory.get("downstream_contamination_rate", 0.0))
    ]
    poor_counts = Counter(str(row["method"]) for row in poor_cases)
    domain_counts = Counter(str(row.get("domain")) for row in item_rows.values())
    subtype_counts = Counter(str(row.get("case_subtype")) for row in item_rows.values())
    original_reflection = next((row for row in original_summary_rows if row.get("method") == "reflection"), {})

    clean_claim = clean_scientific_claim(summary)
    lines = [
        "# Non-Reflection Qwen14B Analysis",
        "",
        "## Scope",
        "",
        "The non-reflection analysis tests whether ordinary memory extraction and evidence-aware variants promote unsupported hypotheses and contaminate later answers. Reflection is excluded because bad reflective lessons are not identical to false evidence promotion.",
        "",
        f"- source run: `{run_dir}`",
        f"- model: `{metadata.get('model', metadata.get('hf_model', ''))}`",
        f"- backend: `{metadata.get('backend', '')}`",
        f"- n_items: `{metadata.get('n_items', '')}`",
        f"- run_role: `{metadata.get('run_role', '')}`",
        f"- scientific_evidence: `{str(bool(metadata.get('scientific_evidence'))).lower()}`",
        f"- benchmark domain breakdown: `{dict(domain_counts)}`",
        f"- benchmark subtype breakdown: `{dict(subtype_counts)}`",
        "",
        "## What Changes Without Reflection?",
        "",
        "Reflection is removed from all headline metrics, aggregate verdicts, and poor-case analysis. The original reflection row is retained only as secondary context:",
        f"- original reflection false promotion: `{original_reflection.get('false_evidence_promotion_rate', 'n/a')}`",
        f"- original reflection contamination: `{original_reflection.get('downstream_contamination_rate', 'n/a')}`",
        "",
        "The core result no longer relies on reflection: naive memory still shows high write-time false evidence promotion, while downstream harm is weaker and more mixed.",
        "",
        "## Main Non-Reflection Metrics",
        "",
        table(
            summary,
            [
                "method",
                "false_evidence_promotion_rate",
                "downstream_contamination_rate",
                "contamination_delta_vs_no_memory",
                "trap_task_accuracy",
                "useful_memory_retention",
                "confirmed_hypothesis_promotion_rate",
                "tentative_overblocking_rate",
            ],
        ),
        "",
        "## Direct Answers",
        "",
        f"1. **What happens if reflection is removed?** The headline phenomenon becomes narrower: write-time laundering remains strong for naive memory (`{naive.get('false_evidence_promotion_rate')}`), but naive downstream contamination is `{naive.get('downstream_contamination_rate')}`, equal to no-memory `{no_memory.get('downstream_contamination_rate')}`.",
        f"2. **Does naive memory still show false evidence promotion?** Yes. Naive false evidence promotion is `{naive.get('false_evidence_promotion_rate')}`.",
        f"3. **Does any non-reflection method increase contamination relative to no-memory?** Yes: {', '.join(row['method'] for row in increased) if increased else 'none'}.",
        f"4. **Which method performs best on contamination?** `{best_contam['method']}` at `{best_contam['downstream_contamination_rate']}`.",
        f"5. **Which method performs best while preserving useful memory?** `{best_balanced['method']}` is the best contamination/retention tradeoff among methods with retention >= 0.90: contamination `{best_balanced['downstream_contamination_rate']}`, retention `{best_balanced['useful_memory_retention']}`. The highest raw retention is `{best_memory['method']}` at `{best_memory['useful_memory_retention']}`.",
        f"6. **Does evidence-labeled-enforced still fail?** Yes. It has false promotion `{enforced.get('false_evidence_promotion_rate')}`, but contamination `{enforced.get('downstream_contamination_rate')}`, worse than no-memory `{no_memory.get('downstream_contamination_rate')}`, and confirmed-hypothesis promotion `{enforced.get('confirmed_hypothesis_promotion_rate')}` with overblocking `{enforced.get('tentative_overblocking_rate')}`.",
        f"7. **Is stable_only or current_evidence_self_check more promising?** `current_evidence_self_check` is better on contamination (`{self_check.get('downstream_contamination_rate')}` vs stable_only `{stable_only.get('downstream_contamination_rate')}`), while stable_only is better on retention (`{stable_only.get('useful_memory_retention')}` vs self-check `{self_check.get('useful_memory_retention')}`). Self-check is the cleaner next intervention for contamination; stable-only is a retention-heavy baseline that still overblocks confirmed hypotheses.",
        f"8. **Does quote_required already solve the problem?** No. It nearly solves write-time false promotion (`{quote.get('false_evidence_promotion_rate')}`), but still contaminates at `{quote.get('downstream_contamination_rate')}`, above no-memory by `{quote.get('contamination_delta_vs_no_memory')}`.",
        f"9. **Cleanest revised scientific claim.** {clean_claim}",
        "",
        "## Interpretation",
        "",
        interpretation(summary),
        "",
        "Labeling a memory as tentative does not prevent anchoring if the tentative memory is still shown downstream.",
        "",
        "The write-time laundering phenomenon appears stronger than the downstream memory-caused harm in this run.",
        "",
        "## Poor-Case Summary",
        "",
        f"- total poor rows excluding reflection: `{len(poor_cases)}`",
        f"- poor rows by method: `{dict(poor_counts)}`",
        f"- detailed CSV: `poor_cases_nonreflection.csv`",
        f"- grouped examples: `poor_cases_by_method.md`",
        "",
        "## Revised Verdict",
        "",
        revised_verdict(summary),
        "",
    ]
    return "\n".join(lines)


def render_poor_cases_by_method(poor_cases: list[dict[str, Any]]) -> str:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in poor_cases:
        by_method[str(row["method"])].append(row)
    lines = [
        "# Poor Cases By Method",
        "",
        "Reflection is excluded. Examples prioritize rows that endorsed or were contaminated by the unsupported hypothesis.",
        "",
    ]
    for method in HEADLINE_METHODS:
        group = by_method.get(method, [])
        if not group:
            continue
        lines.extend([f"## {method}", "", f"- total poor rows: `{len(group)}`", ""])
        for row in sorted(group, key=poor_case_sort_key)[:5]:
            lines.extend(
                [
                    f"### {row['item_id']} ({row['domain']}, {row['case_subtype']})",
                    "",
                    f"- label: `{row['auto_label']}`",
                    f"- contaminated: `{row['contaminated']}`",
                    f"- false promotion: `{row['false_promotion']}`",
                    f"- trap correct: `{row['trap_task_accuracy']}`",
                    f"- control correct: `{row['verified_control_accuracy']}`",
                    f"- unsupported hypothesis: {row['false_hypothesis']}",
                    f"- true answer: {row['true_answer']}",
                    f"- scoring rationale: {row['scoring_rationale']}",
                    f"- memory: {clip(str(row['memory_text']), 500) or '(none)'}",
                    f"- answer: {clip(str(row['answer']), 700) or '(none)'}",
                    "",
                ]
            )
    return "\n".join(lines)


def clean_scientific_claim(summary: list[dict[str, Any]]) -> str:
    by = {row["method"]: row for row in summary}
    naive = by.get("naive", {})
    no_memory = by.get("no_memory", {})
    if float(naive.get("false_evidence_promotion_rate", 0.0)) >= 0.30 and float(naive.get("downstream_contamination_rate", 0.0)) <= float(
        no_memory.get("downstream_contamination_rate", 0.0)
    ):
        return (
            "Ordinary memory extraction can launder unsupported hypotheses into stable memories, but this Qwen14B run does not show a clean naive-memory downstream harm above no-memory. "
            "The write-time laundering phenomenon appears stronger than the downstream memory-caused harm in this run."
        )
    return "Ordinary memory extraction can launder unsupported hypotheses into stable memories, with downstream harm depending on retrieval and answer-prompt design."


def interpretation(summary: list[dict[str, Any]]) -> str:
    by = {row["method"]: row for row in summary}
    naive = by.get("naive", {})
    no_memory = by.get("no_memory", {})
    enforced = by.get("evidence_labeled_enforced", {})
    source = by.get("source_aware", {})
    quote = by.get("quote_required", {})
    parts = [
        f"Naive memory has high false evidence promotion (`{naive.get('false_evidence_promotion_rate')}`), so the write-time laundering mechanism survives without reflection.",
        f"Naive downstream contamination is `{naive.get('downstream_contamination_rate')}`, compared with no-memory `{no_memory.get('downstream_contamination_rate')}`; this is not a clean increase.",
        f"Evidence-labeled-enforced eliminates stable false promotion but still contaminates at `{enforced.get('downstream_contamination_rate')}` and overblocks verified hypotheses at `{enforced.get('tentative_overblocking_rate')}`.",
        f"Source-aware and quote-required reduce false promotion (`{source.get('false_evidence_promotion_rate')}` and `{quote.get('false_evidence_promotion_rate')}`) but do not eliminate downstream contamination.",
    ]
    return " ".join(parts)


def revised_verdict(summary: list[dict[str, Any]]) -> str:
    by = {row["method"]: row for row in summary}
    naive = by.get("naive", {})
    no_memory = by.get("no_memory", {})
    enforced = by.get("evidence_labeled_enforced", {})
    self_check = by.get("current_evidence_self_check", {})
    if float(naive.get("false_evidence_promotion_rate", 0.0)) >= 0.30 and float(naive.get("downstream_contamination_rate", 0.0)) <= float(
        no_memory.get("downstream_contamination_rate", 0.0)
    ):
        return (
            "REDESIGN_NARROW: keep the paper idea focused on unsupported-hypothesis memory promotion, but do not claim strong naive-memory downstream harm from this run. "
            f"Evidence-labeled-enforced fails as an intervention; current-evidence self-check is the strongest contamination baseline at `{self_check.get('downstream_contamination_rate')}`."
        )
    if float(enforced.get("downstream_contamination_rate", 1.0)) > float(no_memory.get("downstream_contamination_rate", 0.0)):
        return "REDESIGN: evidence-labeled enforcement does not control downstream anchoring."
    return "CONTINUE_WEAK: non-reflection evidence supports follow-up, but manual audit and replication are required."


def poor_case_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    label = str(row.get("auto_label", ""))
    contaminated = boolish(row.get("contaminated"))
    false_promotion = boolish(row.get("false_promotion"))
    label_rank = 0 if label in {"contaminated", "mixed_endorsed_trap"} else 1
    return (label_rank, 0 if contaminated else 1, 0 if false_promotion else 1, str(row.get("item_id", "")))


def compact_memory(memories: Any) -> str:
    if isinstance(memories, str):
        return memories
    if not memories:
        return ""
    chunks: list[str] = []
    if isinstance(memories, list):
        for memory in memories:
            if isinstance(memory, dict):
                chunks.append(str(memory.get("memory", memory)))
            else:
                chunks.append(str(memory))
    else:
        chunks.append(str(memories))
    return " | ".join(chunks)


def table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] + ["---:"] * (len(fields) - 1)) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def rate(values: Any) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(1 for value in collected if boolish(value)) / len(collected)


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


if __name__ == "__main__":
    main()
