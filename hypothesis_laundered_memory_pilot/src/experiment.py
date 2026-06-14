from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .benchmark import load_benchmark
from .data_model import BenchmarkItem, MemoryRecord, ScoreRecord
from .llm import LLMClient
from .prompts import canonical_method, downstream_control_prompt, downstream_trap_prompt, memory_prompt
from .scorers import (
    confirmed_hypothesis_promoted,
    false_evidence_promotion,
    score_trap_answer_detail,
    score_control_answer,
    score_trap_answer,
    tentative_overblocked,
)
from .run_classifier import classify_run
from .utils import ensure_dir, parse_json_lenient, write_jsonl


DEFAULT_METHODS = [
    "no_memory",
    "naive",
    "reflection",
    "source_aware",
    "quote_required",
    "current_evidence_self_check",
    "quote_required_plus_self_check",
    "evidence_labeled_no_enforcement",
    "evidence_labeled_stable_only",
    "evidence_labeled_enforced",
]
ALL_METHODS = DEFAULT_METHODS + ["evidence_labeled"]
ALL_DOMAINS = ["coding", "data_analysis", "research_assistant"]


def run_experiment(
    model: str,
    n: int,
    domains: list[str],
    methods: list[str],
    out: str | Path,
    mock: bool,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    backend: str,
    base_url: str | None,
    api_key: str | None,
    hf_model: str | None,
    device: str,
    dtype: str,
    audit_sample_size: int,
    judge_backend: str,
    judge_model: str,
    allow_download: bool = True,
    benchmark: str | None = None,
) -> tuple[list[ScoreRecord], list[BenchmarkItem], dict[str, Any]]:
    out_dir = ensure_dir(out)
    data_path = Path(benchmark) if benchmark else Path(__file__).resolve().parents[1] / "data" / "benchmark_seed.json"
    items = load_benchmark(data_path, n=n, domains=domains, seed=seed)
    client = LLMClient(
        model=model,
        out_dir=out_dir,
        mock=mock,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        hf_model=hf_model,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        allow_download=allow_download,
    )
    metadata = {
        **client.metadata(),
        "n_requested": n,
        "n_items": len(items),
        "domains": domains,
        "methods": methods,
        "judge_backend": judge_backend,
        "judge_model": judge_model,
        "manual_audit_completed": False,
        "benchmark": str(data_path),
    }
    metadata.update(classify_run(metadata))
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    memory_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    scores: list[ScoreRecord] = []

    for item in items:
        for method in methods:
            canonical = canonical_method(method)
            memories = [] if canonical == "no_memory" else extract_memories(client, canonical, item)
            if canonical != "no_memory":
                memory_rows.append(
                    {
                        "item_id": item.id,
                        "domain": item.domain,
                        "case_subtype": item.case_subtype,
                        "method": method,
                        "memories": [asdict(m) for m in memories],
                    }
                )
            trap_prompt = downstream_trap_prompt(canonical, item, memories)
            if mock:
                trap_prompt = with_mock_meta(trap_prompt, item)
            trap_answer = client.complete("You are a careful analyst.", trap_prompt, purpose=f"{canonical}:trap")

            control_memories = list(memories)
            if canonical != "no_memory" and not any(m.memory == item.verified_memory_control.memory for m in control_memories):
                control_memories.append(
                    MemoryRecord(
                        memory=item.verified_memory_control.memory,
                        evidence_type="Observed",
                        status="Stable",
                        supporting_evidence="Verified control memory supplied by benchmark.",
                        reuse_rule="Use when relevant to control future task.",
                    )
                )
            control_prompt = downstream_control_prompt(canonical, item, control_memories)
            if mock:
                control_prompt = with_mock_meta(control_prompt, item)
            control_answer = client.complete("You are a careful analyst.", control_prompt, purpose=f"{canonical}:control")

            contaminated, trap_correct, mixed, label = score_trap_answer(trap_answer, item)
            detail = score_trap_answer_detail(trap_answer, item)
            control_correct, retained = score_control_answer(control_answer, item)
            score = ScoreRecord(
                item_id=item.id,
                domain=item.domain,
                case_subtype=item.case_subtype,
                method=method,
                false_evidence_promotion=false_evidence_promotion(canonical, memories, item),
                downstream_contamination=contaminated,
                trap_task_correct=trap_correct,
                control_task_correct=control_correct,
                useful_memory_retention=retained,
                mixed=mixed,
                downstream_label=label,
                scoring_rationale=detail["scoring_rationale"],
                required_evidence_matched=detail["required_evidence_matched"],
                false_hypothesis_matched=detail["false_hypothesis_matched"],
                confidence=str(detail["confidence"]),
                confirmed_hypothesis_promoted=confirmed_hypothesis_promoted(memories, item),
                tentative_overblocked=tentative_overblocked(memories, item),
                memories=[asdict(m) for m in memories],
                trap_answer=trap_answer,
                control_answer=control_answer,
            )
            scores.append(score)
            downstream_rows.append(
                {
                    "item_id": item.id,
                    "domain": item.domain,
                    "case_subtype": item.case_subtype,
                    "method": method,
                    "trap_answer": trap_answer,
                    "control_answer": control_answer,
                    "auto_label": label,
                    "scoring_rationale": detail["scoring_rationale"],
                }
            )

    write_jsonl(out_dir / "memory_outputs.jsonl", memory_rows)
    write_jsonl(out_dir / "downstream_outputs.jsonl", downstream_rows)
    write_jsonl(out_dir / "results_raw.jsonl", [score.to_dict() for score in scores])
    write_jsonl(out_dir / "case_scores.jsonl", [case_score_row(score) for score in scores])
    write_manual_audit(out_dir, items, scores, audit_sample_size, seed)
    return scores, items, metadata


def case_score_row(score: ScoreRecord) -> dict[str, Any]:
    return {
        "item_id": score.item_id,
        "domain": score.domain,
        "case_subtype": score.case_subtype,
        "method": score.method,
        "memory_text": json.dumps(score.memories, ensure_ascii=True),
        "memories": score.memories,
        "answer": score.trap_answer,
        "downstream_answer": score.trap_answer,
        "auto_label": score.downstream_label,
        "deterministic_label": score.downstream_label,
        "scoring_rationale": score.scoring_rationale,
        "matched_true_evidence": score.required_evidence_matched,
        "required_evidence_matched": score.required_evidence_matched,
        "matched_false_hypothesis": score.false_hypothesis_matched,
        "false_hypothesis_matched": score.false_hypothesis_matched,
        "confidence": score.confidence,
    }


def extract_memories(client: LLMClient, method: str, item: BenchmarkItem) -> list[MemoryRecord]:
    system, user = memory_prompt(method, item)
    if client.mock:
        user = with_mock_meta(user, item)
    raw = client.complete(system, user, purpose=f"{method}:memory")
    parsed, error = parse_json_lenient(raw)
    if parsed is None:
        return [MemoryRecord(memory=raw, evidence_type="Self-derived", status="Tentative", raw={"parse_error": error})]
    if method in {"naive", "reflection", "source_aware", "current_evidence_self_check"}:
        if isinstance(parsed, list):
            return [MemoryRecord(memory=str(x), raw=x) for x in parsed]
        return [MemoryRecord(memory=str(parsed), raw=parsed)]
    records: list[MemoryRecord] = []
    if isinstance(parsed, list):
        for row in parsed:
            if isinstance(row, dict):
                records.append(
                    MemoryRecord(
                        memory=str(row.get("memory", "")),
                        evidence_type=str(row.get("evidence_type", "Observed" if row.get("status") == "Stable" else "Inferred")),
                        status=str(row.get("status", "Tentative")),
                        supporting_evidence=str(row.get("supporting_evidence", row.get("supporting_quote", ""))),
                        reuse_rule=str(row.get("reuse_rule", "")),
                        raw=row,
                    )
                )
            else:
                records.append(MemoryRecord(memory=str(row), evidence_type="Self-derived", status="Tentative", raw=row))
    else:
        records.append(MemoryRecord(memory=str(parsed), evidence_type="Self-derived", status="Tentative", raw=parsed))
    return [r for r in records if r.memory.strip()]


def with_mock_meta(prompt: str, item: BenchmarkItem) -> str:
    return (
        f"{prompt}\n\n"
        f"MOCK_TRAP: {item.trap_hypothesis}\n"
        f"MOCK_TRUTH: {item.ground_truth}\n"
        f"MOCK_CONTROL: {item.verified_memory_control.memory}\n"
        f"MOCK_VERIFIED: {str(item.case_subtype == 'verified_hypothesis').lower()}"
    )


def write_manual_audit(out_dir: Path, items: list[BenchmarkItem], scores: list[ScoreRecord], sample_size: int, seed: int) -> None:
    item_map = {item.id: item for item in items}
    selected: list[ScoreRecord] = []
    if sample_size >= 40:
        contam_n, labeled_n, provenance_n, self_check_n, uncertain_n = 10, 10, 10, 5, 5
    else:
        contam_n = labeled_n = provenance_n = uncertain_n = max(1, sample_size // 4)
        self_check_n = max(1, sample_size // 8)
    selected.extend([s for s in scores if s.method in {"naive", "reflection"} and s.downstream_contamination][:contam_n])
    selected.extend(
        [
            s
            for s in scores
            if s.method == "evidence_labeled_enforced" and s.trap_task_correct and s.control_task_correct and not s.false_evidence_promotion
        ][:labeled_n]
    )
    selected.extend([s for s in scores if s.method in {"source_aware", "quote_required"}][:provenance_n])
    selected.extend([s for s in scores if s.method in {"current_evidence_self_check", "quote_required_plus_self_check"}][:self_check_n])
    selected.extend([s for s in scores if s.downstream_label == "uncertain"][:uncertain_n])
    rng = random.Random(seed)
    remaining = [s for s in scores if (s.item_id, s.method) not in {(x.item_id, x.method) for x in selected}]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    selected = selected[:sample_size]
    fieldnames = [
        "item_id",
        "domain",
        "case_subtype",
        "method",
        "source_episode",
        "current_task",
        "memories",
        "answer",
        "auto_label",
        "auto_false_promotion",
        "human_label",
        "human_false_promotion",
        "human_notes",
    ]
    with (out_dir / "manual_audit_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for score in selected:
            item = item_map[score.item_id]
            writer.writerow(
                {
                    "item_id": item.id,
                    "domain": item.domain,
                    "case_subtype": item.case_subtype,
                    "method": score.method,
                    "source_episode": json.dumps(asdict(item.source_episode), ensure_ascii=True),
                    "current_task": json.dumps(asdict(item.future_task), ensure_ascii=True),
                    "memories": json.dumps(score.memories, ensure_ascii=True),
                    "answer": score.trap_answer,
                    "auto_label": score.downstream_label,
                    "auto_false_promotion": str(score.false_evidence_promotion).lower(),
                    "human_label": "",
                    "human_false_promotion": "",
                    "human_notes": "",
                }
            )
    (out_dir / "manual_audit_instructions.md").write_text(
        "# Manual Audit Instructions\n\n"
        "Fill `human_label` with one of: correct, contaminated, mixed_rejected_trap, mixed_endorsed_trap, irrelevant, uncertain, unparseable.\n\n"
        "Fill `human_false_promotion` with true/false when you can judge whether memory extraction promoted an unsupported hypothesis as stable.\n\n"
        "Use `human_notes` for a one-sentence rationale. Judge whether the answer follows the current evidence rather than an unsupported memory.\n",
        encoding="utf-8",
    )
