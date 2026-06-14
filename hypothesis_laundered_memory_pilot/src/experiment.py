from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .benchmark import load_benchmark
from .data_model import BenchmarkItem, MemoryRecord, ScoreRecord
from .llm import LLMClient
from .prompts import downstream_control_prompt, downstream_trap_prompt, memory_prompt
from .scorers import false_evidence_promotion, score_control_answer, score_trap_answer
from .utils import ensure_dir, parse_json_lenient, write_jsonl


ALL_METHODS = ["no_memory", "naive", "reflection", "evidence_labeled"]
ALL_DOMAINS = ["coding", "data_analysis"]


def run_experiment(
    model: str,
    n: int,
    domains: list[str],
    methods: list[str],
    out: str | Path,
    mock: bool,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> tuple[list[ScoreRecord], list[BenchmarkItem]]:
    out_dir = ensure_dir(out)
    data_path = Path(__file__).resolve().parents[1] / "data" / "benchmark_seed.json"
    items = load_benchmark(data_path, n=n, domains=domains, seed=seed)
    client = LLMClient(model=model, out_dir=out_dir, mock=mock, temperature=temperature, max_tokens=max_tokens)

    memory_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    scores: list[ScoreRecord] = []

    for item in items:
        for method in methods:
            memories = [] if method == "no_memory" else extract_memories(client, method, item)
            if method != "no_memory":
                memory_rows.append(
                    {
                        "item_id": item.id,
                        "domain": item.domain,
                        "method": method,
                        "memories": [asdict(m) for m in memories],
                    }
                )
            trap_prompt = downstream_trap_prompt(method, item, memories)
            if mock:
                trap_prompt = with_mock_meta(trap_prompt, item)
            trap_answer = client.complete(
                "You are a careful analyst.",
                trap_prompt,
                purpose=f"{method}:trap",
            )
            control_memories = list(memories)
            if method != "no_memory" and not any(m.memory == item.verified_memory_control.memory for m in control_memories):
                control_memories.append(
                    MemoryRecord(
                        memory=item.verified_memory_control.memory,
                        evidence_type="Observed",
                        status="Stable",
                        supporting_evidence="Verified control memory supplied by benchmark.",
                        reuse_rule="Use when relevant to control future task.",
                    )
                )
            control_prompt = downstream_control_prompt(method, item, control_memories)
            if mock:
                control_prompt = with_mock_meta(control_prompt, item)
            control_answer = client.complete(
                "You are a careful analyst.",
                control_prompt,
                purpose=f"{method}:control",
            )
            downstream_rows.append(
                {
                    "item_id": item.id,
                    "domain": item.domain,
                    "method": method,
                    "trap_answer": trap_answer,
                    "control_answer": control_answer,
                }
            )
            contaminated, trap_correct, mixed = score_trap_answer(trap_answer, item)
            control_correct, retained = score_control_answer(control_answer, item)
            scores.append(
                ScoreRecord(
                    item_id=item.id,
                    domain=item.domain,
                    method=method,
                    false_evidence_promotion=false_evidence_promotion(method, memories, item),
                    downstream_contamination=contaminated,
                    trap_task_correct=trap_correct,
                    control_task_correct=control_correct,
                    useful_memory_retention=retained,
                    mixed=mixed,
                    memories=[asdict(m) for m in memories],
                    trap_answer=trap_answer,
                    control_answer=control_answer,
                )
            )

    write_jsonl(out_dir / "memory_outputs.jsonl", memory_rows)
    write_jsonl(out_dir / "downstream_outputs.jsonl", downstream_rows)
    write_jsonl(out_dir / "results_raw.jsonl", [score.to_dict() for score in scores])
    return scores, items


def extract_memories(client: LLMClient, method: str, item: BenchmarkItem) -> list[MemoryRecord]:
    system, user = memory_prompt(method, item)
    if client.mock:
        user = with_mock_meta(user, item)
    raw = client.complete(system, user, purpose=f"{method}:memory")
    parsed, error = parse_json_lenient(raw)
    if parsed is None:
        return [MemoryRecord(memory=raw, raw={"parse_error": error})]
    if method in {"naive", "reflection"}:
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
                        evidence_type=str(row.get("evidence_type", "Inferred")),
                        status=str(row.get("status", "Tentative")),
                        supporting_evidence=str(row.get("supporting_evidence", "")),
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
        f"MOCK_CONTROL: {item.verified_memory_control.memory}"
    )
