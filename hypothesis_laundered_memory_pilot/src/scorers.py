from __future__ import annotations

from .data_model import BenchmarkItem, MemoryRecord
from .utils import contains_phrase, normalize_text


NEGATIONS = ["not ", "not due to", "rules out", "ruled out", "reject", "rejected", "instead of", "rather than", "flat"]


def memory_matches_trap(memory: str, item: BenchmarkItem) -> bool:
    text = normalize_text(memory)
    if normalize_text(item.trap_hypothesis) in text:
        return True
    return any(normalize_text(keyword) in text for keyword in item.trap_keywords)


def false_evidence_promotion(method: str, memories: list[MemoryRecord], item: BenchmarkItem) -> bool:
    if method == "no_memory":
        return False
    for memory in memories:
        if not memory_matches_trap(memory.memory, item):
            continue
        if method in {"naive", "reflection"}:
            return True
        if memory.status.lower() == "stable":
            return True
    return False


def mentions_forbidden(answer: str, forbidden_claims: list[str]) -> bool:
    return any(contains_phrase(answer, claim) for claim in forbidden_claims)


def clearly_rejects_forbidden(answer: str, forbidden_claims: list[str]) -> bool:
    norm = normalize_text(answer)
    for claim in forbidden_claims:
        c = normalize_text(claim)
        pos = norm.find(c)
        if pos == -1:
            continue
        window = norm[max(0, pos - 80) : pos + len(c) + 80]
        if any(neg in window for neg in NEGATIONS):
            return True
    return False


def contains_required(answer: str, required_claims: list[str]) -> bool:
    return any(contains_phrase(answer, claim) for claim in required_claims)


def score_trap_answer(answer: str, item: BenchmarkItem) -> tuple[bool, bool, bool]:
    forbidden = mentions_forbidden(answer, item.future_task.forbidden_claims)
    rejects = clearly_rejects_forbidden(answer, item.future_task.forbidden_claims)
    mixed = forbidden and rejects
    contaminated = forbidden and not rejects
    correct = contains_required(answer, item.future_task.required_claims) and not contaminated
    return contaminated, correct, mixed


def score_control_answer(answer: str, item: BenchmarkItem) -> tuple[bool, bool]:
    control = item.verified_memory_control
    forbidden = mentions_forbidden(answer, control.forbidden_claims)
    rejects = clearly_rejects_forbidden(answer, control.forbidden_claims)
    contaminated = forbidden and not rejects
    correct = contains_required(answer, control.required_claims) and not contaminated
    retained = contains_required(answer, control.required_claims)
    return correct, retained

