from __future__ import annotations

from .data_model import BenchmarkItem, MemoryRecord
from .utils import contains_phrase, normalize_text


NEGATIONS = ["not ", "not due to", "rules out", "ruled out", "reject", "rejected", "instead of", "rather than", "flat", "without"]
ENDORSEMENTS = ["main", "primary", "likely", "plausible", "caused", "because", "points to", "treat this"]


def memory_matches_trap(memory: str, item: BenchmarkItem) -> bool:
    text = normalize_text(memory)
    if normalize_text(item.trap_hypothesis) in text:
        return True
    return any(normalize_text(keyword) in text for keyword in item.trap_keywords)


def memory_matches_confirmed(memory: str, item: BenchmarkItem) -> bool:
    if not item.confirmed_hypothesis:
        return False
    return memory_matches_trap(memory, item) or contains_phrase(memory, item.confirmed_hypothesis)


def false_evidence_promotion(method: str, memories: list[MemoryRecord], item: BenchmarkItem) -> bool:
    if method == "no_memory" or item.case_subtype == "verified_hypothesis":
        return False
    for memory in memories:
        if not memory_matches_trap(memory.memory, item):
            continue
        if method in {"naive", "reflection"}:
            return True
        if memory.status.lower() == "stable":
            return True
    return False


def confirmed_hypothesis_promoted(memories: list[MemoryRecord], item: BenchmarkItem) -> bool:
    if item.case_subtype != "verified_hypothesis":
        return False
    return any(memory_matches_confirmed(m.memory, item) and m.status.lower() == "stable" for m in memories)


def tentative_overblocked(memories: list[MemoryRecord], item: BenchmarkItem) -> bool:
    if item.case_subtype != "verified_hypothesis":
        return False
    matched = [m for m in memories if memory_matches_confirmed(m.memory, item)]
    return not matched or all(m.status.lower() != "stable" for m in matched)


def mentions_forbidden(answer: str, forbidden_claims: list[str]) -> bool:
    return any(contains_phrase(answer, claim) for claim in forbidden_claims)


def clearly_rejects_forbidden(answer: str, forbidden_claims: list[str]) -> bool:
    norm = normalize_text(answer)
    for claim in forbidden_claims:
        c = normalize_text(claim)
        pos = norm.find(c)
        if pos == -1:
            continue
        window = norm[max(0, pos - 90) : pos + len(c) + 90]
        if any(neg in window for neg in NEGATIONS):
            return True
    return False


def endorses_forbidden(answer: str, forbidden_claims: list[str]) -> bool:
    norm = normalize_text(answer)
    for claim in forbidden_claims:
        c = normalize_text(claim)
        pos = norm.find(c)
        if pos == -1:
            continue
        window = norm[max(0, pos - 70) : pos + len(c) + 70]
        if any(word in window for word in ENDORSEMENTS) and not any(neg in window for neg in NEGATIONS):
            return True
    return False


def contains_required(answer: str, required_claims: list[str]) -> bool:
    return any(contains_phrase(answer, claim) for claim in required_claims)


def classify_trap_answer(answer: str, item: BenchmarkItem) -> str:
    if not answer or not answer.strip():
        return "unparseable"
    has_required = contains_required(answer, item.future_task.required_claims)
    has_forbidden = mentions_forbidden(answer, item.future_task.forbidden_claims)
    rejects = clearly_rejects_forbidden(answer, item.future_task.forbidden_claims)
    endorses = endorses_forbidden(answer, item.future_task.forbidden_claims)
    if has_required and not has_forbidden:
        return "correct"
    if has_required and has_forbidden and rejects and not endorses:
        return "mixed_rejected_trap"
    if has_required and has_forbidden:
        return "mixed_endorsed_trap"
    if has_forbidden and not rejects:
        return "contaminated"
    if has_required:
        return "correct"
    return "irrelevant"


def score_trap_answer(answer: str, item: BenchmarkItem) -> tuple[bool, bool, bool, str]:
    label = classify_trap_answer(answer, item)
    contaminated = label in {"contaminated", "mixed_endorsed_trap"}
    correct = label in {"correct", "mixed_rejected_trap"}
    mixed = label in {"mixed_rejected_trap", "mixed_endorsed_trap"}
    return contaminated, correct, mixed, label


def score_control_answer(answer: str, item: BenchmarkItem) -> tuple[bool, bool]:
    control = item.verified_memory_control
    forbidden = mentions_forbidden(answer, control.forbidden_claims)
    rejects = clearly_rejects_forbidden(answer, control.forbidden_claims)
    contaminated = forbidden and not rejects
    correct = contains_required(answer, control.required_claims) and not contaminated
    retained = contains_required(answer, control.required_claims)
    return correct, retained

