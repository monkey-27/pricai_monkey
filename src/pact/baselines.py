"""Offline baselines for the PACT pilot."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from pact.schema import Prediction, ProspectiveActionContract

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "before",
    "for",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "or",
    "the",
    "this",
    "to",
    "when",
    "with",
}

FAMILY_HINTS: dict[str, set[str]] = {
    "research_ideation": {"pricai", "paper", "neurips", "idea", "submission", "llm", "benchmark", "tool"},
    "food_safety": {"food", "dessert", "thai", "potluck", "granola", "bakery", "cake", "restaurant", "takeout"},
    "code_security": {"flask", "auth", "login", "secret", "cookie", "session", "middleware"},
    "travel_planning": {"japan", "india", "nepal", "tokyo", "seoul", "taipei", "istanbul", "baku", "doha", "trip", "layover"},
    "medical_caution": {"chest", "antibiotic", "rash", "headache", "medicine", "symptom", "allergic", "health"},
    "email_rewriting": {"email", "manager", "client", "slack", "coworker", "recruiter", "rewrite", "respond"},
    "benchmark_novelty": {"benchmark", "suite", "dataset", "tasks", "measure", "browser", "stand", "research"},
    "scheduling": {"schedule", "meeting", "appointment", "call", "flight", "calendar", "tomorrow", "2pm"},
    "current_facts": {"current", "latest", "today", "new", "still", "last", "night", "won", "release", "ceo", "bitcoin"},
    "admissions_cs": {"cs", "uiuc", "berkeley", "eecs", "purdue", "college", "admissions", "major", "tech"},
}

NEAR_MISS_PATTERNS = (
    "what is ",
    "what does ",
    "explain ",
    "why do ",
    "why does ",
    "summarize the difference",
    "give tips",
)
ALREADY_PATTERNS = ("already", "just checked", "has already", "were checked", "has already been")
CONFLICT_PATTERNS = (
    "do not mention",
    "make it happen",
    "disabling",
    "with certainty",
    "harmless",
    "severe",
    "two separate meetings at 2pm",
    "guest with a severe",
)
WRONG_SCOPE_HINTS = {
    "research_ideation": {"history essay", "railroads"},
    "food_safety": {"photo", "poster"},
    "code_security": {"health-check", "public"},
    "travel_planning": {"boston", "vermont", "road trip"},
    "medical_caution": {"fictional", "scene"},
    "email_rewriting": {"fantasy", "tavern", "archaic"},
    "benchmark_novelty": {"laptops", "battery", "keyboard"},
    "scheduling": {"cron", "syntax"},
    "current_facts": {"1900"},
    "admissions_cs": {"history phd"},
}


@dataclass(frozen=True)
class EpisodeInput:
    """Prediction-safe episode view.

    It deliberately excludes labels and scoring-only keyword fields.
    """

    episode_id: str
    history_summary: str
    current_query: str

    @property
    def text(self) -> str:
        return f"{self.history_summary} {self.current_query}"


class Method(Protocol):
    name: str

    def predict(
        self, contracts: list[ProspectiveActionContract], episode: EpisodeInput
    ) -> Prediction:
        ...


def public_episode(episode: object) -> EpisodeInput:
    return EpisodeInput(
        episode_id=getattr(episode, "episode_id"),
        history_summary=getattr(episode, "history_summary"),
        current_query=getattr(episode, "current_query"),
    )


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def text_has_any(text: str, phrases: set[str] | tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def overlap_score(a: str, b: str) -> float:
    left, right = set(tokenize(a)), set(tokenize(b))
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def best_contract_by_overlap(
    contracts: list[ProspectiveActionContract], text: str
) -> tuple[ProspectiveActionContract, float]:
    scored = [(contract, overlap_score(contract.raw_text, text)) for contract in contracts]
    scored.sort(key=lambda item: (item[1], item[0].contract_id), reverse=True)
    return scored[0]


def family_hint_score(contract: ProspectiveActionContract, text: str) -> float:
    tokens = set(tokenize(text))
    hints = FAMILY_HINTS.get(contract.family, set())
    if not hints:
        return 0.0
    return min(1.0, len(tokens & hints) / 2.0)


def compiled_action(contract: ProspectiveActionContract) -> str:
    return f"Required plan: {contract.action} Verification check: {contract.check}"


def deterministic_response(contract: ProspectiveActionContract, prefix: str = "") -> str:
    return (
        f"{prefix}{compiled_action(contract)} I will handle the user's request only after "
        f"that contracted step is addressed."
    ).strip()


def simple_contract_checker(contract: ProspectiveActionContract, response: str) -> bool:
    response_tokens = set(tokenize(response))
    action_tokens = [token for token in tokenize(contract.action) if len(token) > 4]
    if not action_tokens:
        return False
    return len(response_tokens & set(action_tokens)) >= min(2, len(set(action_tokens)))


class NoMemoryBaseline:
    name = "NoMemoryBaseline"

    def predict(self, contracts: list[ProspectiveActionContract], episode: EpisodeInput) -> Prediction:
        return Prediction(
            method=self.name,
            episode_id=episode.episode_id,
            contract_id="none",
            predicted_state="suppress",
            confidence=0.9,
            response="I will answer the request in a generic way without using stored commitments.",
            satisfied=False,
            repaired=False,
            rationale="No stored memory is consulted.",
        )


class KeywordTriggerBaseline:
    name = "KeywordTriggerBaseline"

    def __init__(self, min_overlap: int = 2) -> None:
        self.min_overlap = min_overlap

    def predict(self, contracts: list[ProspectiveActionContract], episode: EpisodeInput) -> Prediction:
        query_tokens = set(tokenize(episode.text))
        best = contracts[0]
        best_count = -1
        for contract in contracts:
            contract_tokens = set(tokenize(f"{contract.family} {contract.cue} {contract.action}"))
            count = len(query_tokens & contract_tokens)
            if count > best_count:
                best, best_count = contract, count
        fires = best_count >= self.min_overlap
        response = deterministic_response(best) if fires else "No trigger words are strong enough; proceeding normally."
        return Prediction(
            method=self.name,
            episode_id=episode.episode_id,
            contract_id=best.contract_id if fires else "none",
            predicted_state="fire" if fires else "suppress",
            confidence=min(0.95, best_count / 5.0),
            response=response,
            satisfied=simple_contract_checker(best, response) if fires else False,
            repaired=False,
            rationale=f"keyword overlap={best_count}",
        )


class TfidfMemoryBaseline:
    name = "TfidfMemoryBaseline"

    def __init__(self, threshold: float = 0.19) -> None:
        self.threshold = threshold

    def predict(self, contracts: list[ProspectiveActionContract], episode: EpisodeInput) -> Prediction:
        contract, score = self._best(contracts, episode.text)
        fires = score >= self.threshold
        response = (
            deterministic_response(contract, prefix="Retrieved memory. ")
            if fires
            else "Retrieved memories do not cross threshold; proceeding normally."
        )
        return Prediction(
            method=self.name,
            episode_id=episode.episode_id,
            contract_id=contract.contract_id if fires else "none",
            predicted_state="fire" if fires else "suppress",
            confidence=score,
            response=response,
            satisfied=simple_contract_checker(contract, response) if fires else False,
            repaired=False,
            rationale=f"top-1 memory similarity={score:.3f}",
        )

    def _best(
        self, contracts: list[ProspectiveActionContract], text: str
    ) -> tuple[ProspectiveActionContract, float]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            docs = [contract.raw_text for contract in contracts] + [text]
            matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)
            sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
            idx = int(sims.argmax())
            return contracts[idx], float(sims[idx])
        except Exception:
            return best_contract_by_overlap(contracts, text)


class ContractPromptHeuristicBaseline:
    name = "ContractPromptHeuristicBaseline"

    def predict(self, contracts: list[ProspectiveActionContract], episode: EpisodeInput) -> Prediction:
        contract, score = best_contract_by_overlap(contracts, episode.text)
        score = max(score, family_hint_score(contract, episode.text))
        lower = episode.text.lower()
        if text_has_any(lower, ALREADY_PATTERNS):
            state = "already_satisfied"
        elif text_has_any(lower, CONFLICT_PATTERNS):
            state = "conflict"
        elif text_has_any(lower, NEAR_MISS_PATTERNS) or text_has_any(
            lower, WRONG_SCOPE_HINTS.get(contract.family, set())
        ):
            state = "suppress"
        else:
            state = "fire" if score >= 0.35 else "suppress"
        response = deterministic_response(contract) if state in {"fire", "conflict"} else "No contracted action is needed."
        return Prediction(
            method=self.name,
            episode_id=episode.episode_id,
            contract_id=contract.contract_id if state != "suppress" else "none",
            predicted_state=state,
            confidence=min(0.95, score + 0.15),
            response=response,
            satisfied=simple_contract_checker(contract, response) if state == "fire" else False,
            repaired=False,
            rationale=f"heuristic contract score={score:.3f}; state={state}",
        )


def get_baseline(name: str) -> Method:
    lookup: dict[str, Method] = {
        "NoMemoryBaseline": NoMemoryBaseline(),
        "KeywordTriggerBaseline": KeywordTriggerBaseline(),
        "TfidfMemoryBaseline": TfidfMemoryBaseline(),
        "ContractPromptHeuristicBaseline": ContractPromptHeuristicBaseline(),
    }
    return lookup[name]

