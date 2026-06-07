"""PACT method and ablations."""

from __future__ import annotations

from dataclasses import dataclass

from pact.baselines import (
    ALREADY_PATTERNS,
    CONFLICT_PATTERNS,
    FAMILY_HINTS,
    NEAR_MISS_PATTERNS,
    WRONG_SCOPE_HINTS,
    EpisodeInput,
    best_contract_by_overlap,
    compiled_action,
    deterministic_response,
    family_hint_score,
    overlap_score,
    simple_contract_checker,
    text_has_any,
)
from pact.schema import Prediction, ProspectiveActionContract


@dataclass(frozen=True)
class PamResult:
    state: str
    confidence: float
    rationale: str


class PACTFull:
    name = "PACTFull"

    def __init__(
        self,
        use_guard: bool = True,
        use_checker: bool = True,
        raw_memory: bool = False,
        name: str | None = None,
    ) -> None:
        self.use_guard = use_guard
        self.use_checker = use_checker
        self.raw_memory = raw_memory
        if name:
            self.name = name

    def predict(self, contracts: list[ProspectiveActionContract], episode: EpisodeInput) -> Prediction:
        contract, retrieval_score = self.retrieve(contracts, episode)
        pam = self.pam(contract, episode, retrieval_score)
        response = self.generate_response(contract, pam)
        satisfied = simple_contract_checker(contract, response) if pam.state == "fire" else False
        repaired = False
        if pam.state == "fire" and self.use_checker and not satisfied:
            response = self.repair(contract, response)
            repaired = True
            satisfied = simple_contract_checker(contract, response)
        return Prediction(
            method=self.name,
            episode_id=episode.episode_id,
            contract_id=contract.contract_id if pam.state != "suppress" else "none",
            predicted_state=pam.state,
            confidence=pam.confidence,
            response=response,
            satisfied=satisfied,
            repaired=repaired,
            rationale=pam.rationale,
        )

    def retrieve(
        self, contracts: list[ProspectiveActionContract], episode: EpisodeInput
    ) -> tuple[ProspectiveActionContract, float]:
        if self.raw_memory:
            return best_contract_by_overlap(contracts, episode.text)
        scored = []
        for contract in contracts:
            lexical = overlap_score(contract.raw_text, episode.text)
            hint = family_hint_score(contract, episode.text)
            scored.append((contract, max(lexical, hint)))
        scored.sort(key=lambda item: (item[1], item[0].contract_id), reverse=True)
        return scored[0]

    def pam(
        self, contract: ProspectiveActionContract, episode: EpisodeInput, retrieval_score: float
    ) -> PamResult:
        text = episode.text.lower()
        cue_match = retrieval_score
        guard_match = 0.0 if self.raw_memory else family_hint_score(contract, text)
        action_relevance = overlap_score(contract.action, text)
        near_miss_penalty = 0.35 if text_has_any(text, NEAR_MISS_PATTERNS) else 0.0
        wrong_scope_penalty = (
            0.45 if self.use_guard and text_has_any(text, WRONG_SCOPE_HINTS.get(contract.family, set())) else 0.0
        )
        already = text_has_any(text, ALREADY_PATTERNS)
        conflict = text_has_any(text, CONFLICT_PATTERNS)
        if already:
            return PamResult(
                state="already_satisfied",
                confidence=0.9,
                rationale="already-satisfied detector matched prior completed check",
            )
        if conflict:
            return PamResult(
                state="conflict",
                confidence=0.88,
                rationale="conflict detector matched request that violates the contract",
            )
        score = cue_match + 0.45 * guard_match + 0.25 * action_relevance - near_miss_penalty - wrong_scope_penalty
        threshold = 0.39 if self.use_guard else 0.25
        state = "fire" if score >= threshold else "suppress"
        return PamResult(
            state=state,
            confidence=max(0.05, min(0.98, score)),
            rationale=(
                f"cue={cue_match:.2f} guard={guard_match:.2f} action={action_relevance:.2f} "
                f"near_penalty={near_miss_penalty:.2f} wrong_scope_penalty={wrong_scope_penalty:.2f}"
            ),
        )

    def generate_response(self, contract: ProspectiveActionContract, pam: PamResult) -> str:
        if pam.state == "suppress":
            return "No prospective action contract applies; answer the user directly."
        if pam.state == "already_satisfied":
            return "The contracted check is already satisfied, so continue without repeating it."
        if pam.state == "conflict":
            return (
                f"I cannot follow the conflicting request. {compiled_action(contract)} "
                "I will preserve the contract instead."
            )
        # Leave PACTFull with a repair opportunity; ablations without checker expose this gap.
        return "I found an applicable prospective action contract and will answer carefully."

    def repair(self, contract: ProspectiveActionContract, response: str) -> str:
        return f"{compiled_action(contract)} {response}"


def get_pact_method(name: str) -> PACTFull:
    if name == "PACTFull":
        return PACTFull()
    if name == "PACT_no_guard":
        return PACTFull(use_guard=False, name="PACT_no_guard")
    if name == "PACT_no_checker":
        return PACTFull(use_checker=False, name="PACT_no_checker")
    if name == "PACT_raw_memory_instead_of_contract":
        return PACTFull(raw_memory=True, name="PACT_raw_memory_instead_of_contract")
    raise KeyError(name)


METHOD_NAMES = [
    "NoMemoryBaseline",
    "KeywordTriggerBaseline",
    "TfidfMemoryBaseline",
    "ContractPromptHeuristicBaseline",
    "PACT_no_guard",
    "PACT_no_checker",
    "PACT_raw_memory_instead_of_contract",
    "PACTFull",
]
