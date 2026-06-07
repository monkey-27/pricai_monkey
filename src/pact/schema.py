"""Validated schema objects for PACT-Causal-520."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CASE_TYPES = {
    "direct_trigger",
    "indirect_trigger",
    "near_miss",
    "wrong_scope",
    "conflict",
    "already_satisfied",
    "contract_swap",
}
SET_TYPES = {"controlled", "paraphrase", "naturalistic"}
SPLITS = {"dev", "test"}
GOLD_STATES = {"fire", "suppress", "conflict", "already_satisfied"}
PRIORITIES = {"low", "medium", "high", "safety"}
STATUSES = {"active", "inactive"}


def _str(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{key} must be non-empty")
    return value


def _list(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


@dataclass(frozen=True)
class ProspectiveActionContract:
    contract_id: str
    family: str
    cue: str
    guard: str
    action: str
    check: str
    priority: str
    status: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProspectiveActionContract":
        priority = _str(data, "priority")
        status = _str(data, "status")
        if priority not in PRIORITIES:
            raise ValueError(f"invalid priority: {priority}")
        if status not in STATUSES:
            raise ValueError(f"invalid status: {status}")
        return cls(
            contract_id=_str(data, "contract_id"),
            family=_str(data, "family"),
            cue=_str(data, "cue"),
            guard=_str(data, "guard"),
            action=_str(data, "action"),
            check=_str(data, "check"),
            priority=priority,
            status=status,
        )

    @property
    def raw_text(self) -> str:
        return " ".join([self.family, self.cue, self.guard, self.action, self.check, self.priority])

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_id": self.contract_id,
            "family": self.family,
            "cue": self.cue,
            "guard": self.guard,
            "action": self.action,
            "check": self.check,
            "priority": self.priority,
            "status": self.status,
        }


@dataclass(frozen=True)
class InferenceEpisode:
    """Prediction-safe view. It omits labels and scoring-only metadata."""

    episode_id: str
    history_summary: str
    current_query: str
    available_contract_ids: list[str]

    @property
    def text(self) -> str:
        return f"{self.history_summary} {self.current_query}"


@dataclass(frozen=True)
class Episode:
    episode_id: str
    contract_id: str
    family: str
    case_type: str
    set_type: str
    split: str
    contrast_group_id: str
    contrast_role: str
    paraphrase_group_id: str
    history_summary: str
    current_query: str
    available_contract_ids: list[str]
    target_contract_id: str
    distractor_contract_ids: list[str]
    gold_state: str
    gold_contract_id: str
    expected_action_keywords: list[str]
    forbidden_action_keywords: list[str]
    completion_rubric: str
    priority_expectation: str
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Episode":
        case_type = _str(data, "case_type")
        set_type = _str(data, "set_type", "controlled")
        split = _str(data, "split", "test")
        gold_state = _str(data, "gold_state")
        if case_type not in CASE_TYPES:
            raise ValueError(f"invalid case_type: {case_type}")
        if set_type not in SET_TYPES:
            raise ValueError(f"invalid set_type: {set_type}")
        if split not in SPLITS:
            raise ValueError(f"invalid split: {split}")
        if gold_state not in GOLD_STATES:
            raise ValueError(f"invalid gold_state: {gold_state}")
        contract_id = _str(data, "contract_id")
        return cls(
            episode_id=_str(data, "episode_id"),
            contract_id=contract_id,
            family=_str(data, "family"),
            case_type=case_type,
            set_type=set_type,
            split=split,
            contrast_group_id=_str(data, "contrast_group_id", "none"),
            contrast_role=_str(data, "contrast_role", "none"),
            paraphrase_group_id=_str(data, "paraphrase_group_id", "none"),
            history_summary=_str(data, "history_summary"),
            current_query=_str(data, "current_query"),
            available_contract_ids=_list(data, "available_contract_ids") or [contract_id],
            target_contract_id=_str(data, "target_contract_id", contract_id),
            distractor_contract_ids=_list(data, "distractor_contract_ids"),
            gold_state=gold_state,
            gold_contract_id=_str(data, "gold_contract_id", contract_id),
            expected_action_keywords=[item.lower() for item in _list(data, "expected_action_keywords")],
            forbidden_action_keywords=[item.lower() for item in _list(data, "forbidden_action_keywords")],
            completion_rubric=_str(data, "completion_rubric", "required keywords must be present"),
            priority_expectation=_str(data, "priority_expectation", "normal"),
            notes=_str(data, "notes", "deterministic case"),
        )

    def to_inference(self) -> InferenceEpisode:
        return InferenceEpisode(
            episode_id=self.episode_id,
            history_summary=self.history_summary,
            current_query=self.current_query,
            available_contract_ids=list(self.available_contract_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "contract_id": self.contract_id,
            "family": self.family,
            "case_type": self.case_type,
            "set_type": self.set_type,
            "split": self.split,
            "contrast_group_id": self.contrast_group_id,
            "contrast_role": self.contrast_role,
            "paraphrase_group_id": self.paraphrase_group_id,
            "history_summary": self.history_summary,
            "current_query": self.current_query,
            "available_contract_ids": self.available_contract_ids,
            "target_contract_id": self.target_contract_id,
            "distractor_contract_ids": self.distractor_contract_ids,
            "gold_state": self.gold_state,
            "gold_contract_id": self.gold_contract_id,
            "expected_action_keywords": self.expected_action_keywords,
            "forbidden_action_keywords": self.forbidden_action_keywords,
            "completion_rubric": self.completion_rubric,
            "priority_expectation": self.priority_expectation,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Prediction:
    method: str
    episode_id: str
    predicted_contract_id: str
    predicted_state: str
    confidence: float
    response: str
    action_completed: bool
    repaired: bool
    rationale: str

    @property
    def contract_id(self) -> str:
        return self.predicted_contract_id

    @property
    def satisfied(self) -> bool:
        return self.action_completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "episode_id": self.episode_id,
            "predicted_contract_id": self.predicted_contract_id,
            "predicted_state": self.predicted_state,
            "confidence": round(float(self.confidence), 4),
            "response": self.response,
            "action_completed": self.action_completed,
            "repaired": self.repaired,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class EvalResult:
    method: str
    metrics: dict[str, float]
    predictions: list[Prediction]

