"""Small validated data objects for the PACT pilot."""

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
}
GOLD_STATES = {"fire", "suppress", "conflict", "already_satisfied"}
PRIORITIES = {"low", "medium", "high"}
STATUSES = {"active", "inactive"}


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _keyword_list(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return [item.strip().lower() for item in value if item.strip()]


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
        priority = _required_str(data, "priority")
        status = _required_str(data, "status")
        if priority not in PRIORITIES:
            raise ValueError(f"invalid priority: {priority}")
        if status not in STATUSES:
            raise ValueError(f"invalid status: {status}")
        return cls(
            contract_id=_required_str(data, "contract_id"),
            family=_required_str(data, "family"),
            cue=_required_str(data, "cue"),
            guard=_required_str(data, "guard"),
            action=_required_str(data, "action"),
            check=_required_str(data, "check"),
            priority=priority,
            status=status,
        )

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

    @property
    def raw_text(self) -> str:
        return " ".join(
            [self.family, self.cue, self.guard, self.action, self.check, self.priority]
        )


@dataclass(frozen=True)
class Episode:
    episode_id: str
    contract_id: str
    family: str
    case_type: str
    history_summary: str
    current_query: str
    gold_state: str
    expected_action_keywords: list[str]
    forbidden_action_keywords: list[str]
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Episode":
        case_type = _required_str(data, "case_type")
        gold_state = _required_str(data, "gold_state")
        if case_type not in CASE_TYPES:
            raise ValueError(f"invalid case_type: {case_type}")
        if gold_state not in GOLD_STATES:
            raise ValueError(f"invalid gold_state: {gold_state}")
        return cls(
            episode_id=_required_str(data, "episode_id"),
            contract_id=_required_str(data, "contract_id"),
            family=_required_str(data, "family"),
            case_type=case_type,
            history_summary=_required_str(data, "history_summary"),
            current_query=_required_str(data, "current_query"),
            gold_state=gold_state,
            expected_action_keywords=_keyword_list(data, "expected_action_keywords"),
            forbidden_action_keywords=_keyword_list(data, "forbidden_action_keywords"),
            notes=_required_str(data, "notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "contract_id": self.contract_id,
            "family": self.family,
            "case_type": self.case_type,
            "history_summary": self.history_summary,
            "current_query": self.current_query,
            "gold_state": self.gold_state,
            "expected_action_keywords": self.expected_action_keywords,
            "forbidden_action_keywords": self.forbidden_action_keywords,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Prediction:
    method: str
    episode_id: str
    contract_id: str
    predicted_state: str
    confidence: float
    response: str
    satisfied: bool
    repaired: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "episode_id": self.episode_id,
            "contract_id": self.contract_id,
            "predicted_state": self.predicted_state,
            "confidence": round(float(self.confidence), 4),
            "response": self.response,
            "satisfied": self.satisfied,
            "repaired": self.repaired,
            "rationale": self.rationale,
        }

