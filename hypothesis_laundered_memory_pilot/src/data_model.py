from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Domain = Literal["coding", "data_analysis", "research_assistant"]
CaseSubtype = Literal["false_hypothesis", "verified_hypothesis", "ambiguous_hypothesis"]
Method = Literal[
    "no_memory",
    "naive",
    "reflection",
    "source_aware",
    "quote_required",
    "evidence_labeled",
    "evidence_labeled_no_enforcement",
    "evidence_labeled_stable_only",
    "evidence_labeled_enforced",
]


@dataclass
class SourceEpisode:
    user_task: str
    evidence: str
    agent_trace: str
    final_response: str


@dataclass
class FutureTask:
    user_task: str
    current_evidence: str
    correct_answer: str
    forbidden_claims: list[str]
    required_claims: list[str]


@dataclass
class VerifiedMemoryControl:
    memory: str
    future_task: str
    correct_answer: str
    required_claims: list[str]
    forbidden_claims: list[str]


@dataclass
class BenchmarkItem:
    id: str
    domain: Domain
    source_episode: SourceEpisode
    trap_hypothesis: str
    trap_keywords: list[str]
    ground_truth: str
    future_task: FutureTask
    verified_memory_control: VerifiedMemoryControl
    case_subtype: CaseSubtype = "false_hypothesis"
    confirmed_hypothesis: str = ""
    verification_evidence: str = ""
    trap_is_later_verified: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkItem":
        return cls(
            id=data["id"],
            domain=data["domain"],
            source_episode=SourceEpisode(**data["source_episode"]),
            trap_hypothesis=data["trap_hypothesis"],
            trap_keywords=list(data["trap_keywords"]),
            ground_truth=data["ground_truth"],
            future_task=FutureTask(**data["future_task"]),
            verified_memory_control=VerifiedMemoryControl(**data["verified_memory_control"]),
            case_subtype=data.get("case_subtype", "false_hypothesis"),
            confirmed_hypothesis=data.get("confirmed_hypothesis", ""),
            verification_evidence=data.get("verification_evidence", ""),
            trap_is_later_verified=bool(data.get("trap_is_later_verified", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRecord:
    memory: str
    evidence_type: str = "StableByDefault"
    status: str = "Stable"
    supporting_evidence: str = ""
    reuse_rule: str = ""
    raw: Any = None


@dataclass
class ScoreRecord:
    item_id: str
    domain: str
    case_subtype: str
    method: str
    false_evidence_promotion: bool
    downstream_contamination: bool
    trap_task_correct: bool
    control_task_correct: bool
    useful_memory_retention: bool
    mixed: bool
    downstream_label: str = "unparseable"
    scoring_rationale: str = ""
    required_evidence_matched: list[str] = field(default_factory=list)
    false_hypothesis_matched: list[str] = field(default_factory=list)
    confirmed_hypothesis_promoted: bool = False
    tentative_overblocked: bool = False
    memories: list[dict[str, Any]] = field(default_factory=list)
    trap_answer: str = ""
    control_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
