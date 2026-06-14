from __future__ import annotations

from typing import Any


SMOKE_MODEL_MARKERS = [
    "tiny-gpt2",
    "distilgpt2",
    "/gpt2",
    "gpt2",
    "tiny-random",
    "random",
    "0.5b",
]

INSTRUCT_MARKERS = [
    "instruct",
    "chat",
    "coder",
    "it",
]


def classify_run(metadata: dict[str, Any]) -> dict[str, Any]:
    mock = bool(metadata.get("mock"))
    n_items = int(metadata.get("n_items") or metadata.get("n_requested") or 0)
    model = str(metadata.get("hf_model") or metadata.get("model") or "").lower()
    audit_completed = bool(metadata.get("manual_audit_completed", False))
    if mock:
        return {
            "run_role": "mock_pipeline_validation",
            "scientific_evidence": False,
            "research_verdict": "MOCK_ONLY",
            "classification_reason": "Mock outputs are programmed and validate code paths only.",
        }
    if n_items < 40 or is_smoke_model(model) or not is_instruct_model(model):
        return {
            "run_role": "plumbing_smoke",
            "scientific_evidence": False,
            "research_verdict": "PLUMBING_ONLY",
            "classification_reason": "Run is too small or uses a non-instruct/tiny model.",
        }
    if n_items >= 80 and audit_completed:
        return {
            "run_role": "paper_candidate_evidence",
            "scientific_evidence": True,
            "research_verdict": "PAPER_CANDIDATE",
            "classification_reason": "Run uses an instruct model, full benchmark, and completed manual audit.",
        }
    return {
        "run_role": "preliminary_experiment",
        "scientific_evidence": True,
        "research_verdict": "CONTINUE_WEAK",
        "classification_reason": "Run uses a real instruct/chat model with enough examples, but still needs audit/replication.",
    }


def is_smoke_model(model_name: str) -> bool:
    compact = model_name.lower()
    return any(marker in compact for marker in SMOKE_MODEL_MARKERS)


def is_instruct_model(model_name: str) -> bool:
    compact = model_name.lower()
    return any(marker in compact for marker in INSTRUCT_MARKERS)
