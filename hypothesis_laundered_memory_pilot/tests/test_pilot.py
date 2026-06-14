from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark import build_seed_items
from src.experiment import DEFAULT_METHODS, run_experiment, with_mock_meta
from src.llm import LLMClient
from src.prompts import downstream_trap_prompt, memory_prompt
from src.scorers import classify_trap_answer
from src.utils import parse_json_lenient


def test_benchmark_has_80_items_and_domains() -> None:
    items = build_seed_items()
    assert len(items) == 80
    assert sum(1 for item in items if item.domain == "data_analysis") == 30
    assert sum(1 for item in items if item.domain == "coding") == 30
    assert sum(1 for item in items if item.domain == "research_assistant") == 20
    assert sum(1 for item in items if item.case_subtype == "verified_hypothesis") >= 15


def test_real_prompt_rendering_has_no_mock_or_answer_key_terms() -> None:
    item = build_seed_items()[0]
    prompts = [memory_prompt("naive", item)[1], downstream_trap_prompt("evidence_labeled_enforced", item, [])]
    blocked = ["MOCK_", "trap hypothesis", "forbidden claim", "ground truth", "required_claims", "forbidden_claims"]
    for prompt in prompts:
        for term in blocked:
            assert term not in prompt


def test_mock_prompt_rendering_adds_metadata() -> None:
    prompt = with_mock_meta("hello", build_seed_items()[0])
    assert "MOCK_TRAP:" in prompt
    assert "MOCK_TRUTH:" in prompt
    assert "MOCK_CONTROL:" in prompt


def test_json_parsing_handles_malformed_wrapped_output() -> None:
    parsed, error = parse_json_lenient("prefix [\"a\", \"b\"] suffix")
    assert error is None
    assert parsed == ["a", "b"]
    parsed, error = parse_json_lenient("not json")
    assert parsed is None
    assert error


def test_evidence_labeled_parsing_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "run"
    scores, items, metadata = run_experiment(
        model="mock",
        n=1,
        domains=["coding"],
        methods=["evidence_labeled_enforced"],
        out=out,
        mock=True,
        seed=1,
        temperature=0,
        top_p=1,
        max_new_tokens=100,
        backend="openai_compatible",
        base_url=None,
        api_key=None,
        hf_model=None,
        device="auto",
        dtype="auto",
        audit_sample_size=1,
        judge_backend="none",
        judge_model="same",
    )
    assert scores
    assert scores[0].memories
    assert metadata["mock"] is True


def test_scoring_distinguishes_contaminated_and_rejected() -> None:
    item = build_seed_items()[0]
    contaminated = f"The main explanation is {item.future_task.forbidden_claims[1]}."
    rejected = f"Current evidence rules out {item.future_task.forbidden_claims[1]}; use {item.future_task.required_claims[0]}."
    assert classify_trap_answer(contaminated, item) == "contaminated"
    assert classify_trap_answer(rejected, item) == "mixed_rejected_trap"


def test_summary_and_manual_audit_created(tmp_path: Path) -> None:
    from src.report import write_summary

    out = tmp_path / "run"
    scores, items, metadata = run_experiment(
        model="mock",
        n=4,
        domains=["coding", "data_analysis"],
        methods=DEFAULT_METHODS[:3],
        out=out,
        mock=True,
        seed=2,
        temperature=0,
        top_p=1,
        max_new_tokens=100,
        backend="openai_compatible",
        base_url=None,
        api_key=None,
        hf_model=None,
        device="auto",
        dtype="auto",
        audit_sample_size=3,
        judge_backend="none",
        judge_model="same",
    )
    write_summary(out, scores, items, DEFAULT_METHODS[:3], metadata)
    assert (out / "summary.csv").exists()
    assert (out / "summary.md").exists()
    assert (out / "manual_audit_sample.csv").exists()
    assert (out / "manual_audit_instructions.md").exists()
    assert json.loads((out / "run_metadata.json").read_text())["scientific_evidence"] is False


def test_transformers_backend_missing_dependency_or_model_error(tmp_path: Path) -> None:
    client = LLMClient(model="x", out_dir=tmp_path, backend="transformers", hf_model=None, mock=False, api_key="dummy")
    with pytest.raises(RuntimeError, match="hf-model|transformers"):
        client.complete("s", "u")


def test_openai_compatible_accepts_base_url_and_api_key(tmp_path: Path) -> None:
    client = LLMClient(
        model="local-model",
        out_dir=tmp_path,
        backend="openai_compatible",
        base_url="http://localhost:8000/v1",
        api_key="dummy",
        mock=False,
    )
    assert client.base_url == "http://localhost:8000/v1"
    assert client.api_key == "dummy"
