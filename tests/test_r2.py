import hashlib
import json

from pact.dataset import ROOT, build_contracts, build_episodes, write_dataset
from pact.pact import R2Config, PACTR2, detect_intent_family
from pact.run_eval import OUTPUT_DIR, run, tune_r2_config


def test_intent_family_gate_detects_match_mismatch_unknown():
    assert detect_intent_family("Can you polish this work email?", "").family == "email_rewriting"
    assert detect_intent_family("What is a pleasant blue object?", "").family == "unknown"
    contracts = build_contracts()
    email = next(c for c in contracts if c.family == "email_rewriting")
    food = next(c for c in contracts if c.family == "food_safety")
    method = PACTR2("intent", R2Config(intent_family_confidence_threshold=0.4))
    assert method._intent_matches(detect_intent_family("rewrite this client note", ""), email)
    assert method._intent_mismatches(detect_intent_family("rewrite this client note", ""), food)


def test_specificity_gate_blocks_weak_contract_even_with_action_words():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.case_type == "contract_swap")
    pred = PACTR2("specificity", R2Config(specificity_floor=0.50, base_floor=0.50)).predict(contracts, ep.to_inference())
    assert pred.predicted_state == "suppress"
    assert "specificity_gate" in pred.rationale or "base_gate" in pred.rationale


def test_conditional_bonus_requires_plausible_base():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.case_type == "contract_swap")
    strict = PACTR2("conditional", R2Config(base_floor=0.90, bonus_multiplier=0.35)).predict(contracts, ep.to_inference())
    loose = PACTR2("conditional", R2Config(base_floor=0.01, specificity_floor=0.01, bonus_multiplier=0.35)).predict(contracts, ep.to_inference())
    assert strict.predicted_state == "suppress"
    assert loose.predicted_state in {"fire", "suppress", "conflict", "already_satisfied"}


def test_state_action_split_detects_conflict_with_action_contract():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.gold_state == "conflict")
    pred = PACTR2("state").predict(contracts, ep.to_inference())
    assert pred.predicted_state == "conflict"
    assert pred.predicted_contract_id != "none"
    assert "meta_state=conflict" in pred.rationale


def test_dev_tuning_saves_frozen_config():
    contracts = build_contracts()
    config = tune_r2_config(contracts, build_episodes("pact_causal_520"), seed=0)
    saved = json.loads((OUTPUT_DIR / "r2_best_config.json").read_text())
    assert saved["config"]["specificity_floor"] == config.specificity_floor
    assert saved["dev_result"]


def test_r2_outputs_and_dataset_stable():
    dataset_path = ROOT / "data" / "pact_causal_520" / "episodes.jsonl"
    before = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    write_dataset("pact_causal_520")
    after = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert before == after
    run(dataset="pact_causal_520", methods="r2", split="test", audit=True, bootstrap_iters=20)
    for name in [
        "r2_variant_metrics.csv",
        "r2_threshold_search.csv",
        "r2_best_config.json",
        "r2_error_transition.csv",
        "r2_fixed_errors.csv",
        "r2_new_errors.csv",
        "r2_contract_swap_errors.csv",
        "r2_conflict_errors.csv",
        "r2_family_metrics.csv",
        "manual_audit_r2_template.csv",
        "audit_r2.md",
    ]:
        assert (OUTPUT_DIR / name).exists()
