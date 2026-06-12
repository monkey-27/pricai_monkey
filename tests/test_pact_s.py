import csv
import json

from pact.dataset import build_contracts, build_episodes
from pact.pact import PACTS, PACTSConfig, get_method
from pact.run_eval import OUTPUT_DIR, run


def _rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_pact_s_config_dicts_are_not_shared():
    left = PACTSConfig()
    right = PACTSConfig()
    left.broadness["x"] = 1.0
    assert "x" not in right.broadness


def test_null_candidate_exists_and_can_win():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.case_type == "contract_swap")
    selection = PACTS("full", PACTSConfig(null_margin=0.05, selection_threshold=0.10, null_prior=0.10)).select(contracts, ep.to_inference())
    assert any(item.contract_id == "NULL" for item in selection.candidates)
    assert selection.selected is None


def test_null_and_second_margin_suppress_weak_or_ambiguous_cases():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.case_type == "indirect_trigger")
    null_strict = PACTS("null_margin", PACTSConfig(null_margin=1.0, selection_threshold=0.01, null_prior=0.0)).predict(contracts, ep.to_inference())
    second_strict = PACTS("second_margin", PACTSConfig(contract_margin=1.0, selection_threshold=0.01, null_prior=0.0)).predict(contracts, ep.to_inference())
    assert null_strict.predicted_state == "suppress"
    assert second_strict.predicted_state == "suppress"


def test_broadness_penalty_and_zscore_safe_zero_std():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.case_type == "indirect_trigger")
    target = contracts[0]
    loose = PACTS("broadness_penalty", PACTSConfig(broadness_alpha=0.0)).score_contract(target, ep.to_inference())
    penalized = PACTS("broadness_penalty", PACTSConfig(broadness_alpha=1.0, broadness={target.contract_id: 0.5})).score_contract(target, ep.to_inference())
    assert penalized.adjusted_score < loose.adjusted_score
    zpred = PACTS("zscore_calibration", PACTSConfig(z_std={target.contract_id: 0.0})).predict(contracts, ep.to_inference())
    assert zpred.predicted_state in {"fire", "suppress", "conflict", "already_satisfied"}


def test_pairwise_and_meta_state_toy_cases():
    contracts = build_contracts()
    episodes = build_episodes("pact_causal_520")
    indirect = next(e for e in episodes if e.case_type == "indirect_trigger")
    conflict = next(e for e in episodes if e.gold_state == "conflict")
    already = next(e for e in episodes if e.gold_state == "already_satisfied")
    method = PACTS("full", PACTSConfig(null_margin=0.0, contract_margin=0.0, selection_threshold=0.01, null_prior=0.0))
    assert method.predict(contracts, indirect.to_inference()).predicted_contract_id == indirect.gold_contract_id
    assert method.predict(contracts, conflict.to_inference()).predicted_state == "conflict"
    assert method.predict(contracts, already.to_inference()).predicted_state == "already_satisfied"
    pairwise = PACTS("pairwise_ranker", PACTSConfig(use_pairwise=True, selection_threshold=0.01, null_prior=0.0))
    selection = pairwise.select(contracts, indirect.to_inference())
    assert selection.top is not None
    assert selection.top.contract_id == indirect.gold_contract_id


def test_field_masking_hides_intended_fields():
    contracts = build_contracts()
    ep = next(e for e in build_episodes("pact_causal_520") if e.case_type == "indirect_trigger")
    contract = contracts[0]
    text_masked = PACTS("contract_text_masked").score_contract(contract, ep.to_inference())
    family_masked = PACTS("family_masked").score_contract(contract, ep.to_inference())
    assert text_masked.cue_match == 0.0
    assert text_masked.guard_match == 0.0
    assert text_masked.action_match == 0.0
    assert text_masked.check_match == 0.0
    assert family_masked.family_match == 0.0


def test_pact_s_output_surface_and_summary_columns():
    run(dataset="pact_causal_520", methods="pact_s", split="test", audit=True, bootstrap_iters=5)
    for name in [
        "pact_s_summary.csv",
        "pact_s_threshold_search.csv",
        "pact_s_best_config.json",
        "pact_s_topk_ranking_trace.csv",
        "pact_s_contract_broadness.csv",
        "pact_s_field_masking.csv",
        "pact_s_multi_contract_stress.csv",
        "pact_s_pool_composition_stress.csv",
        "pact_s_conflict_taxonomy.csv",
        "pact_s_target_completion_mismatches.csv",
        "pact_s_selector_variant_metrics.csv",
        "pact_s_pairwise_learning.csv",
        "pact_s_contract_order_invariance.csv",
        "pact_s_contract_duplication.csv",
        "pact_s_safety_priority_matrix.csv",
        "pact_s_naturalistic_failure_taxonomy.csv",
        "manual_audit_pact_s_template.csv",
        "audit_pact_s.md",
    ]:
        assert (OUTPUT_DIR / name).exists()
    summary = _rows("pact_s_summary.csv")
    methods = {row["method"] for row in summary}
    assert "PACT_S_full" in methods
    assert "PACT_S_family_only" in methods
    required = {"strict_e2e", "multi_contract_pool12_strict_e2e", "NULL_accuracy", "correct_contract_selection_rate"}
    assert required <= set(summary[0])
    config = json.loads((OUTPUT_DIR / "pact_s_best_config.json").read_text())
    assert config["dev_result"]


def test_pact_s_diagnostic_required_values():
    run(dataset="pact_causal_520", methods="pact_s", split="test", audit=False, bootstrap_iters=5)
    assert {row["pool_composition"] for row in _rows("pact_s_pool_composition_stress.csv")} == {
        "random_distractors",
        "same_domain_distractors",
        "broad_distractors",
        "action_similar_distractors",
        "guard_similar_distractors",
        "conflict_inducing_distractors",
    }
    assert {int(row["pool_size"]) for row in _rows("pact_s_multi_contract_stress.csv")} == {1, 3, 6, 12}
    assert all(row["expected_selection"] == "NULL" for row in _rows("pact_s_null_dominant_pool.csv")[:5])
    assert _rows("pact_s_conflict_taxonomy.csv")
    assert _rows("manual_audit_pact_s_template.csv")[0]["audit_should_choose_NULL"] == ""
