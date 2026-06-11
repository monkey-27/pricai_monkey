import csv
import json

from pact.dataset import build_contracts, build_episodes
from pact.pact import ContractOnlyClassifier, LearnedPAM, QueryPlusFamilyClassifier, get_method
from pact.run_eval import OUTPUT_DIR, run


def _rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_d3_composition_variants_exist_and_run():
    contracts = build_contracts()
    episode = next(ep for ep in build_episodes("pact_causal_520") if ep.case_type == "indirect_trigger")
    for name in [
        "PACT_intent_plus_state",
        "PACT_intent_plus_state_checker",
        "PACT_intent_plus_state_family_compiler",
    ]:
        pred = get_method(name).predict(contracts, episode.to_inference())
        assert pred.method == name
        assert pred.predicted_state in {"fire", "suppress", "conflict", "already_satisfied"}


def test_d3_controls_have_restricted_inputs():
    contracts = build_contracts()
    episode = next(ep for ep in build_episodes("pact_causal_520") if ep.case_type == "indirect_trigger")
    changed = episode.to_inference().__class__(
        episode_id=episode.episode_id,
        history_summary="unrelated history",
        current_query="unrelated query",
        available_contract_ids=episode.available_contract_ids,
    )
    contract_only = ContractOnlyClassifier()
    assert contract_only.predict(contracts, episode.to_inference()).to_dict() == contract_only.predict(contracts, changed).to_dict()
    family_pred = QueryPlusFamilyClassifier().predict(contracts, episode.to_inference())
    assert "family only" in family_pred.rationale
    assert "contract text" not in family_pred.rationale


def test_learned_pam_is_deterministic_fallback():
    contracts = build_contracts()
    episode = next(ep for ep in build_episodes("pact_causal_520") if ep.case_type == "indirect_trigger")
    method = LearnedPAM()
    assert method.predict(contracts, episode.to_inference()).to_dict() == method.predict(contracts, episode.to_inference()).to_dict()


def test_d3_output_files_and_oracle_marking():
    run(dataset="pact_causal_520", methods="d3", split="test", audit=True, bootstrap_iters=20)
    for name in [
        "diagnostic_summary.csv",
        "oracle_ceiling.csv",
        "composition_ablation.csv",
        "learned_pam_results.csv",
        "learned_pam_feature_report.json",
        "learned_pam_errors.csv",
        "query_contract_controls.csv",
        "multi_contract_stress.csv",
        "score_anatomy.csv",
        "score_anatomy_by_error_type.csv",
        "score_anatomy_by_family.csv",
        "score_anatomy_false_fire_cases.csv",
        "conflict_taxonomy.csv",
        "target_completion_mismatches.csv",
        "naturalistic_failures.csv",
        "naturalistic_metrics.csv",
        "d3_error_transition.csv",
        "d3_fixed_errors.csv",
        "d3_new_errors.csv",
        "manual_audit_d3_template.csv",
        "audit_d3.md",
    ]:
        assert (OUTPUT_DIR / name).exists()
    oracle = _rows("oracle_ceiling.csv")
    assert oracle
    assert all(row["oracle_unfair"] == "True" for row in oracle)
    feature_report = json.loads((OUTPUT_DIR / "learned_pam_feature_report.json").read_text())
    assert feature_report["training_split"] == "dev_only"
    assert not feature_report["test_labels_seen_at_prediction_time"]


def test_d3_stress_taxonomy_score_and_manual_columns():
    run(dataset="pact_causal_520", methods="d3", split="test", audit=False, bootstrap_iters=20)
    assert {int(row["pool_size"]) for row in _rows("multi_contract_stress.csv")} == {1, 3, 6, 12}
    score_fields = set(_rows("score_anatomy.csv")[0])
    for field in [
        "retrieval_score",
        "guard_score",
        "action_score",
        "intent_family_score",
        "bonus",
        "base_score",
        "specificity",
        "final_score",
        "fire_threshold",
    ]:
        assert field in score_fields
    taxonomy = _rows("conflict_taxonomy.csv")
    assert taxonomy
    assert all(row["conflict_subtype"] for row in taxonomy)
    manual_fields = set(_rows("manual_audit_d3_template.csv")[0])
    for field in [
        "audit_question_gold_label_valid",
        "audit_question_d3_fixed_for_right_reason",
        "audit_question_wrong_contract_issue",
        "manual_notes",
    ]:
        assert field in manual_fields


def test_d3_transition_labels_and_target_mismatch_columns():
    run(dataset="pact_causal_520", methods="d3", split="test", audit=False, bootstrap_iters=20)
    labels = {row["transition"] for row in _rows("d3_error_transition.csv")}
    assert labels <= {"fixed", "regressed", "unchanged_correct", "unchanged_wrong"}
    mismatch_fields = set(_rows("target_completion_mismatches.csv")[0])
    for field in ["expected_action_keywords", "completion_rubric", "checker_result", "target_completion_result"]:
        assert field in mismatch_fields
