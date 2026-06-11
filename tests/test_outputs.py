from pact.run_eval import OUTPUT_DIR, run


def test_required_output_files_created():
    run(dataset="pact_causal_520", methods="r2", split="test", audit=True, bootstrap_iters=20)
    for name in [
        "predictions.csv",
        "metrics_main.json",
        "metrics_by_family.csv",
        "metrics_by_case_type.csv",
        "metrics_by_set_type.csv",
        "paired_comparisons.csv",
        "bootstrap_ci.json",
        "mcnemar_tests.json",
        "permutation_tests.json",
        "sanity_checks.json",
        "errors_pact.csv",
        "errors_strongest_baseline.csv",
        "manual_audit_sample.csv",
        "manual_audit_completed_template.csv",
        "method_differences.csv",
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
        "audit_dataset.md",
        "audit_baselines.md",
        "audit_causality.md",
        "audit_metrics.md",
        "audit_reproducibility.md",
        "audit_research_value.md",
        "audit_report.md",
    ]:
        assert (OUTPUT_DIR / name).exists()

    manual = (OUTPUT_DIR / "manual_audit_completed_template.csv").read_text()
    assert "strongest_baseline_method" in manual
    assert "audit_question_wrong_contract_issue" in manual
    assert "PACTFull" in manual
    assert "pact_error" in manual or "baseline_failure_pact_success" in manual
    research = (OUTPUT_DIR / "audit_research_value.md").read_text()
    assert "wrong_contract_false_trigger_le_0.10" in research
    assert "Conflict caveat:" in research
    metrics_audit = (OUTPUT_DIR / "audit_metrics.md").read_text()
    assert "Contract-swap false-trigger rate" in metrics_audit
    assert "Strict end-to-end success" in metrics_audit
