# PACT-S Audit Agent

Simulated subagent: PACT-S Audit Agent.
Dataset unchanged: PACT-S reuses frozen pact_causal_520 and does not edit labels, splits, rubrics, or case types.
Dev-only tuning: pact_s_best_config.json is selected on dev and reused for test/all predictions.
Final mechanism under test: NULL-aware competitive selection plus family-specific execution.
PACT_S_full summary: {"NULL_accuracy": 0.6323529411764706, "behavioral_e2e": 0.7096153846153846, "conflict_detection": 0.275, "conflict_safe_action": 0.45, "correct_contract_selection_rate": 0.6884615384615385, "fixed_vs_D3_best": 11, "indirect_strict_success": 0.29545454545454547, "method": "PACT_S_full", "multi_contract_pool12_strict_e2e": 0.6711538461538461, "multi_contract_pool12_wrong_contract_ft": 0.19230769230769232, "naturalistic_strict_success": 0.75, "naturalistic_wrong_contract_ft": 0.0, "regressed_vs_D3_best": 124, "strict_e2e": 0.6942307692307692, "target_completion": 0.32727272727272727, "weighted_utility": 0.5711538461538461, "wrong_contract_false_trigger_rate": 0.0}.
Pool-size-12 PACT_S_full: {"correct_contract_selection_rate": 0.6634615384615384, "indirect_success": 0.29545454545454547, "method": "PACT_S_full", "pool_size": 12, "strict_e2e": 0.6711538461538461, "wrong_contract_false_trigger_rate": 0.19230769230769232}.
Pool-size-12 D3 comparison: {"correct_contract_selection_rate": 0.573076923076923, "indirect_success": 0.3560606060606061, "method": "PACT_intent_plus_state_family_compiler", "pool_size": 12, "strict_e2e": 0.5711538461538461, "wrong_contract_false_trigger_rate": 0.9807692307692307}.
Success checks: {"conflict_safe_ge_0.90": false, "indirect_strict_ge_0.88": false, "naturalistic_strict_ge_0.75": true, "naturalistic_wrong_contract_ft_le_0.15": true, "pool12_strict_ge_0.75": false, "pool12_wrong_contract_ft_le_0.20": true, "query_plus_family_below_pact_s": false, "strict_e2e_ge_0.90": false, "target_completion_ge_0.90": false, "wrong_contract_ft_le_0.05": true}.
Broadest remaining contract: {"average_score_on_gold": 0.28205378630795136, "average_score_on_non_gold": 0.059917505766702586, "broadness": 0.09554137632368548, "contract_id": "legal_policy_uncertainty", "false_fire_count": 2, "family": "legal_policy_caution", "overactivation_rate": 0.004073319755600814, "specificity_ratio": 4.707368617882198}.
Naturalistic failure count: 10.
Target-completion mismatch count: 0.
Research decision: NARROW_OR_KILL.
Manual audit status: manual_audit_pact_s_template.csv is a template, not completed human evidence.
Caveat: do not claim PACT-S fixes the mechanism unless pool-size-12 stress, naturalistic wrong-contract rate, and manual audit all support it.
