# Research-Value Audit Agent

Simulated subagent: Research-Value Audit Agent.
Strongest ordinary baseline: TfidfRawMemory.
Strongest contract-aware baseline: ContractCompilerOnly.
Strongest non-oracle baseline: TfidfRawMemory.
PACTFull metrics: {"already_satisfied_accuracy": 1.0, "checker_repair_gain": 0.9397590361445783, "conflict_accuracy": 0.5, "conflict_as_fire_rate": 0.45, "conflict_as_suppress_rate": 0.05, "conflict_detection_accuracy": 0.5, "conflict_safe_action_accuracy": 0.875, "contract_swap_sensitivity": 0.7692307692307693, "end_to_end_success": 0.8826923076923077, "end_to_end_success_behavioral": 0.9134615384615384, "end_to_end_success_indirect": 0.8712121212121212, "end_to_end_success_near_miss": 0.9846153846153847, "end_to_end_success_strict": 0.8826923076923077, "end_to_end_success_wrong_scope": 0.9743589743589743, "false_trigger_rate": 0.019230769230769232, "false_trigger_rate_excluding_contract_swap": 0.019230769230769232, "false_trigger_rate_including_contract_swap": 0.06153846153846154, "fire_f1": 0.8736842105263158, "fire_precision": 0.83, "fire_recall": 0.9222222222222223, "indirect_action_completion": 0.8712121212121212, "indirect_trigger_recall": 0.9242424242424242, "irrelevant_action_completion_rate": 0.057971014492753624, "near_miss_false_trigger_rate": 0.015384615384615385, "overall_false_trigger_rate": 0.06153846153846154, "paraphrase_consistency": 0.07499999999999996, "per_family_variance": 0.11577869019388885, "predicted_contract_action_completion_rate": 1.0, "target_action_completion_rate": 0.8681818181818182, "trigger_accuracy": 0.9038461538461539, "weighted_utility": 0.7182692307692308, "wrong_contract_action_completion_rate": 0.23076923076923078, "wrong_contract_false_trigger_rate": 0.23076923076923078, "wrong_scope_false_trigger_rate": 0.02564102564102564}.
Preregistered checks: {"checker_gain_ge_0.15": true, "ci_low_ge_0.10": true, "conflict_safe_action_ge_0.75": true, "false_trigger_le_0.10": true, "gain_ge_0.20": true, "guard_gain_ge_0.10": true, "indirect_success_ge_0.75": true, "irrelevant_action_completion_le_0.10": true, "paraphrase_drop_le_0.10": true, "shuffle_drop_ge_0.25": true, "wrong_contract_false_trigger_le_0.10": false}.
Triggered failures: ['wrong_contract_false_trigger_le_0.10'].
Conflict caveat: not triggered.
What the stricter eval changes relative to the previous report: contract-swap false triggers, irrelevant wrong-contract actions, and conflict detection are no longer hidden inside aggregate false-trigger or end-to-end metrics.
Decision: CONTINUE_WEAK.
Next experiment: run the same causal set with a small local model-backed PAM while preserving blinded InferenceEpisode inputs.
Result: PASS
