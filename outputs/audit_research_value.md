# Research-Value Audit Agent

Simulated subagent: Research-Value Audit Agent.
Strongest ordinary baseline: .
Strongest contract-aware baseline: .
Strongest non-oracle baseline: .
PACTFull metrics: {"already_satisfied_accuracy": 1.0, "checker_repair_gain": 0.9099099099099099, "conflict_accuracy": 0.46875, "contract_swap_sensitivity": 0.7804878048780488, "end_to_end_success": 0.8735632183908046, "end_to_end_success_indirect": 0.8765432098765432, "end_to_end_success_near_miss": 0.9873417721518988, "end_to_end_success_wrong_scope": 1.0, "false_trigger_rate": 0.007936507936507936, "fire_f1": 0.8740157480314961, "fire_precision": 0.8102189781021898, "fire_recall": 0.9487179487179487, "indirect_action_completion": 0.8765432098765432, "indirect_trigger_recall": 0.9629629629629629, "paraphrase_consistency": 0.11071428571428565, "per_family_variance": 0.11601555075879352, "trigger_accuracy": 0.9051724137931034, "weighted_utility": 0.6910919540229885, "wrong_scope_false_trigger_rate": 0.0}.
Preregistered checks: {"checker_gain_ge_0.15": true, "ci_low_ge_0.10": false, "false_trigger_le_0.10": true, "gain_ge_0.20": true, "guard_gain_ge_0.10": true, "indirect_success_ge_0.75": true, "paraphrase_drop_le_0.10": true, "shuffle_drop_ge_0.25": true}.
Triggered failures: ['ci_low_ge_0.10'].
Decision: CONTINUE_WEAK.
Next experiment: run the same causal set with a small local model-backed PAM while preserving blinded InferenceEpisode inputs.
Result: PASS
