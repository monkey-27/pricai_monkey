# Baseline Fairness Audit Agent

Simulated subagent: Baseline Fairness Audit Agent.
Methods evaluated: ['ContractCheckerOnly', 'ContractClassifierOnly', 'ContractCompilerOnly', 'ContractOnlyClassifier', 'ContractPromptHeuristic', 'ContractShufflePACT', 'FullHistory', 'KeywordTrigger', 'LLMContractClassifier', 'LLMContractSelfCheck', 'LLMFullHistory', 'LLMRawMemoryRAG', 'LabelPermutationSanity', 'LearnedPAM', 'LearnedPAM_plus_checker', 'LearnedPAM_plus_family_compiler', 'NoMemory', 'PACTFull', 'PACTFull_current', 'PACT_R2_full', 'PACT_S_broadness_penalty', 'PACT_S_contract_text_masked', 'PACT_S_family_masked', 'PACT_S_family_only', 'PACT_S_full', 'PACT_S_margin_abstain', 'PACT_S_margins', 'PACT_S_multi_select_top2', 'PACT_S_no_NULL', 'PACT_S_null_margin', 'PACT_S_null_only', 'PACT_S_pairwise_ranker', 'PACT_S_second_margin', 'PACT_S_zscore_calibration', 'PACT_conditional_bonus', 'PACT_intent_family_gate', 'PACT_intent_plus_state', 'PACT_intent_plus_state_checker', 'PACT_intent_plus_state_family_compiler', 'PACT_no_checker', 'PACT_no_compiler', 'PACT_no_conflict_resolver', 'PACT_no_guard', 'PACT_raw_memory', 'PACT_specificity_gate', 'PACT_state_action_split', 'QueryOnlyClassifier', 'QueryPlusContractClassifier', 'QueryPlusFamilyClassifier', 'QueryPlusWrongContractOnly', 'RawMemorySelfCheck', 'TfidfRawMemory'].
Predictors receive InferenceEpisode only.
Thresholds are fixed constants intended as dev-tuned before test reporting.
Strongest baseline is selected by indirect end-to-end success among ordinary baselines.
Result: PASS
