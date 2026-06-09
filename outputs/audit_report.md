# PACT-Causal-520 Evaluation

method | end_to_end_success_strict | end_to_end_success_behavioral | indirect_action_completion | false_trigger_rate_including_contract_swap | wrong_contract_false_trigger_rate | conflict_detection_accuracy
--- | --- | --- | --- | --- | --- | ---
NoMemory | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000
KeywordTrigger | 0.583 | 0.590 | 0.008 | 0.000 | 0.000 | 0.000
TfidfRawMemory | 0.560 | 0.575 | 0.038 | 0.035 | 0.000 | 0.000
FullHistory | 0.608 | 0.608 | 0.000 | 0.000 | 0.000 | 0.500
RawMemorySelfCheck | 0.585 | 0.594 | 0.000 | 0.000 | 0.000 | 0.000
ContractPromptHeuristic | 0.606 | 0.606 | 0.000 | 0.000 | 0.000 | 0.500
ContractClassifierOnly | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.500
ContractCompilerOnly | 0.585 | 0.594 | 0.008 | 0.004 | 0.000 | 0.000
ContractCheckerOnly | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000
PACTFull | 0.883 | 0.913 | 0.871 | 0.062 | 0.231 | 0.500
PACT_no_guard | 0.794 | 0.825 | 0.894 | 0.250 | 0.346 | 0.500
PACT_no_checker | 0.583 | 0.583 | 0.000 | 0.062 | 0.231 | 0.500
PACT_no_compiler | 0.840 | 0.863 | 0.765 | 0.062 | 0.231 | 0.500
PACT_raw_memory | 0.667 | 0.685 | 0.205 | 0.038 | 0.135 | 0.500
PACT_no_conflict_resolver | 0.846 | 0.910 | 0.871 | 0.062 | 0.231 | 0.000
QueryOnlyClassifier | 0.540 | 0.540 | 0.000 | 0.073 | 0.365 | 0.500
ContractShufflePACT | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.500
LabelPermutationSanity | 0.288 | 0.288 | 0.000 | 0.508 | 0.519 | 0.250
LLMFullHistory | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000
LLMRawMemoryRAG | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000
LLMContractClassifier | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000
LLMContractSelfCheck | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000
