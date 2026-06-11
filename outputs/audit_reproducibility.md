# Reproducibility Audit Agent

Simulated subagent: Reproducibility Audit Agent.
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q` exit=0
```
.............................                                            [100%]
29 passed in 21.82s
```
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pact.dataset --dataset pact_causal_520 --write` exit=0
```
wrote /Users/Arjun/Documents/monkeyy1/pricai_monkey/data/pact_causal_520/contracts.json
wrote /Users/Arjun/Documents/monkeyy1/pricai_monkey/data/pact_causal_520/episodes.jsonl
dataset=pact_causal_520 contracts=12 episodes=520
```
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pact.run_eval --dataset pact_causal_520 --methods PACTFull,QueryOnlyClassifier,ContractShufflePACT --split test --bootstrap-iters 50` exit=0
```
method | end_to_end_success_strict | end_to_end_success_behavioral | indirect_action_completion | false_trigger_rate_including_contract_swap | wrong_contract_false_trigger_rate | conflict_detection_accuracy
--- | --- | --- | --- | --- | --- | ---
PACTFull | 0.874 | 0.914 | 0.877 | 0.060 | 0.220 | 0.469
QueryOnlyClassifier | 0.532 | 0.532 | 0.000 | 0.084 | 0.341 | 0.469
ContractShufflePACT | 0.572 | 0.572 | 0.000 | 0.000 | 0.000 | 0.469
```
README commands use python3 -m alternatives for pip/pytest.
Result: PASS
