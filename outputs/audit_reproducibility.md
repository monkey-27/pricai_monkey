# Reproducibility Audit Agent

Simulated subagent: Reproducibility Audit Agent.

`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q` exit=0
stdout:
```
..........                                                               [100%]
10 passed in 0.02s
```
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pact.dataset --write` exit=0
stdout:
```
wrote /Users/Arjun/Documents/monkeyy1/pricai_monkey/data/contracts.json
wrote /Users/Arjun/Documents/monkeyy1/pricai_monkey/data/episodes.jsonl
contracts=10 episodes=100
```
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pact.run_eval --methods all` exit=0
stdout:
```
method | trigger_accuracy | fire_precision | fire_recall | indirect_trigger_recall | false_trigger_rate_near_wrong | action_completion_rate_indirect_fire
--- | --- | --- | --- | --- | --- | ---
NoMemoryBaseline | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000
KeywordTriggerBaseline | 0.420 | 0.425 | 0.340 | 0.333 | 0.375 | 0.133
TfidfMemoryBaseline | 0.370 | 0.143 | 0.020 | 0.000 | 0.100 | 0.000
ContractPromptHeuristicBaseline | 0.710 | 1.000 | 0.420 | 0.267 | 0.000 | 0.267
PACT_no_guard | 0.870 | 0.794 | 1.000 | 1.000 | 0.325 | 0.967
PACT_no_checker | 0.950 | 0.909 | 1.000 | 1.000 | 0.125 | 0.000
PACT_raw_memory_instead_of_contract | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000
PACTFull | 0.950 | 0.909 | 1.000 | 1.000 | 0.125 | 0.967
```

Result: PASS
