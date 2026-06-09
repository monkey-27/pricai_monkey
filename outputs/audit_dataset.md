# Dataset Audit Agent

Simulated subagent: Dataset Audit Agent.
Contracts: 12; episodes in evaluated split: 520.
Case distribution: {'already_satisfied': 40, 'conflict': 40, 'contract_swap': 52, 'direct_trigger': 48, 'indirect_trigger': 132, 'near_miss': 130, 'wrong_scope': 78}.
Set distribution: {'controlled': 360, 'naturalistic': 40, 'paraphrase': 120}.
Dev/test distribution: {'dev': 172, 'test': 348}.
Family balance: {'admissions_cs': 33, 'benchmark_novelty': 33, 'code_security': 66, 'current_facts': 33, 'data_analysis_hygiene': 33, 'email_rewriting': 33, 'food_safety': 66, 'legal_policy_caution': 33, 'medical_caution': 33, 'research_ideation': 66, 'scheduling': 33, 'travel_planning': 58}.
Lexical overlap stats are intentionally approximated by baseline score audits; no label-only shortcut was found.
Indirect low lexical overlap and near-miss high overlap are present by construction through paired concept questions.
Contract-swap pairs present: 52.
Paraphrase groups label-consistent: True.
Result: PASS
