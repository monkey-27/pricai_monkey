# PACT-Causal-520

PACT-Causal-520 is a deterministic pilot benchmark for **Prospective Action
Contracts**: memory items that are not merely facts to retrieve, but delayed
instructions that should fire only when the right future situation occurs.

The motivating question is:

> Can an agent remember a standing instruction and decide when it should become
> an action, without firing it in the wrong context?

PACT represents a selected memory as a contract:

```text
When cue C occurs, if guard G is satisfied, perform action A, then verify check K.
```

The repo evaluates whether contract-aware machinery improves over ordinary
memory retrieval, keyword triggers, full-history baselines, and weaker
contract-aware ablations.

## What This Repo Contains

- A deterministic 520-episode benchmark, `pact_causal_520`.
- A legacy 100-episode prototype, `pact100_legacy`.
- Contract-aware methods and ablations under `src/pact/`.
- Ordinary memory and trigger baselines.
- Statistical tests, bootstrap intervals, sanity controls, and audit reports.
- Saved output artifacts under `outputs/`.

The default evaluation makes **no external API calls**. LLM hooks exist as
disabled stubs, but the current pilot is offline and reproducible.

## Why The Benchmark Exists

Many agent-memory systems can retrieve a relevant instruction. PACT-Causal-520
tests a harder behavior:

1. Store a standing contract.
2. Observe a later user query.
3. Decide whether the contract's cue and guard apply.
4. Fire the required action only when appropriate.
5. Verify that the response satisfied the contract.

This separates "the memory was semantically nearby" from "the memory caused the
right future behavior."

## Dataset Summary

`pact_causal_520` contains 520 episodes:

| Set | Episodes | Purpose |
|---|---:|---|
| Controlled causal set | 360 | Main causal trigger and guard tests |
| Paraphrase robustness set | 120 | Surface-form variation of contract cues |
| Naturalistic transcript set | 40 | More realistic conversational context |

There are 12 contract families. Examples include:

- research ideation: check closest prior work before developing a paper idea;
- food safety: check peanut/tree-nut and cross-contamination risk;
- code security: inspect Flask auth for hardcoded secrets and session safety;
- travel planning: check visa, passport, and entry constraints;
- current facts: verify unstable facts before answering confidently;
- data analysis hygiene: check leakage, missingness, split validity, and metric
  choice before interpreting results.

## Methods

Ordinary baselines:

- `NoMemory`
- `KeywordTrigger`
- `TfidfRawMemory`
- `FullHistory`
- `RawMemorySelfCheck`

Contract-aware baselines:

- `ContractPromptHeuristic`
- `ContractClassifierOnly`
- `ContractCompilerOnly`
- `ContractCheckerOnly`

PACT variants:

- `PACTFull`
- `PACTFull_current`
- `PACT_specificity_gate`
- `PACT_conditional_bonus`
- `PACT_intent_family_gate`
- `PACT_state_action_split`
- `PACT_R2_full`
- `PACT_intent_plus_state`
- `PACT_intent_plus_state_checker`
- `PACT_intent_plus_state_family_compiler`
- `LearnedPAM`
- `LearnedPAM_plus_checker`
- `LearnedPAM_plus_family_compiler`
- `PACT_no_guard`
- `PACT_no_checker`
- `PACT_no_compiler`
- `PACT_raw_memory`
- `PACT_no_conflict_resolver`

Sanity controls:

- `QueryOnlyClassifier`
- `ContractOnlyClassifier`
- `QueryPlusFamilyClassifier`
- `QueryPlusContractClassifier`
- `QueryPlusWrongContractOnly`
- `ContractShufflePACT`
- `LabelPermutationSanity`

## Main Metrics

The primary endpoint is end-to-end prospective success, especially on indirect
triggers. Key metrics include:

- `end_to_end_success_indirect`
- `indirect_action_completion`
- `false_trigger_rate_excluding_contract_swap`
- `false_trigger_rate_including_contract_swap`
- `wrong_contract_false_trigger_rate`
- `fire_precision`
- `paraphrase_consistency`
- `contract_shuffle_drop`

The anti-leakage boundary is strict: predictors receive only an
`InferenceEpisode` with `episode_id`, `history_summary`, `current_query`, and
`available_contract_ids`. Prediction code must not inspect labels or
scoring-only fields.

## Saved Pilot Result

The strict eval repair showed why aggregate false-trigger metrics were too
flattering: ordinary near-miss and wrong-scope false fires were low, but
contract-swap cases exposed wrong-contract activations. That is why reports now
separate ordinary false triggers from wrong-contract false triggers and strict
success from behavioral success.

The saved run in `outputs/` reports the latest all-split evaluation. Before
R2, `PACTFull_current` had strong indirect completion but a nontrivial
wrong-contract false-trigger rate. PACT-R2 tests whether that weakness is a
mechanism bug rather than a fundamental flaw.

## PACT-R2

PACT-R2 is a calibrated applicability intervention. It does not change the
dataset. It adds:

- a specificity gate, so weakly matched contracts cannot fire because of a
  generic action bonus;
- a conditional bonus, so bonuses amplify plausible contracts instead of
  creating activations from nothing;
- an intent-family gate, so a likely user intent family can suppress mismatched
  contracts;
- a state/action split, so explicit conflict detection is separated from safe
  action execution.

R2 thresholds are tuned on the dev split only. The search writes:

- `outputs/r2_threshold_search.csv`
- `outputs/r2_best_config.json`

The saved config is then reused unchanged for test/all reporting.

Run the focused R2 comparison:

```bash
python3 -m pact.run_eval \
  --dataset pact_causal_520 \
  --methods r2 \
  --split test \
  --audit
```

R2 output files include:

- `outputs/r2_variant_metrics.csv`
- `outputs/r2_error_transition.csv`
- `outputs/r2_fixed_errors.csv`
- `outputs/r2_new_errors.csv`
- `outputs/r2_contract_swap_errors.csv`
- `outputs/r2_conflict_errors.csv`
- `outputs/r2_family_metrics.csv`
- `outputs/manual_audit_r2_template.csv`
- `outputs/audit_r2.md`

Interpretation:

- `r2_variant_metrics.csv` compares all R2 variants on primary R2 metrics.
- `r2_error_transition.csv` shows every current-vs-R2 changed episode and
  labels it as fixed, regressed, unchanged-correct, or unchanged-wrong.
- `r2_fixed_errors.csv` and `r2_new_errors.csv` are the fastest way to inspect
  the intervention tradeoff.
- `manual_audit_r2_template.csv` is still a template for human review, not
  completed manual evidence.

## PACT-D3

PACT-D3 is a diagnostic disentanglement suite. It does not try to make the
headline metric look better; it asks where the remaining failures come from:
contract selection, conflict/state prediction, action compilation, completion
scoring, deterministic PAM limits, or dataset/control leakage.

D3 adds:

- composition ablations, especially `PACT_intent_plus_state` and
  `PACT_intent_plus_state_family_compiler`;
- oracle/unfair ceiling variants, clearly marked as diagnostic only;
- an offline deterministic `LearnedPAM` mini-probe and checker/compiler
  variants;
- query/contract controls that restrict which information each classifier can
  see;
- multi-contract retrieval stress views for contract pools of 1, 3, 6, and 12;
- score anatomy, conflict taxonomy, naturalistic-only, and target-completion
  mismatch exports.

Run the focused D3 diagnostic suite:

```bash
python3 -m pact.run_eval \
  --dataset pact_causal_520 \
  --methods d3 \
  --split test \
  --audit
```

The ordinary all-method commands also include the D3 methods and write D3
reports:

```bash
python3 -m pact.run_eval \
  --dataset pact_causal_520 \
  --methods all \
  --split test \
  --audit
```

D3 output files include:

- `outputs/diagnostic_summary.csv`
- `outputs/composition_ablation.csv`
- `outputs/oracle_ceiling.csv`
- `outputs/learned_pam_results.csv`
- `outputs/learned_pam_feature_report.json`
- `outputs/learned_pam_errors.csv`
- `outputs/query_contract_controls.csv`
- `outputs/multi_contract_stress.csv`
- `outputs/score_anatomy.csv`
- `outputs/score_anatomy_by_error_type.csv`
- `outputs/score_anatomy_by_family.csv`
- `outputs/score_anatomy_false_fire_cases.csv`
- `outputs/conflict_taxonomy.csv`
- `outputs/target_completion_mismatches.csv`
- `outputs/naturalistic_metrics.csv`
- `outputs/naturalistic_failures.csv`
- `outputs/d3_error_transition.csv`
- `outputs/d3_fixed_errors.csv`
- `outputs/d3_new_errors.csv`
- `outputs/manual_audit_d3_template.csv`
- `outputs/audit_d3.md`

Interpretation:

- `diagnostic_summary.csv` is the one-table view across current PACT, R2,
  composition variants, LearnedPAM, controls, and oracle ceilings.
- `composition_ablation.csv` tests whether a lighter intent/state composition
  beats full R2 or current PACT.
- `oracle_ceiling.csv` is unfair by design; use it to diagnose bottlenecks, not
  to claim a fair baseline win.
- `query_contract_controls.csv` checks whether query-only or family-only
  controls are too strong, which would threaten the contract-dependence claim.
- `multi_contract_stress.csv` tests whether contract selection degrades as the
  available memory pool grows.
- `target_completion_mismatches.csv` isolates cases where state and contract
  selection look right but target-action scoring still fails.
- `manual_audit_d3_template.csv` is a human-audit template, not completed
  manual evidence.

## Saved Pilot Result

The saved run in `outputs/` reports:

| Method | End-to-end success | Indirect success | Indirect action completion | False trigger rate |
|---|---:|---:|---:|---:|
| `PACTFull_current` | see `metrics_main.json` | see `metrics_main.json` | see `metrics_main.json` | see strict split metrics |
| `PACT_R2_full` | see `metrics_main.json` | see `metrics_main.json` | see `metrics_main.json` | see strict split metrics |
| `QueryOnlyClassifier` | 0.532 | 0.000 | 0.000 | 0.000 |
| `ContractShufflePACT` | 0.572 | 0.000 | 0.000 | 0.000 |

The research-value audit decision is `CONTINUE_WEAK`.

Interpretation:

- PACT variants substantially improve indirect action completion, but strict
  interpretation depends on wrong-contract false-trigger control and conflict
  detection, not just ordinary near-miss suppression.
- The contract-shuffle sanity check degrades performance, which supports that
  the model is using contract identity rather than only query priors.
- The preregistered strong decision did not pass because the bootstrap lower
  bound check failed. Treat this as a promising pilot, not final validation.

## Quick Start

Install the package in editable mode:

```bash
python3 -m pip install -e .
```

Generate the final dataset:

```bash
python3 -m pact.dataset --dataset pact_causal_520 --write
```

Run the final evaluation on the test split:

```bash
python3 -m pact.run_eval \
  --dataset pact_causal_520 \
  --methods all \
  --split test \
  --audit
```

Run all splits:

```bash
python3 -m pact.run_eval \
  --dataset pact_causal_520 \
  --methods all \
  --split all \
  --audit
```

Run tests:

```bash
python3 -m pytest -q
```

Useful focused comparison:

```bash
python3 -m pact.run_eval \
  --dataset pact_causal_520 \
  --methods PACTFull,FullHistory \
  --split dev \
  --bootstrap-iters 1000 \
  --seed 0
```

The legacy dataset is still available:

```bash
python3 -m pact.dataset --dataset pact100_legacy --write
```

## Output Files

Final evaluation writes:

- `outputs/predictions.csv`
- `outputs/metrics_main.json`
- `outputs/metrics_by_family.csv`
- `outputs/metrics_by_case_type.csv`
- `outputs/metrics_by_set_type.csv`
- `outputs/paired_comparisons.csv`
- `outputs/bootstrap_ci.json`
- `outputs/mcnemar_tests.json`
- `outputs/permutation_tests.json`
- `outputs/sanity_checks.json`
- `outputs/errors_pact.csv`
- `outputs/errors_strongest_baseline.csv`
- `outputs/manual_audit_sample.csv`
- `outputs/manual_audit_completed_template.csv`
- `outputs/method_differences.csv`
- `outputs/r2_variant_metrics.csv`
- `outputs/r2_threshold_search.csv`
- `outputs/r2_best_config.json`
- `outputs/r2_error_transition.csv`
- `outputs/r2_fixed_errors.csv`
- `outputs/r2_new_errors.csv`
- `outputs/r2_contract_swap_errors.csv`
- `outputs/r2_conflict_errors.csv`
- `outputs/r2_family_metrics.csv`
- `outputs/manual_audit_r2_template.csv`
- `outputs/diagnostic_summary.csv`
- `outputs/composition_ablation.csv`
- `outputs/oracle_ceiling.csv`
- `outputs/learned_pam_results.csv`
- `outputs/learned_pam_feature_report.json`
- `outputs/learned_pam_errors.csv`
- `outputs/query_contract_controls.csv`
- `outputs/multi_contract_stress.csv`
- `outputs/score_anatomy.csv`
- `outputs/score_anatomy_by_error_type.csv`
- `outputs/score_anatomy_by_family.csv`
- `outputs/score_anatomy_false_fire_cases.csv`
- `outputs/conflict_taxonomy.csv`
- `outputs/target_completion_mismatches.csv`
- `outputs/naturalistic_failures.csv`
- `outputs/naturalistic_metrics.csv`
- `outputs/d3_error_transition.csv`
- `outputs/d3_fixed_errors.csv`
- `outputs/d3_new_errors.csv`
- `outputs/manual_audit_d3_template.csv`
- `outputs/audit_dataset.md`
- `outputs/audit_baselines.md`
- `outputs/audit_causality.md`
- `outputs/audit_metrics.md`
- `outputs/audit_reproducibility.md`
- `outputs/audit_research_value.md`
- `outputs/audit_report.md`
- `outputs/audit_r2.md`
- `outputs/audit_d3.md`

## Preregistered Decision Rule

The research-value audit emits one of:

- `CONTINUE_STRONG`
- `CONTINUE_WEAK`
- `REFORMULATE`
- `KILL`

`CONTINUE_STRONG` requires all preregistered thresholds to pass, including
indirect success, low false-trigger rate, gain over the strongest non-oracle
baseline, bootstrap lower bound, ablation gains, paraphrase robustness, and
contract-shuffle degradation.

Do not treat a weaker decision as final validation.

The D3 diagnostic audit emits one of:

- `DETERMINISTIC_PACT_READY`
- `USE_LEARNED_PAM`
- `FIX_COMPILER_CHECKER`
- `FIX_CONTRACT_SELECTION`
- `NARROW_CLAIM`
- `KILL`

The D3 decision is diagnostic. A positive deterministic decision still needs
the caveats in `outputs/audit_d3.md`, especially naturalistic performance,
remaining conflict failures, target-completion mismatches, and the unfinished
manual audit template.

## Code Map

```text
src/pact/schema.py       dataclasses for contracts, episodes, and predictions
src/pact/dataset.py      deterministic dataset generation
src/pact/baselines.py    ordinary and contract-aware baselines
src/pact/pact.py         PACTFull and ablation implementations
src/pact/scoring.py      metrics and endpoint scoring
src/pact/stats.py        bootstrap, McNemar, permutation tests
src/pact/sanity.py       query-only, shuffle, and label-permutation controls
src/pact/audits.py       audit report generation
src/pact/run_eval.py     evaluation CLI
tests/                   regression tests for dataset, scoring, and outputs
```

## Limitations

- The current run is deterministic and offline; it does not prove behavior for
  a live LLM agent.
- Contract families are synthetic but designed to cover realistic trigger,
  guard, and conflict patterns.
- `CONTINUE_WEAK` means the pilot is worth extending, not that the claim is
  settled.
- The next natural experiment is a small model-backed PACT/agent run while
  preserving the blinded `InferenceEpisode` input boundary.
