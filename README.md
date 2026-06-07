# PACT-Causal-520

Final offline pilot for **Prospective Action Contracts for LLM Agents**.

PACT treats selected memories as delayed behavioral contracts:

> When cue C occurs, if guard G is satisfied, perform action A, then verify check K.

The primary claim is that ordinary memory retrieval can remember an instruction, but cannot reliably decide when that instruction should become an action. This repo evaluates that claim with a deterministic, no-API pilot.

## Commands

Install:

```bash
python3 -m pip install -e .
```

Generate legacy dataset:

```bash
python3 -m pact.dataset --dataset pact100_legacy --write
```

Generate final dataset:

```bash
python3 -m pact.dataset --dataset pact_causal_520 --write
```

Run final evaluation:

```bash
python3 -m pact.run_eval --dataset pact_causal_520 --methods all --split test --audit
```

Run all splits:

```bash
python3 -m pact.run_eval --dataset pact_causal_520 --methods all --split all --audit
```

Run tests:

```bash
python3 -m pytest -q
```

Useful options:

```bash
python3 -m pact.run_eval --dataset pact_causal_520 --methods PACTFull,FullHistory --split dev --bootstrap-iters 1000 --seed 0
```

## Dataset

`pact_causal_520` has 520 episodes:

- Controlled causal set: 360 episodes
- Paraphrase robustness set: 120 episodes
- Naturalistic transcript set: 40 episodes

There are 12 contract families. The original 10 are preserved, with two harder families added:

- `legal_policy_caution`
- `data_analysis_hygiene`

The old 100-episode prototype remains available as `pact100_legacy`.

## Methods

Ordinary baselines:

- `NoMemory`
- `KeywordTrigger`
- `TfidfRawMemory`
- `FullHistory`
- `RawMemorySelfCheck`

Strong contract-aware baselines:

- `ContractPromptHeuristic`
- `ContractClassifierOnly`
- `ContractCompilerOnly`
- `ContractCheckerOnly`

PACT variants:

- `PACTFull`
- `PACT_no_guard`
- `PACT_no_checker`
- `PACT_no_compiler`
- `PACT_raw_memory`
- `PACT_no_conflict_resolver`

Sanity controls:

- `QueryOnlyClassifier`
- `ContractShufflePACT`
- `LabelPermutationSanity`

LLM hooks are present as disabled stubs. The default run makes no external API calls.

## Outputs

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
- `outputs/audit_dataset.md`
- `outputs/audit_baselines.md`
- `outputs/audit_causality.md`
- `outputs/audit_metrics.md`
- `outputs/audit_reproducibility.md`
- `outputs/audit_research_value.md`
- `outputs/audit_report.md`

## Primary Endpoint

Primary metric: end-to-end prospective success.

Headline:

**Indirect trigger action completion at low false-trigger rate.**

Key test metrics:

- `end_to_end_success_indirect`
- `indirect_action_completion`
- `false_trigger_rate`
- `paraphrase_consistency`
- `contract_shuffle_drop` in `sanity_checks.json`

## Preregistered Decision

The research-value audit emits exactly one of:

- `CONTINUE_STRONG`
- `CONTINUE_WEAK`
- `REFORMULATE`
- `KILL`

`CONTINUE_STRONG` requires all preregistered thresholds to pass on test, including indirect success, low false-trigger rate, gain over the strongest non-oracle baseline, bootstrap lower bound, component ablation gains, paraphrase robustness, and contract-shuffle degradation.

Do not treat a weaker decision as validation.

## Anti-Leakage Boundary

Predictors receive only `InferenceEpisode`:

- `episode_id`
- `history_summary`
- `current_query`
- `available_contract_ids`

Prediction modules must not inspect label or scoring-only fields. Tests and audits scan `baselines.py` and `pact.py` for forbidden field references.

