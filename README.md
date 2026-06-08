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
- `PACT_no_guard`
- `PACT_no_checker`
- `PACT_no_compiler`
- `PACT_raw_memory`
- `PACT_no_conflict_resolver`

Sanity controls:

- `QueryOnlyClassifier`
- `ContractShufflePACT`
- `LabelPermutationSanity`

## Main Metrics

The primary endpoint is end-to-end prospective success, especially on indirect
triggers. Key metrics include:

- `end_to_end_success_indirect`
- `indirect_action_completion`
- `false_trigger_rate`
- `fire_precision`
- `paraphrase_consistency`
- `contract_shuffle_drop`

The anti-leakage boundary is strict: predictors receive only an
`InferenceEpisode` with `episode_id`, `history_summary`, `current_query`, and
`available_contract_ids`. Prediction code must not inspect labels or
scoring-only fields.

## Saved Pilot Result

The saved run in `outputs/` reports:

| Method | End-to-end success | Indirect success | Indirect action completion | False trigger rate |
|---|---:|---:|---:|---:|
| `PACTFull` | 0.874 | 0.877 | 0.877 | 0.008 |
| `QueryOnlyClassifier` | 0.532 | 0.000 | 0.000 | 0.000 |
| `ContractShufflePACT` | 0.572 | 0.000 | 0.000 | 0.000 |

The research-value audit decision is `CONTINUE_WEAK`.

Interpretation:

- PACTFull substantially improves indirect action completion while keeping
  false triggers low in this deterministic pilot.
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
- `outputs/audit_dataset.md`
- `outputs/audit_baselines.md`
- `outputs/audit_causality.md`
- `outputs/audit_metrics.md`
- `outputs/audit_reproducibility.md`
- `outputs/audit_research_value.md`
- `outputs/audit_report.md`

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

