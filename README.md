# PACT Pilot

Fast offline pilot for **Prospective Action Contracts for LLM Agents**.

PACT treats selected memories as delayed behavioral contracts:

> When cue C occurs, if guard G is satisfied, perform action A, then verify check K.

The pilot tests whether structured contracts improve delayed intention activation and action completion compared with ordinary no-memory, keyword, and TF-IDF memory baselines.

## Install

```bash
pip install -e .
```

Optional packages:

```bash
pip install -e ".[dev,data]"
```

The default run uses only Python stdlib. If `scikit-learn` is installed, the TF-IDF baseline uses it; otherwise it falls back to a deterministic bag-of-words cosine score.

## Generate Dataset

```bash
python -m pact.dataset --write
```

This writes:

- `data/contracts.json`
- `data/episodes.jsonl`

The dataset has 10 contract families and 100 episodes. Each family has:

- 2 direct triggers
- 3 indirect triggers
- 3 near misses
- 1 wrong scope
- 1 conflict or already-satisfied case

## Run Evaluation

```bash
python -m pact.run_eval --methods all --audit
```

Outputs:

- `outputs/predictions.csv`
- `outputs/metrics.json`
- `outputs/audit_report.md`
- `outputs/audit_dataset.md`
- `outputs/audit_baselines.md`
- `outputs/audit_metrics.md`
- `outputs/audit_reproducibility.md`
- `outputs/audit_research_value.md`

## Run Tests

```bash
pytest -q
```

## Methods

- `NoMemoryBaseline`: always suppresses; sanity baseline.
- `KeywordTriggerBaseline`: fires on simple keyword overlap.
- `TfidfMemoryBaseline`: retrieves top-1 raw contract memory and fires above a fixed threshold.
- `ContractPromptHeuristicBaseline`: deterministic contract-aware heuristic.
- `PACT_no_guard`: ablation without guard penalty.
- `PACT_no_checker`: ablation without repair.
- `PACT_raw_memory_instead_of_contract`: ablation using unstructured raw-memory retrieval.
- `PACTFull`: structured retrieval, deterministic PAM, action compiler, checker, and repair.

## Headline Metric

The important pilot metric is:

**Action completion under indirect triggers at low false-trigger rate.**

Inspect these fields in `outputs/metrics.json`:

- `action_completion_rate_indirect_fire`
- `indirect_trigger_recall`
- `false_trigger_rate_near_wrong`

The direction is promising only if PACT improves indirect-trigger action completion while keeping near-miss and wrong-scope false triggers low.

## Anti-Leakage Rule

Prediction methods receive only `EpisodeInput`:

- `episode_id`
- `history_summary`
- `current_query`

They must not inspect:

- `gold_state`
- `case_type`
- `expected_action_keywords`
- `forbidden_action_keywords`

Tests and the baseline audit scan prediction modules for those field names.

## Audit Model

This environment did not expose true subagents, so the pilot simulates subagents as separate named audit phases in `src/pact/audits.py`:

- Dataset Audit Agent
- Baseline Fairness Audit Agent
- Metric Audit Agent
- Reproducibility Audit Agent
- Research-Value Audit Agent

Each audit writes its own markdown report in `outputs/`.

