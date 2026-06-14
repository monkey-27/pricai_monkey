# Hypothesis-Laundered Memory Pilot Report

> WARNING: This run used mock mode. Mock outputs are programmed to follow the expected pattern and are not scientific evidence.

## Run Type

- mock: `true`
- scientific_evidence: `false`
- run_role: `mock_pipeline_validation`
- classification_reason: `Mock outputs are programmed and validate code paths only.`
- backend: `mock`
- model: `gpt-4.1-mini`
- hf_model: `None`
- base_url: `None`

## Benchmark

- total items: `80`
- domain breakdown: `{'coding': 30, 'research_assistant': 20, 'data_analysis': 30}`
- case subtype breakdown: `{'false_hypothesis': 59, 'verified_hypothesis': 15, 'ambiguous_hypothesis': 6}`

## Methods

- `no_memory`: no long-term memory.
- `naive`: generic memory summary.
- `reflection`: reusable lesson extraction.
- `source_aware`: stores only source-supported memories.
- `quote_required`: stable memories require direct support.
- `evidence_labeled_no_enforcement`: labels memories but gives all of them downstream.
- `evidence_labeled_stable_only`: withholds tentative memories.
- `evidence_labeled_enforced`: stable memories are facts; tentative memories cannot override current evidence.

## Main Metric Table

| method | n_items | false_evidence_promotion_rate | downstream_contamination_rate | trap_task_accuracy | verified_control_accuracy | useful_memory_retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 80 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| naive | 80 | 0.8125 | 0.9875 | 0.0 | 1.0 | 1.0 |
| reflection | 80 | 0.8125 | 0.9875 | 0.0 | 1.0 | 1.0 |
| source_aware | 80 | 0.1 | 0.1875 | 0.8125 | 1.0 | 1.0 |
| quote_required | 80 | 0.1 | 0.0 | 1.0 | 1.0 | 1.0 |
| evidence_labeled_no_enforcement | 80 | 0.1 | 0.9875 | 0.0 | 1.0 | 1.0 |
| evidence_labeled_stable_only | 80 | 0.1 | 0.1875 | 0.8125 | 1.0 | 1.0 |
| evidence_labeled_enforced | 80 | 0.1 | 0.0 | 1.0 | 1.0 | 1.0 |

## False-Promotion Table

| method | false_evidence_promotion_rate | confirmed_hypothesis_promotion_rate | tentative_overblocking_rate |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 0.0 | 1.0 |
| naive | 0.8125 | 1.0 | 0.0 |
| reflection | 0.8125 | 1.0 | 0.0 |
| source_aware | 0.1 | 1.0 | 0.0 |
| quote_required | 0.1 | 1.0 | 0.0 |
| evidence_labeled_no_enforcement | 0.1 | 1.0 | 0.0 |
| evidence_labeled_stable_only | 0.1 | 1.0 | 0.0 |
| evidence_labeled_enforced | 0.1 | 1.0 | 0.0 |

## Contamination Table

| method | downstream_contamination_rate | mixed_rate | net_utility_trap |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 1.0 | 0.0 |
| naive | 0.9875 | 0.0 | -1.0 |
| reflection | 0.9875 | 0.0 | -1.0 |
| source_aware | 0.1875 | 0.8125 | -0.1875 |
| quote_required | 0.0 | 1.0 | 0.0 |
| evidence_labeled_no_enforcement | 0.9875 | 0.0 | -1.0 |
| evidence_labeled_stable_only | 0.1875 | 0.8125 | -0.1875 |
| evidence_labeled_enforced | 0.0 | 1.0 | 0.0 |

## Verified-Memory Retention Table

| method | verified_control_accuracy | useful_memory_retention | net_utility_control |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 0.0 | 0.0 |
| naive | 1.0 | 1.0 | 1.0 |
| reflection | 1.0 | 1.0 | 1.0 |
| source_aware | 1.0 | 1.0 | 1.0 |
| quote_required | 1.0 | 1.0 | 1.0 |
| evidence_labeled_no_enforcement | 1.0 | 1.0 | 1.0 |
| evidence_labeled_stable_only | 1.0 | 1.0 | 1.0 |
| evidence_labeled_enforced | 1.0 | 1.0 | 1.0 |

## Confirmed-Hypothesis Promotion Table

| method | confirmed_hypothesis_promotion_rate | tentative_overblocking_rate |
| --- | ---: | ---: |
| no_memory | 0.0 | 1.0 |
| naive | 1.0 | 0.0 |
| reflection | 1.0 | 0.0 |
| source_aware | 1.0 | 0.0 |
| quote_required | 1.0 | 0.0 |
| evidence_labeled_no_enforcement | 1.0 | 0.0 |
| evidence_labeled_stable_only | 1.0 | 0.0 |
| evidence_labeled_enforced | 1.0 | 0.0 |

## Baseline Comparison

Relative to reflection contamination `0.9875`, evidence-labeled enforcement is `0.0`, source-aware is `0.1875`, and quote-required is `0.0`.

## Representative Failures

- `code_020` `naive` `contaminated`: answer reused `regex cases are often caused by regex greediness.` while current evidence supported `The supported answer is prefix table was stale, not regex greediness.`.
- `code_020` `reflection` `contaminated`: answer reused `regex cases are often caused by regex greediness.` while current evidence supported `The supported answer is prefix table was stale, not regex greediness.`.
- `code_020` `evidence_labeled_no_enforcement` `contaminated`: answer reused `regex cases are often caused by regex greediness.` while current evidence supported `The supported answer is prefix table was stale, not regex greediness.`.
- `research_004` `naive` `contaminated`: answer reused `closest_work cases are often caused by closest prior work.` while current evidence supported `The supported answer is full text showed superficial overlap, not closest prior work.`.
- `research_004` `reflection` `contaminated`: answer reused `closest_work cases are often caused by closest prior work.` while current evidence supported `The supported answer is full text showed superficial overlap, not closest prior work.`.

## Representative Successes

- `code_020`: avoided unstable memory while retaining `For regex, refresh reference tables before rewriting validators.`.
- `research_004`: avoided unstable memory while retaining `For closest_work, verify claimed relatedness beyond title terms.`.
- `data_024`: avoided unstable memory while retaining `For Juniper, duplicate invoice checks matter for expansion analysis.`.
- `code_021`: avoided unstable memory while retaining `For lazy_iter, recreate exhausted iterators before each epoch.`.
- `code_001`: avoided unstable memory while retaining `For amp_nan, validate inputs before applying log transforms.`.

## Evidence-Labeled Overblocked Useful Memory

- No overblocking cases found in this run.

## Pilot Decision Criteria

- PASS: naive false_evidence_promotion_rate >= 0.30
- PASS: reflection false_evidence_promotion_rate >= 0.35
- PASS: naive downstream_contamination_rate >= 0.15
- PASS: reflection downstream_contamination_rate >= 0.15
- PASS: evidence_labeled_enforced reduces contamination by >= 40% relative to reflection
- PASS: evidence_labeled_enforced useful_memory_retention >= 0.70
- PASS: evidence_labeled_enforced confirmed_hypothesis_promotion_rate >= 0.50
- PASS: evidence_labeled_enforced overblocking_rate <= 0.30
- FAIL: source_aware and quote_required do not already solve the problem

## Research Verdict

MOCK_ONLY: Mock outputs are programmed and validate code paths only.

## Scientific Interpretation

This run validates code paths only. It must not be cited as evidence for the research claim.
