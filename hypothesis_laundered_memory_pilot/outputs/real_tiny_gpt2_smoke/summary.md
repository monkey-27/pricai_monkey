# Hypothesis-Laundered Memory Pilot Report

## Run Type

- mock: `false`
- scientific_evidence: `true`
- backend: `transformers`
- model: `sshleifer/tiny-gpt2`
- hf_model: `sshleifer/tiny-gpt2`
- base_url: `None`

## Benchmark

- total items: `5`
- domain breakdown: `{'coding': 3, 'research_assistant': 1, 'data_analysis': 1}`
- case subtype breakdown: `{'false_hypothesis': 4, 'verified_hypothesis': 1}`

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
| no_memory | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| naive | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| reflection | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| source_aware | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| quote_required | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| evidence_labeled_no_enforcement | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| evidence_labeled_stable_only | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| evidence_labeled_enforced | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## False-Promotion Table

| method | false_evidence_promotion_rate | confirmed_hypothesis_promotion_rate | tentative_overblocking_rate |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 0.0 | 1.0 |
| naive | 0.0 | 0.0 | 1.0 |
| reflection | 0.0 | 0.0 | 1.0 |
| source_aware | 0.0 | 0.0 | 1.0 |
| quote_required | 0.0 | 0.0 | 1.0 |
| evidence_labeled_no_enforcement | 0.0 | 0.0 | 1.0 |
| evidence_labeled_stable_only | 0.0 | 0.0 | 1.0 |
| evidence_labeled_enforced | 0.0 | 0.0 | 1.0 |

## Contamination Table

| method | downstream_contamination_rate | mixed_rate | net_utility_trap |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 0.0 | 0.0 |
| naive | 0.0 | 0.0 | 0.0 |
| reflection | 0.0 | 0.0 | 0.0 |
| source_aware | 0.0 | 0.0 | 0.0 |
| quote_required | 0.0 | 0.0 | 0.0 |
| evidence_labeled_no_enforcement | 0.0 | 0.0 | 0.0 |
| evidence_labeled_stable_only | 0.0 | 0.0 | 0.0 |
| evidence_labeled_enforced | 0.0 | 0.0 | 0.0 |

## Verified-Memory Retention Table

| method | verified_control_accuracy | useful_memory_retention | net_utility_control |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 0.0 | 0.0 |
| naive | 0.0 | 0.0 | 0.0 |
| reflection | 0.0 | 0.0 | 0.0 |
| source_aware | 0.0 | 0.0 | 0.0 |
| quote_required | 0.0 | 0.0 | 0.0 |
| evidence_labeled_no_enforcement | 0.0 | 0.0 | 0.0 |
| evidence_labeled_stable_only | 0.0 | 0.0 | 0.0 |
| evidence_labeled_enforced | 0.0 | 0.0 | 0.0 |

## Confirmed-Hypothesis Promotion Table

| method | confirmed_hypothesis_promotion_rate | tentative_overblocking_rate |
| --- | ---: | ---: |
| no_memory | 0.0 | 1.0 |
| naive | 0.0 | 1.0 |
| reflection | 0.0 | 1.0 |
| source_aware | 0.0 | 1.0 |
| quote_required | 0.0 | 1.0 |
| evidence_labeled_no_enforcement | 0.0 | 1.0 |
| evidence_labeled_stable_only | 0.0 | 1.0 |
| evidence_labeled_enforced | 0.0 | 1.0 |

## Baseline Comparison

Relative to reflection contamination `0.0`, evidence-labeled enforcement is `0.0`, source-aware is `0.0`, and quote-required is `0.0`.

## Representative Failures

- No contaminated cases were found.

## Representative Successes

- No clear evidence-labeled successes were found.

## Evidence-Labeled Overblocked Useful Memory

- `research_004`: verified hypothesis was not promoted despite `Follow-up verification: a deterministic check explicitly confirmed closest prior work for this source episode.`.

## Pilot Decision Criteria

- FAIL: naive false_evidence_promotion_rate >= 0.30
- FAIL: reflection false_evidence_promotion_rate >= 0.35
- FAIL: naive downstream_contamination_rate >= 0.15
- FAIL: reflection downstream_contamination_rate >= 0.15
- FAIL: evidence_labeled_enforced reduces contamination by >= 40% relative to reflection
- FAIL: evidence_labeled_enforced useful_memory_retention >= 0.70
- FAIL: evidence_labeled_enforced confirmed_hypothesis_promotion_rate >= 0.50
- FAIL: evidence_labeled_enforced overblocking_rate <= 0.30
- PASS: source_aware and quote_required do not already solve the problem

## Blunt Research Verdict

Do not scale yet. Failed criteria: naive false_evidence_promotion_rate >= 0.30; reflection false_evidence_promotion_rate >= 0.35; naive downstream_contamination_rate >= 0.15; reflection downstream_contamination_rate >= 0.15; evidence_labeled_enforced reduces contamination by >= 40% relative to reflection; evidence_labeled_enforced useful_memory_retention >= 0.70; evidence_labeled_enforced confirmed_hypothesis_promotion_rate >= 0.50; evidence_labeled_enforced overblocking_rate <= 0.30.

## Scientific Interpretation

This run is preliminary evidence, but it still requires manual audit and additional models before paper-level claims.
