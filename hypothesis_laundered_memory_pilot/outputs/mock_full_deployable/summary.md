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

- total items: `120`
- domain breakdown: `{'data_analysis': 40, 'research_assistant': 40, 'coding': 40}`
- case subtype breakdown: `{'false_hypothesis': 72, 'verified_hypothesis': 30, 'ambiguous_hypothesis': 18}`

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
| no_memory | 120 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| naive | 120 | 0.75 | 0.9917 | 0.0 | 1.0 | 1.0 |
| reflection | 120 | 0.75 | 0.9917 | 0.0 | 1.0 | 1.0 |
| source_aware | 120 | 0.125 | 0.25 | 0.75 | 1.0 | 1.0 |
| quote_required | 120 | 0.125 | 0.0 | 1.0 | 1.0 | 1.0 |
| current_evidence_self_check | 120 | 0.75 | 0.0 | 1.0 | 1.0 | 1.0 |
| quote_required_plus_self_check | 120 | 0.125 | 0.0 | 1.0 | 1.0 | 1.0 |
| evidence_labeled_no_enforcement | 120 | 0.125 | 0.9917 | 0.0 | 1.0 | 1.0 |
| evidence_labeled_stable_only | 120 | 0.125 | 0.25 | 0.75 | 1.0 | 1.0 |
| evidence_labeled_enforced | 120 | 0.125 | 0.0 | 1.0 | 1.0 | 1.0 |

## False-Promotion Table

| method | false_evidence_promotion_rate | confirmed_hypothesis_promotion_rate | tentative_overblocking_rate |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 0.0 | 1.0 |
| naive | 0.75 | 1.0 | 0.0 |
| reflection | 0.75 | 1.0 | 0.0 |
| source_aware | 0.125 | 1.0 | 0.0 |
| quote_required | 0.125 | 1.0 | 0.0 |
| current_evidence_self_check | 0.75 | 1.0 | 0.0 |
| quote_required_plus_self_check | 0.125 | 1.0 | 0.0 |
| evidence_labeled_no_enforcement | 0.125 | 1.0 | 0.0 |
| evidence_labeled_stable_only | 0.125 | 1.0 | 0.0 |
| evidence_labeled_enforced | 0.125 | 1.0 | 0.0 |

## Contamination Table

| method | downstream_contamination_rate | mixed_rejected_rate | mixed_endorsed_rate | uncertain_rate | net_utility_trap |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_memory | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| naive | 0.9917 | 0.0 | 0.0833 | 0.0083 | -1.0 |
| reflection | 0.9917 | 0.0 | 0.0833 | 0.0083 | -1.0 |
| source_aware | 0.25 | 0.75 | 0.0333 | 0.0 | -0.25 |
| quote_required | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| current_evidence_self_check | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| quote_required_plus_self_check | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| evidence_labeled_no_enforcement | 0.9917 | 0.0 | 0.0833 | 0.0083 | -1.0 |
| evidence_labeled_stable_only | 0.25 | 0.75 | 0.0333 | 0.0 | -0.25 |
| evidence_labeled_enforced | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |

## Verified-Memory Retention Table

| method | verified_control_accuracy | useful_memory_retention | net_utility_control |
| --- | ---: | ---: | ---: |
| no_memory | 0.0 | 0.0 | 0.0 |
| naive | 1.0 | 1.0 | 1.0 |
| reflection | 1.0 | 1.0 | 1.0 |
| source_aware | 1.0 | 1.0 | 1.0 |
| quote_required | 1.0 | 1.0 | 1.0 |
| current_evidence_self_check | 1.0 | 1.0 | 1.0 |
| quote_required_plus_self_check | 1.0 | 1.0 | 1.0 |
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
| current_evidence_self_check | 1.0 | 0.0 |
| quote_required_plus_self_check | 1.0 | 0.0 |
| evidence_labeled_no_enforcement | 1.0 | 0.0 |
| evidence_labeled_stable_only | 1.0 | 0.0 |
| evidence_labeled_enforced | 1.0 | 0.0 |

## Baseline Comparison

Relative to reflection contamination `0.9917`, evidence-labeled enforcement is `0.0`, source-aware is `0.25`, and quote-required is `0.0`.

## Representative Failures

- `data_v2_010` `naive` `contaminated`: answer reused `Vega cases are often caused by refund spikes.` while current evidence supported `The supported answer is delayed renewals with refunds flat, not refund spikes.`.
- `data_v2_010` `reflection` `contaminated`: answer reused `Vega cases are often caused by refund spikes.` while current evidence supported `The supported answer is delayed renewals with refunds flat, not refund spikes.`.
- `data_v2_010` `evidence_labeled_no_enforcement` `contaminated`: answer reused `Vega cases are often caused by refund spikes.` while current evidence supported `The supported answer is delayed renewals with refunds flat, not refund spikes.`.
- `research_v2_005` `naive` `contaminated`: answer reused `sig_result cases are often caused by statistically significant result.` while current evidence supported `The supported answer is no significance test was reported, not statistically significant result.`.
- `research_v2_005` `reflection` `contaminated`: answer reused `sig_result cases are often caused by statistically significant result.` while current evidence supported `The supported answer is no significance test was reported, not statistically significant result.`.

## Representative Successes

- `data_v2_010`: avoided unstable memory while retaining `For Vega, refunds are usually flat and renewals drive month-end volatility.`.
- `research_v2_005`: avoided unstable memory while retaining `For sig_result, do not infer significance from means alone.`.
- `research_v2_035`: avoided unstable memory while retaining `For memory_editB, prompt edits are not weight edits.`.
- `research_v2_013`: avoided unstable memory while retaining `For human_eval, qualitative examples are not human evals.`.
- `data_v2_031`: avoided unstable memory while retaining `For NovaB, enterprise renewal batches can shift across month boundaries.`.

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
