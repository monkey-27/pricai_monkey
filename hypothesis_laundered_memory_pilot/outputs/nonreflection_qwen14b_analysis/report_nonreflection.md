# Non-Reflection Qwen14B Analysis

## Scope

The non-reflection analysis tests whether ordinary memory extraction and evidence-aware variants promote unsupported hypotheses and contaminate later answers. Reflection is excluded because bad reflective lessons are not identical to false evidence promotion.

- source run: `outputs/modal_qwen_qwen2_5_14b_instruct_v2_n80_clean_20260623_rerun1/modal_qwen_qwen2_5_14b_instruct_v2_n80_clean_20260623_rerun1`
- model: `Qwen/Qwen2.5-14B-Instruct`
- backend: `transformers`
- n_items: `80`
- run_role: `preliminary_experiment`
- scientific_evidence: `true`
- benchmark domain breakdown: `{'data_analysis': 26, 'research_assistant': 23, 'coding': 31}`
- benchmark subtype breakdown: `{'false_hypothesis': 53, 'verified_hypothesis': 18, 'ambiguous_hypothesis': 9}`

## What Changes Without Reflection?

Reflection is removed from all headline metrics, aggregate verdicts, and poor-case analysis. The original reflection row is retained only as secondary context:
- original reflection false promotion: `0.45`
- original reflection contamination: `0.4125`

The core result no longer relies on reflection: naive memory still shows high write-time false evidence promotion, while downstream harm is weaker and more mixed.

## Main Non-Reflection Metrics

| method | false_evidence_promotion_rate | downstream_contamination_rate | contamination_delta_vs_no_memory | trap_task_accuracy | useful_memory_retention | confirmed_hypothesis_promotion_rate | tentative_overblocking_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 0.0 | 0.3 | 0.0 | 0.7 | 0.5875 | 0.0 | 1.0 |
| naive | 0.7375 | 0.3 | 0.0 | 0.7 | 0.95 | 0.6111 | 0.3889 |
| source_aware | 0.15 | 0.325 | 0.025 | 0.675 | 0.9625 | 0.7778 | 0.2222 |
| quote_required | 0.05 | 0.375 | 0.075 | 0.625 | 0.9875 | 0.5556 | 0.4444 |
| evidence_labeled_no_enforcement | 0.0 | 0.475 | 0.175 | 0.525 | 0.9875 | 0.0 | 1.0 |
| evidence_labeled_stable_only | 0.0 | 0.325 | 0.025 | 0.675 | 1.0 | 0.0 | 1.0 |
| evidence_labeled_enforced | 0.0 | 0.475 | 0.175 | 0.525 | 0.925 | 0.0 | 1.0 |
| current_evidence_self_check | 0.675 | 0.275 | -0.025 | 0.725 | 0.95 | 0.6111 | 0.3889 |
| quote_required_plus_self_check | 0.05 | 0.325 | 0.025 | 0.675 | 0.9375 | 0.5556 | 0.4444 |

## Direct Answers

1. **What happens if reflection is removed?** The headline phenomenon becomes narrower: write-time laundering remains strong for naive memory (`0.7375`), but naive downstream contamination is `0.3`, equal to no-memory `0.3`.
2. **Does naive memory still show false evidence promotion?** Yes. Naive false evidence promotion is `0.7375`.
3. **Does any non-reflection method increase contamination relative to no-memory?** Yes: source_aware, quote_required, evidence_labeled_no_enforcement, evidence_labeled_stable_only, evidence_labeled_enforced, quote_required_plus_self_check.
4. **Which method performs best on contamination?** `current_evidence_self_check` at `0.275`.
5. **Which method performs best while preserving useful memory?** `current_evidence_self_check` is the best contamination/retention tradeoff among methods with retention >= 0.90: contamination `0.275`, retention `0.95`. The highest raw retention is `evidence_labeled_stable_only` at `1.0`.
6. **Does evidence-labeled-enforced still fail?** Yes. It has false promotion `0.0`, but contamination `0.475`, worse than no-memory `0.3`, and confirmed-hypothesis promotion `0.0` with overblocking `1.0`.
7. **Is stable_only or current_evidence_self_check more promising?** `current_evidence_self_check` is better on contamination (`0.275` vs stable_only `0.325`), while stable_only is better on retention (`1.0` vs self-check `0.95`). Self-check is the cleaner next intervention for contamination; stable-only is a retention-heavy baseline that still overblocks confirmed hypotheses.
8. **Does quote_required already solve the problem?** No. It nearly solves write-time false promotion (`0.05`), but still contaminates at `0.375`, above no-memory by `0.075`.
9. **Cleanest revised scientific claim.** Ordinary memory extraction can launder unsupported hypotheses into stable memories, but this Qwen14B run does not show a clean naive-memory downstream harm above no-memory. The write-time laundering phenomenon appears stronger than the downstream memory-caused harm in this run.

## Interpretation

Naive memory has high false evidence promotion (`0.7375`), so the write-time laundering mechanism survives without reflection. Naive downstream contamination is `0.3`, compared with no-memory `0.3`; this is not a clean increase. Evidence-labeled-enforced eliminates stable false promotion but still contaminates at `0.475` and overblocks verified hypotheses at `1.0`. Source-aware and quote-required reduce false promotion (`0.15` and `0.05`) but do not eliminate downstream contamination.

Labeling a memory as tentative does not prevent anchoring if the tentative memory is still shown downstream.

The write-time laundering phenomenon appears stronger than the downstream memory-caused harm in this run.

## Poor-Case Summary

- total poor rows excluding reflection: `294`
- poor rows by method: `{'source_aware': 29, 'quote_required_plus_self_check': 30, 'evidence_labeled_no_enforcement': 38, 'evidence_labeled_enforced': 44, 'naive': 27, 'quote_required': 31, 'current_evidence_self_check': 24, 'no_memory': 45, 'evidence_labeled_stable_only': 26}`
- detailed CSV: `poor_cases_nonreflection.csv`
- grouped examples: `poor_cases_by_method.md`

## Revised Verdict

REDESIGN_NARROW: keep the paper idea focused on unsupported-hypothesis memory promotion, but do not claim strong naive-memory downstream harm from this run. Evidence-labeled-enforced fails as an intervention; current-evidence self-check is the strongest contamination baseline at `0.275`.
