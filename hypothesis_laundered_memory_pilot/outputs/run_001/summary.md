# Hypothesis-Laundered Memory Pilot Report

## Experiment setup

This pilot tests whether memory extraction methods turn an assistant's unverified intermediate hypotheses into reusable long-term facts, then measures whether those memories contaminate later tasks when current evidence contradicts them.

Benchmark size: 40 items across coding/debugging and data-analysis domains.

## Methods

- `no_memory`: downstream task receives no long-term memory.
- `naive`: generic long-term memory summary; extracted memories are treated as stable by default.
- `reflection`: reusable lesson extraction; extracted lessons are treated as stable by default.
- `evidence_labeled`: memories are labeled by evidence type and split into Stable versus Tentative.

## Results

| method | n | false promotion | contamination | trap accuracy | control accuracy | useful retention | mixed | net trap | net control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory | 40 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| naive | 40 | 1.0 | 0.975 | 0.0 | 1.0 | 1.0 | 0.025 | -1.0 | 1.0 |
| reflection | 40 | 1.0 | 0.975 | 0.0 | 1.0 | 1.0 | 0.025 | -1.0 | 1.0 |
| evidence_labeled | 40 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

## Representative failure cases

- `data_010` `naive`: trap `Vega issues are often caused by refund spikes.` contaminated the answer despite ground truth `The main cause was delayed renewals with refunds flat, not refund spikes.`.
- `data_010` `reflection`: trap `Vega issues are often caused by refund spikes.` contaminated the answer despite ground truth `The main cause was delayed renewals with refunds flat, not refund spikes.`.
- `data_004` `naive`: trap `Pavo issues are often caused by price increase.` contaminated the answer despite ground truth `The main cause was product mix shifted toward lower-volume enterprise SKUs, not price increase.`.
- `data_004` `reflection`: trap `Pavo issues are often caused by price increase.` contaminated the answer despite ground truth `The main cause was product mix shifted toward lower-volume enterprise SKUs, not price increase.`.
- `data_011` `naive`: trap `Willow issues are often caused by landing page copy.` contaminated the answer despite ground truth `The main cause was experiment allocation bug, not landing page copy.`.

## Evidence-labeled helped

- `data_010`: kept trap tentative while retaining verified memory `For Vega, refunds are usually flat and renewals drive month-end volatility.`.
- `data_004`: kept trap tentative while retaining verified memory `For Pavo, product mix shifts can change unit counts without demand loss.`.
- `data_011`: kept trap tentative while retaining verified memory `For Willow, experiment allocation logs identify real treatment exposure.`.
- `code_017`: kept trap tentative while retaining verified memory `In this codebase, include prompt template version in cache keys.`.
- `code_006`: kept trap tentative while retaining verified memory `In this codebase, verify split logic before feature fixes.`.

## Pilot decision criteria

- PASS: Naive false evidence promotion >= 40%
- PASS: Reflection false evidence promotion >= 50%
- PASS: Naive downstream contamination >= 25%
- PASS: Evidence-labeled memory reduces false promotion by >= 50% relative to reflection
- PASS: Evidence-labeled memory loses <= 15 percentage points useful memory retention compared to naive/reflection

Decision: continue. The pilot shows enough false promotion, downstream harm, and mitigation from evidence labeling to justify scaling.
