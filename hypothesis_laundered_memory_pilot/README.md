# Hypothesis-Laundered Memory Pilot

This repo tests the paper idea **Hypothesis-Laundered Memory: When LLM Agents Store Their Own Reasoning Traces as Evidence**.

The research question is whether memory-augmented LLM agents turn their own unverified intermediate hypotheses into stable long-term memories, then reuse those false memories when later evidence contradicts them.

## Important Evidence Warning

Mock mode is only a pipeline validation tool.

```text
WARNING: This run used mock mode. Mock outputs are programmed to follow the expected pattern and are not scientific evidence.
```

Only completed non-mock runs against real open instruct/chat models with `n >= 80` should be treated as preliminary experimental evidence. Even those require manual audit and more than one model family before paper-level claims.

Runs are classified automatically:

- `mock_pipeline_validation`: mock mode, never scientific evidence.
- `plumbing_smoke`: tiny/non-instruct models or `n < 80`, never scientific evidence.
- `preliminary_experiment`: completed real instruct/chat model with `n >= 80`.
- `paper_candidate_evidence`: real instruct/chat model with `n >= 80` and completed manual audit.
- `failed_run` / `incomplete_run`: attempted runs without complete outputs, never scientific evidence.

Current saved verdict: `NO_SCIENTIFIC_RUNS`. The repo is deployable, but the saved artifacts do not yet contain completed scientific evidence from real instruct models.

## Benchmark

The default seed benchmark has 80 examples:

- 30 data-analysis / business analytics cases
- 30 coding / debugging cases
- 20 research-assistant / literature-review cases

Each item contains a source episode, a plausible unsupported hypothesis, a future task where current evidence matters, and a verified control task where memory should help. It also includes case subtypes:

- `false_hypothesis`
- `verified_hypothesis`
- `ambiguous_hypothesis`

Verified-hypothesis cases check whether methods can promote a hypothesis once explicit verification appears, instead of becoming uselessly conservative.

The paper-grade pilot benchmark is:

```text
data/benchmark_v2.json
```

It has 120 examples: 40 coding/debugging, 40 data-analysis/business analytics, and 40 research-assistant/literature-review cases, with false, verified, and ambiguous hypothesis subtypes.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For direct Transformers inference, also install compatible local ML dependencies:

```bash
pip install transformers torch accelerate
```

## Mock Smoke Test

```bash
python run_pilot.py --mock --n 20 --out outputs/mock_smoke
```

This validates code paths only. Do not cite mock numbers as evidence.

Full deployable mock validation:

```bash
scripts/run_mock_full.sh
```

## Local OpenAI-Compatible Run

Use this with vLLM, llama.cpp server, LM Studio, Ollama OpenAI-compatible mode, or any local OpenAI-compatible server:

```bash
python run_pilot.py \
  --backend openai_compatible \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --model Qwen2.5-7B-Instruct \
  --n 80 \
  --out outputs/real_qwen7b_001 \
  --temperature 0
```

Environment variables also work:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy
```

## Transformers Run

```bash
python run_pilot.py \
  --backend transformers \
  --hf-model Qwen/Qwen2.5-7B-Instruct \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n 80 \
  --out outputs/real_qwen25_7b_001 \
  --temperature 0
```

Small-model smoke tests, such as `scripts/run_transformers_small_smoke.sh`, are useful for plumbing but are not final scientific evidence.

Model configs live in:

```text
configs/open_models.yaml
```

The CLI supports:

```bash
python run_pilot.py \
  --backend transformers \
  --models-config configs/open_models.yaml \
  --model-tier recommended_small \
  --allow-download false \
  --n 80 \
  --out outputs/real_configured_model
```

When `--allow-download false`, Transformers uses local files only and unavailable models should be skipped by the sweep runner rather than faked.

## Memory Methods

Default methods:

```text
no_memory,naive,reflection,source_aware,quote_required,current_evidence_self_check,quote_required_plus_self_check,evidence_labeled_no_enforcement,evidence_labeled_stable_only,evidence_labeled_enforced
```

`evidence_labeled` remains as a backward-compatible alias for `evidence_labeled_enforced`.

## Outputs

Each run writes:

- `run_metadata.json`
- `results_raw.jsonl`
- `case_scores.jsonl`
- `memory_outputs.jsonl`
- `downstream_outputs.jsonl`
- `summary.csv`
- `summary.md`
- `llm_cache.jsonl`
- `manual_audit_sample.csv`
- `manual_audit_instructions.md`

`run_metadata.json` includes:

```json
{
  "mock": false,
  "scientific_evidence": true
}
```

For mock runs, `scientific_evidence` is `false`.

## Metrics

Core metrics:

- `false_evidence_promotion_rate`
- `downstream_contamination_rate`
- `trap_task_accuracy`
- `verified_control_accuracy`
- `useful_memory_retention`
- `confirmed_hypothesis_promotion_rate`
- `tentative_overblocking_rate`

Downstream labels:

- `correct`
- `contaminated`
- `mixed_rejected_trap`
- `mixed_endorsed_trap`
- `irrelevant`
- `unparseable`

## Manual Audit

Each run creates a manual audit sample:

```text
outputs/<run>/manual_audit_sample.csv
outputs/<run>/manual_audit_instructions.md
```

After filling `human_label` and `human_notes`, summarize agreement:

```bash
python -m src.audit_summarizer \
  --audit outputs/<run>/manual_audit_completed.csv \
  --out outputs/<run>/manual_audit_summary.md
```

The audit summarizer reports exact label agreement, contamination agreement, false-promotion agreement when annotated, and examples where automatic scoring disagrees with human labels.

## Scripts

```bash
scripts/run_mock_full.sh
scripts/run_smoke_tiny_transformers.sh
scripts/run_local_openai_compatible.sh
scripts/run_transformers_model.sh
scripts/run_transformers_qwen7b.sh
scripts/run_transformers_small_smoke.sh
scripts/run_recommended_open_models.sh
scripts/run_all_available.sh
scripts/run_scientific_pilot.sh
```

One-command local sweep:

```bash
scripts/run_all_available.sh
```

It runs mock validation, a tiny Transformers smoke if possible, probes a local OpenAI-compatible endpoint, checks configured HF models, skips unavailable models with reasons, and writes:

```text
outputs/experiment_index.md
```

Scientific pilot runner:

```bash
bash scripts/run_scientific_pilot.sh --allow-download true --target minimum
```

Modal helper for cloud GPU attempts:

```bash
python3 -m modal run scripts/modal_scientific_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --n 80 --max-new-tokens 120
```

If a model is gated, unavailable, interrupted, or too slow to produce complete artifacts, the run must be recorded as skipped or incomplete. Do not treat partial metadata as evidence.

## Continuation Criteria

For real open-model runs, continue only if:

```text
naive false_evidence_promotion_rate >= 0.30
reflection false_evidence_promotion_rate >= 0.35
naive downstream_contamination_rate >= 0.15
reflection downstream_contamination_rate >= 0.15
evidence_labeled_enforced reduces contamination by >= 40% relative to reflection
evidence_labeled_enforced useful_memory_retention >= 0.70
evidence_labeled_enforced confirmed_hypothesis_promotion_rate >= 0.50
evidence_labeled_enforced overblocking_rate <= 0.30
source_aware and quote_required do not already solve the problem
```

If source-aware or quote-required provenance performs almost as well as evidence-labeled enforcement, the proposed method is probably not novel enough.

Reports include `Research Verdict` with one of:

```text
MOCK_ONLY
PLUMBING_ONLY
KILL
REDESIGN
CONTINUE_WEAK
CONTINUE_STRONG
PAPER_CANDIDATE
```

## Git Policy For Outputs

Outputs are ignored by default except `.gitkeep`. Do not commit huge real-model caches by accident.

To commit selected small summaries:

```bash
git add outputs/<run>/summary.csv outputs/<run>/summary.md outputs/<run>/run_metadata.json outputs/<run>/manual_audit_sample.csv outputs/experiment_index.md
```

## Test Commands

```bash
python run_pilot.py --mock --n 20 --out outputs/mock_smoke
python -m compileall -q src run_pilot.py
python -m pytest -q
git diff --check
```
