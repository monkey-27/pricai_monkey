# Hypothesis-Laundered Memory Pilot

This repo tests the paper idea **Hypothesis-Laundered Memory: When LLM Agents Store Their Own Reasoning Traces as Evidence**.

The research question is whether memory-augmented LLM agents turn their own unverified intermediate hypotheses into stable long-term memories, then reuse those false memories when later evidence contradicts them.

## Important Evidence Warning

Mock mode is only a pipeline validation tool.

```text
WARNING: This run used mock mode. Mock outputs are programmed to follow the expected pattern and are not scientific evidence.
```

Only non-mock runs against open/local models should be treated as preliminary experimental evidence. Even those require manual audit and more than one model before paper-level claims.

## Benchmark

The generated seed benchmark has 80 examples:

- 30 data-analysis / business analytics cases
- 30 coding / debugging cases
- 20 research-assistant / literature-review cases

Each item contains a source episode, a plausible unsupported hypothesis, a future task where current evidence matters, and a verified control task where memory should help. It also includes case subtypes:

- `false_hypothesis`
- `verified_hypothesis`
- `ambiguous_hypothesis`

Verified-hypothesis cases check whether methods can promote a hypothesis once explicit verification appears, instead of becoming uselessly conservative.

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

## Memory Methods

Default methods:

```text
no_memory,naive,reflection,source_aware,quote_required,evidence_labeled_no_enforcement,evidence_labeled_stable_only,evidence_labeled_enforced
```

`evidence_labeled` remains as a backward-compatible alias for `evidence_labeled_enforced`.

## Outputs

Each run writes:

- `run_metadata.json`
- `results_raw.jsonl`
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

## Scripts

```bash
scripts/run_mock_full.sh
scripts/run_local_openai_compatible.sh
scripts/run_transformers_qwen7b.sh
scripts/run_transformers_small_smoke.sh
```

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

## Git Policy For Outputs

Outputs are ignored by default except `.gitkeep`. Do not commit huge real-model caches by accident.

To commit selected small summaries:

```bash
git add outputs/<run>/summary.csv outputs/<run>/summary.md outputs/<run>/manual_audit_sample.csv
```

## Test Commands

```bash
python run_pilot.py --mock --n 20 --out outputs/mock_smoke
python -m compileall -q src run_pilot.py
python -m pytest -q
git diff --check
```

