# Hypothesis-Laundered Memory Pilot

This repo is a fast pilot for the paper idea **Hypothesis-Laundered Memory: When LLM Agents Store Their Own Reasoning Traces as Evidence**.

The pilot asks a narrow question: can ordinary memory extraction methods tell the difference between externally supported evidence and an assistant's own unverified intermediate hypotheses? It then tests whether those hypotheses become stable long-term memories and contaminate later reasoning when fresh evidence contradicts them.

## Why This Matters

Memory-augmented agents are often asked to summarize prior work into reusable facts. That is useful when the memory comes from user statements, tool outputs, tests, calculations, or cited sources. It is dangerous when the memory is only a hypothesis that appeared in the assistant's reasoning trace.

This pilot calls that failure mode **hypothesis laundering**: a speculative explanation becomes a durable memory, and later tasks treat it as evidence.

## What The Benchmark Tests

The benchmark contains 40 controlled items:

- 20 data-analysis / business analytics cases
- 20 coding / debugging cases

Each item has:

- a source episode with external evidence, assistant trace, and final response
- a plausible but unverified trap hypothesis
- a future trap task where current evidence contradicts that hypothesis
- a verified control memory and future task where remembering should help

The benchmark is intentionally not one-sided. Many examples contain hidden data or code issues, and every item includes a verified control memory so a method cannot win by storing nothing.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Mock Mode

Mock mode does not require API keys and exercises the full pipeline:

```bash
python run_pilot.py --mock --n 10 --out outputs/mock_run
```

Full mock benchmark:

```bash
python run_pilot.py --mock --n 40 --out outputs/mock_full
```

## Run Real LLM Mode

Set an OpenAI-compatible API key:

```bash
cp .env.example .env
export OPENAI_API_KEY=...
python run_pilot.py --model gpt-4.1-mini --n 40 --out outputs/run_001
```

LLM calls are cached under the output directory in `llm_cache.jsonl`, keyed by prompt hash.

## CLI

```bash
python run_pilot.py \
  --model gpt-4.1-mini \
  --n 40 \
  --domains coding,data_analysis \
  --methods no_memory,naive,reflection,evidence_labeled \
  --out outputs/run_001 \
  --seed 42 \
  --temperature 0 \
  --max-tokens 800
```

## Memory Methods

- `no_memory`: no long-term memory is passed downstream.
- `naive`: generic memory summary. Any extracted memory is treated as stable.
- `reflection`: reflective lesson extraction. Any extracted lesson is treated as stable.
- `evidence_labeled`: labels memories as Observed, Executed, Cited, Inferred, Hypothesized, or Self-derived. Only Observed, Executed, and Cited memories may be Stable.

## Metrics

The run writes:

- `results_raw.jsonl`
- `memory_outputs.jsonl`
- `downstream_outputs.jsonl`
- `summary.csv`
- `summary.md`
- `llm_cache.jsonl`

Key metrics:

- **False Evidence Promotion**: the trap hypothesis was stored as a stable/reusable fact.
- **Downstream Contamination**: the future answer used the false trap despite contradictory current evidence.
- **Trap Task Accuracy**: the future answer identified the correct cause or fix.
- **Verified Control Accuracy**: the system used a genuinely verified memory correctly.
- **Useful Memory Retention**: the memory method retained verified information that helped the control task.
- **Net Memory Utility**: memory accuracy minus no-memory accuracy on trap and control tasks.

## Scale-Up Criteria

The report recommends continuing only if:

```text
Naive false evidence promotion >= 40%
Reflection false evidence promotion >= 50%
Naive downstream contamination >= 25%
Evidence-labeled memory reduces false promotion by >= 50% relative to reflection
Evidence-labeled memory loses <= 15 percentage points useful memory retention compared to naive/reflection
```

If the criteria fail, `summary.md` names which ones failed.

## Repo Map

```text
hypothesis_laundered_memory_pilot/
  run_pilot.py              # CLI entry point
  src/benchmark.py          # controlled benchmark generation/loading
  src/llm.py                # OpenAI-compatible wrapper plus mock mode/cache
  src/prompts.py            # memory and downstream prompts
  src/scorers.py            # deterministic scoring
  src/experiment.py         # end-to-end run orchestration
  src/report.py             # summary CSV and Markdown report
  data/benchmark_seed.json  # generated seed benchmark, 40 items
  outputs/                  # run artifacts
```

