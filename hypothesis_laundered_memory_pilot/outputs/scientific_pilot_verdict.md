# Scientific Pilot Verdict

Final verdict: `NO_SCIENTIFIC_RUNS`

## Models Actually Run

- `mock_full_deployable`: model=`gpt-4.1-mini`, backend=`mock`, n=`120`, role=`mock_pipeline_validation`, scientific=`False`
- `mock_full_latest`: model=`gpt-4.1-mini`, backend=`mock`, n=`80`, role=`mock_pipeline_validation`, scientific=`False`
- `mock_smoke`: model=`gpt-4.1-mini`, backend=`mock`, n=`20`, role=`mock_pipeline_validation`, scientific=`False`
- `modal_google_gemma_3_1b_it_v2_n80`: model=`google/gemma-3-1b-it`, backend=`transformers`, n=`80`, role=`failed_run`, scientific=`False`
- `real_qwen_qwen2_5_1_5b_instruct_v2_n40`: model=`Qwen/Qwen2.5-1.5B-Instruct`, backend=`transformers`, n=`40`, role=`incomplete_run`, scientific=`False`
- `real_qwen_qwen2_5_7b_instruct_v2_n120`: model=`Qwen/Qwen2.5-7B-Instruct`, backend=`transformers`, n=`120`, role=`incomplete_run`, scientific=`False`
- `real_tiny_gpt2_smoke`: model=`sshleifer/tiny-gpt2`, backend=`transformers`, n=`5`, role=`incomplete_run`, scientific=`False`
- `smoke_tiny_gpt2_deployable`: model=`sshleifer/tiny-gpt2`, backend=`transformers`, n=`5`, role=`plumbing_smoke`, scientific=`False`

## Models Skipped

- `local-open-instruct` for `local_openai_compatible_v2_n120`: No server reachable at http://localhost:8000/v1
- `google/gemma-3-1b-it` for `modal_google_gemma_3_1b_it_v2_n80`: Hugging Face returned 401 Unauthorized for gated repository without HF_TOKEN.
- `Qwen/Qwen2.5-1.5B-Instruct` for `modal_qwen_qwen2_5_1_5b_instruct_v2_n80`: Modal A10G Transformers job was attempted at n=80 but interrupted after sustained runtime without completed artifacts.
- `mistralai/Mistral-7B-Instruct-v0.3` for `modal_mistralai_mistral_7b_instruct_v0_3_v2_n80`: Modal A10G Transformers job loaded and generated at n=80 but was interrupted after sustained runtime without completed artifacts.

- Minimum pilot achieved: `false`
- Strong pilot achieved: `false`
- Scientific model count: `0`
- Scientific model families: `none`

## Main Aggregate Numbers

- No scientific aggregate rows because no qualifying real instruct-model runs completed.

## Strongest Baseline Threat

No scientific aggregate exists. Baseline threat cannot be evaluated yet.

## Manual Audit Status

Manual audit samples were generated, but no completed human audit file was found. Manual audit is still required before paper claims.

## Exact Next Step

Run at least two real instruct/chat models with `n >= 80` using `scripts/run_scientific_pilot.sh --allow-download true --target minimum`, or provide a reachable local OpenAI-compatible endpoint serving such models.
