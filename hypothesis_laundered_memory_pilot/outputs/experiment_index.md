# Experiment Index

| run_name | backend | model | n_items | mock | run_role | scientific_evidence | main_metrics | verdict | summary |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| mock_full_deployable | mock | gpt-4.1-mini | 120 | True | mock_pipeline_validation | False | false_evidence_promotion_rate=0.125<br>downstream_contamination_rate=0.0<br>useful_memory_retention=1.0<br>confirmed_hypothesis_promotion_rate=1.0<br>tentative_overblocking_rate=0.0 | MOCK_ONLY | [summary.md](outputs/mock_full_deployable/summary.md) |
| mock_full_latest | mock | gpt-4.1-mini | 80 | True | mock_pipeline_validation | False | false_evidence_promotion_rate=0.1<br>downstream_contamination_rate=0.0<br>useful_memory_retention=1.0<br>confirmed_hypothesis_promotion_rate=1.0<br>tentative_overblocking_rate=0.0 | MOCK_ONLY | [summary.md](outputs/mock_full_latest/summary.md) |
| mock_smoke | mock | gpt-4.1-mini | 20 | True | mock_pipeline_validation | False | false_evidence_promotion_rate=0.1<br>downstream_contamination_rate=0.0<br>useful_memory_retention=1.0<br>confirmed_hypothesis_promotion_rate=1.0<br>tentative_overblocking_rate=0.0 | MOCK_ONLY | [summary.md](outputs/mock_smoke/summary.md) |
| modal_google_gemma_3_1b_it_v2_n80 | transformers | google/gemma-3-1b-it | 80 | False | incomplete_run | False | NA | NO_SCIENTIFIC_RUNS | [summary.md](outputs/modal_google_gemma_3_1b_it_v2_n80/summary.md) |
| real_qwen_qwen2_5_1_5b_instruct_v2_n40 | transformers | Qwen/Qwen2.5-1.5B-Instruct | 40 | False | incomplete_run | False | NA | NO_SCIENTIFIC_RUNS | [summary.md](outputs/real_qwen_qwen2_5_1_5b_instruct_v2_n40/summary.md) |
| real_qwen_qwen2_5_7b_instruct_v2_n120 | transformers | Qwen/Qwen2.5-7B-Instruct | 120 | False | incomplete_run | False | NA | NO_SCIENTIFIC_RUNS | [summary.md](outputs/real_qwen_qwen2_5_7b_instruct_v2_n120/summary.md) |
| real_tiny_gpt2_smoke | transformers | sshleifer/tiny-gpt2 | 5 | False | incomplete_run | False | false_evidence_promotion_rate=0.0<br>downstream_contamination_rate=0.0<br>useful_memory_retention=0.0<br>confirmed_hypothesis_promotion_rate=0.0<br>tentative_overblocking_rate=1.0 | NO_SCIENTIFIC_RUNS | [summary.md](outputs/real_tiny_gpt2_smoke/summary.md) |
| smoke_tiny_gpt2_deployable | transformers | sshleifer/tiny-gpt2 | 5 | False | plumbing_smoke | False | false_evidence_promotion_rate=0.0<br>downstream_contamination_rate=0.0<br>useful_memory_retention=0.0<br>confirmed_hypothesis_promotion_rate=0.0<br>tentative_overblocking_rate=0.0 | PLUMBING_ONLY | [summary.md](outputs/smoke_tiny_gpt2_deployable/summary.md) |

## Skipped Runs

- `local_openai_compatible_v2_n120` `local-open-instruct` skipped: No server reachable at http://localhost:8000/v1
- `modal_google_gemma_3_1b_it_v2_n80` `google/gemma-3-1b-it` skipped: Hugging Face returned 401 Unauthorized for gated repository without HF_TOKEN.
- `modal_qwen_qwen2_5_1_5b_instruct_v2_n80` `Qwen/Qwen2.5-1.5B-Instruct` skipped: Modal A10G Transformers job was attempted at n=80 but interrupted after sustained runtime without completed artifacts.
- `modal_mistralai_mistral_7b_instruct_v0_3_v2_n80` `mistralai/Mistral-7B-Instruct-v0.3` skipped: Modal A10G Transformers job loaded and generated at n=80 but was interrupted after sustained runtime without completed artifacts.
