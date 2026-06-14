# Experiment Index

| run_name | backend | model | n_items | mock | run_role | scientific_evidence | main_metrics | verdict | summary |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| mock_full_deployable | mock | gpt-4.1-mini | 80 | True | mock_pipeline_validation | False | false_evidence_promotion_rate=0.1<br>downstream_contamination_rate=0.0<br>useful_memory_retention=1.0<br>confirmed_hypothesis_promotion_rate=1.0<br>tentative_overblocking_rate=0.0 | MOCK_ONLY | [summary.md](outputs/mock_full_deployable/summary.md) |
| mock_full_latest | mock | gpt-4.1-mini | 80 | True | mock_pipeline_validation | False | false_evidence_promotion_rate=0.1<br>downstream_contamination_rate=0.0<br>useful_memory_retention=1.0<br>confirmed_hypothesis_promotion_rate=1.0<br>tentative_overblocking_rate=0.0 | MOCK_ONLY | [summary.md](outputs/mock_full_latest/summary.md) |
| mock_smoke | mock | gpt-4.1-mini | 20 | True | mock_pipeline_validation | False | false_evidence_promotion_rate=0.1<br>downstream_contamination_rate=0.0<br>useful_memory_retention=1.0<br>confirmed_hypothesis_promotion_rate=1.0<br>tentative_overblocking_rate=0.0 | MOCK_ONLY | [summary.md](outputs/mock_smoke/summary.md) |
| real_tiny_gpt2_smoke | transformers | sshleifer/tiny-gpt2 | 5 | False | plumbing_smoke | False | false_evidence_promotion_rate=0.0<br>downstream_contamination_rate=0.0<br>useful_memory_retention=0.0<br>confirmed_hypothesis_promotion_rate=0.0<br>tentative_overblocking_rate=1.0 | PLUMBING_ONLY | [summary.md](outputs/real_tiny_gpt2_smoke/summary.md) |
| smoke_tiny_gpt2_deployable | transformers | sshleifer/tiny-gpt2 | 5 | False | plumbing_smoke | False | false_evidence_promotion_rate=0.0<br>downstream_contamination_rate=0.0<br>useful_memory_retention=0.0<br>confirmed_hypothesis_promotion_rate=0.0<br>tentative_overblocking_rate=1.0 | PLUMBING_ONLY | [summary.md](outputs/smoke_tiny_gpt2_deployable/summary.md) |

## Skipped Runs

- `local_openai_compatible_n80` `local-open-model` skipped: No server reachable at http://localhost:8000/v1
- `real_qwen_qwen2_5_1_5b_instruct_n80` `Qwen/Qwen2.5-1.5B-Instruct` skipped: Model not present in local Hugging Face cache and downloads are disabled by runner.
- `real_qwen_qwen2_5_3b_instruct_n80` `Qwen/Qwen2.5-3B-Instruct` skipped: Model not present in local Hugging Face cache and downloads are disabled by runner.
- `real_qwen_qwen2_5_7b_instruct_n80` `Qwen/Qwen2.5-7B-Instruct` skipped: Model not present in local Hugging Face cache and downloads are disabled by runner.
- `real_qwen_qwen2_5_coder_7b_instruct_n80` `Qwen/Qwen2.5-Coder-7B-Instruct` skipped: Model not present in local Hugging Face cache and downloads are disabled by runner.
