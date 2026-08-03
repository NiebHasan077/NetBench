| label | profile_type | device | model_name | question_count | avg_total_time_s | avg_retrieval_time_s | avg_generation_time_s | avg_tokens_per_second | avg_ms_per_token | avg_ttft_s | peak_gpu_vram_mb | peak_cpu_ram_mb | load_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rag-gpu-benchmark10-S6-gemma-3-27b-RAG-Instruct-Official | rag | cuda | gemma-3-27b-it | 10 | 32.06 | 1.6043 | 30.4557 | 7.7 |  |  | 17858 | 9528 | 45.03 |
| rag-gpu-benchmark10-S8-gemma-3-27b-RAG-LoRA-CPT-SFT | rag | cuda | gemma-3-27b-instruct-lora-cpt-then-sft-merged | 10 | 30.9594 | 1.5835 | 29.3759 | 7.11 |  |  | 17858 | 9701 | 44.57 |
| rag-cpu-benchmark10-S6-gemma-3-27b-RAG-Instruct-Official | rag | cpu | gemma-3-27b-it | 10 | 287.1736 | 11.3414 | 275.8323 | 0.86 |  |  | 0 | 112227 | 34.43 |
| rag-cpu-benchmark10-S8-gemma-3-27b-RAG-LoRA-CPT-SFT | rag | cpu | gemma-3-27b-instruct-lora-cpt-then-sft-merged | 10 | 259.2971 | 11.1842 | 248.1128 | 0.83 |  |  | 0 | 112468 | 31.86 |
