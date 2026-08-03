| label | profile_type | device | model_name | question_count | avg_total_time_s | avg_retrieval_time_s | avg_generation_time_s | avg_tokens_per_second | avg_ms_per_token | avg_ttft_s | peak_gpu_vram_mb | peak_cpu_ram_mb | load_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rag-gpu-benchmark10-S6-Qwen3.5-27B-RAG-Instruct-Official | rag | cuda | Qwen3.5-27B | 10 | 41.8129 | 1.0469 | 40.7661 | 9.01 |  |  | 16242 | 11492 | 251.39 |
| rag-gpu-benchmark10-S8-Qwen3.5-27B-RAG-LoRA-CPT-SFT | rag | cuda | Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 10 | 28.6105 | 1.0046 | 27.6059 | 7.69 |  |  | 16242 | 9949 | 35.29 |
| rag-cpu-benchmark10-S6-Qwen3.5-27B-RAG-Instruct-Official | rag | cpu | Qwen3.5-27B | 10 | 199.61 | 10.7777 | 188.8323 | 1.88 |  |  | 0 | 55892 | 87.66 |
| rag-cpu-benchmark10-S8-Qwen3.5-27B-RAG-LoRA-CPT-SFT | rag | cpu | Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 10 | 238.6765 | 10.9464 | 227.7301 | 0.95 |  |  | 0 | 109199 | 32.39 |
