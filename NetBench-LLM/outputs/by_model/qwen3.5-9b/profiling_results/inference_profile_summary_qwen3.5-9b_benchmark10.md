| label | profile_type | device | model_name | question_count | avg_total_time_s | avg_retrieval_time_s | avg_generation_time_s | avg_tokens_per_second | avg_ms_per_token | avg_ttft_s | peak_gpu_vram_mb | peak_cpu_ram_mb | load_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct-cpu-benchmark10-S1-Qwen3.5-9B-Instruct-Official | direct | cpu | Qwen3.5-9B | 10 | 79.7924 |  | 79.7924 | 3.21 | 311.69 | 0.0012 | 0 | 35403 | 2.86 |
| direct-cpu-benchmark10-S4-Qwen3.5-9B-CPT-SFT | direct | cpu | Qwen3.5-9B-cpt-full-sft | 10 | 79.9329 |  | 79.9329 | 3.2 | 312.24 | 0.0011 | 0 | 35411 | 2.65 |
| direct-cpu-benchmark10-S5-Qwen3.5-9B-LoRA-CPT-SFT | direct | cpu | Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 10 | 78.5198 |  | 78.5198 | 3.2 | 312.03 | 0.0011 | 0 | 35484 | 2.73 |
| direct-gpu-benchmark10-S1-Qwen3.5-9B-Instruct-Official | direct | cuda | Qwen3.5-9B | 10 | 9.4738 |  | 9.4738 | 27.03 | 37.01 | 0.0013 | 6940 | 2204 | 9.84 |
| direct-gpu-benchmark10-S4-Qwen3.5-9B-CPT-SFT | direct | cuda | Qwen3.5-9B-cpt-full-sft | 10 | 9.7986 |  | 9.7986 | 26.86 | 38.28 | 0.0012 | 6940 | 2203 | 9.58 |
| direct-gpu-benchmark10-S5-Qwen3.5-9B-LoRA-CPT-SFT | direct | cuda | Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 10 | 7.8317 |  | 7.8317 | 27.08 | 36.95 | 0.0013 | 6940 | 2223 | 9.5 |
| rag-cpu-benchmark10-S7-Qwen3.5-9B-RAG-CPT-SFT | rag | cpu | Qwen3.5-9B-cpt-full-sft | 10 | 112.8593 | 11.3032 | 101.556 | 2.24 |  |  | 0 | 40869 | 11.34 |
| rag-cpu-benchmark10-S8-Qwen3.5-9B-RAG-LoRA-CPT-SFT | rag | cpu | Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 10 | 111.9674 | 12.1093 | 99.8581 | 2.19 |  |  | 0 | 40910 | 11.5 |
| rag-gpu-benchmark10-S7-Qwen3.5-9B-RAG-CPT-SFT | rag | cuda | Qwen3.5-9B-cpt-full-sft | 10 | 12.8773 | 2.2526 | 10.6247 | 21.49 |  |  | 11778 | 5043 | 13.67 |
| rag-gpu-benchmark10-S8-Qwen3.5-9B-RAG-LoRA-CPT-SFT | rag | cuda | Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 10 | 12.2858 | 2.1307 | 10.1551 | 21.79 |  |  | 11778 | 5064 | 12.9 |
