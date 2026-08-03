| label | profile_type | device | model_name | question_count | avg_total_time_s | avg_retrieval_time_s | avg_generation_time_s | avg_tokens_per_second | avg_ms_per_token | avg_ttft_s | peak_gpu_vram_mb | peak_cpu_ram_mb | load_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct-gpu-benchmark10-S1-Qwen3.5-27B-Instruct-Official | direct | cuda | Qwen3.5-27B | 10 | 19.5622 |  | 19.5622 | 14.39 | 76.42 | 0.0012 | 11229 | 3687 | 27.8 |
| direct-gpu-benchmark10-S3-Qwen3.5-27B-Instruct-Full-HPN-SFT | direct | cuda | Qwen3.5-27B-instruct-full-hpn-sft | 10 | 19.6081 |  | 19.6081 | 14.36 | 76.59 | 0.0012 | 11229 | 3680 | 28.38 |
| direct-gpu-benchmark10-S5-Qwen3.5-27B-LoRA-CPT-SFT | direct | cuda | Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 10 | 14.2383 |  | 14.2383 | 13.77 | 80.22 | 0.0012 | 11229 | 3520 | 32.68 |
| direct-cpu-benchmark10-S1-Qwen3.5-27B-Instruct-Official | direct | cpu | Qwen3.5-27B | 10 | 116.0132 |  | 116.0132 | 2.21 | 453.18 | 0.0013 | 0 | 50074 | 92.45 |
| direct-cpu-benchmark10-S3-Qwen3.5-27B-Instruct-Full-HPN-SFT | direct | cpu | Qwen3.5-27B-instruct-full-hpn-sft | 10 | 108.1861 |  | 108.1861 | 2.29 | 436.1 | 0.0012 | 0 | 50035 | 92.96 |
| direct-cpu-benchmark10-S5-Qwen3.5-27B-LoRA-CPT-SFT | direct | cpu | Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 10 | 106.9817 |  | 106.9817 | 1.71 | 584.71 | 0.0011 | 0 | 103755 | 25.29 |
| rag-gpu-benchmark10-S6-Qwen3.5-27B-RAG-Instruct-Official | rag | cuda | Qwen3.5-27B | 10 | 41.8129 | 1.0469 | 40.7661 | 9.01 |  |  | 16242 | 11492 | 251.39 |
| rag-gpu-benchmark10-S8-Qwen3.5-27B-RAG-LoRA-CPT-SFT | rag | cuda | Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 10 | 28.6105 | 1.0046 | 27.6059 | 7.69 |  |  | 16242 | 9949 | 35.29 |
| rag-cpu-benchmark10-S6-Qwen3.5-27B-RAG-Instruct-Official | rag | cpu | Qwen3.5-27B | 10 | 199.61 | 10.7777 | 188.8323 | 1.88 |  |  | 0 | 55892 | 87.66 |
| rag-cpu-benchmark10-S8-Qwen3.5-27B-RAG-LoRA-CPT-SFT | rag | cpu | Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 10 | 238.6765 | 10.9464 | 227.7301 | 0.95 |  |  | 0 | 109199 | 32.39 |
