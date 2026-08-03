| label | profile_type | device | model_name | question_count | avg_total_time_s | avg_retrieval_time_s | avg_generation_time_s | avg_tokens_per_second | avg_ms_per_token | avg_ttft_s | peak_gpu_vram_mb | peak_cpu_ram_mb | load_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct-gpu-benchmark10-S1-gemma-3-27b-Instruct-Official | direct | cuda | gemma-3-27b-it | 10 | 18.8868 |  | 18.8868 | 14.84 | 75.58 | 0.0035 | 13020 | 3477 | 35.89 |
| direct-gpu-benchmark10-S3-gemma-3-27b-Instruct-Full-HPN-SFT | direct | cuda | gemma-3-27b-instruct-full-hpn-sft | 10 | 14.1131 |  | 14.1131 | 14.62 | 77.8 | 0.0036 | 13023 | 3606 | 35.46 |
| direct-gpu-benchmark10-S5-gemma-3-27b-LoRA-CPT-SFT | direct | cuda | gemma-3-27b-instruct-lora-cpt-then-sft-merged | 10 | 14.7027 |  | 14.7027 | 13.58 | 86.05 | 0.0034 | 13023 | 3604 | 38.62 |
| direct-cpu-benchmark10-S1-gemma-3-27b-Instruct-Official | direct | cpu | gemma-3-27b-it | 10 | 163.1219 |  | 163.1219 | 1.48 | 674.76 | 0.0018 | 0 | 105794 | 35.38 |
| direct-cpu-benchmark10-S3-gemma-3-27b-Instruct-Full-HPN-SFT | direct | cpu | gemma-3-27b-instruct-full-hpn-sft | 10 | 105.6696 |  | 105.6696 | 1.58 | 632.27 | 0.0017 | 0 | 105817 | 26.1 |
| direct-cpu-benchmark10-S5-gemma-3-27b-LoRA-CPT-SFT | direct | cpu | gemma-3-27b-instruct-lora-cpt-then-sft-merged | 10 | 125.6471 |  | 125.6471 | 1.59 | 627.29 | 0.0018 | 0 | 105844 | 23.41 |
| rag-gpu-benchmark10-S6-gemma-3-27b-RAG-Instruct-Official | rag | cuda | gemma-3-27b-it | 10 | 32.06 | 1.6043 | 30.4557 | 7.7 |  |  | 17858 | 9528 | 45.03 |
| rag-gpu-benchmark10-S8-gemma-3-27b-RAG-LoRA-CPT-SFT | rag | cuda | gemma-3-27b-instruct-lora-cpt-then-sft-merged | 10 | 30.9594 | 1.5835 | 29.3759 | 7.11 |  |  | 17858 | 9701 | 44.57 |
| rag-cpu-benchmark10-S6-gemma-3-27b-RAG-Instruct-Official | rag | cpu | gemma-3-27b-it | 10 | 287.1736 | 11.3414 | 275.8323 | 0.86 |  |  | 0 | 112227 | 34.43 |
| rag-cpu-benchmark10-S8-gemma-3-27b-RAG-LoRA-CPT-SFT | rag | cpu | gemma-3-27b-instruct-lora-cpt-then-sft-merged | 10 | 259.2971 | 11.1842 | 248.1128 | 0.83 |  |  | 0 | 112468 | 31.86 |
