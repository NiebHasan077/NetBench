| label | profile_type | device | model_name | question_count | avg_total_time_s | avg_retrieval_time_s | avg_generation_time_s | avg_tokens_per_second | avg_ms_per_token | avg_ttft_s | peak_gpu_vram_mb | peak_cpu_ram_mb | load_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct-gpu-benchmark10-S1-Llama-3.2-1B-Instruct-Official | direct | cuda | Llama-3.2-1B-Instruct | 10 | 2.3693 |  | 2.3693 | 108.1 | 9.26 | 0.001 | 1215 | 2052 | 1.84 |
| direct-gpu-benchmark10-S4-Llama-3.2-1B-CPT-SFT | direct | cuda | Llama-3.2-1B-cpt-full-sft | 10 | 1.0201 |  | 1.0201 | 103.29 | 9.69 | 0.001 | 1215 | 2045 | 1.94 |
| direct-gpu-benchmark10-S5-Llama-3.2-1B-LoRA-CPT-SFT | direct | cuda | Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 10 | 1.2669 |  | 1.2669 | 101.96 | 9.81 | 0.0011 | 1215 | 2051 | 1.97 |
| direct-cpu-benchmark10-S1-Llama-3.2-1B-Instruct-Official | direct | cpu | Llama-3.2-1B-Instruct | 10 | 13.5506 |  | 13.5506 | 18.19 | 54.98 | 0.001 | 0 | 5919 | 1.68 |
| direct-cpu-benchmark10-S4-Llama-3.2-1B-CPT-SFT | direct | cpu | Llama-3.2-1B-cpt-full-sft | 10 | 5.8905 |  | 5.8905 | 18.24 | 54.85 | 0.001 | 0 | 5926 | 1.44 |
| direct-cpu-benchmark10-S5-Llama-3.2-1B-LoRA-CPT-SFT | direct | cpu | Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 10 | 6.8356 |  | 6.8356 | 18.48 | 54.11 | 0.001 | 0 | 5924 | 1.44 |
| rag-gpu-benchmark10-S7-Llama-3.2-1B-RAG-CPT-SFT | rag | cuda | Llama-3.2-1B-cpt-full-sft | 10 | 4.5673 | 2.0617 | 2.5055 | 63.3 |  |  | 6087 | 4747 | 11.0 |
| rag-gpu-benchmark10-S8-Llama-3.2-1B-RAG-LoRA-CPT-SFT | rag | cuda | Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 10 | 4.9177 | 2.074 | 2.8437 | 72.48 |  |  | 6087 | 4756 | 10.62 |
| rag-cpu-benchmark10-S7-Llama-3.2-1B-RAG-CPT-SFT | rag | cpu | Llama-3.2-1B-cpt-full-sft | 10 | 24.1505 | 10.1793 | 13.9712 | 10.84 |  |  | 0 | 11292 | 10.5 |
| rag-cpu-benchmark10-S8-Llama-3.2-1B-RAG-LoRA-CPT-SFT | rag | cpu | Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 10 | 27.5921 | 10.1528 | 17.4393 | 12.17 |  |  | 0 | 11300 | 9.97 |
