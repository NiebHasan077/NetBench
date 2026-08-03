# HPN-QA Inter-Judge Agreement: `gpt-5.1` vs `gemini-3.5-flash`

Paired **63 models** (14679 question-judgements). Both judges scored identical answers at `temperature=0.0`.

> Agreement measures *consistency* between the two judges, not correctness. It supports a robustness claim only.

## System-level (leaderboard robustness)

| Metric | Value |
|---|---|
| Kendall tau-b (model ranking) | **0.913** |
| Spearman rho (model ranking)  | **0.987** |
| Pearson r (mean overall)      | 0.988 |
| Models compared               | 63 |

## Per-question agreement (pooled)

| Dimension | n | Pearson | Spearman | MAE | Quadratic-weighted kappa |
|---|---|---|---|---|---|
| correctness | 14679 | 0.880 | 0.865 | 0.408 | 0.870 |
| completeness | 14679 | 0.864 | 0.866 | 0.484 | 0.843 |
| clarity | 14679 | 0.606 | 0.579 | 0.468 | 0.558 |
| conciseness | 14679 | 0.491 | 0.471 | 0.668 | 0.411 |
| overall | 14679 | 0.887 | 0.873 | 0.483 | n/a |

_kappa is reported for the four integer 1-5 dimensions; `overall` is a continuous weighted average._

## Per-model leaderboard (mean overall under each judge)

| Model | gpt-5.1 mean | rank | gemini-3.5-flash mean | rank | rank Δ |
|---|---|---|---|---|---|
| RAG-Qwen3.5-27B | 4.376 | 1 | 4.783 | 1 | +0 |
| RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 4.359 | 2 | 4.755 | 2 | +0 |
| RAG-gemma-3-27b-it | 4.353 | 3 | 4.666 | 4 | -1 |
| claude-sonnet-4-6 | 4.289 | 4 | 4.665 | 5 | -1 |
| gemini-2.5-pro | 4.282 | 5 | 4.616 | 8 | -3 |
| RAG-gemma-3-12b-it | 4.264 | 6 | 4.566 | 10 | -4 |
| RAG-Qwen3.5-9B | 4.220 | 7 | 4.680 | 3 | +4 |
| RAG-Qwen3.5-9B-cpt-full-sft | 4.212 | 8 | 4.638 | 6 | +2 |
| RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 4.173 | 9 | 4.625 | 7 | +2 |
| gpt-4o | 4.167 | 10 | 4.442 | 16 | -6 |
| Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 4.093 | 11 | 4.405 | 17 | -6 |
| RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged | 4.068 | 12 | 4.495 | 13 | -1 |
| RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged | 4.067 | 13 | 4.524 | 11 | +2 |
| Qwen3.5-27B | 4.059 | 14 | 4.584 | 9 | +5 |
| Qwen3.5-27B-instruct-lora-sft-merged | 4.044 | 15 | 4.328 | 21 | -6 |
| RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged | 4.043 | 16 | 4.494 | 14 | +2 |
| RAG-Llama-3.1-8B-cpt-full-sft | 4.042 | 17 | 4.458 | 15 | +2 |
| RAG-gemma-3-12b-cpt-full-sft | 4.038 | 18 | 4.507 | 12 | +6 |
| Qwen3.5-27B-instruct-full-hpn-sft | 3.996 | 19 | 4.333 | 20 | -1 |
| RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged | 3.922 | 20 | 4.386 | 18 | +2 |
| RAG-Llama-3.1-8B-Instruct | 3.889 | 21 | 4.151 | 25 | -4 |
| RAG-Qwen3.5-2B | 3.857 | 22 | 4.380 | 19 | +3 |
| Qwen3.5-9B-cpt-full-sft | 3.837 | 23 | 4.145 | 26 | -3 |
| Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 3.825 | 24 | 4.194 | 24 | +0 |
| RAG-Qwen3.5-2B-cpt-full-sft | 3.821 | 25 | 4.260 | 23 | +2 |
| gemma-3-27b-it | 3.786 | 26 | 4.130 | 27 | -1 |
| Qwen3.5-9B-instruct-lora-sft-merged | 3.745 | 27 | 3.968 | 30 | -3 |
| Qwen3.5-9B-instruct-full-hpn-sft | 3.741 | 28 | 4.035 | 29 | -1 |
| gemma-3-27b-instruct-lora-cpt-then-sft-merged | 3.733 | 29 | 4.045 | 28 | +1 |
| Qwen3.5-9B | 3.691 | 30 | 4.287 | 22 | +8 |
| gemma-3-27b-instruct-lora-sft-merged | 3.658 | 31 | 3.958 | 31 | +0 |
| gemma-3-12b-it | 3.621 | 32 | 3.884 | 32 | +0 |
| Llama-3.1-8B-cpt-full-sft | 3.586 | 33 | 3.751 | 40 | -7 |
| Llama-3.1-8B-instruct-lora-cpt-then-sft-merged | 3.586 | 34 | 3.770 | 36 | -2 |
| Llama-3.1-8B-instruct-lora-sft-merged | 3.584 | 35 | 3.765 | 37 | -2 |
| gemma-3-12b-cpt-full-sft | 3.573 | 36 | 3.783 | 35 | +1 |
| gemma-3-12b-instruct-lora-cpt-then-sft-merged | 3.572 | 37 | 3.843 | 33 | +4 |
| gemma-3-12b-instruct-full-hpn-sft | 3.563 | 38 | 3.759 | 39 | -1 |
| gemma-3-12b-instruct-lora-sft-merged | 3.538 | 39 | 3.764 | 38 | +1 |
| gemma-3-27b-instruct-full-hpn-sft | 3.536 | 40 | 3.786 | 34 | +6 |
| Qwen3.5-2B-cpt-full-sft | 3.498 | 41 | 3.633 | 41 | +0 |
| Llama-3.1-8B-instruct-full-hpn-sft | 3.468 | 42 | 3.617 | 42 | +0 |
| Qwen3.5-2B-instruct-lora-cpt-then-sft-merged | 3.406 | 43 | 3.503 | 43 | +0 |
| Qwen3.5-2B-instruct-full-hpn-sft | 3.363 | 44 | 3.423 | 44 | +0 |
| Llama-3.1-8B-Instruct | 3.336 | 45 | 3.341 | 47 | -2 |
| Qwen3.5-2B-instruct-lora-sft-merged | 3.298 | 46 | 3.397 | 46 | +0 |
| RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 3.140 | 47 | 3.417 | 45 | +2 |
| RAG-gemma-3-1b-Instruct | 3.078 | 48 | 3.038 | 50 | -2 |
| RAG-Llama-3.2-1B-cpt-full-sft | 3.041 | 49 | 3.184 | 48 | +1 |
| RAG-Llama-3.2-1B-Instruct | 2.990 | 50 | 2.927 | 52 | -2 |
| RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged | 2.951 | 51 | 3.119 | 49 | +2 |
| RAG-gemma-3-1b-cpt-full-sft | 2.893 | 52 | 3.035 | 51 | +1 |
| Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 2.890 | 53 | 2.748 | 54 | -1 |
| Llama-3.2-1B-cpt-full-sft | 2.851 | 54 | 2.626 | 57 | -3 |
| gemma-3-1b-instruct-lora-cpt-then-sft-merged | 2.825 | 55 | 2.713 | 55 | +0 |
| Llama-3.2-1B-instruct-lora-sft-merged | 2.801 | 56 | 2.639 | 56 | +0 |
| Llama-3.2-1B-instruct-full-hpn-sft | 2.756 | 57 | 2.575 | 58 | -1 |
| gemma-3-1b-instruct-full-hpn-sft | 2.717 | 58 | 2.436 | 60 | -2 |
| Qwen3.5-2B | 2.712 | 59 | 2.857 | 53 | +6 |
| gemma-3-1b-instruct-lora-sft-merged | 2.706 | 60 | 2.548 | 59 | +1 |
| gemma-3-1b-cpt-full-sft | 2.604 | 61 | 2.350 | 61 | +0 |
| gemma-3-1b-Instruct | 2.528 | 62 | 2.258 | 62 | +0 |
| Llama-3.2-1B-Instruct | 2.511 | 63 | 2.138 | 63 | +0 |
