# HPN-QA Inter-Judge Agreement: `gpt-5.1` vs `gemini-3.5-flash`

Paired **63 models** (14679 question-judgements). Both judges scored identical answers at `temperature=0.0`.

> Agreement measures *consistency* between the two judges, not correctness. It supports a robustness claim only.

## System-level (leaderboard robustness)

| Metric | Value |
|---|---|
| Kendall tau-b (model ranking) | **0.931** |
| Spearman rho (model ranking)  | **0.991** |
| Pearson r (mean overall)      | 0.989 |
| Models compared               | 63 |

## Per-question agreement (pooled)

| Dimension | n | Pearson | Spearman | MAE | Quadratic-weighted kappa |
|---|---|---|---|---|---|
| correctness | 14679 | 0.880 | 0.865 | 0.408 | 0.870 |
| completeness | 14679 | 0.864 | 0.866 | 0.484 | 0.843 |
| clarity | 14679 | 0.606 | 0.579 | 0.468 | 0.558 |
| conciseness | 14679 | 0.491 | 0.471 | 0.668 | 0.411 |
| overall | 14679 | 0.886 | 0.875 | 0.440 | n/a |

_kappa is reported for the four integer 1-5 dimensions; `overall` is a continuous weighted average._

## Per-model leaderboard (mean overall under each judge)

| Model | gpt-5.1 mean | rank | gemini-3.5-flash mean | rank | rank Δ |
|---|---|---|---|---|---|
| RAG-Qwen3.5-27B | 4.528 | 1 | 4.782 | 1 | +0 |
| RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 4.465 | 2 | 4.752 | 2 | +0 |
| RAG-gemma-3-27b-it | 4.425 | 3 | 4.664 | 5 | -2 |
| claude-sonnet-4-6 | 4.403 | 4 | 4.666 | 4 | +0 |
| gemini-2.5-pro | 4.358 | 5 | 4.615 | 8 | -3 |
| RAG-Qwen3.5-9B | 4.358 | 6 | 4.679 | 3 | +3 |
| RAG-Qwen3.5-9B-cpt-full-sft | 4.324 | 7 | 4.638 | 6 | +1 |
| RAG-gemma-3-12b-it | 4.303 | 8 | 4.566 | 10 | -2 |
| RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 4.270 | 9 | 4.625 | 7 | +2 |
| gpt-4o | 4.220 | 10 | 4.439 | 16 | -6 |
| RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged | 4.142 | 11 | 4.524 | 11 | +0 |
| Qwen3.5-27B | 4.141 | 12 | 4.584 | 9 | +3 |
| RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged | 4.138 | 13 | 4.493 | 13 | +0 |
| RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged | 4.107 | 14 | 4.490 | 14 | +0 |
| RAG-gemma-3-12b-cpt-full-sft | 4.101 | 15 | 4.508 | 12 | +3 |
| RAG-Llama-3.1-8B-cpt-full-sft | 4.088 | 16 | 4.453 | 15 | +1 |
| Qwen3.5-27B-instruct-lora-cpt-then-sft-merged | 4.086 | 17 | 4.402 | 17 | +0 |
| Qwen3.5-27B-instruct-lora-sft-merged | 4.030 | 18 | 4.328 | 21 | -3 |
| RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged | 4.001 | 19 | 4.382 | 18 | +1 |
| Qwen3.5-27B-instruct-full-hpn-sft | 3.990 | 20 | 4.329 | 20 | +0 |
| RAG-Llama-3.1-8B-Instruct | 3.962 | 21 | 4.148 | 25 | -4 |
| RAG-Qwen3.5-2B-cpt-full-sft | 3.896 | 22 | 4.262 | 23 | -1 |
| RAG-Qwen3.5-2B | 3.881 | 23 | 4.380 | 19 | +4 |
| Qwen3.5-9B-cpt-full-sft | 3.825 | 24 | 4.142 | 26 | -2 |
| Qwen3.5-9B-instruct-lora-cpt-then-sft-merged | 3.806 | 25 | 4.191 | 24 | +1 |
| gemma-3-27b-it | 3.796 | 26 | 4.118 | 27 | -1 |
| Qwen3.5-9B-instruct-lora-sft-merged | 3.721 | 27 | 3.965 | 30 | -3 |
| gemma-3-27b-instruct-lora-cpt-then-sft-merged | 3.714 | 28 | 4.040 | 28 | +0 |
| Qwen3.5-9B | 3.713 | 29 | 4.281 | 22 | +7 |
| Qwen3.5-9B-instruct-full-hpn-sft | 3.711 | 30 | 4.036 | 29 | +1 |
| gemma-3-27b-instruct-lora-sft-merged | 3.636 | 31 | 3.960 | 31 | +0 |
| gemma-3-12b-it | 3.614 | 32 | 3.879 | 32 | +0 |
| Llama-3.1-8B-cpt-full-sft | 3.562 | 33 | 3.748 | 40 | -7 |
| Llama-3.1-8B-instruct-lora-cpt-then-sft-merged | 3.553 | 34 | 3.769 | 36 | -2 |
| Llama-3.1-8B-instruct-lora-sft-merged | 3.546 | 35 | 3.763 | 37 | -2 |
| gemma-3-12b-instruct-lora-cpt-then-sft-merged | 3.543 | 36 | 3.840 | 33 | +3 |
| gemma-3-12b-cpt-full-sft | 3.541 | 37 | 3.779 | 35 | +2 |
| gemma-3-12b-instruct-full-hpn-sft | 3.523 | 38 | 3.752 | 39 | -1 |
| gemma-3-27b-instruct-full-hpn-sft | 3.500 | 39 | 3.785 | 34 | +5 |
| gemma-3-12b-instruct-lora-sft-merged | 3.489 | 40 | 3.759 | 38 | +2 |
| Qwen3.5-2B-cpt-full-sft | 3.448 | 41 | 3.626 | 41 | +0 |
| Llama-3.1-8B-instruct-full-hpn-sft | 3.424 | 42 | 3.611 | 42 | +0 |
| Qwen3.5-2B-instruct-lora-cpt-then-sft-merged | 3.352 | 43 | 3.496 | 43 | +0 |
| Qwen3.5-2B-instruct-full-hpn-sft | 3.315 | 44 | 3.419 | 44 | +0 |
| Llama-3.1-8B-Instruct | 3.304 | 45 | 3.339 | 47 | -2 |
| Qwen3.5-2B-instruct-lora-sft-merged | 3.242 | 46 | 3.395 | 46 | +0 |
| RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 3.123 | 47 | 3.410 | 45 | +2 |
| RAG-gemma-3-1b-Instruct | 3.015 | 48 | 3.032 | 50 | -2 |
| RAG-Llama-3.2-1B-cpt-full-sft | 3.011 | 49 | 3.179 | 48 | +1 |
| RAG-Llama-3.2-1B-Instruct | 2.930 | 50 | 2.922 | 52 | -2 |
| RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged | 2.910 | 51 | 3.116 | 49 | +2 |
| RAG-gemma-3-1b-cpt-full-sft | 2.839 | 52 | 3.028 | 51 | +1 |
| Llama-3.2-1B-instruct-lora-cpt-then-sft-merged | 2.824 | 53 | 2.747 | 54 | -1 |
| Llama-3.2-1B-cpt-full-sft | 2.766 | 54 | 2.614 | 57 | -3 |
| gemma-3-1b-instruct-lora-cpt-then-sft-merged | 2.744 | 55 | 2.711 | 55 | +0 |
| Llama-3.2-1B-instruct-lora-sft-merged | 2.734 | 56 | 2.639 | 56 | +0 |
| Llama-3.2-1B-instruct-full-hpn-sft | 2.677 | 57 | 2.573 | 58 | -1 |
| Qwen3.5-2B | 2.634 | 58 | 2.857 | 53 | +5 |
| gemma-3-1b-instruct-full-hpn-sft | 2.628 | 59 | 2.432 | 60 | -1 |
| gemma-3-1b-instruct-lora-sft-merged | 2.620 | 60 | 2.548 | 59 | +1 |
| gemma-3-1b-cpt-full-sft | 2.518 | 61 | 2.350 | 61 | +0 |
| Llama-3.2-1B-Instruct | 2.403 | 62 | 2.136 | 63 | -1 |
| gemma-3-1b-Instruct | 2.403 | 63 | 2.251 | 62 | +1 |
