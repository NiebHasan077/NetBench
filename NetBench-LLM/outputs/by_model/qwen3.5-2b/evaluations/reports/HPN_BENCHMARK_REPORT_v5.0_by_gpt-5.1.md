# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-05-13 17:15:08
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `Qwen3.5-2B` (242 scored questions)
  2. `RAG-Qwen3.5-2B` (242 scored questions)
  3. `Qwen3.5-2B-instruct-lora-sft-merged` (242 scored questions)
  4. `Qwen3.5-2B-cpt-full-sft` (242 scored questions)
  5. `RAG-Qwen3.5-2B-cpt-full-sft` (242 scored questions)
  6. `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  7. `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  8. `Qwen3.5-2B-instruct-full-hpn-sft` (242 scored questions)

## Executive Summary

| Dimension | `Qwen3.5-2B` | `RAG-Qwen3.5-2B` | `Qwen3.5-2B-instruct-lora-sft-merged` | `Qwen3.5-2B-cpt-full-sft` | `RAG-Qwen3.5-2B-cpt-full-sft` | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | `Qwen3.5-2B-instruct-full-hpn-sft` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 2.04 | 2.51 | 3.03 | 3.27 | 3.88 | 3.17 | 4.05 | 3.15 |
| **Completeness** | 1.48 | 2.01 | 2.84 | 3.08 | 3.86 | 2.96 | 4.04 | 2.93 |
| **Clarity** | 2.36 | 2.65 | 4.14 | 4.27 | 4.17 | 4.20 | 4.12 | 4.19 |
| **Conciseness** | 1.94 | 2.43 | 3.35 | 3.40 | 3.25 | 3.38 | 3.09 | 3.36 |
| **Overall** | 2.03 | 2.45 | 3.28 | 3.48 | 3.80 | 3.39 | 3.89 | 3.36 |

## Scores by Category

| Category | `Qwen3.5-2B` overall | `RAG-Qwen3.5-2B` overall | `Qwen3.5-2B-instruct-lora-sft-merged` overall | `Qwen3.5-2B-cpt-full-sft` overall | `RAG-Qwen3.5-2B-cpt-full-sft` overall | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-2B-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.25 | 2.15 | 3.45 | 3.71 | 3.63 | 3.51 | 3.88 | 3.49 |
| BDP-Based Reasoning and Window Sizing | 2.19 | 1.95 | 3.51 | 3.55 | 3.52 | 3.33 | 3.83 | 3.48 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.02 | 2.57 | 3.19 | 3.44 | 3.78 | 3.37 | 3.86 | 3.32 |
| Concurrency Tuning and Scaling | 1.77 | 2.50 | 3.31 | 3.36 | 3.77 | 3.72 | 3.86 | 3.50 |
| Dataset Partitioning and Mixed Workloads | 2.12 | 1.66 | 3.10 | 3.40 | 4.20 | 2.88 | 4.20 | 3.36 |
| Fairness, Stability, and Shared Networks | 2.00 | 2.45 | 3.38 | 3.36 | 3.56 | 3.43 | 3.54 | 3.67 |
| Parallelism and Large-File Optimization | 2.28 | 2.56 | 3.37 | 3.74 | 3.99 | 4.10 | 3.49 | 4.04 |
| Pipelining and Small-File Optimization | 1.98 | 2.37 | 3.06 | 3.16 | 3.85 | 3.14 | 4.02 | 3.14 |
| Practical HPN Scenarios and Design | 1.85 | 2.54 | 3.00 | 3.24 | 3.45 | 3.13 | 3.58 | 2.86 |
| Transfer Parameters: Definitions and Roles | 1.94 | 2.77 | 3.28 | 3.54 | 4.17 | 3.38 | 4.19 | 3.28 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-2B` overall | `RAG-Qwen3.5-2B` overall | `Qwen3.5-2B-instruct-lora-sft-merged` overall | `Qwen3.5-2B-cpt-full-sft` overall | `RAG-Qwen3.5-2B-cpt-full-sft` overall | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-2B-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 2.28 | 2.44 | 3.76 | 3.91 | 3.91 | 3.92 | 3.97 | 3.80 |
| medium | 1.97 | 2.44 | 3.11 | 3.42 | 3.74 | 3.22 | 3.85 | 3.27 |
| hard | 1.85 | 2.47 | 2.97 | 3.06 | 3.76 | 3.02 | 3.85 | 2.98 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.04 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 2.51 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 3.03 | 1.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.27 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 3.88 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.17 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.05 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 3.15 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 1.48 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 2.01 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 2.84 | 1.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.08 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 3.86 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 2.96 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.04 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 2.93 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.36 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 2.65 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 4.14 | 2.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 4.27 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 4.17 | 3.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.20 | 3.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.12 | 3.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 4.19 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 1.94 | 1.00 | 4.00 |
| `RAG-Qwen3.5-2B` | 2.43 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 3.35 | 2.00 | 4.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.40 | 2.00 | 4.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 3.25 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.38 | 2.00 | 4.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.09 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 3.36 | 2.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
