# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:58
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `Qwen3.5-9B` (233 scored questions)
  2. `Qwen3.5-9B-cpt-full-sft` (233 scored questions)
  3. `Qwen3.5-9B-instruct-full-hpn-sft` (233 scored questions)
  4. `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  5. `Qwen3.5-9B-instruct-lora-sft-merged` (233 scored questions)
  6. `RAG-Qwen3.5-9B` (233 scored questions)
  7. `RAG-Qwen3.5-9B-cpt-full-sft` (233 scored questions)
  8. `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` (233 scored questions)

## Executive Summary

| Dimension | `Qwen3.5-9B` | `Qwen3.5-9B-cpt-full-sft` | `Qwen3.5-9B-instruct-full-hpn-sft` | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | `Qwen3.5-9B-instruct-lora-sft-merged` | `RAG-Qwen3.5-9B` | `RAG-Qwen3.5-9B-cpt-full-sft` | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 3.73 | 3.79 | 3.64 | 3.78 | 3.68 | 4.47 | 4.49 | 4.45 |
| **Completeness** | 3.66 | 3.50 | 3.34 | 3.46 | 3.33 | 4.51 | 4.39 | 4.33 |
| **Clarity** | 4.24 | 4.53 | 4.48 | 4.51 | 4.48 | 4.58 | 4.43 | 4.35 |
| **Conciseness** | 2.74 | 3.54 | 3.58 | 3.54 | 3.55 | 3.02 | 3.24 | 3.21 |
| **Overall** | 3.69 | 3.84 | 3.74 | 3.83 | 3.75 | 4.22 | 4.21 | 4.17 |

## Scores by Category

| Category | `Qwen3.5-9B` overall | `Qwen3.5-9B-cpt-full-sft` overall | `Qwen3.5-9B-instruct-full-hpn-sft` overall | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-9B` overall | `RAG-Qwen3.5-9B-cpt-full-sft` overall | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.76 | 3.82 | 3.82 | 3.75 | 3.86 | 4.36 | 4.29 | 4.36 |
| BDP-Based Reasoning and Window Sizing | 3.94 | 3.93 | 4.08 | 3.98 | 3.83 | 4.15 | 4.13 | 4.01 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.63 | 3.74 | 3.65 | 3.68 | 3.62 | 4.18 | 4.19 | 4.12 |
| Concurrency Tuning and Scaling | 4.02 | 3.76 | 3.67 | 3.91 | 3.84 | 4.27 | 4.16 | 3.96 |
| Dataset Partitioning and Mixed Workloads | 3.84 | 3.78 | 3.74 | 4.18 | 3.62 | 4.34 | 4.32 | 4.12 |
| Fairness, Stability, and Shared Networks | 3.54 | 3.77 | 3.47 | 3.69 | 3.53 | 4.20 | 4.20 | 4.07 |
| Parallelism and Large-File Optimization | 3.78 | 4.29 | 4.29 | 4.34 | 4.01 | 3.79 | 4.02 | 3.98 |
| Pipelining and Small-File Optimization | 3.74 | 3.94 | 3.52 | 3.73 | 3.61 | 4.16 | 4.04 | 4.37 |
| Practical HPN Scenarios and Design | 3.44 | 3.73 | 3.32 | 3.62 | 3.52 | 4.07 | 3.93 | 3.86 |
| Transfer Parameters: Definitions and Roles | 3.63 | 3.87 | 3.81 | 3.92 | 3.88 | 4.35 | 4.42 | 4.40 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-9B` overall | `Qwen3.5-9B-cpt-full-sft` overall | `Qwen3.5-9B-instruct-full-hpn-sft` overall | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-9B` overall | `RAG-Qwen3.5-9B-cpt-full-sft` overall | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.01 | 4.24 | 4.08 | 4.19 | 4.09 | 4.26 | 4.16 | 4.14 |
| medium | 3.78 | 3.80 | 3.77 | 3.76 | 3.73 | 4.23 | 4.29 | 4.18 |
| hard | 3.18 | 3.42 | 3.30 | 3.49 | 3.36 | 4.17 | 4.17 | 4.20 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 3.73 | 1.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 3.79 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.64 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.78 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.68 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 4.47 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.49 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.45 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 3.66 | 1.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 3.50 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.34 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.46 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.33 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 4.51 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.39 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.33 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 4.24 | 3.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 4.53 | 4.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 4.48 | 4.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.51 | 4.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 4.48 | 3.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 4.58 | 3.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.43 | 3.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.35 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 2.74 | 2.00 | 4.00 |
| `Qwen3.5-9B-cpt-full-sft` | 3.54 | 3.00 | 4.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.58 | 3.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.54 | 3.00 | 4.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.55 | 2.00 | 4.00 |
| `RAG-Qwen3.5-9B` | 3.02 | 2.00 | 4.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 3.24 | 2.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.21 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
