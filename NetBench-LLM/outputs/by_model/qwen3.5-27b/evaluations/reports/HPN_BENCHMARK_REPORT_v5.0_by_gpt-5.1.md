# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-11 16:57:15
**Judge model:** `gpt-5.1`
**Models compared:** 6
  1. `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  2. `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  3. `Qwen3.5-27B-instruct-full-hpn-sft` (242 scored questions)
  4. `RAG-Qwen3.5-27B` (242 scored questions)
  5. `Qwen3.5-27B` (242 scored questions)
  6. `Qwen3.5-27B-instruct-lora-sft-merged` (242 scored questions)

## Executive Summary

| Dimension | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | `Qwen3.5-27B-instruct-full-hpn-sft` | `RAG-Qwen3.5-27B` | `Qwen3.5-27B` | `Qwen3.5-27B-instruct-lora-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 4.13 | 4.62 | 3.99 | 4.50 | 3.39 | 4.05 |
| **Completeness** | 3.74 | 4.52 | 3.68 | 4.49 | 2.67 | 3.68 |
| **Clarity** | 4.68 | 4.55 | 4.65 | 4.61 | 3.10 | 4.69 |
| **Conciseness** | 3.68 | 3.39 | 3.61 | 3.13 | 2.40 | 3.64 |
| **Overall** | 4.09 | 4.35 | 4.00 | 4.24 | 3.05 | 4.04 |

## Scores by Category

| Category | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-27B-instruct-full-hpn-sft` overall | `RAG-Qwen3.5-27B` overall | `Qwen3.5-27B` overall | `Qwen3.5-27B-instruct-lora-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.11 | 4.42 | 3.99 | 4.38 | 3.12 | 4.05 |
| BDP-Based Reasoning and Window Sizing | 4.26 | 4.42 | 4.23 | 4.33 | 3.27 | 4.40 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.07 | 4.33 | 3.96 | 4.15 | 3.00 | 4.02 |
| Concurrency Tuning and Scaling | 4.07 | 4.26 | 4.30 | 4.34 | 2.89 | 4.02 |
| Dataset Partitioning and Mixed Workloads | 4.00 | 4.36 | 4.04 | 4.38 | 3.18 | 4.02 |
| Fairness, Stability, and Shared Networks | 3.91 | 4.03 | 3.94 | 4.30 | 2.87 | 3.98 |
| Parallelism and Large-File Optimization | 4.29 | 4.02 | 4.35 | 3.88 | 3.37 | 4.44 |
| Pipelining and Small-File Optimization | 3.92 | 4.38 | 3.85 | 4.48 | 2.77 | 3.92 |
| Practical HPN Scenarios and Design | 3.71 | 4.19 | 3.77 | 4.16 | 2.48 | 3.76 |
| Transfer Parameters: Definitions and Roles | 4.21 | 4.51 | 3.93 | 4.22 | 3.24 | 3.96 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-27B-instruct-full-hpn-sft` overall | `RAG-Qwen3.5-27B` overall | `Qwen3.5-27B` overall | `Qwen3.5-27B-instruct-lora-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.32 | 4.31 | 4.26 | 4.13 | 3.36 | 4.32 |
| medium | 4.10 | 4.38 | 4.01 | 4.32 | 2.97 | 4.07 |
| hard | 3.80 | 4.35 | 3.68 | 4.26 | 2.82 | 3.68 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.13 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.62 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 3.99 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.50 | 1.00 | 5.00 |
| `Qwen3.5-27B` | 3.39 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.05 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 3.74 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.52 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 3.68 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.49 | 1.00 | 5.00 |
| `Qwen3.5-27B` | 2.67 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 3.68 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.68 | 4.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.55 | 4.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 4.65 | 4.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.61 | 1.00 | 5.00 |
| `Qwen3.5-27B` | 3.10 | 2.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.69 | 4.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 3.68 | 3.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 3.39 | 2.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 3.61 | 2.00 | 4.00 |
| `RAG-Qwen3.5-27B` | 3.13 | 1.00 | 5.00 |
| `Qwen3.5-27B` | 2.40 | 1.00 | 4.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 3.64 | 3.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
