# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:56
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `Qwen3.5-2B` (233 scored questions)
  2. `Qwen3.5-2B-cpt-full-sft` (233 scored questions)
  3. `Qwen3.5-2B-instruct-full-hpn-sft` (233 scored questions)
  4. `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  5. `Qwen3.5-2B-instruct-lora-sft-merged` (233 scored questions)
  6. `RAG-Qwen3.5-2B` (233 scored questions)
  7. `RAG-Qwen3.5-2B-cpt-full-sft` (233 scored questions)
  8. `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` (233 scored questions)

## Executive Summary

| Dimension | `Qwen3.5-2B` | `Qwen3.5-2B-cpt-full-sft` | `Qwen3.5-2B-instruct-full-hpn-sft` | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | `Qwen3.5-2B-instruct-lora-sft-merged` | `RAG-Qwen3.5-2B` | `RAG-Qwen3.5-2B-cpt-full-sft` | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 2.33 | 3.31 | 3.16 | 3.20 | 3.06 | 3.78 | 3.92 | 4.11 |
| **Completeness** | 2.42 | 3.10 | 2.93 | 2.97 | 2.85 | 3.96 | 3.89 | 4.08 |
| **Clarity** | 3.68 | 4.27 | 4.18 | 4.21 | 4.14 | 4.27 | 4.18 | 4.12 |
| **Conciseness** | 2.39 | 3.40 | 3.35 | 3.38 | 3.34 | 3.27 | 3.25 | 3.09 |
| **Overall** | 2.71 | 3.50 | 3.36 | 3.41 | 3.30 | 3.86 | 3.82 | 3.92 |

## Scores by Category

| Category | `Qwen3.5-2B` overall | `Qwen3.5-2B-cpt-full-sft` overall | `Qwen3.5-2B-instruct-full-hpn-sft` overall | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-2B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-2B` overall | `RAG-Qwen3.5-2B-cpt-full-sft` overall | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.80 | 3.71 | 3.49 | 3.51 | 3.45 | 3.92 | 3.63 | 3.88 |
| BDP-Based Reasoning and Window Sizing | 2.81 | 3.51 | 3.44 | 3.28 | 3.46 | 3.86 | 3.48 | 3.80 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.72 | 3.47 | 3.33 | 3.38 | 3.23 | 3.63 | 3.80 | 3.87 |
| Concurrency Tuning and Scaling | 3.00 | 3.36 | 3.50 | 3.72 | 3.31 | 3.97 | 3.77 | 3.86 |
| Dataset Partitioning and Mixed Workloads | 2.74 | 3.40 | 3.36 | 2.88 | 3.10 | 4.22 | 4.20 | 4.20 |
| Fairness, Stability, and Shared Networks | 2.48 | 3.45 | 3.65 | 3.53 | 3.48 | 3.61 | 3.63 | 3.62 |
| Parallelism and Large-File Optimization | 3.17 | 3.69 | 3.98 | 4.09 | 3.53 | 4.10 | 3.96 | 3.67 |
| Pipelining and Small-File Optimization | 2.53 | 3.21 | 3.13 | 3.18 | 3.10 | 3.97 | 3.88 | 4.06 |
| Practical HPN Scenarios and Design | 2.63 | 3.27 | 2.87 | 3.19 | 3.01 | 3.48 | 3.49 | 3.62 |
| Transfer Parameters: Definitions and Roles | 2.62 | 3.58 | 3.33 | 3.43 | 3.28 | 4.14 | 4.24 | 4.25 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-2B` overall | `Qwen3.5-2B-cpt-full-sft` overall | `Qwen3.5-2B-instruct-full-hpn-sft` overall | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-2B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-2B` overall | `RAG-Qwen3.5-2B-cpt-full-sft` overall | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3.12 | 3.94 | 3.81 | 3.95 | 3.76 | 3.98 | 3.92 | 3.99 |
| medium | 2.59 | 3.42 | 3.26 | 3.22 | 3.15 | 3.89 | 3.75 | 3.90 |
| hard | 2.42 | 3.10 | 2.98 | 3.05 | 2.98 | 3.66 | 3.80 | 3.87 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.33 | 1.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.31 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 3.16 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.20 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 3.06 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 3.78 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 3.92 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.11 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.42 | 1.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.10 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 2.93 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 2.97 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 2.85 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 3.96 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 3.89 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.08 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 3.68 | 2.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 4.27 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 4.18 | 3.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.21 | 3.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 4.14 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 4.27 | 3.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 4.18 | 3.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.12 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.39 | 1.00 | 4.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.40 | 2.00 | 4.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 3.35 | 2.00 | 4.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.38 | 2.00 | 4.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 3.34 | 2.00 | 4.00 |
| `RAG-Qwen3.5-2B` | 3.27 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 3.25 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.09 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
