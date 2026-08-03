# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:59
**Judge model:** `gemini-3.5-flash`
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
| **Correctness** | 4.25 | 4.05 | 3.93 | 4.15 | 3.87 | 4.74 | 4.69 | 4.68 |
| **Completeness** | 4.18 | 3.82 | 3.68 | 3.85 | 3.58 | 4.74 | 4.62 | 4.62 |
| **Clarity** | 4.81 | 4.85 | 4.78 | 4.83 | 4.76 | 4.89 | 4.88 | 4.84 |
| **Conciseness** | 3.66 | 4.06 | 4.02 | 4.09 | 3.93 | 3.84 | 4.01 | 3.97 |
| **Overall** | 4.29 | 4.15 | 4.03 | 4.19 | 3.97 | 4.68 | 4.64 | 4.62 |

## Scores by Category

| Category | `Qwen3.5-9B` overall | `Qwen3.5-9B-cpt-full-sft` overall | `Qwen3.5-9B-instruct-full-hpn-sft` overall | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-9B` overall | `RAG-Qwen3.5-9B-cpt-full-sft` overall | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.42 | 4.12 | 4.13 | 4.12 | 4.03 | 4.78 | 4.64 | 4.76 |
| BDP-Based Reasoning and Window Sizing | 4.42 | 4.16 | 4.29 | 4.22 | 4.03 | 4.64 | 4.52 | 4.51 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.38 | 4.19 | 3.95 | 4.20 | 3.95 | 4.67 | 4.67 | 4.63 |
| Concurrency Tuning and Scaling | 4.22 | 3.96 | 4.16 | 3.91 | 4.13 | 4.48 | 4.54 | 4.19 |
| Dataset Partitioning and Mixed Workloads | 4.34 | 3.94 | 4.28 | 4.14 | 3.41 | 4.90 | 4.94 | 4.94 |
| Fairness, Stability, and Shared Networks | 3.95 | 4.02 | 3.63 | 4.02 | 3.63 | 4.65 | 4.73 | 4.55 |
| Parallelism and Large-File Optimization | 4.60 | 4.67 | 4.83 | 4.80 | 4.30 | 4.21 | 4.09 | 4.29 |
| Pipelining and Small-File Optimization | 4.16 | 3.87 | 3.67 | 3.87 | 3.60 | 4.60 | 4.58 | 4.79 |
| Practical HPN Scenarios and Design | 4.11 | 4.04 | 3.74 | 4.11 | 3.76 | 4.73 | 4.48 | 4.49 |
| Transfer Parameters: Definitions and Roles | 4.21 | 4.21 | 4.06 | 4.31 | 4.13 | 4.76 | 4.79 | 4.75 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-9B` overall | `Qwen3.5-9B-cpt-full-sft` overall | `Qwen3.5-9B-instruct-full-hpn-sft` overall | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-9B` overall | `RAG-Qwen3.5-9B-cpt-full-sft` overall | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.68 | 4.59 | 4.52 | 4.60 | 4.41 | 4.76 | 4.57 | 4.60 |
| medium | 4.41 | 4.09 | 4.04 | 4.10 | 3.95 | 4.64 | 4.69 | 4.63 |
| hard | 3.64 | 3.70 | 3.46 | 3.86 | 3.48 | 4.64 | 4.64 | 4.65 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 4.25 | 1.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 4.05 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.93 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.15 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.87 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 4.74 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.69 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.68 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 4.18 | 1.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 3.82 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.68 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.85 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.58 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 4.74 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.62 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.62 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 4.81 | 3.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 4.85 | 2.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 4.78 | 3.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.83 | 2.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 4.76 | 3.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 4.89 | 3.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.88 | 3.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.84 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-9B` | 3.66 | 2.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 4.06 | 2.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 4.02 | 2.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.09 | 2.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.93 | 2.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 3.84 | 2.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.01 | 2.00 | 5.00 |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.97 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
