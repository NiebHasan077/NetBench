# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:56
**Judge model:** `gemini-3.5-flash`
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
| **Correctness** | 2.55 | 3.44 | 3.22 | 3.33 | 3.18 | 4.29 | 4.24 | 4.41 |
| **Completeness** | 2.69 | 3.26 | 3.00 | 3.05 | 2.97 | 4.41 | 4.22 | 4.39 |
| **Clarity** | 3.85 | 4.50 | 4.39 | 4.43 | 4.39 | 4.70 | 4.62 | 4.64 |
| **Conciseness** | 2.62 | 3.73 | 3.53 | 3.64 | 3.51 | 4.00 | 3.73 | 3.71 |
| **Overall** | 2.86 | 3.63 | 3.42 | 3.50 | 3.40 | 4.38 | 4.26 | 4.39 |

## Scores by Category

| Category | `Qwen3.5-2B` overall | `Qwen3.5-2B-cpt-full-sft` overall | `Qwen3.5-2B-instruct-full-hpn-sft` overall | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-2B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-2B` overall | `RAG-Qwen3.5-2B-cpt-full-sft` overall | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.08 | 3.97 | 3.78 | 3.66 | 3.74 | 4.35 | 4.03 | 4.34 |
| BDP-Based Reasoning and Window Sizing | 2.83 | 3.49 | 3.28 | 3.18 | 3.49 | 4.22 | 3.87 | 4.08 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.85 | 3.65 | 3.36 | 3.62 | 3.41 | 4.17 | 4.29 | 4.39 |
| Concurrency Tuning and Scaling | 3.20 | 3.54 | 3.66 | 3.66 | 3.37 | 4.57 | 4.09 | 4.21 |
| Dataset Partitioning and Mixed Workloads | 2.90 | 3.34 | 3.48 | 3.24 | 3.14 | 4.80 | 4.82 | 4.86 |
| Fairness, Stability, and Shared Networks | 2.69 | 3.54 | 3.70 | 3.63 | 3.49 | 4.19 | 4.16 | 4.21 |
| Parallelism and Large-File Optimization | 3.21 | 4.12 | 4.44 | 4.27 | 3.69 | 4.34 | 4.12 | 3.84 |
| Pipelining and Small-File Optimization | 2.25 | 3.11 | 2.72 | 2.94 | 2.84 | 4.52 | 4.17 | 4.57 |
| Practical HPN Scenarios and Design | 2.92 | 3.34 | 3.02 | 3.16 | 3.29 | 4.27 | 4.05 | 4.17 |
| Transfer Parameters: Definitions and Roles | 2.81 | 3.74 | 3.41 | 3.59 | 3.29 | 4.68 | 4.68 | 4.74 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-2B` overall | `Qwen3.5-2B-cpt-full-sft` overall | `Qwen3.5-2B-instruct-full-hpn-sft` overall | `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-2B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-2B` overall | `RAG-Qwen3.5-2B-cpt-full-sft` overall | `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3.57 | 4.30 | 4.15 | 4.25 | 4.01 | 4.55 | 4.35 | 4.48 |
| medium | 2.69 | 3.44 | 3.21 | 3.24 | 3.21 | 4.41 | 4.19 | 4.31 |
| hard | 2.27 | 3.14 | 2.90 | 3.03 | 2.96 | 4.14 | 4.26 | 4.39 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.55 | 1.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.44 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 3.22 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.33 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 3.18 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 4.29 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 4.24 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.41 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.69 | 1.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.26 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 3.00 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.05 | 1.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 2.97 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 4.41 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 4.22 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.39 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 3.85 | 2.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 4.50 | 3.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 4.39 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.43 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 4.39 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 4.70 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 4.62 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 4.64 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-2B` | 2.62 | 1.00 | 5.00 |
| `Qwen3.5-2B-cpt-full-sft` | 3.73 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-full-hpn-sft` | 3.53 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.64 | 2.00 | 5.00 |
| `Qwen3.5-2B-instruct-lora-sft-merged` | 3.51 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B` | 4.00 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-cpt-full-sft` | 3.73 | 1.00 | 5.00 |
| `RAG-Qwen3.5-2B-instruct-lora-cpt-then-sft-merged` | 3.71 | 1.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
