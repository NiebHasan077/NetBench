# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:59
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `gemma-3-12b-cpt-full-sft` (233 scored questions)
  2. `gemma-3-12b-instruct-full-hpn-sft` (233 scored questions)
  3. `gemma-3-12b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  4. `gemma-3-12b-instruct-lora-sft-merged` (233 scored questions)
  5. `gemma-3-12b-it` (233 scored questions)
  6. `RAG-gemma-3-12b-cpt-full-sft` (233 scored questions)
  7. `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  8. `RAG-gemma-3-12b-it` (233 scored questions)

## Executive Summary

| Dimension | `gemma-3-12b-cpt-full-sft` | `gemma-3-12b-instruct-full-hpn-sft` | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | `gemma-3-12b-instruct-lora-sft-merged` | `gemma-3-12b-it` | `RAG-gemma-3-12b-cpt-full-sft` | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | `RAG-gemma-3-12b-it` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 3.45 | 3.42 | 3.42 | 3.42 | 3.52 | 4.18 | 4.26 | 4.44 |
| **Completeness** | 3.18 | 3.16 | 3.21 | 3.09 | 3.43 | 4.12 | 4.15 | 4.13 |
| **Clarity** | 4.33 | 4.29 | 4.35 | 4.29 | 4.41 | 4.30 | 4.30 | 4.62 |
| **Conciseness** | 3.39 | 3.50 | 3.42 | 3.34 | 2.96 | 3.31 | 3.27 | 3.66 |
| **Overall** | 3.57 | 3.56 | 3.57 | 3.54 | 3.62 | 4.04 | 4.07 | 4.26 |

## Scores by Category

| Category | `gemma-3-12b-cpt-full-sft` overall | `gemma-3-12b-instruct-full-hpn-sft` overall | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-lora-sft-merged` overall | `gemma-3-12b-it` overall | `RAG-gemma-3-12b-cpt-full-sft` overall | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-12b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.48 | 3.57 | 3.39 | 3.61 | 3.65 | 3.95 | 4.18 | 4.45 |
| BDP-Based Reasoning and Window Sizing | 3.58 | 3.58 | 3.53 | 3.62 | 3.92 | 3.77 | 3.75 | 4.14 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.72 | 3.60 | 3.72 | 3.67 | 3.53 | 4.01 | 4.02 | 4.17 |
| Concurrency Tuning and Scaling | 3.86 | 3.58 | 3.40 | 3.68 | 4.10 | 4.14 | 4.06 | 4.18 |
| Dataset Partitioning and Mixed Workloads | 3.30 | 3.42 | 3.30 | 3.28 | 3.54 | 4.14 | 4.30 | 4.38 |
| Fairness, Stability, and Shared Networks | 3.45 | 3.33 | 3.33 | 3.27 | 3.66 | 3.87 | 3.83 | 4.17 |
| Parallelism and Large-File Optimization | 3.88 | 4.06 | 4.00 | 4.16 | 4.04 | 3.81 | 4.10 | 3.74 |
| Pipelining and Small-File Optimization | 3.32 | 3.50 | 3.51 | 3.24 | 3.09 | 3.82 | 3.89 | 4.25 |
| Practical HPN Scenarios and Design | 3.33 | 3.28 | 3.42 | 3.44 | 3.38 | 3.89 | 3.85 | 4.15 |
| Transfer Parameters: Definitions and Roles | 3.58 | 3.63 | 3.66 | 3.42 | 3.58 | 4.41 | 4.39 | 4.49 |

## Scores by Difficulty

| Difficulty | `gemma-3-12b-cpt-full-sft` overall | `gemma-3-12b-instruct-full-hpn-sft` overall | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-lora-sft-merged` overall | `gemma-3-12b-it` overall | `RAG-gemma-3-12b-cpt-full-sft` overall | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-12b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.05 | 3.95 | 4.04 | 4.00 | 4.10 | 4.10 | 4.19 | 4.31 |
| medium | 3.45 | 3.47 | 3.42 | 3.45 | 3.59 | 4.02 | 4.00 | 4.29 |
| hard | 3.20 | 3.25 | 3.26 | 3.13 | 3.10 | 3.98 | 4.03 | 4.17 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 3.45 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.42 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.42 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.42 | 1.00 | 5.00 |
| `gemma-3-12b-it` | 3.52 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.18 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.26 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 4.44 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 3.18 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.16 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.21 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.09 | 1.00 | 5.00 |
| `gemma-3-12b-it` | 3.43 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.12 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.15 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 4.13 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 4.33 | 4.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 4.29 | 3.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.35 | 3.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 4.29 | 2.00 | 5.00 |
| `gemma-3-12b-it` | 4.41 | 4.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.30 | 3.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.30 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 4.62 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 3.39 | 2.00 | 4.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.50 | 2.00 | 4.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.42 | 2.00 | 4.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.34 | 2.00 | 4.00 |
| `gemma-3-12b-it` | 2.96 | 2.00 | 4.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 3.31 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.27 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 3.66 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
