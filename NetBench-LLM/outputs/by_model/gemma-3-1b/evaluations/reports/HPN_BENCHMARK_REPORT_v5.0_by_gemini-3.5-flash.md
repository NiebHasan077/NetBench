# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:55
**Judge model:** `gemini-3.5-flash`
**Models compared:** 8
  1. `gemma-3-1b-cpt-full-sft` (233 scored questions)
  2. `gemma-3-1b-Instruct` (233 scored questions)
  3. `gemma-3-1b-instruct-full-hpn-sft` (233 scored questions)
  4. `gemma-3-1b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  5. `gemma-3-1b-instruct-lora-sft-merged` (233 scored questions)
  6. `RAG-gemma-3-1b-cpt-full-sft` (232 scored questions)
  7. `RAG-gemma-3-1b-Instruct` (233 scored questions)
  8. `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` (233 scored questions)

## Executive Summary

| Dimension | `gemma-3-1b-cpt-full-sft` | `gemma-3-1b-Instruct` | `gemma-3-1b-instruct-full-hpn-sft` | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | `gemma-3-1b-instruct-lora-sft-merged` | `RAG-gemma-3-1b-cpt-full-sft` | `RAG-gemma-3-1b-Instruct` | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 2.07 | 1.91 | 2.14 | 2.49 | 2.27 | 2.94 | 2.88 | 3.03 |
| **Completeness** | 1.88 | 1.86 | 2.00 | 2.29 | 2.09 | 2.91 | 2.82 | 2.99 |
| **Clarity** | 3.47 | 3.46 | 3.51 | 3.65 | 3.66 | 3.51 | 3.72 | 3.57 |
| **Conciseness** | 2.64 | 2.34 | 2.74 | 2.95 | 2.82 | 2.84 | 2.88 | 2.94 |
| **Overall** | 2.35 | 2.26 | 2.44 | 2.71 | 2.55 | 3.04 | 3.04 | 3.12 |

## Scores by Category

| Category | `gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-Instruct` overall | `gemma-3-1b-instruct-full-hpn-sft` overall | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-1b-instruct-lora-sft-merged` overall | `RAG-gemma-3-1b-cpt-full-sft` overall | `RAG-gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.56 | 2.58 | 2.61 | 3.10 | 2.75 | 3.18 | 3.09 | 3.03 |
| BDP-Based Reasoning and Window Sizing | 1.92 | 1.91 | 2.02 | 2.30 | 2.10 | 2.34 | 2.56 | 2.24 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.43 | 2.44 | 2.41 | 2.80 | 2.64 | 3.02 | 2.85 | 3.07 |
| Concurrency Tuning and Scaling | 2.79 | 2.46 | 3.11 | 3.18 | 3.02 | 3.19 | 3.20 | 3.41 |
| Dataset Partitioning and Mixed Workloads | 2.26 | 1.90 | 2.10 | 2.24 | 1.86 | 2.88 | 3.20 | 2.74 |
| Fairness, Stability, and Shared Networks | 2.41 | 1.96 | 2.38 | 2.41 | 2.23 | 2.66 | 2.95 | 3.08 |
| Parallelism and Large-File Optimization | 2.18 | 2.12 | 2.93 | 2.59 | 3.10 | 2.83 | 2.81 | 2.91 |
| Pipelining and Small-File Optimization | 2.04 | 1.79 | 2.36 | 2.21 | 2.17 | 2.78 | 2.89 | 3.03 |
| Practical HPN Scenarios and Design | 2.35 | 2.52 | 2.49 | 2.75 | 2.82 | 3.36 | 2.94 | 3.12 |
| Transfer Parameters: Definitions and Roles | 2.43 | 2.22 | 2.45 | 2.85 | 2.58 | 3.45 | 3.53 | 3.73 |

## Scores by Difficulty

| Difficulty | `gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-Instruct` overall | `gemma-3-1b-instruct-full-hpn-sft` overall | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-1b-instruct-lora-sft-merged` overall | `RAG-gemma-3-1b-cpt-full-sft` overall | `RAG-gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 2.80 | 2.80 | 3.05 | 3.42 | 3.22 | 3.40 | 3.58 | 3.55 |
| medium | 2.11 | 2.06 | 2.21 | 2.37 | 2.24 | 2.88 | 2.89 | 2.78 |
| hard | 2.19 | 1.92 | 2.06 | 2.40 | 2.23 | 2.87 | 2.63 | 3.14 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 2.07 | 1.00 | 5.00 |
| `gemma-3-1b-Instruct` | 1.91 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 2.14 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.49 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 2.27 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.94 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.88 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.03 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 1.88 | 1.00 | 5.00 |
| `gemma-3-1b-Instruct` | 1.86 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 2.00 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.29 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 2.09 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.91 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.82 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.99 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 3.47 | 1.00 | 5.00 |
| `gemma-3-1b-Instruct` | 3.46 | 2.00 | 5.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 3.51 | 2.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.65 | 2.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 3.66 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 3.51 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 3.72 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.57 | 1.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 2.64 | 1.00 | 5.00 |
| `gemma-3-1b-Instruct` | 2.34 | 1.00 | 4.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 2.74 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.95 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 2.82 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.84 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.88 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.94 | 1.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
