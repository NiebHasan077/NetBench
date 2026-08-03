# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:55
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `gemma-3-1b-cpt-full-sft` (233 scored questions)
  2. `gemma-3-1b-Instruct` (233 scored questions)
  3. `gemma-3-1b-instruct-full-hpn-sft` (233 scored questions)
  4. `gemma-3-1b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  5. `gemma-3-1b-instruct-lora-sft-merged` (233 scored questions)
  6. `RAG-gemma-3-1b-cpt-full-sft` (233 scored questions)
  7. `RAG-gemma-3-1b-Instruct` (233 scored questions)
  8. `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` (233 scored questions)

## Executive Summary

| Dimension | `gemma-3-1b-cpt-full-sft` | `gemma-3-1b-Instruct` | `gemma-3-1b-instruct-full-hpn-sft` | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | `gemma-3-1b-instruct-lora-sft-merged` | `RAG-gemma-3-1b-cpt-full-sft` | `RAG-gemma-3-1b-Instruct` | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 2.13 | 2.05 | 2.26 | 2.46 | 2.23 | 2.70 | 2.86 | 2.78 |
| **Completeness** | 2.05 | 2.03 | 2.17 | 2.31 | 2.14 | 2.61 | 2.72 | 2.72 |
| **Clarity** | 3.74 | 3.70 | 3.80 | 3.79 | 3.84 | 3.52 | 3.88 | 3.52 |
| **Conciseness** | 3.02 | 2.34 | 3.15 | 3.10 | 3.19 | 2.75 | 2.78 | 2.80 |
| **Overall** | 2.60 | 2.53 | 2.72 | 2.82 | 2.71 | 2.89 | 3.08 | 2.95 |

## Scores by Category

| Category | `gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-Instruct` overall | `gemma-3-1b-instruct-full-hpn-sft` overall | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-1b-instruct-lora-sft-merged` overall | `RAG-gemma-3-1b-cpt-full-sft` overall | `RAG-gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.64 | 2.62 | 2.87 | 3.00 | 2.68 | 3.01 | 3.13 | 2.88 |
| BDP-Based Reasoning and Window Sizing | 2.35 | 2.46 | 2.45 | 2.60 | 2.45 | 2.48 | 2.71 | 2.45 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.70 | 2.68 | 2.67 | 2.87 | 2.69 | 2.86 | 2.93 | 2.93 |
| Concurrency Tuning and Scaling | 2.81 | 2.87 | 3.43 | 3.27 | 3.03 | 3.09 | 3.20 | 3.18 |
| Dataset Partitioning and Mixed Workloads | 2.42 | 2.48 | 2.70 | 2.68 | 2.46 | 2.72 | 3.04 | 2.80 |
| Fairness, Stability, and Shared Networks | 2.52 | 2.43 | 2.62 | 2.65 | 2.61 | 2.75 | 2.95 | 2.72 |
| Parallelism and Large-File Optimization | 2.64 | 2.51 | 2.88 | 2.51 | 3.04 | 2.77 | 3.04 | 2.80 |
| Pipelining and Small-File Optimization | 2.54 | 2.13 | 2.69 | 2.58 | 2.75 | 2.70 | 3.09 | 2.83 |
| Practical HPN Scenarios and Design | 2.60 | 2.62 | 2.78 | 2.78 | 2.76 | 3.00 | 2.94 | 2.84 |
| Transfer Parameters: Definitions and Roles | 2.65 | 2.41 | 2.71 | 2.93 | 2.78 | 3.14 | 3.47 | 3.41 |

## Scores by Difficulty

| Difficulty | `gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-Instruct` overall | `gemma-3-1b-instruct-full-hpn-sft` overall | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-1b-instruct-lora-sft-merged` overall | `RAG-gemma-3-1b-cpt-full-sft` overall | `RAG-gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 2.85 | 2.79 | 3.11 | 3.21 | 3.12 | 3.09 | 3.50 | 3.20 |
| medium | 2.49 | 2.42 | 2.58 | 2.68 | 2.53 | 2.84 | 2.98 | 2.75 |
| hard | 2.50 | 2.39 | 2.46 | 2.59 | 2.48 | 2.75 | 2.74 | 2.96 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 2.13 | 1.00 | 5.00 |
| `gemma-3-1b-Instruct` | 2.05 | 1.00 | 4.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 2.26 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.46 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 2.23 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.70 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.86 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.78 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 2.05 | 1.00 | 5.00 |
| `gemma-3-1b-Instruct` | 2.03 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 2.17 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.31 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 2.14 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.61 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.72 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.72 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 3.74 | 1.00 | 4.00 |
| `gemma-3-1b-Instruct` | 3.70 | 2.00 | 4.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 3.80 | 2.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.79 | 2.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 3.84 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 3.52 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 3.88 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.52 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-cpt-full-sft` | 3.02 | 1.00 | 5.00 |
| `gemma-3-1b-Instruct` | 2.34 | 1.00 | 3.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 3.15 | 2.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.10 | 2.00 | 4.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 3.19 | 2.00 | 4.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.75 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.78 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.80 | 2.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
