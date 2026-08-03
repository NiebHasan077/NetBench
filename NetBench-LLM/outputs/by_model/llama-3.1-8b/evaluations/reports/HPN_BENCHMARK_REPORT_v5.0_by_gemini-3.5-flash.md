# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:57
**Judge model:** `gemini-3.5-flash`
**Models compared:** 8
  1. `Llama-3.1-8B-cpt-full-sft` (233 scored questions)
  2. `Llama-3.1-8B-Instruct` (233 scored questions)
  3. `Llama-3.1-8B-instruct-full-hpn-sft` (233 scored questions)
  4. `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  5. `Llama-3.1-8B-instruct-lora-sft-merged` (233 scored questions)
  6. `RAG-Llama-3.1-8B-cpt-full-sft` (233 scored questions)
  7. `RAG-Llama-3.1-8B-Instruct` (233 scored questions)
  8. `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` (233 scored questions)

## Executive Summary

| Dimension | `Llama-3.1-8B-cpt-full-sft` | `Llama-3.1-8B-Instruct` | `Llama-3.1-8B-instruct-full-hpn-sft` | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | `Llama-3.1-8B-instruct-lora-sft-merged` | `RAG-Llama-3.1-8B-cpt-full-sft` | `RAG-Llama-3.1-8B-Instruct` | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 3.55 | 3.15 | 3.42 | 3.61 | 3.61 | 4.46 | 4.25 | 4.50 |
| **Completeness** | 3.30 | 2.99 | 3.16 | 3.36 | 3.36 | 4.38 | 4.17 | 4.46 |
| **Clarity** | 4.69 | 4.40 | 4.55 | 4.62 | 4.60 | 4.71 | 4.44 | 4.76 |
| **Conciseness** | 4.00 | 3.03 | 3.83 | 3.93 | 3.91 | 4.13 | 3.09 | 4.00 |
| **Overall** | 3.75 | 3.34 | 3.62 | 3.77 | 3.77 | 4.46 | 4.15 | 4.49 |

## Scores by Category

| Category | `Llama-3.1-8B-cpt-full-sft` overall | `Llama-3.1-8B-Instruct` overall | `Llama-3.1-8B-instruct-full-hpn-sft` overall | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.1-8B-instruct-lora-sft-merged` overall | `RAG-Llama-3.1-8B-cpt-full-sft` overall | `RAG-Llama-3.1-8B-Instruct` overall | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.80 | 3.37 | 3.74 | 3.92 | 3.84 | 4.46 | 4.06 | 4.47 |
| BDP-Based Reasoning and Window Sizing | 3.52 | 3.43 | 3.19 | 3.42 | 3.57 | 4.14 | 3.71 | 4.33 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.97 | 3.36 | 3.60 | 3.81 | 3.94 | 4.46 | 4.14 | 4.52 |
| Concurrency Tuning and Scaling | 4.18 | 3.67 | 3.97 | 3.89 | 3.93 | 4.42 | 4.18 | 4.10 |
| Dataset Partitioning and Mixed Workloads | 3.66 | 2.92 | 2.98 | 3.78 | 3.96 | 4.72 | 4.40 | 4.70 |
| Fairness, Stability, and Shared Networks | 3.30 | 3.09 | 3.55 | 3.53 | 3.36 | 4.37 | 3.96 | 4.37 |
| Parallelism and Large-File Optimization | 4.20 | 4.33 | 4.37 | 4.39 | 4.33 | 3.72 | 4.07 | 3.92 |
| Pipelining and Small-File Optimization | 3.38 | 2.79 | 3.48 | 3.58 | 3.02 | 4.29 | 4.22 | 4.55 |
| Practical HPN Scenarios and Design | 3.55 | 2.78 | 3.63 | 3.55 | 3.74 | 4.41 | 3.99 | 4.34 |
| Transfer Parameters: Definitions and Roles | 3.80 | 3.49 | 3.73 | 3.92 | 3.81 | 4.81 | 4.54 | 4.80 |

## Scores by Difficulty

| Difficulty | `Llama-3.1-8B-cpt-full-sft` overall | `Llama-3.1-8B-Instruct` overall | `Llama-3.1-8B-instruct-full-hpn-sft` overall | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.1-8B-instruct-lora-sft-merged` overall | `RAG-Llama-3.1-8B-cpt-full-sft` overall | `RAG-Llama-3.1-8B-Instruct` overall | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.35 | 3.98 | 4.22 | 4.39 | 4.47 | 4.46 | 4.44 | 4.56 |
| medium | 3.67 | 3.22 | 3.35 | 3.53 | 3.55 | 4.44 | 4.14 | 4.40 |
| hard | 3.18 | 2.77 | 3.32 | 3.42 | 3.27 | 4.48 | 3.83 | 4.55 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.55 | 1.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 3.15 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.42 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.61 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.61 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.46 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 4.25 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.50 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.30 | 1.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 2.99 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.16 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.36 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.36 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.38 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 4.17 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.46 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 4.69 | 3.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 4.40 | 2.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 4.55 | 3.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.62 | 2.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 4.60 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.71 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 4.44 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.76 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 4.00 | 2.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 3.03 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.83 | 2.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.93 | 2.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.91 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.13 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 3.09 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.00 | 1.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
