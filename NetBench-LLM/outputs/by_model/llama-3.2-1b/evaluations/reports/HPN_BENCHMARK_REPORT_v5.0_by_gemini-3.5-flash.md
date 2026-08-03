# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:54
**Judge model:** `gemini-3.5-flash`
**Models compared:** 8
  1. `Llama-3.2-1B-cpt-full-sft` (233 scored questions)
  2. `Llama-3.2-1B-Instruct` (233 scored questions)
  3. `Llama-3.2-1B-instruct-full-hpn-sft` (233 scored questions)
  4. `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  5. `Llama-3.2-1B-instruct-lora-sft-merged` (233 scored questions)
  6. `RAG-Llama-3.2-1B-cpt-full-sft` (233 scored questions)
  7. `RAG-Llama-3.2-1B-Instruct` (233 scored questions)
  8. `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` (233 scored questions)

## Executive Summary

| Dimension | `Llama-3.2-1B-cpt-full-sft` | `Llama-3.2-1B-Instruct` | `Llama-3.2-1B-instruct-full-hpn-sft` | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | `Llama-3.2-1B-instruct-lora-sft-merged` | `RAG-Llama-3.2-1B-cpt-full-sft` | `RAG-Llama-3.2-1B-Instruct` | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 2.33 | 1.80 | 2.27 | 2.44 | 2.36 | 3.08 | 2.74 | 3.36 |
| **Completeness** | 2.15 | 1.72 | 2.10 | 2.30 | 2.20 | 2.98 | 2.63 | 3.29 |
| **Clarity** | 3.69 | 3.37 | 3.71 | 3.83 | 3.67 | 3.64 | 3.84 | 3.78 |
| **Conciseness** | 2.97 | 2.27 | 2.90 | 3.17 | 3.05 | 3.22 | 2.70 | 3.22 |
| **Overall** | 2.63 | 2.14 | 2.58 | 2.75 | 2.64 | 3.18 | 2.93 | 3.42 |

## Scores by Category

| Category | `Llama-3.2-1B-cpt-full-sft` overall | `Llama-3.2-1B-Instruct` overall | `Llama-3.2-1B-instruct-full-hpn-sft` overall | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-sft-merged` overall | `RAG-Llama-3.2-1B-cpt-full-sft` overall | `RAG-Llama-3.2-1B-Instruct` overall | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.91 | 2.28 | 2.97 | 2.71 | 2.73 | 2.90 | 2.82 | 3.35 |
| BDP-Based Reasoning and Window Sizing | 2.36 | 2.14 | 2.27 | 2.40 | 2.21 | 2.63 | 2.53 | 2.73 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.73 | 2.05 | 2.72 | 2.87 | 2.72 | 2.97 | 3.05 | 3.18 |
| Concurrency Tuning and Scaling | 2.93 | 2.49 | 2.43 | 2.90 | 2.78 | 3.33 | 3.19 | 4.07 |
| Dataset Partitioning and Mixed Workloads | 2.07 | 2.02 | 2.12 | 3.20 | 2.14 | 2.98 | 2.56 | 2.52 |
| Fairness, Stability, and Shared Networks | 2.35 | 1.94 | 2.20 | 2.69 | 2.55 | 2.93 | 2.15 | 3.29 |
| Parallelism and Large-File Optimization | 2.60 | 2.30 | 2.99 | 3.24 | 3.31 | 3.67 | 2.98 | 3.52 |
| Pipelining and Small-File Optimization | 2.29 | 1.88 | 2.07 | 2.02 | 2.29 | 2.71 | 2.49 | 3.09 |
| Practical HPN Scenarios and Design | 2.41 | 1.98 | 2.76 | 2.69 | 2.93 | 3.18 | 2.77 | 3.46 |
| Transfer Parameters: Definitions and Roles | 2.77 | 2.25 | 2.56 | 2.89 | 2.65 | 3.90 | 3.43 | 4.09 |

## Scores by Difficulty

| Difficulty | `Llama-3.2-1B-cpt-full-sft` overall | `Llama-3.2-1B-Instruct` overall | `Llama-3.2-1B-instruct-full-hpn-sft` overall | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-sft-merged` overall | `RAG-Llama-3.2-1B-cpt-full-sft` overall | `RAG-Llama-3.2-1B-Instruct` overall | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3.17 | 2.64 | 3.23 | 3.31 | 3.26 | 3.52 | 3.50 | 3.71 |
| medium | 2.35 | 1.97 | 2.29 | 2.51 | 2.45 | 2.93 | 2.65 | 3.36 |
| hard | 2.40 | 1.79 | 2.24 | 2.46 | 2.20 | 3.18 | 2.67 | 3.15 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 2.33 | 1.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 1.80 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 2.27 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.44 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 2.36 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 3.08 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 2.74 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.36 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 2.15 | 1.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 1.72 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 2.10 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.30 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 2.20 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 2.98 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 2.63 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.29 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 3.69 | 2.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 3.37 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 3.71 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.83 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 3.67 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 3.64 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 3.84 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.78 | 1.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 2.97 | 2.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 2.27 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 2.90 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.17 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 3.05 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 3.22 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 2.70 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.22 | 1.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
