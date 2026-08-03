# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:54
**Judge model:** `gpt-5.1`
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
| **Correctness** | 2.42 | 2.02 | 2.29 | 2.48 | 2.35 | 2.80 | 2.70 | 3.02 |
| **Completeness** | 2.33 | 1.98 | 2.21 | 2.42 | 2.33 | 2.72 | 2.61 | 2.95 |
| **Clarity** | 3.87 | 3.74 | 3.88 | 3.90 | 3.86 | 3.74 | 3.91 | 3.63 |
| **Conciseness** | 3.25 | 2.53 | 3.20 | 3.25 | 3.21 | 3.28 | 2.86 | 3.03 |
| **Overall** | 2.85 | 2.51 | 2.76 | 2.89 | 2.80 | 3.04 | 2.99 | 3.14 |

## Scores by Category

| Category | `Llama-3.2-1B-cpt-full-sft` overall | `Llama-3.2-1B-Instruct` overall | `Llama-3.2-1B-instruct-full-hpn-sft` overall | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-sft-merged` overall | `RAG-Llama-3.2-1B-cpt-full-sft` overall | `RAG-Llama-3.2-1B-Instruct` overall | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.97 | 2.58 | 2.98 | 2.92 | 2.86 | 2.81 | 2.85 | 3.05 |
| BDP-Based Reasoning and Window Sizing | 2.74 | 2.54 | 2.62 | 2.68 | 2.57 | 2.69 | 2.73 | 2.70 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.87 | 2.42 | 2.77 | 2.87 | 2.83 | 2.97 | 3.04 | 2.98 |
| Concurrency Tuning and Scaling | 3.07 | 2.89 | 2.73 | 3.20 | 2.87 | 3.29 | 3.32 | 3.68 |
| Dataset Partitioning and Mixed Workloads | 2.60 | 2.68 | 2.60 | 3.28 | 2.36 | 2.76 | 2.70 | 2.64 |
| Fairness, Stability, and Shared Networks | 2.76 | 2.49 | 2.66 | 2.99 | 2.73 | 2.80 | 2.56 | 2.96 |
| Parallelism and Large-File Optimization | 2.67 | 2.37 | 3.06 | 3.17 | 3.23 | 3.41 | 3.00 | 3.27 |
| Pipelining and Small-File Optimization | 2.94 | 2.31 | 2.54 | 2.32 | 2.69 | 2.75 | 2.86 | 2.88 |
| Practical HPN Scenarios and Design | 2.71 | 2.36 | 2.84 | 2.80 | 2.98 | 2.82 | 2.91 | 3.21 |
| Transfer Parameters: Definitions and Roles | 2.91 | 2.60 | 2.72 | 3.01 | 2.81 | 3.55 | 3.30 | 3.58 |

## Scores by Difficulty

| Difficulty | `Llama-3.2-1B-cpt-full-sft` overall | `Llama-3.2-1B-Instruct` overall | `Llama-3.2-1B-instruct-full-hpn-sft` overall | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-sft-merged` overall | `RAG-Llama-3.2-1B-cpt-full-sft` overall | `RAG-Llama-3.2-1B-Instruct` overall | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3.18 | 2.82 | 3.20 | 3.19 | 3.22 | 3.26 | 3.33 | 3.36 |
| medium | 2.69 | 2.44 | 2.57 | 2.76 | 2.68 | 2.90 | 2.83 | 3.08 |
| hard | 2.70 | 2.27 | 2.52 | 2.73 | 2.49 | 3.01 | 2.82 | 2.97 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 2.42 | 1.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 2.02 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 2.29 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.48 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 2.35 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 2.80 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 2.70 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.02 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 2.33 | 1.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 1.98 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 2.21 | 1.00 | 4.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.42 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 2.33 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 2.72 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 2.61 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.95 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 3.87 | 2.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 3.74 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 3.88 | 3.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.90 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 3.86 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 3.74 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 3.91 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.63 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.2-1B-cpt-full-sft` | 3.25 | 2.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 2.53 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 3.20 | 2.00 | 4.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.25 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 3.21 | 2.00 | 4.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 3.28 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-Instruct` | 2.86 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.03 | 1.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
