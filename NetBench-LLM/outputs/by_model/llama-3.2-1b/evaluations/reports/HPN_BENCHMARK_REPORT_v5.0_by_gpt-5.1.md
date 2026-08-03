# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-05-11 14:28:37
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `RAG-Llama-3.2-1B-Instruct` (242 scored questions)
  2. `Llama-3.2-1B-cpt-full-sft` (242 scored questions)
  3. `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  4. `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  5. `Llama-3.2-1B-instruct-lora-sft-merged` (242 scored questions)
  6. `Llama-3.2-1B-Instruct` (242 scored questions)
  7. `RAG-Llama-3.2-1B-cpt-full-sft` (242 scored questions)
  8. `Llama-3.2-1B-instruct-full-hpn-sft` (242 scored questions)

## Executive Summary

| Dimension | `RAG-Llama-3.2-1B-Instruct` | `Llama-3.2-1B-cpt-full-sft` | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | `Llama-3.2-1B-instruct-lora-sft-merged` | `Llama-3.2-1B-Instruct` | `RAG-Llama-3.2-1B-cpt-full-sft` | `Llama-3.2-1B-instruct-full-hpn-sft` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 2.68 | 2.40 | 2.98 | 2.45 | 2.33 | 1.99 | 2.76 | 2.29 |
| **Completeness** | 2.59 | 2.31 | 2.92 | 2.39 | 2.32 | 1.97 | 2.69 | 2.20 |
| **Clarity** | 3.90 | 3.87 | 3.63 | 3.91 | 3.86 | 3.73 | 3.75 | 3.88 |
| **Conciseness** | 2.85 | 3.26 | 3.04 | 3.27 | 3.22 | 2.52 | 3.29 | 3.22 |
| **Overall** | 2.97 | 2.84 | 3.11 | 2.87 | 2.79 | 2.49 | 3.02 | 2.75 |

## Scores by Category

| Category | `RAG-Llama-3.2-1B-Instruct` overall | `Llama-3.2-1B-cpt-full-sft` overall | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-sft-merged` overall | `Llama-3.2-1B-Instruct` overall | `RAG-Llama-3.2-1B-cpt-full-sft` overall | `Llama-3.2-1B-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.85 | 2.97 | 3.05 | 2.92 | 2.86 | 2.58 | 2.81 | 2.98 |
| BDP-Based Reasoning and Window Sizing | 2.78 | 2.72 | 2.67 | 2.66 | 2.55 | 2.52 | 2.66 | 2.69 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.03 | 2.85 | 2.98 | 2.84 | 2.82 | 2.40 | 2.96 | 2.75 |
| Concurrency Tuning and Scaling | 3.32 | 3.07 | 3.68 | 3.20 | 2.87 | 2.89 | 3.29 | 2.73 |
| Dataset Partitioning and Mixed Workloads | 2.70 | 2.60 | 2.64 | 3.28 | 2.36 | 2.68 | 2.76 | 2.60 |
| Fairness, Stability, and Shared Networks | 2.52 | 2.71 | 2.92 | 2.92 | 2.68 | 2.44 | 2.74 | 2.61 |
| Parallelism and Large-File Optimization | 2.89 | 2.81 | 3.13 | 3.27 | 3.33 | 2.30 | 3.26 | 3.13 |
| Pipelining and Small-File Optimization | 2.84 | 2.92 | 2.84 | 2.28 | 2.63 | 2.26 | 2.88 | 2.49 |
| Practical HPN Scenarios and Design | 2.90 | 2.71 | 3.15 | 2.76 | 2.92 | 2.34 | 2.80 | 2.80 |
| Transfer Parameters: Definitions and Roles | 3.25 | 2.89 | 3.55 | 2.98 | 2.79 | 2.60 | 3.50 | 2.71 |

## Scores by Difficulty

| Difficulty | `RAG-Llama-3.2-1B-Instruct` overall | `Llama-3.2-1B-cpt-full-sft` overall | `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.2-1B-instruct-lora-sft-merged` overall | `Llama-3.2-1B-Instruct` overall | `RAG-Llama-3.2-1B-cpt-full-sft` overall | `Llama-3.2-1B-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3.31 | 3.15 | 3.33 | 3.15 | 3.18 | 2.78 | 3.22 | 3.20 |
| medium | 2.82 | 2.69 | 3.06 | 2.76 | 2.70 | 2.43 | 2.88 | 2.58 |
| hard | 2.81 | 2.69 | 2.94 | 2.69 | 2.46 | 2.25 | 3.00 | 2.49 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-Instruct` | 2.68 | 1.00 | 5.00 |
| `Llama-3.2-1B-cpt-full-sft` | 2.40 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.98 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.45 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 2.33 | 1.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 1.99 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 2.76 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 2.29 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-Instruct` | 2.59 | 1.00 | 5.00 |
| `Llama-3.2-1B-cpt-full-sft` | 2.31 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.92 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 2.39 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 2.32 | 1.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 1.97 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 2.69 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 2.20 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-Instruct` | 3.90 | 2.00 | 5.00 |
| `Llama-3.2-1B-cpt-full-sft` | 3.87 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.63 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.91 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 3.86 | 2.00 | 5.00 |
| `Llama-3.2-1B-Instruct` | 3.73 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 3.75 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 3.88 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-Instruct` | 2.85 | 2.00 | 5.00 |
| `Llama-3.2-1B-cpt-full-sft` | 3.26 | 2.00 | 5.00 |
| `RAG-Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.04 | 1.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-cpt-then-sft-merged` | 3.27 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-lora-sft-merged` | 3.22 | 2.00 | 4.00 |
| `Llama-3.2-1B-Instruct` | 2.52 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-cpt-full-sft` | 3.29 | 2.00 | 5.00 |
| `Llama-3.2-1B-instruct-full-hpn-sft` | 3.22 | 2.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
