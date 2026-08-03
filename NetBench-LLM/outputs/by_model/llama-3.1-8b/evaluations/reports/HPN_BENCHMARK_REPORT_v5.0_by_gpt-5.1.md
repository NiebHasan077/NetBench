# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:54:57
**Judge model:** `gpt-5.1`
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
| **Correctness** | 3.44 | 3.15 | 3.27 | 3.44 | 3.42 | 4.18 | 4.09 | 4.23 |
| **Completeness** | 3.15 | 3.05 | 3.07 | 3.18 | 3.19 | 4.04 | 3.97 | 4.12 |
| **Clarity** | 4.40 | 4.20 | 4.24 | 4.34 | 4.31 | 4.29 | 4.26 | 4.27 |
| **Conciseness** | 3.64 | 2.91 | 3.48 | 3.53 | 3.55 | 3.48 | 2.85 | 3.26 |
| **Overall** | 3.59 | 3.34 | 3.47 | 3.59 | 3.58 | 4.04 | 3.89 | 4.04 |

## Scores by Category

| Category | `Llama-3.1-8B-cpt-full-sft` overall | `Llama-3.1-8B-Instruct` overall | `Llama-3.1-8B-instruct-full-hpn-sft` overall | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.1-8B-instruct-lora-sft-merged` overall | `RAG-Llama-3.1-8B-cpt-full-sft` overall | `RAG-Llama-3.1-8B-Instruct` overall | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.72 | 3.37 | 3.59 | 3.81 | 3.65 | 4.06 | 3.85 | 4.07 |
| BDP-Based Reasoning and Window Sizing | 3.45 | 3.49 | 3.27 | 3.47 | 3.48 | 3.79 | 3.57 | 3.85 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.68 | 3.28 | 3.40 | 3.48 | 3.66 | 3.97 | 3.93 | 4.02 |
| Concurrency Tuning and Scaling | 3.88 | 3.67 | 3.82 | 3.40 | 3.76 | 3.92 | 3.83 | 3.77 |
| Dataset Partitioning and Mixed Workloads | 3.54 | 3.28 | 3.34 | 3.82 | 3.78 | 4.24 | 4.16 | 4.10 |
| Fairness, Stability, and Shared Networks | 3.24 | 3.23 | 3.33 | 3.27 | 3.33 | 3.87 | 3.77 | 3.81 |
| Parallelism and Large-File Optimization | 3.87 | 3.91 | 3.98 | 3.89 | 3.78 | 3.77 | 3.92 | 3.86 |
| Pipelining and Small-File Optimization | 3.57 | 3.04 | 3.41 | 3.63 | 3.52 | 3.95 | 3.78 | 4.10 |
| Practical HPN Scenarios and Design | 3.37 | 2.78 | 3.29 | 3.40 | 3.40 | 3.77 | 3.68 | 3.78 |
| Transfer Parameters: Definitions and Roles | 3.61 | 3.45 | 3.57 | 3.74 | 3.61 | 4.46 | 4.15 | 4.38 |

## Scores by Difficulty

| Difficulty | `Llama-3.1-8B-cpt-full-sft` overall | `Llama-3.1-8B-Instruct` overall | `Llama-3.1-8B-instruct-full-hpn-sft` overall | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.1-8B-instruct-lora-sft-merged` overall | `RAG-Llama-3.1-8B-cpt-full-sft` overall | `RAG-Llama-3.1-8B-Instruct` overall | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.05 | 3.79 | 3.94 | 4.06 | 4.11 | 4.11 | 4.10 | 4.11 |
| medium | 3.51 | 3.21 | 3.28 | 3.36 | 3.43 | 4.03 | 3.85 | 3.97 |
| hard | 3.16 | 3.01 | 3.20 | 3.36 | 3.21 | 3.98 | 3.69 | 4.07 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.44 | 1.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 3.15 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.27 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.44 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.42 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.18 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 4.09 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.23 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.15 | 1.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 3.05 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.07 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.18 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.19 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.04 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 3.97 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.12 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 4.40 | 3.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 4.20 | 3.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 4.24 | 3.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.34 | 3.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 4.31 | 3.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.29 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 4.26 | 3.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.27 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.64 | 2.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 2.91 | 2.00 | 4.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.48 | 2.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.53 | 3.00 | 4.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.55 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 3.48 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 2.85 | 1.00 | 4.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.26 | 1.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
