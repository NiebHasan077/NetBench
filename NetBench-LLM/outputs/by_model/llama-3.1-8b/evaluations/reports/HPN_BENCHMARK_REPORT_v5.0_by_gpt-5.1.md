# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-05-24 16:33:39
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `Llama-3.1-8B-cpt-full-sft` (242 scored questions)
  2. `Llama-3.1-8B-Instruct` (242 scored questions)
  3. `Llama-3.1-8B-instruct-full-hpn-sft` (242 scored questions)
  4. `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  5. `Llama-3.1-8B-instruct-lora-sft-merged` (242 scored questions)
  6. `RAG-Llama-3.1-8B-cpt-full-sft` (242 scored questions)
  7. `RAG-Llama-3.1-8B-Instruct` (242 scored questions)
  8. `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` (242 scored questions)

## Executive Summary

| Dimension | `Llama-3.1-8B-cpt-full-sft` | `Llama-3.1-8B-Instruct` | `Llama-3.1-8B-instruct-full-hpn-sft` | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | `Llama-3.1-8B-instruct-lora-sft-merged` | `RAG-Llama-3.1-8B-cpt-full-sft` | `RAG-Llama-3.1-8B-Instruct` | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 3.41 | 3.13 | 3.26 | 3.42 | 3.39 | 4.14 | 4.05 | 4.18 |
| **Completeness** | 3.13 | 3.03 | 3.06 | 3.16 | 3.17 | 4.01 | 3.95 | 4.09 |
| **Clarity** | 4.39 | 4.21 | 4.25 | 4.34 | 4.31 | 4.29 | 4.26 | 4.27 |
| **Conciseness** | 3.64 | 2.90 | 3.49 | 3.53 | 3.56 | 3.48 | 2.86 | 3.27 |
| **Overall** | 3.57 | 3.33 | 3.46 | 3.57 | 3.57 | 4.02 | 3.87 | 4.02 |

## Scores by Category

| Category | `Llama-3.1-8B-cpt-full-sft` overall | `Llama-3.1-8B-Instruct` overall | `Llama-3.1-8B-instruct-full-hpn-sft` overall | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.1-8B-instruct-lora-sft-merged` overall | `RAG-Llama-3.1-8B-cpt-full-sft` overall | `RAG-Llama-3.1-8B-Instruct` overall | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.72 | 3.37 | 3.59 | 3.81 | 3.65 | 4.06 | 3.85 | 4.07 |
| BDP-Based Reasoning and Window Sizing | 3.49 | 3.54 | 3.32 | 3.52 | 3.48 | 3.83 | 3.59 | 3.88 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.62 | 3.28 | 3.39 | 3.48 | 3.63 | 3.94 | 3.92 | 4.00 |
| Concurrency Tuning and Scaling | 3.88 | 3.67 | 3.82 | 3.40 | 3.76 | 3.92 | 3.83 | 3.77 |
| Dataset Partitioning and Mixed Workloads | 3.54 | 3.28 | 3.34 | 3.82 | 3.78 | 4.24 | 4.16 | 4.10 |
| Fairness, Stability, and Shared Networks | 3.31 | 3.19 | 3.24 | 3.17 | 3.24 | 3.79 | 3.66 | 3.73 |
| Parallelism and Large-File Optimization | 3.94 | 3.93 | 4.04 | 3.96 | 3.86 | 3.85 | 3.99 | 3.93 |
| Pipelining and Small-File Optimization | 3.55 | 3.01 | 3.40 | 3.55 | 3.56 | 3.95 | 3.81 | 4.08 |
| Practical HPN Scenarios and Design | 3.34 | 2.74 | 3.26 | 3.36 | 3.37 | 3.75 | 3.63 | 3.70 |
| Transfer Parameters: Definitions and Roles | 3.56 | 3.43 | 3.56 | 3.71 | 3.58 | 4.40 | 4.11 | 4.33 |

## Scores by Difficulty

| Difficulty | `Llama-3.1-8B-cpt-full-sft` overall | `Llama-3.1-8B-Instruct` overall | `Llama-3.1-8B-instruct-full-hpn-sft` overall | `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall | `Llama-3.1-8B-instruct-lora-sft-merged` overall | `RAG-Llama-3.1-8B-cpt-full-sft` overall | `RAG-Llama-3.1-8B-Instruct` overall | `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.05 | 3.78 | 3.93 | 4.04 | 4.06 | 4.09 | 4.07 | 4.10 |
| medium | 3.49 | 3.21 | 3.28 | 3.37 | 3.42 | 4.01 | 3.84 | 3.96 |
| hard | 3.14 | 2.98 | 3.18 | 3.33 | 3.20 | 3.96 | 3.69 | 4.02 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.41 | 1.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 3.13 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.26 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.42 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.39 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.14 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 4.05 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.18 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.13 | 1.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 3.03 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.06 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.16 | 1.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.17 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.01 | 1.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 3.95 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.09 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 4.39 | 3.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 4.21 | 3.00 | 5.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 4.25 | 3.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.34 | 3.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 4.31 | 3.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 4.29 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 4.26 | 3.00 | 5.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 4.27 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Llama-3.1-8B-cpt-full-sft` | 3.64 | 2.00 | 5.00 |
| `Llama-3.1-8B-Instruct` | 2.90 | 2.00 | 4.00 |
| `Llama-3.1-8B-instruct-full-hpn-sft` | 3.49 | 2.00 | 5.00 |
| `Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.53 | 3.00 | 4.00 |
| `Llama-3.1-8B-instruct-lora-sft-merged` | 3.56 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-cpt-full-sft` | 3.48 | 2.00 | 5.00 |
| `RAG-Llama-3.1-8B-Instruct` | 2.86 | 1.00 | 4.00 |
| `RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged` | 3.27 | 1.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
