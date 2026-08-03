# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-05-19 00:01:48
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  2. `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  3. `Qwen3.5-9B-cpt-full-sft` (242 scored questions)
  4. `RAG-Qwen3.5-9B` (242 scored questions)
  5. `RAG-Qwen3.5-9B-cpt-full-sft` (242 scored questions)
  6. `Qwen3.5-9B-instruct-lora-sft-merged` (242 scored questions)
  7. `Qwen3.5-9B` (242 scored questions)
  8. `Qwen3.5-9B-instruct-full-hpn-sft` (242 scored questions)

## Executive Summary

| Dimension | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | `Qwen3.5-9B-cpt-full-sft` | `RAG-Qwen3.5-9B` | `RAG-Qwen3.5-9B-cpt-full-sft` | `Qwen3.5-9B-instruct-lora-sft-merged` | `Qwen3.5-9B` | `Qwen3.5-9B-instruct-full-hpn-sft` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 4.41 | 3.77 | 3.76 | 2.86 | 4.44 | 3.67 | 3.04 | 3.64 |
| **Completeness** | 4.29 | 3.45 | 3.49 | 2.33 | 4.35 | 3.33 | 2.38 | 3.35 |
| **Clarity** | 4.35 | 4.50 | 4.53 | 2.90 | 4.43 | 4.48 | 2.90 | 4.49 |
| **Conciseness** | 3.22 | 3.55 | 3.55 | 2.67 | 3.24 | 3.55 | 2.31 | 3.59 |
| **Overall** | 4.15 | 3.82 | 3.82 | 2.74 | 4.18 | 3.74 | 2.80 | 3.74 |

## Scores by Category

| Category | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-cpt-full-sft` overall | `RAG-Qwen3.5-9B` overall | `RAG-Qwen3.5-9B-cpt-full-sft` overall | `Qwen3.5-9B-instruct-lora-sft-merged` overall | `Qwen3.5-9B` overall | `Qwen3.5-9B-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.36 | 3.75 | 3.82 | 2.84 | 4.29 | 3.86 | 2.70 | 3.82 |
| BDP-Based Reasoning and Window Sizing | 4.04 | 4.00 | 3.95 | 2.67 | 4.15 | 3.86 | 3.05 | 4.10 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.10 | 3.68 | 3.72 | 2.58 | 4.15 | 3.59 | 2.73 | 3.65 |
| Concurrency Tuning and Scaling | 3.96 | 3.91 | 3.76 | 2.44 | 4.16 | 3.84 | 2.67 | 3.67 |
| Dataset Partitioning and Mixed Workloads | 4.12 | 4.18 | 3.78 | 2.42 | 4.32 | 3.62 | 2.68 | 3.74 |
| Fairness, Stability, and Shared Networks | 3.97 | 3.76 | 3.65 | 2.60 | 4.07 | 3.43 | 2.59 | 3.37 |
| Parallelism and Large-File Optimization | 4.04 | 4.37 | 4.30 | 2.64 | 4.10 | 4.07 | 3.02 | 4.34 |
| Pipelining and Small-File Optimization | 4.40 | 3.73 | 3.85 | 2.37 | 4.07 | 3.63 | 3.24 | 3.52 |
| Practical HPN Scenarios and Design | 3.80 | 3.58 | 3.65 | 2.85 | 3.87 | 3.45 | 2.40 | 3.30 |
| Transfer Parameters: Definitions and Roles | 4.35 | 3.85 | 3.90 | 3.06 | 4.37 | 3.91 | 2.91 | 3.84 |

## Scores by Difficulty

| Difficulty | `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-9B-cpt-full-sft` overall | `RAG-Qwen3.5-9B` overall | `RAG-Qwen3.5-9B-cpt-full-sft` overall | `Qwen3.5-9B-instruct-lora-sft-merged` overall | `Qwen3.5-9B` overall | `Qwen3.5-9B-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.13 | 4.18 | 4.22 | 2.51 | 4.14 | 4.08 | 2.94 | 4.07 |
| medium | 4.15 | 3.75 | 3.82 | 2.82 | 4.26 | 3.74 | 2.90 | 3.79 |
| hard | 4.18 | 3.50 | 3.38 | 2.88 | 4.12 | 3.33 | 2.51 | 3.29 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.41 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.77 | 1.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 3.76 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 2.86 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.44 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.67 | 1.00 | 5.00 |
| `Qwen3.5-9B` | 3.04 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.64 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.29 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.45 | 1.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 3.49 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 2.33 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.35 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.33 | 1.00 | 5.00 |
| `Qwen3.5-9B` | 2.38 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.35 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.35 | 3.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 4.50 | 2.00 | 5.00 |
| `Qwen3.5-9B-cpt-full-sft` | 4.53 | 4.00 | 5.00 |
| `RAG-Qwen3.5-9B` | 2.90 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 4.43 | 3.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 4.48 | 3.00 | 5.00 |
| `Qwen3.5-9B` | 2.90 | 1.00 | 5.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 4.49 | 4.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.22 | 2.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-cpt-then-sft-merged` | 3.55 | 3.00 | 4.00 |
| `Qwen3.5-9B-cpt-full-sft` | 3.55 | 3.00 | 4.00 |
| `RAG-Qwen3.5-9B` | 2.67 | 1.00 | 5.00 |
| `RAG-Qwen3.5-9B-cpt-full-sft` | 3.24 | 2.00 | 5.00 |
| `Qwen3.5-9B-instruct-lora-sft-merged` | 3.55 | 2.00 | 4.00 |
| `Qwen3.5-9B` | 2.31 | 1.00 | 4.00 |
| `Qwen3.5-9B-instruct-full-hpn-sft` | 3.59 | 3.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
