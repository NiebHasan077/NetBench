# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:55:01
**Judge model:** `gpt-5.1`
**Models compared:** 6
  1. `Qwen3.5-27B` (233 scored questions)
  2. `Qwen3.5-27B-instruct-full-hpn-sft` (233 scored questions)
  3. `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  4. `Qwen3.5-27B-instruct-lora-sft-merged` (233 scored questions)
  5. `RAG-Qwen3.5-27B` (233 scored questions)
  6. `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` (233 scored questions)

## Executive Summary

| Dimension | `Qwen3.5-27B` | `Qwen3.5-27B-instruct-full-hpn-sft` | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | `Qwen3.5-27B-instruct-lora-sft-merged` | `RAG-Qwen3.5-27B` | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 4.29 | 4.00 | 4.14 | 4.06 | 4.67 | 4.64 |
| **Completeness** | 4.12 | 3.67 | 3.75 | 3.67 | 4.67 | 4.54 |
| **Clarity** | 4.50 | 4.65 | 4.68 | 4.69 | 4.72 | 4.55 |
| **Conciseness** | 2.91 | 3.60 | 3.68 | 3.65 | 3.15 | 3.39 |
| **Overall** | 4.06 | 4.00 | 4.09 | 4.04 | 4.38 | 4.36 |

## Scores by Category

| Category | `Qwen3.5-27B` overall | `Qwen3.5-27B-instruct-full-hpn-sft` overall | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-27B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-27B` overall | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.17 | 3.99 | 4.11 | 4.05 | 4.45 | 4.42 |
| BDP-Based Reasoning and Window Sizing | 4.28 | 4.21 | 4.24 | 4.38 | 4.44 | 4.41 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.94 | 3.96 | 4.09 | 4.04 | 4.29 | 4.36 |
| Concurrency Tuning and Scaling | 4.21 | 4.30 | 4.07 | 4.02 | 4.40 | 4.26 |
| Dataset Partitioning and Mixed Workloads | 4.22 | 4.04 | 4.00 | 4.02 | 4.40 | 4.36 |
| Fairness, Stability, and Shared Networks | 4.24 | 3.93 | 3.87 | 3.96 | 4.34 | 4.12 |
| Parallelism and Large-File Optimization | 4.26 | 4.37 | 4.26 | 4.40 | 4.00 | 3.93 |
| Pipelining and Small-File Optimization | 3.79 | 3.82 | 3.92 | 3.88 | 4.55 | 4.35 |
| Practical HPN Scenarios and Design | 3.80 | 3.82 | 3.80 | 3.81 | 4.34 | 4.23 |
| Transfer Parameters: Definitions and Roles | 4.04 | 3.92 | 4.21 | 3.95 | 4.42 | 4.50 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-27B` overall | `Qwen3.5-27B-instruct-full-hpn-sft` overall | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-27B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-27B` overall | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.30 | 4.26 | 4.32 | 4.32 | 4.30 | 4.32 |
| medium | 4.12 | 3.99 | 4.09 | 4.07 | 4.47 | 4.38 |
| hard | 3.69 | 3.68 | 3.83 | 3.69 | 4.31 | 4.36 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 4.29 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 4.00 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.14 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.06 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.67 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.64 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 4.12 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 3.67 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 3.75 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 3.67 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.67 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.54 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 4.50 | 3.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 4.65 | 4.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.68 | 4.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.69 | 4.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.72 | 3.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.55 | 4.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 2.91 | 2.00 | 4.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 3.60 | 2.00 | 4.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 3.68 | 3.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 3.65 | 3.00 | 4.00 |
| `RAG-Qwen3.5-27B` | 3.15 | 2.00 | 4.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 3.39 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
