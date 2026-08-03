# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-04-20 12:28:20
**Judge model:** `gpt-4o`
**Models compared:** 3
  1. `RAG-gemma-4-e4b-it` (90 scored questions)
  2. `RAG-Qwen3.5-2B-trained-new-instruct` (90 scored questions)
  3. `RAG-Qwen3.5-27B-Instruct` (90 scored questions)

## Executive Summary

| Dimension | `RAG-gemma-4-e4b-it` | `RAG-Qwen3.5-2B-trained-new-instruct` | `RAG-Qwen3.5-27B-Instruct` |
| --- | ---: | ---: | ---: |
| **Correctness** | 4.30 | 3.94 | 4.40 |
| **Completeness** | 3.98 | 3.53 | 4.23 |
| **Clarity** | 4.27 | 4.30 | 4.29 |
| **Conciseness** | 3.41 | 3.73 | 3.37 |
| **Overall** | 4.10 | 3.89 | 4.20 |

## Scores by Category

| Category | `RAG-gemma-4-e4b-it` overall | `RAG-Qwen3.5-2B-trained-new-instruct` overall | `RAG-Qwen3.5-27B-Instruct` overall |
| --- | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.17 | 4.00 | 4.20 |
| BDP-Based Reasoning and Window Sizing | 4.18 | 3.28 | 4.28 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.31 | 3.97 | 4.28 |
| Concurrency Tuning and Scaling | 4.14 | 4.32 | 4.39 |
| Dataset Partitioning and Mixed Workloads | 4.19 | 3.86 | 3.87 |
| Fairness, Stability, and Shared Networks | 4.16 | 3.85 | 4.04 |
| Parallelism and Large-File Optimization | 4.10 | 3.65 | 4.12 |
| Pipelining and Small-File Optimization | 4.13 | 3.95 | 4.40 |
| Practical HPN Scenarios and Design | 3.92 | 3.77 | 4.15 |
| Transfer Parameters: Definitions and Roles | 4.06 | 4.03 | 4.29 |

## Scores by Difficulty

| Difficulty | `RAG-gemma-4-e4b-it` overall | `RAG-Qwen3.5-2B-trained-new-instruct` overall | `RAG-Qwen3.5-27B-Instruct` overall |
| --- | ---: | ---: | ---: |
| easy | 4.15 | 3.89 | 4.39 |
| medium | 4.16 | 3.94 | 4.17 |
| hard | 4.01 | 3.82 | 4.08 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-4-e4b-it` | 4.30 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B-trained-new-instruct` | 3.94 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B-Instruct` | 4.40 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-4-e4b-it` | 3.98 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B-trained-new-instruct` | 3.53 | 2.00 | 5.00 |
| `RAG-Qwen3.5-27B-Instruct` | 4.23 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-4-e4b-it` | 4.27 | 3.00 | 5.00 |
| `RAG-Qwen3.5-2B-trained-new-instruct` | 4.30 | 3.00 | 5.00 |
| `RAG-Qwen3.5-27B-Instruct` | 4.29 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-4-e4b-it` | 3.41 | 2.00 | 5.00 |
| `RAG-Qwen3.5-2B-trained-new-instruct` | 3.73 | 2.00 | 5.00 |
| `RAG-Qwen3.5-27B-Instruct` | 3.37 | 2.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-4o` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
