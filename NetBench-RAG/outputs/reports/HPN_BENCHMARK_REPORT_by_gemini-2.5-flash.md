# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-03-23 14:02:41
**Judge model:** `gemini-2.5-flash`
**Models compared:** 2
  1. `RAG-Llama-3.2-1B-base-instruct` (90 scored questions)
  2. `RAG-Llama-3.2-1B-trained-new-instruct` (90 scored questions)

## Executive Summary

| Dimension | `RAG-Llama-3.2-1B-base-instruct` | `RAG-Llama-3.2-1B-trained-new-instruct` |
| --- | ---: | ---: |
| **Correctness** | 2.18 | 2.42 |
| **Completeness** | 1.84 | 2.00 |
| **Clarity** | 1.44 | 1.50 |
| **Conciseness** | 1.12 | 1.17 |
| **Overall** | 1.79 | 1.96 |

> **Overall winner: `RAG-Llama-3.2-1B-trained-new-instruct`** (delta = +0.17 on the 1–5 scale)

## Scores by Category

| Category | `RAG-Llama-3.2-1B-base-instruct` overall | `RAG-Llama-3.2-1B-trained-new-instruct` overall |
| --- | ---: | ---: |
| Adaptive Optimization Methods | 1.34 | 1.82 |
| BDP, RTT, and Window Reasoning | 1.61 | 1.26 |
| Bottleneck Diagnosis Across I/O and Network | 2.02 | 1.79 |
| Concurrency and Throughput Scaling | 1.66 | 2.10 |
| Fairness, Stability, and Multi-Tenant Sharing | 1.90 | 2.04 |
| Modular Architecture and Buffer Dynamics | 2.16 | 2.58 |
| Network and Transport Foundations | 2.49 | 2.26 |
| Parallelism and Chunking for Large Files | 1.61 | 1.36 |
| Scenario-Based HPN Tuning and Design | 1.21 | 2.08 |
| Small-File Transfer Optimization | 1.88 | 2.31 |

## Scores by Difficulty

| Difficulty | `RAG-Llama-3.2-1B-base-instruct` overall | `RAG-Llama-3.2-1B-trained-new-instruct` overall |
| --- | ---: | ---: |
| easy | 1.84 | 1.99 |
| medium | 2.02 | 1.95 |
| hard | 1.50 | 1.93 |

## Head-to-Head

Comparing per-question scores across 90 common questions.

| Dimension | `RAG-Llama-3.2-1B-base-instruct` Wins | Ties | `RAG-Llama-3.2-1B-trained-new-instruct` Wins |
| --- | ---: | ---: | ---: |
| **Correctness** | 22 | 41 | 27 |
| **Completeness** | 20 | 48 | 22 |
| **Clarity** | 15 | 57 | 18 |
| **Conciseness** | 3 | 81 | 6 |
| **Overall** | 29 | 26 | 35 |

### Head-to-Head by Category (Overall)

| Category | `RAG-Llama-3.2-1B-base-instruct` Wins | Ties | `RAG-Llama-3.2-1B-trained-new-instruct` Wins | `RAG-Llama-3.2-1B-base-instruct` Avg | `RAG-Llama-3.2-1B-trained-new-instruct` Avg | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive Optimization Methods | 1 | 4 | 4 | 1.34 | 1.82 | +0.48 |
| BDP, RTT, and Window Reasoning | 3 | 5 | 1 | 1.61 | 1.26 | -0.35 |
| Bottleneck Diagnosis Across I/O and Network | 5 | 2 | 2 | 2.02 | 1.79 | -0.23 |
| Concurrency and Throughput Scaling | 3 | 1 | 5 | 1.66 | 2.10 | +0.44 |
| Fairness, Stability, and Multi-Tenant Sharing | 2 | 3 | 4 | 1.90 | 2.04 | +0.14 |
| Modular Architecture and Buffer Dynamics | 3 | 2 | 4 | 2.16 | 2.58 | +0.42 |
| Network and Transport Foundations | 5 | 1 | 3 | 2.49 | 2.26 | -0.23 |
| Parallelism and Chunking for Large Files | 5 | 2 | 2 | 1.61 | 1.36 | -0.25 |
| Scenario-Based HPN Tuning and Design | 1 | 3 | 5 | 1.21 | 2.08 | +0.87 |
| Small-File Transfer Optimization | 1 | 3 | 5 | 1.88 | 2.31 | +0.43 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-base-instruct` | 2.18 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-trained-new-instruct` | 2.42 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-base-instruct` | 1.84 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-trained-new-instruct` | 2.00 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-base-instruct` | 1.44 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-trained-new-instruct` | 1.50 | 1.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-Llama-3.2-1B-base-instruct` | 1.12 | 1.00 | 5.00 |
| `RAG-Llama-3.2-1B-trained-new-instruct` | 1.17 | 1.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gemini-2.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
