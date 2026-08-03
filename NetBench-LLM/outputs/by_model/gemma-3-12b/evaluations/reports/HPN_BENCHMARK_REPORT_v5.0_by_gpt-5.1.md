# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-09 00:42:22
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `RAG-gemma-3-12b-it` (242 scored questions)
  2. `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  3. `gemma-3-12b-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  4. `gemma-3-12b-instruct-full-hpn-sft` (242 scored questions)
  5. `gemma-3-12b-it` (242 scored questions)
  6. `gemma-3-12b-cpt-full-sft` (242 scored questions)
  7. `RAG-gemma-3-12b-cpt-full-sft` (242 scored questions)
  8. `gemma-3-12b-instruct-lora-sft-merged` (242 scored questions)

## Executive Summary

| Dimension | `RAG-gemma-3-12b-it` | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | `gemma-3-12b-instruct-full-hpn-sft` | `gemma-3-12b-it` | `gemma-3-12b-cpt-full-sft` | `RAG-gemma-3-12b-cpt-full-sft` | `gemma-3-12b-instruct-lora-sft-merged` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 4.41 | 4.22 | 3.42 | 3.42 | 3.52 | 3.45 | 4.15 | 3.40 |
| **Completeness** | 4.11 | 4.12 | 3.21 | 3.17 | 3.43 | 3.19 | 4.10 | 3.10 |
| **Clarity** | 4.61 | 4.31 | 4.35 | 4.29 | 4.41 | 4.34 | 4.30 | 4.30 |
| **Conciseness** | 3.64 | 3.29 | 3.43 | 3.51 | 2.96 | 3.40 | 3.31 | 3.35 |
| **Overall** | 4.25 | 4.05 | 3.57 | 3.57 | 3.62 | 3.57 | 4.02 | 3.53 |

## Scores by Category

| Category | `RAG-gemma-3-12b-it` overall | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-full-hpn-sft` overall | `gemma-3-12b-it` overall | `gemma-3-12b-cpt-full-sft` overall | `RAG-gemma-3-12b-cpt-full-sft` overall | `gemma-3-12b-instruct-lora-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.45 | 4.18 | 3.39 | 3.57 | 3.65 | 3.48 | 3.95 | 3.61 |
| BDP-Based Reasoning and Window Sizing | 4.17 | 3.78 | 3.57 | 3.62 | 3.95 | 3.62 | 3.80 | 3.65 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.15 | 3.98 | 3.68 | 3.59 | 3.51 | 3.72 | 4.00 | 3.65 |
| Concurrency Tuning and Scaling | 4.18 | 4.06 | 3.40 | 3.58 | 4.10 | 3.86 | 4.14 | 3.68 |
| Dataset Partitioning and Mixed Workloads | 4.38 | 4.30 | 3.30 | 3.42 | 3.54 | 3.30 | 4.14 | 3.28 |
| Fairness, Stability, and Shared Networks | 4.08 | 3.76 | 3.29 | 3.33 | 3.70 | 3.52 | 3.79 | 3.19 |
| Parallelism and Large-File Optimization | 3.81 | 4.17 | 4.08 | 4.07 | 4.08 | 3.95 | 3.89 | 4.16 |
| Pipelining and Small-File Optimization | 4.29 | 3.93 | 3.49 | 3.52 | 3.15 | 3.35 | 3.84 | 3.31 |
| Practical HPN Scenarios and Design | 4.09 | 3.79 | 3.39 | 3.25 | 3.35 | 3.30 | 3.79 | 3.37 |
| Transfer Parameters: Definitions and Roles | 4.46 | 4.35 | 3.68 | 3.65 | 3.56 | 3.54 | 4.37 | 3.41 |

## Scores by Difficulty

| Difficulty | `RAG-gemma-3-12b-it` overall | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-full-hpn-sft` overall | `gemma-3-12b-it` overall | `gemma-3-12b-cpt-full-sft` overall | `RAG-gemma-3-12b-cpt-full-sft` overall | `gemma-3-12b-instruct-lora-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.28 | 4.18 | 4.03 | 3.95 | 4.12 | 4.05 | 4.10 | 3.99 |
| medium | 4.28 | 3.98 | 3.44 | 3.49 | 3.58 | 3.45 | 4.01 | 3.45 |
| hard | 4.15 | 3.99 | 3.24 | 3.23 | 3.10 | 3.20 | 3.94 | 3.11 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-12b-it` | 4.41 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.22 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.42 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.42 | 1.00 | 5.00 |
| `gemma-3-12b-it` | 3.52 | 1.00 | 5.00 |
| `gemma-3-12b-cpt-full-sft` | 3.45 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.15 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.40 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-12b-it` | 4.11 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.12 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.21 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.17 | 1.00 | 5.00 |
| `gemma-3-12b-it` | 3.43 | 1.00 | 5.00 |
| `gemma-3-12b-cpt-full-sft` | 3.19 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.10 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.10 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-12b-it` | 4.61 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.31 | 2.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.35 | 3.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 4.29 | 3.00 | 5.00 |
| `gemma-3-12b-it` | 4.41 | 4.00 | 5.00 |
| `gemma-3-12b-cpt-full-sft` | 4.34 | 4.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.30 | 3.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 4.30 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-12b-it` | 3.64 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.29 | 2.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.43 | 2.00 | 4.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.51 | 2.00 | 4.00 |
| `gemma-3-12b-it` | 2.96 | 2.00 | 4.00 |
| `gemma-3-12b-cpt-full-sft` | 3.40 | 2.00 | 4.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 3.31 | 2.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.35 | 2.00 | 4.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
