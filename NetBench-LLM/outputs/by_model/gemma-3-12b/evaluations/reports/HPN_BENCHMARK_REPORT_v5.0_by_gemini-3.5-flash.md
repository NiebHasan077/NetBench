# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:55:00
**Judge model:** `gemini-3.5-flash`
**Models compared:** 8
  1. `gemma-3-12b-cpt-full-sft` (233 scored questions)
  2. `gemma-3-12b-instruct-full-hpn-sft` (233 scored questions)
  3. `gemma-3-12b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  4. `gemma-3-12b-instruct-lora-sft-merged` (233 scored questions)
  5. `gemma-3-12b-it` (233 scored questions)
  6. `RAG-gemma-3-12b-cpt-full-sft` (233 scored questions)
  7. `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  8. `RAG-gemma-3-12b-it` (233 scored questions)

## Executive Summary

| Dimension | `gemma-3-12b-cpt-full-sft` | `gemma-3-12b-instruct-full-hpn-sft` | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | `gemma-3-12b-instruct-lora-sft-merged` | `gemma-3-12b-it` | `RAG-gemma-3-12b-cpt-full-sft` | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | `RAG-gemma-3-12b-it` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 3.63 | 3.58 | 3.72 | 3.64 | 3.72 | 4.52 | 4.52 | 4.61 |
| **Completeness** | 3.45 | 3.36 | 3.46 | 3.37 | 3.66 | 4.48 | 4.47 | 4.49 |
| **Clarity** | 4.59 | 4.65 | 4.65 | 4.60 | 4.69 | 4.79 | 4.76 | 4.83 |
| **Conciseness** | 3.74 | 3.82 | 3.85 | 3.75 | 3.55 | 3.97 | 3.91 | 4.07 |
| **Overall** | 3.78 | 3.76 | 3.84 | 3.76 | 3.88 | 4.51 | 4.49 | 4.57 |

## Scores by Category

| Category | `gemma-3-12b-cpt-full-sft` overall | `gemma-3-12b-instruct-full-hpn-sft` overall | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-lora-sft-merged` overall | `gemma-3-12b-it` overall | `RAG-gemma-3-12b-cpt-full-sft` overall | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-12b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.67 | 3.77 | 3.65 | 3.98 | 3.97 | 4.36 | 4.58 | 4.69 |
| BDP-Based Reasoning and Window Sizing | 3.69 | 3.63 | 3.66 | 3.76 | 4.18 | 4.28 | 4.10 | 4.41 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.08 | 3.91 | 4.14 | 4.01 | 3.96 | 4.53 | 4.54 | 4.51 |
| Concurrency Tuning and Scaling | 4.11 | 4.08 | 3.90 | 3.93 | 4.19 | 4.51 | 4.48 | 4.38 |
| Dataset Partitioning and Mixed Workloads | 3.50 | 3.56 | 3.32 | 3.60 | 3.58 | 4.84 | 4.92 | 4.90 |
| Fairness, Stability, and Shared Networks | 3.64 | 3.43 | 3.65 | 3.31 | 3.89 | 4.49 | 4.17 | 4.43 |
| Parallelism and Large-File Optimization | 4.16 | 4.38 | 4.29 | 4.54 | 4.56 | 4.16 | 4.36 | 4.02 |
| Pipelining and Small-File Optimization | 3.03 | 3.29 | 3.48 | 3.05 | 3.11 | 4.36 | 4.27 | 4.44 |
| Practical HPN Scenarios and Design | 3.69 | 3.63 | 3.82 | 3.75 | 3.53 | 4.46 | 4.39 | 4.64 |
| Transfer Parameters: Definitions and Roles | 3.76 | 3.79 | 3.87 | 3.58 | 3.78 | 4.77 | 4.79 | 4.77 |

## Scores by Difficulty

| Difficulty | `gemma-3-12b-cpt-full-sft` overall | `gemma-3-12b-instruct-full-hpn-sft` overall | `gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-12b-instruct-lora-sft-merged` overall | `gemma-3-12b-it` overall | `RAG-gemma-3-12b-cpt-full-sft` overall | `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-12b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.45 | 4.35 | 4.43 | 4.36 | 4.50 | 4.56 | 4.63 | 4.62 |
| medium | 3.63 | 3.64 | 3.67 | 3.65 | 3.78 | 4.47 | 4.33 | 4.56 |
| hard | 3.23 | 3.24 | 3.42 | 3.24 | 3.31 | 4.51 | 4.59 | 4.51 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 3.63 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.58 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.72 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.64 | 1.00 | 5.00 |
| `gemma-3-12b-it` | 3.72 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.52 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.52 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 4.61 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 3.45 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.36 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.46 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.37 | 1.00 | 5.00 |
| `gemma-3-12b-it` | 3.66 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.48 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.47 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 4.49 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 4.59 | 3.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 4.65 | 2.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.65 | 2.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 4.60 | 2.00 | 5.00 |
| `gemma-3-12b-it` | 4.69 | 3.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 4.79 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 4.76 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 4.83 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-12b-cpt-full-sft` | 3.74 | 1.00 | 5.00 |
| `gemma-3-12b-instruct-full-hpn-sft` | 3.82 | 2.00 | 5.00 |
| `gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.85 | 2.00 | 5.00 |
| `gemma-3-12b-instruct-lora-sft-merged` | 3.75 | 2.00 | 5.00 |
| `gemma-3-12b-it` | 3.55 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-cpt-full-sft` | 3.97 | 2.00 | 5.00 |
| `RAG-gemma-3-12b-instruct-lora-cpt-then-sft-merged` | 3.91 | 1.00 | 5.00 |
| `RAG-gemma-3-12b-it` | 4.07 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
