# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:55:00
**Judge model:** `gpt-5.1`
**Models compared:** 6
  1. `gemma-3-27b-instruct-full-hpn-sft` (233 scored questions)
  2. `gemma-3-27b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  3. `gemma-3-27b-instruct-lora-sft-merged` (233 scored questions)
  4. `gemma-3-27b-it` (233 scored questions)
  5. `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` (233 scored questions)
  6. `RAG-gemma-3-27b-it` (233 scored questions)

## Executive Summary

| Dimension | `gemma-3-27b-instruct-full-hpn-sft` | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | `gemma-3-27b-instruct-lora-sft-merged` | `gemma-3-27b-it` | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | `RAG-gemma-3-27b-it` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 3.41 | 3.70 | 3.58 | 3.76 | 4.26 | 4.55 |
| **Completeness** | 3.12 | 3.35 | 3.30 | 3.59 | 4.20 | 4.38 |
| **Clarity** | 4.30 | 4.44 | 4.39 | 4.54 | 4.28 | 4.67 |
| **Conciseness** | 3.38 | 3.43 | 3.39 | 3.06 | 3.22 | 3.55 |
| **Overall** | 3.54 | 3.73 | 3.66 | 3.79 | 4.07 | 4.35 |

## Scores by Category

| Category | `gemma-3-27b-instruct-full-hpn-sft` overall | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-27b-instruct-lora-sft-merged` overall | `gemma-3-27b-it` overall | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-27b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.58 | 3.68 | 3.73 | 3.91 | 4.16 | 4.50 |
| BDP-Based Reasoning and Window Sizing | 3.61 | 3.77 | 4.05 | 4.02 | 3.70 | 4.32 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.56 | 3.74 | 3.56 | 3.76 | 4.06 | 4.26 |
| Concurrency Tuning and Scaling | 3.39 | 3.86 | 3.63 | 4.21 | 4.29 | 4.28 |
| Dataset Partitioning and Mixed Workloads | 3.64 | 3.82 | 3.46 | 4.06 | 4.10 | 4.54 |
| Fairness, Stability, and Shared Networks | 3.29 | 3.38 | 3.42 | 3.64 | 3.81 | 4.17 |
| Parallelism and Large-File Optimization | 4.08 | 4.12 | 3.93 | 4.02 | 4.01 | 3.93 |
| Pipelining and Small-File Optimization | 3.09 | 3.96 | 3.39 | 3.42 | 3.81 | 4.34 |
| Practical HPN Scenarios and Design | 3.34 | 3.60 | 3.40 | 3.58 | 3.92 | 4.23 |
| Transfer Parameters: Definitions and Roles | 3.62 | 3.73 | 3.70 | 3.69 | 4.39 | 4.56 |

## Scores by Difficulty

| Difficulty | `gemma-3-27b-instruct-full-hpn-sft` overall | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-27b-instruct-lora-sft-merged` overall | `gemma-3-27b-it` overall | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-27b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 3.95 | 4.18 | 4.03 | 4.18 | 4.22 | 4.35 |
| medium | 3.45 | 3.64 | 3.61 | 3.85 | 4.02 | 4.38 |
| hard | 3.17 | 3.35 | 3.29 | 3.23 | 3.97 | 4.32 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.41 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.70 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.58 | 1.00 | 5.00 |
| `gemma-3-27b-it` | 3.76 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.26 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.55 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.12 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.35 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.30 | 1.00 | 5.00 |
| `gemma-3-27b-it` | 3.59 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.20 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.38 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 4.30 | 3.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.44 | 4.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 4.39 | 3.00 | 5.00 |
| `gemma-3-27b-it` | 4.54 | 4.00 | 5.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.28 | 3.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.67 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.38 | 2.00 | 4.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.43 | 2.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.39 | 2.00 | 4.00 |
| `gemma-3-27b-it` | 3.06 | 2.00 | 4.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.22 | 2.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 3.55 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
