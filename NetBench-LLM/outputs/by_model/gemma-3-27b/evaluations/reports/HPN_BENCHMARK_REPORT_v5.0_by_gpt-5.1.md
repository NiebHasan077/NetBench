# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-16 19:36:05
**Judge model:** `gpt-5.1`
**Models compared:** 6
  1. `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  2. `gemma-3-27b-instruct-full-hpn-sft` (242 scored questions)
  3. `gemma-3-27b-instruct-lora-sft-merged` (242 scored questions)
  4. `gemma-3-27b-it` (242 scored questions)
  5. `gemma-3-27b-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  6. `RAG-gemma-3-27b-it` (242 scored questions)

## Executive Summary

| Dimension | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | `gemma-3-27b-instruct-full-hpn-sft` | `gemma-3-27b-instruct-lora-sft-merged` | `gemma-3-27b-it` | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | `RAG-gemma-3-27b-it` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 4.22 | 3.40 | 3.58 | 3.76 | 3.70 | 4.52 |
| **Completeness** | 4.17 | 3.11 | 3.30 | 3.59 | 3.37 | 4.36 |
| **Clarity** | 4.29 | 4.31 | 4.39 | 4.54 | 4.44 | 4.67 |
| **Conciseness** | 3.23 | 3.38 | 3.40 | 3.07 | 3.43 | 3.54 |
| **Overall** | 4.05 | 3.53 | 3.66 | 3.79 | 3.74 | 4.33 |

## Scores by Category

| Category | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-27b-instruct-full-hpn-sft` overall | `gemma-3-27b-instruct-lora-sft-merged` overall | `gemma-3-27b-it` overall | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-27b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.16 | 3.58 | 3.73 | 3.91 | 3.68 | 4.50 |
| BDP-Based Reasoning and Window Sizing | 3.74 | 3.65 | 4.07 | 4.05 | 3.81 | 4.34 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.05 | 3.53 | 3.54 | 3.74 | 3.74 | 4.25 |
| Concurrency Tuning and Scaling | 4.29 | 3.39 | 3.63 | 4.21 | 3.86 | 4.28 |
| Dataset Partitioning and Mixed Workloads | 4.10 | 3.64 | 3.46 | 4.06 | 3.82 | 4.54 |
| Fairness, Stability, and Shared Networks | 3.73 | 3.20 | 3.38 | 3.66 | 3.41 | 4.07 |
| Parallelism and Large-File Optimization | 4.04 | 4.08 | 4.00 | 4.08 | 4.17 | 3.73 |
| Pipelining and Small-File Optimization | 3.88 | 3.08 | 3.43 | 3.38 | 3.95 | 4.38 |
| Practical HPN Scenarios and Design | 3.83 | 3.27 | 3.37 | 3.54 | 3.56 | 4.15 |
| Transfer Parameters: Definitions and Roles | 4.33 | 3.66 | 3.73 | 3.72 | 3.74 | 4.56 |

## Scores by Difficulty

| Difficulty | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-27b-instruct-full-hpn-sft` overall | `gemma-3-27b-instruct-lora-sft-merged` overall | `gemma-3-27b-it` overall | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-27b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.19 | 3.94 | 4.03 | 4.19 | 4.17 | 4.35 |
| medium | 4.00 | 3.47 | 3.62 | 3.86 | 3.67 | 4.35 |
| hard | 3.94 | 3.13 | 3.29 | 3.20 | 3.33 | 4.29 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.22 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.40 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.58 | 1.00 | 5.00 |
| `gemma-3-27b-it` | 3.76 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.70 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.52 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.17 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.11 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.30 | 1.00 | 5.00 |
| `gemma-3-27b-it` | 3.59 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.37 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.36 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.29 | 3.00 | 5.00 |
| `gemma-3-27b-instruct-full-hpn-sft` | 4.31 | 3.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 4.39 | 3.00 | 5.00 |
| `gemma-3-27b-it` | 4.54 | 4.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.44 | 4.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.67 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.23 | 2.00 | 5.00 |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.38 | 2.00 | 4.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.40 | 2.00 | 4.00 |
| `gemma-3-27b-it` | 3.07 | 2.00 | 4.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.43 | 2.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 3.54 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
