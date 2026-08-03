# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-05-14 20:34:00
**Judge model:** `gpt-5.1`
**Models compared:** 8
  1. `gemma-3-1b-Instruct` (242 scored questions)
  2. `RAG-gemma-3-1b-cpt-full-sft` (242 scored questions)
  3. `gemma-3-1b-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  4. `RAG-gemma-3-1b-Instruct` (242 scored questions)
  5. `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` (242 scored questions)
  6. `gemma-3-1b-cpt-full-sft` (242 scored questions)
  7. `gemma-3-1b-instruct-lora-sft-merged` (242 scored questions)
  8. `gemma-3-1b-instruct-full-hpn-sft` (242 scored questions)

## Executive Summary

| Dimension | `gemma-3-1b-Instruct` | `RAG-gemma-3-1b-cpt-full-sft` | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | `RAG-gemma-3-1b-Instruct` | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | `gemma-3-1b-cpt-full-sft` | `gemma-3-1b-instruct-lora-sft-merged` | `gemma-3-1b-instruct-full-hpn-sft` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correctness** | 2.05 | 2.67 | 2.43 | 2.85 | 2.74 | 2.13 | 2.21 | 2.22 |
| **Completeness** | 2.03 | 2.60 | 2.30 | 2.70 | 2.68 | 2.06 | 2.12 | 2.14 |
| **Clarity** | 3.71 | 3.51 | 3.79 | 3.89 | 3.52 | 3.75 | 3.83 | 3.80 |
| **Conciseness** | 2.34 | 2.76 | 3.11 | 2.79 | 2.81 | 3.03 | 3.19 | 3.16 |
| **Overall** | 2.53 | 2.88 | 2.81 | 3.07 | 2.93 | 2.61 | 2.69 | 2.70 |

## Scores by Category

| Category | `gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-instruct-lora-sft-merged` overall | `gemma-3-1b-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 2.62 | 3.01 | 3.00 | 3.13 | 2.88 | 2.64 | 2.68 | 2.87 |
| BDP-Based Reasoning and Window Sizing | 2.52 | 2.48 | 2.60 | 2.77 | 2.49 | 2.41 | 2.50 | 2.46 |
| Bottleneck Diagnosis and End-to-End Reasoning | 2.68 | 2.85 | 2.86 | 2.93 | 2.89 | 2.68 | 2.67 | 2.64 |
| Concurrency Tuning and Scaling | 2.87 | 3.09 | 3.27 | 3.20 | 3.18 | 2.81 | 3.03 | 3.43 |
| Dataset Partitioning and Mixed Workloads | 2.48 | 2.72 | 2.68 | 3.04 | 2.80 | 2.42 | 2.46 | 2.70 |
| Fairness, Stability, and Shared Networks | 2.40 | 2.73 | 2.61 | 2.92 | 2.71 | 2.58 | 2.56 | 2.58 |
| Parallelism and Large-File Optimization | 2.61 | 2.85 | 2.69 | 3.11 | 2.71 | 2.77 | 3.17 | 2.78 |
| Pipelining and Small-File Optimization | 2.12 | 2.75 | 2.58 | 3.05 | 2.79 | 2.55 | 2.68 | 2.63 |
| Practical HPN Scenarios and Design | 2.59 | 2.97 | 2.74 | 2.88 | 2.79 | 2.60 | 2.72 | 2.78 |
| Transfer Parameters: Definitions and Roles | 2.40 | 3.10 | 2.90 | 3.42 | 3.37 | 2.62 | 2.74 | 2.69 |

## Scores by Difficulty

| Difficulty | `gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-1b-Instruct` overall | `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-1b-cpt-full-sft` overall | `gemma-3-1b-instruct-lora-sft-merged` overall | `gemma-3-1b-instruct-full-hpn-sft` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 2.79 | 3.05 | 3.17 | 3.49 | 3.19 | 2.86 | 3.10 | 3.09 |
| medium | 2.44 | 2.84 | 2.69 | 2.98 | 2.74 | 2.49 | 2.55 | 2.56 |
| hard | 2.37 | 2.75 | 2.58 | 2.72 | 2.91 | 2.50 | 2.45 | 2.44 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-Instruct` | 2.05 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.67 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.43 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.85 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.74 | 1.00 | 5.00 |
| `gemma-3-1b-cpt-full-sft` | 2.13 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 2.21 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 2.22 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-Instruct` | 2.03 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.60 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.30 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 2.70 | 1.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.68 | 1.00 | 5.00 |
| `gemma-3-1b-cpt-full-sft` | 2.06 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 2.12 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 2.14 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-Instruct` | 3.71 | 2.00 | 4.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 3.51 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.79 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-Instruct` | 3.89 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.52 | 2.00 | 5.00 |
| `gemma-3-1b-cpt-full-sft` | 3.75 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 3.83 | 2.00 | 5.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 3.80 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-1b-Instruct` | 2.34 | 1.00 | 3.00 |
| `RAG-gemma-3-1b-cpt-full-sft` | 2.76 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 3.11 | 2.00 | 4.00 |
| `RAG-gemma-3-1b-Instruct` | 2.79 | 2.00 | 5.00 |
| `RAG-gemma-3-1b-instruct-lora-cpt-then-sft-merged` | 2.81 | 2.00 | 4.00 |
| `gemma-3-1b-cpt-full-sft` | 3.03 | 1.00 | 5.00 |
| `gemma-3-1b-instruct-lora-sft-merged` | 3.19 | 2.00 | 4.00 |
| `gemma-3-1b-instruct-full-hpn-sft` | 3.16 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gpt-5.1` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
