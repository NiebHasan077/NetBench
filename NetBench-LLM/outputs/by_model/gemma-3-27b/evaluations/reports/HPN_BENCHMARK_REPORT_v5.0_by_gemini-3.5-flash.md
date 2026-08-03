# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:55:01
**Judge model:** `gemini-3.5-flash`
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
| **Correctness** | 3.67 | 3.94 | 3.87 | 4.01 | 4.54 | 4.73 |
| **Completeness** | 3.38 | 3.73 | 3.58 | 3.91 | 4.54 | 4.66 |
| **Clarity** | 4.64 | 4.76 | 4.75 | 4.82 | 4.79 | 4.86 |
| **Conciseness** | 3.73 | 3.94 | 3.87 | 3.75 | 3.91 | 4.03 |
| **Overall** | 3.79 | 4.05 | 3.96 | 4.13 | 4.52 | 4.67 |

## Scores by Category

| Category | `gemma-3-27b-instruct-full-hpn-sft` overall | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-27b-instruct-lora-sft-merged` overall | `gemma-3-27b-it` overall | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-27b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 3.76 | 4.02 | 3.98 | 4.39 | 4.58 | 4.74 |
| BDP-Based Reasoning and Window Sizing | 3.89 | 4.06 | 4.36 | 4.33 | 4.31 | 4.70 |
| Bottleneck Diagnosis and End-to-End Reasoning | 3.87 | 4.16 | 4.06 | 4.22 | 4.52 | 4.64 |
| Concurrency Tuning and Scaling | 3.99 | 4.14 | 3.76 | 4.42 | 4.53 | 4.39 |
| Dataset Partitioning and Mixed Workloads | 3.90 | 4.33 | 3.88 | 4.23 | 4.78 | 4.96 |
| Fairness, Stability, and Shared Networks | 3.55 | 3.60 | 3.53 | 4.11 | 4.18 | 4.43 |
| Parallelism and Large-File Optimization | 4.40 | 4.41 | 4.43 | 4.41 | 4.38 | 3.99 |
| Pipelining and Small-File Optimization | 3.04 | 3.98 | 3.21 | 3.48 | 4.20 | 4.65 |
| Practical HPN Scenarios and Design | 3.70 | 3.91 | 3.85 | 3.91 | 4.49 | 4.64 |
| Transfer Parameters: Definitions and Roles | 3.78 | 4.03 | 3.92 | 3.95 | 4.79 | 4.84 |

## Scores by Difficulty

| Difficulty | `gemma-3-27b-instruct-full-hpn-sft` overall | `gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `gemma-3-27b-instruct-lora-sft-merged` overall | `gemma-3-27b-it` overall | `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` overall | `RAG-gemma-3-27b-it` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.34 | 4.62 | 4.51 | 4.60 | 4.65 | 4.70 |
| medium | 3.67 | 3.96 | 3.90 | 4.16 | 4.49 | 4.70 |
| hard | 3.32 | 3.50 | 3.40 | 3.53 | 4.43 | 4.57 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.67 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.94 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.87 | 1.00 | 5.00 |
| `gemma-3-27b-it` | 4.01 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.54 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.73 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.38 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.73 | 1.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.58 | 1.00 | 5.00 |
| `gemma-3-27b-it` | 3.91 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.54 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.66 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 4.64 | 3.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.76 | 3.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 4.75 | 3.00 | 5.00 |
| `gemma-3-27b-it` | 4.82 | 3.00 | 5.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 4.79 | 2.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.86 | 3.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `gemma-3-27b-instruct-full-hpn-sft` | 3.73 | 2.00 | 5.00 |
| `gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.94 | 2.00 | 5.00 |
| `gemma-3-27b-instruct-lora-sft-merged` | 3.87 | 2.00 | 5.00 |
| `gemma-3-27b-it` | 3.75 | 2.00 | 5.00 |
| `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged` | 3.91 | 1.00 | 5.00 |
| `RAG-gemma-3-27b-it` | 4.03 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
