# HPN Q&A Benchmark — Comparative Report

**Generated:** 2026-06-25 15:55:01
**Judge model:** `gemini-3.5-flash`
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
| **Correctness** | 4.64 | 4.32 | 4.38 | 4.30 | 4.87 | 4.80 |
| **Completeness** | 4.55 | 4.00 | 4.15 | 4.04 | 4.85 | 4.77 |
| **Clarity** | 4.89 | 4.90 | 4.91 | 4.88 | 4.94 | 4.93 |
| **Conciseness** | 3.87 | 4.21 | 4.24 | 4.20 | 3.93 | 4.15 |
| **Overall** | 4.58 | 4.33 | 4.40 | 4.33 | 4.78 | 4.76 |

## Scores by Category

| Category | `Qwen3.5-27B` overall | `Qwen3.5-27B-instruct-full-hpn-sft` overall | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-27B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-27B` overall | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive and Online Optimization | 4.74 | 4.23 | 4.27 | 4.22 | 4.90 | 4.68 |
| BDP-Based Reasoning and Window Sizing | 4.56 | 4.56 | 4.57 | 4.50 | 4.82 | 4.76 |
| Bottleneck Diagnosis and End-to-End Reasoning | 4.63 | 4.41 | 4.44 | 4.45 | 4.71 | 4.84 |
| Concurrency Tuning and Scaling | 4.70 | 4.53 | 4.36 | 4.41 | 4.54 | 4.60 |
| Dataset Partitioning and Mixed Workloads | 4.55 | 4.24 | 4.46 | 4.60 | 4.80 | 4.94 |
| Fairness, Stability, and Shared Networks | 4.70 | 4.07 | 4.11 | 4.19 | 4.75 | 4.53 |
| Parallelism and Large-File Optimization | 4.72 | 4.67 | 4.64 | 4.66 | 4.34 | 4.10 |
| Pipelining and Small-File Optimization | 4.28 | 3.96 | 4.08 | 3.79 | 4.90 | 4.81 |
| Practical HPN Scenarios and Design | 4.50 | 4.25 | 4.36 | 4.16 | 4.88 | 4.72 |
| Transfer Parameters: Definitions and Roles | 4.51 | 4.29 | 4.48 | 4.30 | 4.84 | 4.88 |

## Scores by Difficulty

| Difficulty | `Qwen3.5-27B` overall | `Qwen3.5-27B-instruct-full-hpn-sft` overall | `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall | `Qwen3.5-27B-instruct-lora-sft-merged` overall | `RAG-Qwen3.5-27B` overall | `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 4.76 | 4.62 | 4.64 | 4.63 | 4.72 | 4.70 |
| medium | 4.70 | 4.37 | 4.46 | 4.40 | 4.84 | 4.79 |
| hard | 4.20 | 3.94 | 4.04 | 3.87 | 4.77 | 4.76 |

## Per-Dimension Analysis

### Correctness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 4.64 | 2.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 4.32 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.38 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.30 | 2.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.87 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.80 | 1.00 | 5.00 |

### Completeness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 4.55 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 4.00 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.15 | 1.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.04 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.85 | 1.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.77 | 1.00 | 5.00 |

### Clarity

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 4.89 | 3.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 4.90 | 3.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.91 | 4.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.88 | 2.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 4.94 | 3.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.93 | 2.00 | 5.00 |

### Conciseness

| Model | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| `Qwen3.5-27B` | 3.87 | 2.00 | 5.00 |
| `Qwen3.5-27B-instruct-full-hpn-sft` | 4.21 | 3.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.24 | 3.00 | 5.00 |
| `Qwen3.5-27B-instruct-lora-sft-merged` | 4.20 | 2.00 | 5.00 |
| `RAG-Qwen3.5-27B` | 3.93 | 2.00 | 5.00 |
| `RAG-Qwen3.5-27B-instruct-lora-cpt-then-sft-merged` | 4.15 | 2.00 | 5.00 |

## Scoring Rubric

Each answer was scored by `gemini-3.5-flash` on four dimensions (1–5 scale):

| Dimension | Weight | Description |
| --- | ---: | --- |
| Correctness | 40% | Technical accuracy relative to reference |
| Completeness | 30% | Coverage of key concepts |
| Clarity | 20% | Organization and readability |
| Conciseness | 10% | Focus without unnecessary padding |

**Overall** = Correctness×0.4 + Completeness×0.3 + Clarity×0.2 + Conciseness×0.1
