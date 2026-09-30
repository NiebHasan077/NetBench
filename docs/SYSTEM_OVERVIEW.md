# NetBench — System Overview

*Domain-Adaptive LLMs for High-Performance Networking: Fine-tuning, RAG, and
Evaluation on HPN-Specific Tasks.*

This document is the code-accurate overview of the NetBench system and the
reader's entry point. It replaces the earlier `Bench-LLM.docx` scoping note;
where that note described an aspirational "LLM-as-optimizer / agentic
optimization" capability, **that capability is not implemented in this
repository and is described here only as future work.** For deeper structure
see [ARCHITECTURE.md](ARCHITECTURE.md); for the paper plan see
[STUDY_DESIGN.md](STUDY_DESIGN.md).

## What NetBench is

NetBench is an end-to-end framework for **measuring how domain adaptation
changes an LLM's competence on High-Performance Networking (HPN) tasks**. It
covers the whole chain: acquiring a domain paper corpus, generating a held-out
HPN benchmark, building an instruction dataset, adapting open-weight models with
continued pre-training (CPT), supervised fine-tuning (SFT), and LoRA, serving a
retrieval-augmented (RAG) variant, and scoring everything with a single
strong-judge protocol (GPT-5.1) plus training/inference profiling.

The **primary contribution** is the empirical adaptation study (RQ1–RQ4 below).
The **framework itself** is a secondary, engineering contribution. The HPN
benchmark and its automated generator are a supporting artifact.

## The pipeline (six modules)

```
Collect-papers/   acquire open-access HPN papers (OpenAlex + Semantic Scholar),
                  LLM-filter out-of-domain, dedup against a local paperbase
   └─> Paper2Corpus/   PDF → cleaned JSON text  ([{"text": ...}])
          └─> research_corpus_v3.json   (2,371 paper-text records; not in git)
                ├─> Benchmark-Generator/   corpus → HPN v5.0 benchmark (242 Q)
                ├─> Instruct-FTD/          corpus → instruction SFT train/val JSONL
                └─> NetBench-LLM/  +  NetBench-RAG/
                       model adaptation (CPT/SFT/LoRA), direct + RAG answer
                       generation, GPT-5.1 judging, profiling
```

Each module is independently runnable with its own virtualenv, requirements,
tests, and README. See each module's docs; the cross-module contracts are in
[ARCHITECTURE.md](ARCHITECTURE.md).

## The HPN v5.0 benchmark

- **242 questions** auto-generated from the 2,371-paper corpus by
  `Benchmark-Generator/` (chunk → filter → cards → cluster → question synthesis
  → deterministic + LLM validation → judge calibration → split). Generation uses
  local Ollama models; paid APIs appear only in judge calibration. The benchmark
  output is kept strictly separate from any training data.
- **10 skill categories:** Adaptive and Online Optimization · BDP-Based
  Reasoning and Window Sizing · Bottleneck Diagnosis and End-to-End Reasoning ·
  Concurrency Tuning and Scaling · Dataset Partitioning and Mixed Workloads ·
  Fairness, Stability, and Shared Networks · Parallelism and Large-File
  Optimization · Pipelining and Small-File Optimization · Practical HPN
  Scenarios and Design · Transfer Parameters: Definitions and Roles.
- **3 difficulty levels** (easy / medium / hard) and an expert reference answer
  per question.

## Model roster

Eight open-weight models were adapted and evaluated, spanning three size tiers
and three families, plus three frontier API baselines benchmarked under the
identical protocol.

| Tier | Model (dir) | Base HF repo | CPT-from-base? |
|---|---|---|---|
| Small | llama-3.2-1b | `meta-llama/Llama-3.2-1B` | yes |
| Small | qwen3.5-2b | `Qwen/Qwen3.5-2B-Base` | yes |
| Small | gemma-3-1b | `google/gemma-3-1b-pt` *(inferred — see note)* | yes |
| Medium | llama-3.1-8b | `meta-llama/Llama-3.1-8B` | yes |
| Medium | qwen3.5-9b | `Qwen/Qwen3.5-9B-Base` | yes |
| Medium | gemma-3-12b | `google/gemma-3-12b-pt` | yes |
| Large | qwen3.5-27b | `Qwen/Qwen3.5-27B` (instruct) | **no — disabled by design** |
| Large | gemma-3-27b | `google/gemma-3-27b-it` (instruct) | **no — disabled by design** |
| API baseline | gpt-4o, gemini-2.5-pro, claude-sonnet-4-6 | — | — |

Notes: For the **large (27B) models, full continued pre-training from a base
checkpoint was disabled** (`MODEL_HF_BASE=""`, "S4/S7 disabled" in the run
script): they are adapted from the official *instruct* checkpoint via LoRA, so
the full-CPT variants (P4/P7) do not exist for them. The `gemma-3-1b` base repo
is inferred by family convention because that run's recovered pipeline script is
mislabeled to `gemma-3-12b` (recorded as a known limitation in the run
registry); its consolidated outputs and judged scores are nonetheless the
correct `gemma-3-1b` artifacts.

## Adaptation / evaluation variants (P1–P8)

Each model is evaluated across up to eight variants. P4/P7 are absent for the
large tier (above).

| P | Definition | profiling tag |
|---|---|---|
| P1 | Original Instruct (distribution checkpoint) | S1 |
| P2 | Original Instruct + LoRA HPN-SFT | S2 |
| P3 | Original Instruct + full HPN-SFT | S3 |
| P4 | Base + full HPN-CPT + full HPN-SFT | S4 |
| P5 | Original Instruct + LoRA HPN-CPT + LoRA HPN-SFT | S5 |
| P6 | Original Instruct + RAG | S6 |
| P7 | Base + full HPN-CPT + HPN-SFT + RAG | S7 |
| P8 | Original Instruct + LoRA HPN-CPT + LoRA HPN-SFT + RAG | S8 |

## Evaluation protocol

- **Three phases:** (1) answer generation (local HF models or API models, with
  resume), (2) LLM-as-Judge scoring, (3) comparative report.
- **Judge:** `gpt-5.1`, scoring every answer on **Correctness, Completeness,
  Clarity, Conciseness** (1–5), with an `overall` mean. The judge family is
  deliberately distinct from the generated/adapted models (cross-vendor).
- **Judge validation:** a second judge (`gemini-3.5-flash`) scored every answer
  independently (model-ranking Spearman ρ = 0.987), and two domain experts
  (authors) blind-scored 125 responses (5 systems × 25 stratified questions)
  under the judge's rubric; their system ranking matches GPT-5.1's exactly
  (Kendall τ = 1.0). Reports: `analysis/outputs/HPN_JUDGE_AGREEMENT_*`; study
  design: `analysis/human_eval/README.md`.
- **Coverage:** all 242 questions × 63 model-variants were judged
  (8 models × their variant sets + 3 API baselines); zero missing cells.
- **"Thinking" caveat:** some instruct checkpoints emit chain-of-thought traces
  in the answer field, which the judge penalizes when no final answer follows.
  Whether "thinking" was enabled materially affects scores and must be reported
  per run in the paper's methods.
- **CPT quality:** an LM-adaptation report measures HPN-domain perplexity,
  general (WikiText-2) perplexity, and the forgetting gap before/after CPT.

## Research questions

RQ1–RQ4 are the empirical core; RQ5 is a cross-cutting methodological
contribution. Each maps to artifacts in `analysis/outputs/` (see
`docs/STUDY_DESIGN.md` for the full evidence mapping).

- **RQ1 — Mechanism.** Which adaptation recipe (SFT, CPT+SFT, LoRA SFT, LoRA
  CPT+SFT) moves HPN competence, and does continued pre-training add value
  *beyond* instruction tuning?
- **RQ2 — Heterogeneity.** How consistent are effects across model family,
  scale (1B–27B), skill category, and difficulty?
- **RQ3 — Retrieval vs parametric adaptation.** Holding the model fixed, what
  does retrieval add, and does RAG complement or substitute for fine-tuning?
  (strictly an information-access comparison — closed-book P1–P5 vs open-book
  P6–P8).
- **RQ4 — Frontier gap & inference cost.** Can adapted open-weight models reach
  frontier-API HPN quality, and at what latency/throughput/memory cost?
- **RQ5 — Measurement.** How often does statistical significance overstate
  *practically meaningful* improvement, and how stable are conclusions across
  skill strata and under a single LLM judge?

## Headline findings (from the consolidated results)

Full numbers: [`analysis/outputs/MASTER_RESULTS.md`](../analysis/outputs/MASTER_RESULTS.md)
(regenerate with `analysis/.venv/bin/python analysis/aggregate_scores.py`).
Paired Wilcoxon + bootstrap CIs on per-question `overall`:

- Adaptation gains are **strongly model- and size-dependent.** Full CPT+SFT
  yields large, significant gains on Qwen small/medium (e.g. qwen3.5-2b base→P4
  ≈ +1.44, d_z≈1.6), is **non-significant** on gemma-3-12b, and full SFT
  **reduces** quality on gemma-3-27b.
- **RAG helps broadly but not universally** (an open-book, information-access
  setting — not a pure model-quality gain): a large win for most models, yet for
  qwen3.5-9b RAG-only (P6) does not beat base and is far below closed-book
  CPT+SFT (P4 ≫ P6).
- The small, non-significant deltas (e.g. gemma-3-12b CPT/SFT ≈ 0.05) directly
  motivate the paper's discussion of **what counts as a "significant"
  improvement** in domain-LLM benchmarking.

## Scope and limitations

- **Not implemented (future work):** LLM-as-optimizer / agentic networking
  optimization; multi-seed variance; an API-baseline + RAG (open-book)
  condition for a like-for-like frontier comparison.
- **RAG evaluation** is folded into the per-model GPT-5.1 reports; the standalone
  `NetBench-RAG/outputs` reports are stale (older v4 / different judges) and are
  being reconciled.
- **Reproducibility:** this branch is the aggregated *results-of-record*; the
  immutable `snapshot/*` tags are the source-state provenance. Recovered runs are
  classified `recovered_from_artifact`. Large weights, checkpoints, raw corpora,
  and retrieval indexes are not distributed; they are regenerated by running the
  pipeline. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md), `ARTIFACTS.md`, and
  `experiment_registry/`.
