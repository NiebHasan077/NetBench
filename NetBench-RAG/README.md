# NetBench-RAG

A production-grade **Retrieval-Augmented Generation (RAG)** system for High-Performance Networking (HPN) research. NetBench-RAG retrieves relevant passages from a curated corpus of HPN research papers and generates grounded, citation-backed answers using either GPT-4o or a locally fine-tuned Llama 3.1 8B model.

The system includes a complete three-phase evaluation pipeline benchmarked against a 242-question HPN QA dataset, with results designed to be directly comparable to the sibling [LLM-Training](../LLM-Training/) project.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Benchmark Dataset](#benchmark-dataset)
- [Documentation](#documentation)

---

## Overview

NetBench-RAG implements a **three-stage retrieval pipeline** combined with LLM generation to answer questions about HPN topics such as congestion control algorithms (BBR, CUBIC), RDMA, TCP/IP tuning, and high-throughput network design.

**Key capabilities:**
- Dual retrieval (dense vector + sparse BM25) with Reciprocal Rank Fusion
- Cross-encoder reranking for precise final selection
- Two generation backends: GPT-4o (API) or fine-tuned Llama 3.1 8B (local GPU)
- HPN-domain acronym expansion for improved BM25 retrieval
- Full evaluation suite with LLM-as-Judge scoring (Gemini or OpenAI)
- Excel + Markdown comparative reports compatible with LLM-Training project

---

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: Dual Search                               │
│  ├── Vector Search (Qdrant, top-30)                 │
│  │   └── BAAI/bge-large-en-v1.5 embeddings         │
│  └── BM25 Search (rank_bm25, top-30)               │
│      └── HPN acronym expansion                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: RRF Fusion (k=60, top-20 candidates)     │
│  Reciprocal Rank Fusion merges both rankings        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3: Cross-Encoder Reranking                   │
│  BAAI/bge-reranker-v2-m3 → top-5 final chunks      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  LLM Generation                                     │
│  ├── OpenAI backend: GPT-4o                        │
│  └── Local backend: Llama 3.1 8B (fine-tuned)     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              Grounded Answer
          with inline citations
```

### Corpus Indexing

```
research_corpus_v3.json / PDFs
        │
        ▼
   src/parser.py        ← Extract sections, detect headers
        │
        ▼
   src/chunker.py       ← Token-aware chunks (~1200 tok, 150-tok overlap)
        │
        ▼
   src/embedder.py      ← BAAI/bge-large-en-v1.5 embeddings
        │
     ┌──┴──┐
     ▼     ▼
 Qdrant   BM25       ← Dual indexes (point IDs aligned)
  Store   Index
     │     │
     └──┬──┘
        ▼
  chunks_cache.pkl    ← Fast positional lookup
```

---

## Requirements

- **Python** 3.11+
- **CUDA** 12.x (for GPU-accelerated embedding, reranking, and local LLM)
- **RAM** 32 GB+ recommended
- **GPU** 2× RTX A6000 (or equivalent VRAM ≥ 24 GB for local backend)
- **Disk** ~10 GB for Qdrant store + BM25 index

**API Keys** (for cloud backends):
- OpenAI API key — required for `--model openai` and OpenAI judge models (e.g. `gpt-4o`)
- Google Gemini API key — required for Gemini judge models (default)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd NetBench-RAG

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CUDA 12.8 build of PyTorch)
pip install -r requirements.txt
```

---

## Setup

### 1. API Keys

```bash
# OpenAI key (for GPT-4o generation and optional OpenAI judge)
echo "sk-..." > openai_api_key.txt

# Gemini key (for LLM-as-Judge evaluation, default)
echo "AIza..." > gemini_api_key.txt
```

### 2. Corpus Data

Place the research corpus at:
```
data/raw/research_corpus_v3.json
```

Or place PDF files in:
```
pdfs/
```

### 3. Local Model (optional)

For the local Llama backend, set the model path in `config.yaml`:
```yaml
paths:
  local_model: ../LLM-Training/models/instruction/Llama-3.1-8B-trained-new-instruct
```

### 4. Index the Corpus

```bash
# Index from JSON corpus (default, ~30 min on GPU)
python index_corpus.py

# Index from PDFs
python index_corpus.py --source pdf

# Index from both sources
python index_corpus.py --source both

# Force rebuild (wipe and reindex)
python index_corpus.py --force
```

This creates:
- `data/qdrant_store/` — Qdrant vector index
- `data/bm25_index.pkl` — BM25 keyword index
- `data/chunks_cache.pkl` — Chunk metadata cache

---

## Usage

### Interactive Query REPL

```bash
python query.py
```

Example session:
```
> How does BBR congestion control differ from CUBIC?

[Retrieving...]  3.2s
[Generating...]  1.1s

BBR (Bottleneck Bandwidth and Round-Trip Propagation) differs from
CUBIC in its fundamental model...

Sources:
  [1] BBR: Congestion-Based Congestion Control | Section III
  [2] Comparing Transport Protocols | Section IV.B
```

### Single-Shot Query

```bash
python query.py --question "What is the impact of BDP on TCP throughput?"
```

### Override Backend

```bash
python query.py --backend openai          # Use GPT-4o
python query.py --backend local           # Use local Llama 3.1 8B
```

### Override Top-K Chunks

```bash
python query.py --top-k 3    # Use only top-3 chunks instead of default 5
```

---

## Evaluation Pipeline

The evaluation pipeline consists of three phases. See [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) for full details.

### Phase 1 — Answer Generation

```bash
# Generate answers with GPT-4o
python evaluation/evaluate_rag.py --model openai

# Generate answers with local Llama
python evaluation/evaluate_rag.py --model local

# Specific local model by path
python evaluation/evaluate_rag.py --model /path/to/LLM-Training/models/ModelA

# Multiple models in one run
python evaluation/evaluate_rag.py --model openai local

# Override top-k chunks
python evaluation/evaluate_rag.py --model openai --top_k 3
```

Output: `outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx`

### Phase 2 — LLM-as-Judge Scoring

```bash
# Judge with Gemini (default)
python evaluation/judge_responses.py --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx

# Judge with OpenAI GPT-4o
python evaluation/judge_responses.py \
    --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx \
    --judge_model gpt-4o \
    --openai_api_key_file openai_api_key.txt

# Resume interrupted run (auto-detects existing output)
python evaluation/judge_responses.py --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx
```

Output: `outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx`

### Phase 3 — Comparative Report

```bash
# Compare two RAG backends
python evaluation/benchmark_report.py --judged_files \
    outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx \
    outputs/judged/hpn_judged_RAG-Llama-3.1-8B_v5_general_skills_by_gemini-2.5-flash.xlsx

# Cross-project comparison (RAG vs LLM-Training baseline)
python evaluation/benchmark_report.py --judged_files \
    outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx \
    ../LLM-Training/outputs/judged/hpn_judged_Llama-3.1-8B_v5_general_skills_by_gemini-2.5-flash.xlsx
```

Output:
- `outputs/reports/hpn_benchmark_report_v5_general_skills_by_gemini-2.5-flash.xlsx`
- `outputs/reports/HPN_BENCHMARK_REPORT_v5_general_skills_by_gemini-2.5-flash.md`

### Scoring Rubric

| Dimension    | Weight | Description                              |
|-------------|--------|------------------------------------------|
| Correctness  | 40%    | Technical accuracy vs. reference answer  |
| Completeness | 30%    | Coverage of key concepts                 |
| Clarity      | 20%    | Organization and readability             |
| Conciseness  | 10%    | Appropriate focus, no unnecessary detail |

Scores: 1–5 per dimension. Overall = weighted average.

---

## Configuration

All parameters are in `config.yaml`. Key settings:

```yaml
# Switch generation backend
generation:
  backend: openai    # openai | local

# Adjust retrieval depth
retrieval:
  vector_top_k: 30       # candidates from vector search
  bm25_top_k: 30         # candidates from BM25
  rrf_top_n: 20          # candidates after fusion
  reranker_top_n: 5      # final chunks to LLM

# Evaluation judge
judge:
  model: gemini-2.5-flash-preview-04-17
  rate_limit_rpm: 15     # Gemini free tier: 15 RPM
```

See `config.yaml` for the full reference with all tunable parameters.

---

## Project Structure

```
NetBench-RAG/
├── config.yaml                  # All tunable parameters
├── requirements.txt             # Python dependencies
├── index_corpus.py              # Offline corpus indexing
├── query.py                     # Interactive query REPL
│
├── src/                         # Core RAG library
│   ├── parser.py                # JSON & PDF document parser
│   ├── chunker.py               # Token-aware chunker with overlap
│   ├── embedder.py              # BAAI/bge embedding wrapper
│   ├── store.py                 # Qdrant + BM25 dual-index store
│   ├── retriever.py             # 3-stage retrieval pipeline
│   └── generator.py             # OpenAI + local LLM generation
│
├── evaluation/                  # Benchmark evaluation suite
│   ├── evaluate_rag.py          # Phase 1: answer generation
│   ├── judge_responses.py       # Phase 2: LLM-as-Judge scoring
│   └── benchmark_report.py      # Phase 3: comparative report
│
├── data/
│   ├── raw/                     # Source corpus JSON
│   └── prompts/                 # Benchmark QA datasets
│       ├── hpn_qa_benchmark_v5_general_skills.json   # Active (242 Q)
│       └── hpn_qa_benchmark.json                     # Legacy
│
├── outputs/
│   ├── answers/                 # Phase 1 Excel files
│   ├── judged/                  # Phase 2 Excel files
│   └── reports/                 # Phase 3 reports
│
└── docs/
    ├── QUICKSTART.md            # 5-minute setup guide
    ├── CHAT_GUIDE.md            # Interactive query guide
    └── EVALUATION_GUIDE.md      # Full evaluation pipeline guide
```

---

## Benchmark Dataset

`data/prompts/hpn_qa_benchmark_v5_general_skills.json`

| Property       | Value                                                              |
|---------------|---------------------------------------------------------------------|
| Questions      | 242                                                                |
| Categories     | 10 (congestion control, RDMA, TCP, MPI, SDN, ...)                 |
| Difficulty     | easy / medium / hard                                               |
| Question types | concept, reasoning, comparison, diagnosis, calculation, scenario   |
| Fields         | id, category, difficulty, question_type, question, reference_answer, source_paper, keywords |

---

## Documentation

| File | Description |
|------|-------------|
| [QUICKSTART.md](docs/QUICKSTART.md) | End-to-end setup in 5 minutes |
| [CHAT_GUIDE.md](docs/CHAT_GUIDE.md) | Using the interactive query interface |
| [EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) | Full evaluation pipeline walkthrough |
| [CLAUDE.md](CLAUDE.md) | Claude Code context for AI-assisted development |
