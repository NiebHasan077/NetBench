# NetBench-RAG — development guide

Architecture notes, commands, and configuration for working in this module.

## Project Overview

NetBench-RAG is a production-grade Retrieval-Augmented Generation (RAG) system designed for high-performance networking (HPN) research. It retrieves relevant passages from an HPN research paper corpus and uses an LLM (GPT-4o or a local fine-tuned Llama 3.1 8B) to answer questions grounded in those retrieved passages.

The system is evaluated against a 242-question benchmark (`hpn_qa_benchmark_v5_general_skills.json`) using a three-phase pipeline: answer generation → LLM-as-Judge scoring → comparative report. Results are designed to be **directly comparable** to the sibling `LLM-Training` project.

## Repository Layout

```
NetBench-RAG/
├── DEVELOPMENT.md              # ← you are here
├── config.yaml                 # Central config (all tunable parameters)
├── requirements.txt
├── index_corpus.py             # Offline corpus indexing (run once)
├── query.py                    # Interactive RAG query REPL
├── src/                        # Core RAG library
│   ├── parser.py               # JSON + PDF document parsing
│   ├── chunker.py              # Token-aware chunking with overlap
│   ├── embedder.py             # BAAI/bge-large-en-v1.5 wrapper
│   ├── store.py                # Qdrant vector store + BM25 index
│   ├── retriever.py            # 3-stage retrieval pipeline
│   └── generator.py            # OpenAI / local LLM generation
├── evaluation/                 # Benchmark evaluation suite (Phase 7)
│   ├── evaluate_rag.py         # Phase 1: answer generation
│   ├── judge_responses.py      # Phase 2: LLM-as-Judge scoring
│   └── benchmark_report.py    # Phase 3: comparative report
├── data/
│   ├── raw/                    # research_corpus_v3.json (gitignored)
│   ├── prompts/                # Benchmark QA JSON files
│   ├── qdrant_store/           # Qdrant index (gitignored)
│   ├── bm25_index.pkl          # BM25 index (gitignored)
│   └── chunks_cache.pkl        # Chunk cache (gitignored)
├── outputs/
│   ├── answers/                # Phase 1 Excel output
│   ├── judged/                 # Phase 2 Excel output
│   └── reports/                # Phase 3 Excel + Markdown output
└── docs/                       # Extended documentation
```

## Critical Architecture Notes

### Dual-Index Invariant
**Must be maintained**: `Qdrant point ID == chunks_cache[i] index == BM25 corpus position`
This invariant is established by `index_corpus.py` and relied on by `retriever.py`. Breaking it causes silent retrieval failures.

### Three-Stage Retrieval
1. **Dual Search** — vector (Qdrant, top-30) + keyword (BM25, top-30)
2. **RRF Fusion** — Reciprocal Rank Fusion merges rankings (k=60, top-20)
3. **Cross-Encoder Reranking** — BAAI/bge-reranker-v2-m3 → top-5 final chunks

### Two Generation Backends
- `openai` — GPT-4o via API (requires `openai_api_key.txt`)
- `local` — Fine-tuned Llama 3.1 8B (requires GPU, path in config)

### Evaluation Cross-Comparability
The evaluation scripts use **identical** judge prompts, scoring rubric, Excel schema, and file naming as the `LLM-Training` project. The `benchmark_report.py` can directly ingest judged Excel files from either project for head-to-head comparison.

## Common Commands

```bash
# Activate virtualenv
source venv/bin/activate

# Index corpus (run once, ~30 min on GPU)
python index_corpus.py

# Interactive query
python query.py
python query.py --backend openai --question "How does BBR work?"

# Evaluation pipeline
python evaluation/evaluate_rag.py --model openai
python evaluation/judge_responses.py --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx
python evaluation/benchmark_report.py --judged_files outputs/judged/*.xlsx
```

## Key Configuration (config.yaml)

All parameters are in `config.yaml`. Key sections:
- `paths` — data/output directories, API key file paths
- `generation.backend` — `openai` or `local`
- `retrieval.reranker_top_n` — final chunks passed to LLM (default 5)
- `judge.model` — Judge model for evaluation (default `gemini-2.5-flash`; supports Gemini and OpenAI models)
- `judge.rate_limit_rpm` — API rate limit (default 15 for free tier)

## API Keys

- `openai_api_key.txt` — OpenAI key (gitignored; required for `--backend openai` and OpenAI judge models)
- `gemini_api_key.txt` — Gemini key (gitignored; required for Gemini judge models)

Never commit these files.

## Python Environment

```bash
source venv/bin/activate   # always activate before running any script
python --version           # should be 3.11+
```

## Benchmark Dataset

`data/prompts/hpn_qa_benchmark_v5_general_skills.json`
- 242 questions, 10 categories, easy/medium/hard split
- Fields: `id, category, difficulty, question_type, question, reference_answer, source_paper, keywords`
- `question_type`: concept, reasoning, comparison, diagnosis, calculation, scenario

## Git Notes

Gitignored: `pdfs/`, `venv/`, `data/qdrant_store/`, `data/bm25_index.pkl`, `data/chunks_cache.pkl`, `outputs/answers/`, `outputs/judged/`, `openai_api_key.txt`, `gemini_api_key.txt`, `data/processed/`

`outputs/reports/` is tracked in git (contains benchmark comparison reports).

Large binary artifacts (model weights, Qdrant store) should never be committed.
