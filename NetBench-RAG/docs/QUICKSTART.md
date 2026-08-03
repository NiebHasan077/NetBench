# Quickstart — NetBench-RAG

Get up and running in ~5 minutes (plus ~30 min for corpus indexing).

---

## Prerequisites

- Python 3.11+
- CUDA 12.x GPU (for embedding, reranking, and local LLM)
- OpenAI API key (for GPT-4o backend)
- Google Gemini API key (for evaluation judge, default) or OpenAI API key (for GPT-4o judge)

---

## Step 1 — Install

```bash
cd NetBench-RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 2 — Add API Keys

```bash
echo "sk-proj-..." > openai_api_key.txt       # OpenAI key
echo "AIzaSy..."   > gemini_api_key.txt        # Gemini key (for evaluation)
```

---

## Step 3 — Place Corpus Data

Copy the research corpus to:
```
data/raw/research_corpus_v3.json
```

Or place PDF papers in `pdfs/` if you want to index from PDFs.

---

## Step 4 — Index the Corpus

```bash
python index_corpus.py
```

This takes ~30 minutes on a GPU. It builds:
- `data/qdrant_store/` — dense vector index
- `data/bm25_index.pkl` — sparse BM25 index
- `data/chunks_cache.pkl` — chunk metadata cache

Run `python index_corpus.py --force` to wipe and rebuild from scratch.

---

## Step 5 — Ask a Question

```bash
python query.py
```

Type any HPN question at the prompt:
```
> How does BBR control congestion differently from CUBIC?
```

Press `Ctrl+C` or type `exit` / `quit` to stop.

For a single-shot query without the REPL:
```bash
python query.py --question "What is the role of RDMA in HPC clusters?"
```

---

## Step 6 — Run the Evaluation (Optional)

```bash
# Phase 1: Generate answers for all 242 benchmark questions
python evaluation/evaluate_rag.py --model openai

# Phase 2: Score answers with LLM judge (Gemini default, or use --judge_model gpt-4o for OpenAI)
python evaluation/judge_responses.py --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx

# Phase 3: Build comparative report
python evaluation/benchmark_report.py --judged_files outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx
```

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for full details.

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `source venv/bin/activate` first |
| `FileNotFoundError: openai_api_key.txt` | Create the key file (Step 2) |
| `FileNotFoundError: data/qdrant_store` | Run `python index_corpus.py` first (Step 4) |
| CUDA out of memory | Reduce `retrieval.reranker_top_n` or `retrieval.rrf_top_n` in `config.yaml` |
| Slow BM25 on first query | Expected — BM25 loads the full index into RAM once, subsequent queries are fast |

---

## Configuration Quick Reference

Edit `config.yaml` to change behavior without modifying code.

```yaml
# Switch to local Llama backend
generation:
  backend: local

# Use fewer retrieved chunks (faster, less context)
retrieval:
  reranker_top_n: 3

# Use a different GPT model
generation:
  openai:
    model: gpt-4o-mini
```

---

## Next Steps

- [CHAT_GUIDE.md](CHAT_GUIDE.md) — all query.py options and tips
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) — full evaluation pipeline walkthrough
- [../README.md](../README.md) — architecture overview and full project docs
