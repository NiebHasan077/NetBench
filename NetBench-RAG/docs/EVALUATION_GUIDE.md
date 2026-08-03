# Evaluation Guide — NetBench-RAG

This guide walks through the full three-phase evaluation pipeline for benchmarking RAG system quality.

---

## Overview

The evaluation pipeline mirrors the [LLM-Training](../../LLM-Training/) project's pipeline so results are **directly comparable**. It consists of three sequential phases:

```
Phase 1: evaluate_rag.py       → outputs/answers/hpn_answers_{model}_{benchmark_tag}.xlsx
Phase 2: judge_responses.py    → outputs/judged/hpn_judged_{model}_{benchmark_tag}_by_{judge}.xlsx
Phase 3: benchmark_report.py   → outputs/reports/hpn_benchmark_report_{benchmark_tag}_by_{judge}.xlsx
                                  outputs/reports/HPN_BENCHMARK_REPORT_{benchmark_tag}_by_{judge}.md
```

---

## Benchmark Dataset

**File:** `data/prompts/hpn_qa_benchmark_v5_general_skills.json`

| Property | Value |
|----------|-------|
| Total questions | 242 |
| Categories | 10 HPN topic areas |
| Difficulty | easy / medium / hard |
| Question types | concept, reasoning, comparison, diagnosis, calculation, scenario |

Each question has: `id, category, difficulty, question_type, question, reference_answer, source_paper, keywords`

---

## Phase 1 — Answer Generation (`evaluate_rag.py`)

Runs all benchmark questions through the full RAG pipeline and records answers.

### Run

```bash
source venv/bin/activate

# OpenAI backend (GPT-4o)
python evaluation/evaluate_rag.py --model openai

# Local model (path from config.yaml)
python evaluation/evaluate_rag.py --model local

# Specific local model by path
python evaluation/evaluate_rag.py --model /path/to/LLM-Training/models/ModelA

# Multiple models in one command (names or paths)
python evaluation/evaluate_rag.py --model openai local
python evaluation/evaluate_rag.py --model /path/to/ModelA /path/to/ModelB

# Override number of retrieved chunks
python evaluation/evaluate_rag.py --model openai --top_k 3

# Custom benchmark file or output directory
python evaluation/evaluate_rag.py \
    --model openai \
    --benchmark data/prompts/hpn_qa_benchmark_v5_general_skills.json \
    --output_dir outputs/answers/
```

### Output

`outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx`

The output filename includes a benchmark tag derived from the benchmark filename (`v5_general_skills` for the default benchmark). This keeps outputs from different benchmark versions distinguishable.

**Metadata sheet** — pipeline configuration snapshot:
- backend, model name, timestamp
- retrieval config (top-k, reranker model)
- generation config (temperature, max tokens)
- total questions, mean retrieval/generation times

**Answers sheet** — one row per question with 14 columns:

| Column | Description |
|--------|-------------|
| id | Benchmark question ID |
| category | HPN topic category |
| difficulty | easy / medium / hard |
| question_type | concept / reasoning / comparison / ... |
| question | The question text |
| reference_answer | Ground-truth reference answer |
| model_answer | RAG system's generated answer |
| source_paper | Reference source paper |
| keywords | Domain keywords |
| retrieval_time_sec | Time for 3-stage retrieval |
| generation_time_sec | Time for LLM generation |
| prompt_tokens | Token count sent to LLM |
| retrieved_chunks | Pipe-separated list of "title §section" |
| reranker_scores | Cross-encoder scores for each chunk |

### Notes

- The script loads the Qdrant index and BM25 index on startup. Ensure `python index_corpus.py` has been run first.
- For `--model local` or a path, the model is loaded into GPU memory on first question (~60s).
- When passing a filesystem path to `--model`, config.yaml's `paths.local_model` is overridden for that run only.
- **Checkpoint/resume**: Progress is saved to a `.checkpoint.json` file after every question. If the run is interrupted, simply rerun the same command to resume from where it left off. The checkpoint is deleted automatically after the Excel file is written successfully.

---

## Phase 2 — LLM-as-Judge Scoring (`judge_responses.py`)

Scores each answer against the reference answer using an LLM judge (Gemini or OpenAI).

### Run

```bash
# Judge with Gemini (default)
python evaluation/judge_responses.py --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx

# Judge multiple answer files at once
python evaluation/judge_responses.py \
    --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx \
                   outputs/answers/hpn_answers_RAG-Llama-3.2-1B-trained-new-instruct_v5_general_skills.xlsx

# Judge with OpenAI GPT-4o
python evaluation/judge_responses.py \
    --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx \
    --judge_model gpt-4o \
    --openai_api_key_file openai_api_key.txt

# Override Gemini judge model
python evaluation/judge_responses.py \
    --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx \
    --judge_model gemini-2.5-pro

# Override rate limit
python evaluation/judge_responses.py \
    --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx \
    --requests_per_minute 10
```

### API Key Resolution

The script auto-detects the provider from the model name (gpt-*/o1-*/o3-*/o4-* → OpenAI, otherwise → Gemini).

**Gemini** key lookup order:
1. `--gemini_api_key_file` CLI argument
2. `paths.gemini_api_key` in `config.yaml` (default: `gemini_api_key.txt`)
3. `GEMINI_API_KEY` environment variable

**OpenAI** key lookup order:
1. `--openai_api_key_file` CLI argument
2. `paths.openai_api_key` in `config.yaml` (default: `openai_api_key.txt`)
3. `OPENAI_API_KEY` environment variable

### Scoring Rubric

The judge evaluates each answer on four dimensions (1–5 scale):

| Dimension | Weight | Criteria |
|-----------|--------|----------|
| **Correctness** | 40% | Technical accuracy compared to reference answer |
| **Completeness** | 30% | Coverage of key concepts from reference answer |
| **Clarity** | 20% | Organization, logical flow, and readability |
| **Conciseness** | 10% | Appropriate focus without unnecessary padding |

**Overall score** = weighted average = `0.4×C1 + 0.3×C2 + 0.2×C3 + 0.1×C4`

### Rate Limiting and Retries

- Default: 15 requests/minute (Gemini free tier)
- Exponential backoff on `429 / ResourceExhausted` errors: waits `2^(attempt+1)` seconds, capped at 60s
- Maximum 5 retry attempts per question
- Failed questions are marked with `overall = 0.0` and a note in `justification`

### Resume Capability

If the output file already exists, the script reads it and **skips already-judged questions**. This means you can safely interrupt and re-run without re-scoring questions or wasting API calls.

### Output

`outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx`

The output filename includes the benchmark tag and the judge model name.

**Judged sheet** — one row per question with 15 columns:

| Column | Description |
|--------|-------------|
| id | Benchmark question ID |
| category | HPN topic category |
| difficulty | easy / medium / hard |
| question_type | concept / reasoning / ... |
| question | Question text |
| reference_answer | Ground-truth answer |
| model_answer | RAG system's answer |
| correctness | Score 1–5 |
| completeness | Score 1–5 |
| clarity | Score 1–5 |
| conciseness | Score 1–5 |
| overall | Weighted average |
| justification | Judge's reasoning |
| source_paper | Reference source paper |
| keywords | Domain keywords |

**Metadata sheet** — configuration snapshot including judge model, weights, timestamp.

Color coding in Excel:
- Green: overall ≥ 4.0
- Yellow: 3.0 ≤ overall < 4.0
- Red: overall < 3.0

**Summary sheet** — aggregated statistics:
- Overall averages per dimension + by category + by difficulty
- Min/max/std per group

---

## Phase 3 — Comparative Report (`benchmark_report.py`)

Builds a multi-sheet Excel workbook and Markdown report comparing two or more judged files.

### Run

```bash
# Compare two RAG models
python evaluation/benchmark_report.py --judged_files \
    outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx \
    outputs/judged/hpn_judged_RAG-Llama-3.1-8B-trained-new-instruct_v5_general_skills_by_gemini-2.5-flash.xlsx

# Cross-project comparison: RAG vs LLM-Training baseline
python evaluation/benchmark_report.py --judged_files \
    outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx \
    ../LLM-Training/outputs/judged/hpn_judged_Llama-3.1-8B_v5_general_skills_by_gemini-2.5-flash.xlsx

# Custom output directory
python evaluation/benchmark_report.py --judged_files outputs/judged/*.xlsx \
    --output_dir outputs/reports/
```

### Output Files

#### Excel Report (`hpn_benchmark_report_{benchmark_tag}_by_{judge}.xlsx`)

Five worksheets:

| Sheet | Contents |
|-------|----------|
| **Comparison** | Per-question side-by-side scores for all models, delta column, winner highlight |
| **Overall Summary** | Mean/min/max per scoring dimension per model |
| **By Category** | Average overall score per HPN category per model (pivot table) |
| **By Difficulty** | Average overall score per difficulty level per model |
| **Head-to-Head** | Win/tie/loss counts overall and per category (only when exactly 2 models) |

#### Markdown Report (`HPN_BENCHMARK_REPORT_{benchmark_tag}_by_{judge}.md`)

Sections:
- Executive summary (best model, average scores, category winner)
- Overall score table
- By-category breakdown table
- By-difficulty breakdown table
- Head-to-head analysis (2 models)
- Per-dimension analysis (correctness, completeness, clarity, conciseness)
- Scoring rubric reference

---

## Cross-Project Comparison

Both NetBench-RAG and LLM-Training use identical judge prompts, scoring rubric, and Excel schema. You can pass judged Excel files from either project to `benchmark_report.py`:

```bash
# Model names will appear as "RAG-gpt-4o" vs "Llama-3.1-8B" in the report
python evaluation/benchmark_report.py --judged_files \
    outputs/judged/hpn_judged_RAG-gpt-4o_v5_general_skills_by_gemini-2.5-flash.xlsx \
    ../LLM-Training/outputs/evaluations/judged/hpn_judged_Llama-3.1-8B_v5_general_skills_by_gemini-2.5-flash.xlsx
```

The report directly quantifies how much retrieval augmentation improves over standalone LLM generation.

---

## Full Pipeline Example

```bash
source venv/bin/activate

# Phase 1: Generate answers (multiple models)
python evaluation/evaluate_rag.py --model openai local

# Phase 2: Judge both answer files
python evaluation/judge_responses.py --answer_files outputs/answers/hpn_answers_RAG-gpt-4o_v5_general_skills.xlsx
python evaluation/judge_responses.py --answer_files outputs/answers/hpn_answers_RAG-Llama-3.1-8B-trained-new-instruct_v5_general_skills.xlsx

# Phase 3: Compare
python evaluation/benchmark_report.py --judged_files outputs/judged/*.xlsx
```

Reports are written to `outputs/reports/`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `google.api_core.exceptions.ResourceExhausted` | Rate limit hit; script retries automatically. Reduce `--rpm` if persistent. |
| `FileNotFoundError: gemini_api_key.txt` | Create the file: `echo "AIzaSy..." > gemini_api_key.txt` |
| `KeyError: 'model_answer'` in judge | Ensure the answer file came from Phase 1 with the correct schema |
| Empty "Head-to-Head" sheet | This sheet only appears when exactly 2 judged files are compared |
| `qdrant_client.http.exceptions.UnexpectedResponse` | Index may be corrupted; run `python index_corpus.py --force` |
| Phase 1 slow on `--model local` or a path | Normal — Llama 3.1 8B loads ~30s, then ~5-15s per question |

---

## Configuration Reference

Evaluation-specific settings in `config.yaml`:

```yaml
paths:
  benchmark: data/prompts/hpn_qa_benchmark_v5_general_skills.json
  answers_dir: outputs/answers/
  judged_dir: outputs/judged/
  reports_dir: outputs/reports/
  gemini_api_key: gemini_api_key.txt

judge:
  model: gemini-2.5-flash-preview-04-17
  rate_limit_rpm: 15          # Free tier limit
  retry_attempts: 3
  retry_backoff_base: 2.0     # Exponential backoff base (seconds)

  weights:
    correctness: 0.40
    completeness: 0.30
    clarity: 0.20
    conciseness: 0.10
```
