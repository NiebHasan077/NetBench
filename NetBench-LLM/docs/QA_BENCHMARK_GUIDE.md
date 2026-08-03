# HPN Q&A Benchmark — Full Guide

Evaluate instruction-tuned models on **90 domain-specific questions** covering
10 HPN (High-Performance Networking) skill categories.  An **LLM-as-Judge**
(Gemini or OpenAI) scores every answer on four dimensions, and a final
comparative report ranks the models.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1 — Answer Generation](#phase-1--answer-generation)
4. [Phase 2 — LLM-as-Judge Scoring](#phase-2--llm-as-judge-scoring)
5. [Phase 3 — Comparative Report](#phase-3--comparative-report)
6. [Filtering Judged Results](#filtering-judged-results)
7. [Output File Reference](#output-file-reference)
8. [Scoring Rubric](#scoring-rubric)
9. [Troubleshooting](#troubleshooting)

---

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  Phase 1        │     │  Phase 2        │     │  Phase 3            │
│  hpn_qa_        │ ──▶ │  judge_         │ ──▶ │  benchmark_         │
│  benchmark.py   │     │  responses.py   │     │  report.py          │
│                 │     │                 │     │                     │
│  Generate       │     │  LLM-as-Judge   │     │  Comparative Excel  │
│  answers (.xlsx)│     │  scores (.xlsx) │     │  + Markdown report  │
└─────────────────┘     └─────────────────┘     └─────────────────────┘
```

| Aspect | Details |
|--------|---------|
| Benchmark file | `data/prompts/hpn_qa_benchmark_v4_general_skills.json` (default) |
| Questions | 90 (10 categories × 9 questions) |
| Difficulty levels | Easy, Medium, Hard |
| Question types | Concept, Reasoning, Calculation, Comparison, Diagnosis, Scenario, Task |
| Scoring dimensions | Correctness, Completeness, Clarity, Conciseness (1–5) |
| Judge model | Gemini (default: `gemini-2.5-flash`) or OpenAI (`gpt-4o`, `gpt-4.1`, etc.) |

> **Benchmark versioning**: Output filenames automatically include a benchmark
> tag derived from the JSON filename (e.g. `_v4_general_skills`), so results
> from different benchmark versions are kept separate.

### Categories (v4)

1. Transfer Parameters: Definitions and Roles
2. Concurrency Tuning and Scaling
3. Pipelining and Small-File Optimization
4. Parallelism and Large-File Optimization
5. Dataset Partitioning and Mixed Workloads
6. BDP-Based Reasoning and Window Sizing
7. Bottleneck Diagnosis and End-to-End Reasoning
8. Adaptive and Online Optimization
9. Fairness, Stability, and Shared Networks
10. Practical HPN Scenarios and Design

---

## Prerequisites

| Requirement | How to satisfy |
|-------------|---------------|
| Python 3.10+ | System Python or conda |
| GPU (≥ 24 GB VRAM) | RTX A6000 / A100 / etc. — required for local models only |
| Instruction-tuned models | Run instruction fine-tuning first (see [INSTRUCTION_FINETUNING_GUIDE.md](INSTRUCTION_FINETUNING_GUIDE.md)) — required for local models only |
| Gemini API key | Get from [Google AI Studio](https://aistudio.google.com/apikey), save to `gemini_api_key.txt` in project root — required for Gemini answer generation **and** Gemini judging |
| OpenAI API key | Get from [OpenAI Platform](https://platform.openai.com/api-keys), save to `openai_api_key.txt` in project root — required for GPT-4o answer generation **and** OpenAI judging |
| Python packages | `pip install google-genai openai openpyxl` (or `pip install -r requirements.txt`) |

> **Security**: `gemini_api_key.txt` and `openai_api_key.txt` are listed in `.gitignore` — never commit them.

---

## Phase 1 — Answer Generation

**Script**: `evaluation/hpn_qa_benchmark.py`

Supports two backends that can be mixed freely in a single run:

| Backend | When to use | Model name format |
|---------|-------------|-------------------|
| **Local HuggingFace** | Fine-tuned Llama, Qwen, or Gemma models on-device | Path to model directory (e.g. `models/instruction/Llama-3.1-8B-base-instruct`, `models/instruction/Qwen3.5-2B-trained-new-instruct`, or `models/instruction/gemma-3-4b-instruct`) |
| **OpenAI API** | GPT-4o and other OpenAI models | Model name string (e.g. `gpt-4o`, `gpt-4.1`) |
| **Gemini API** | Gemini 2.5-Pro and other Gemini models | Model name string (e.g. `gemini-2.5-pro`, `gemini-2.5-flash`) |

Provider is auto-detected from the model name: `gpt-`, `o1-`, `o3-`, `o4-` → OpenAI;
`gemini-` → Gemini; everything else → local HuggingFace.

All backends use the same HPN-specialist system prompt and produce the same Excel
schema so downstream Phase 2 and Phase 3 scripts work without modification.

**Resume support**: if an output file already exists, already-answered questions
are skipped and only missing or empty/error answers are regenerated. Safe to
re-run after a failure without duplicating work.

### Usage

```bash
# Local instruction-tuned model
python evaluation/hpn_qa_benchmark.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct

# Multiple local models (loaded, benchmarked, and unloaded sequentially)
python evaluation/hpn_qa_benchmark.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
                 models/instruction/Llama-3.1-8B-trained-new-instruct

# GPT-4o via OpenAI API
python evaluation/hpn_qa_benchmark.py \
    --model_path gpt-4o \
    --openai_api_key_file openai_api_key.txt

# Gemini 2.5-Pro via Google API
python evaluation/hpn_qa_benchmark.py \
    --model_path gemini-2.5-pro \
    --gemini_api_key_file gemini_api_key.txt

# Mix local + API in one run
python evaluation/hpn_qa_benchmark.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct gpt-4o \
    --openai_api_key_file openai_api_key.txt

# Custom benchmark file or generation settings
python evaluation/hpn_qa_benchmark.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
    --benchmark data/prompts/hpn_qa_benchmark.json \
    --max_new_tokens 512 --temperature 0.0
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | *required* | One or more model paths (local) or API model names |
| `--benchmark` | `data/prompts/hpn_qa_benchmark_v4_general_skills.json` | Benchmark JSON file |
| `--output_dir` | `outputs/evaluations/answers` | Output directory |
| `--max_new_tokens` | `512` | Max tokens per answer (local models and OpenAI) |
| `--temperature` | `0.0` | Sampling temperature (0.0 = deterministic for local/OpenAI) |
| `--top_p` | `0.9` | Nucleus sampling top-p (local models only) |
| `--gemini_api_key_file` | `gemini_api_key.txt` | Path to Gemini API key file |
| `--openai_api_key_file` | `openai_api_key.txt` | Path to OpenAI API key file |
| `--requests_per_minute` | `15` | Rate limit for API calls (Gemini and OpenAI) |

### What happens — local models

1. Reads the benchmark JSON
2. Checks for an existing output file — skips already-answered questions (resume)
3. Loads model + tokenizer with `device_map="auto"`
4. **Auto-detects the model family** (Llama, Qwen, or Gemma) from the model path and 
   selects the correct prompt template (Open-Orca, ChatML, or Gemma turns) and
   stop strings automatically — no user action required
5. Generates answers (greedy by default, max 512 tokens, stop strings + repetition penalty enabled)
6. Merges new answers with any existing ones and saves Excel
7. Unloads model to free VRAM before the next model

### What happens — API models

1. Reads the benchmark JSON
2. Checks for an existing output file — skips already-answered questions (resume)
3. Initialises the appropriate API client (OpenAI or Gemini)
4. Calls the API for each remaining question with rate limiting (`--requests_per_minute`)
   and exponential backoff on 429 / 503 errors
5. Merges and saves Excel

> **Gemini thinking models** (e.g. `gemini-2.5-pro`): internal reasoning tokens
> count against `max_output_tokens`. The script automatically adds an 8192-token
> thinking buffer on top of `--max_new_tokens` so reasoning does not crowd out
> the actual response. `temperature=0.0` is also omitted for Gemini thinking
> models as it causes silent empty output; the API default is used instead.

> **Gemini temperature note**: `temperature=0.0` is passed to all non-thinking
> Gemini models (e.g. `gemini-2.5-flash`) for deterministic output. Only
> thinking models (identified by their model name) have `temperature` omitted.
> For maximum reproducibility when comparing runs, always use the same Gemini
> model and avoid mixing thinking vs. non-thinking models in the same benchmark.

### Cross-judge workflow

To compare local models against SOTA models fairly, use a **cross-judge** setup:

- All local models + GPT-4o answers → judged by **Gemini** → Report A
- All local models + Gemini answers → judged by **GPT-4o** → Report B

Scores within each report are directly comparable because the judge is the same.
Do **not** include a model's own answers in the run where it is the judge
(self-judging bias).

### Output

`hpn_answers_<model_name>_<benchmark_tag>.xlsx` with two sheets:

| Sheet | Contents |
|-------|----------|
| **Metadata** | Model path, benchmark file, timestamp, question count, generation config |
| **Answers** | id, category, difficulty, question, reference_answer, model_answer, source_paper, keywords, generation_time_sec |

> The `<benchmark_tag>` is derived from the benchmark filename. For the default
> v4 benchmark, outputs are named e.g.
> `hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx` or
> `hpn_answers_gpt-4o_v4_general_skills.xlsx`.

---

## Phase 2 — LLM-as-Judge Scoring

**Script**: `evaluation/judge_responses.py`

Sends each `(question, reference_answer, model_answer)` triple to a judge LLM
and asks it to score on four dimensions plus provide a justification.

The judge supports two providers:
- **Gemini** (default): models like `gemini-2.5-flash`, `gemini-2.5-pro`
- **OpenAI**: models like `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`

Provider is automatically detected from the model name prefix (`gpt-`, `o1-`, `o3-`, `o4-` → OpenAI; everything else → Gemini).

The judge prompt uses a **justification-first** format (chain-of-thought before
scores), treats the reference answer as an exemplar rather than the only valid
answer, and includes full 5-level rubrics for each scoring dimension.

### Usage

```bash
# Judge one file with default Gemini judge
python evaluation/judge_responses.py \
    --answer_files outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx

# Judge multiple files in one run
python evaluation/judge_responses.py \
    --answer_files outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx \
                   outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-trained-new-instruct_v4_general_skills.xlsx

# Use a different Gemini model
python evaluation/judge_responses.py \
    --answer_files outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx \
    --judge_model gemini-2.5-pro

# Use OpenAI GPT-4o as judge
python evaluation/judge_responses.py \
    --answer_files outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx \
    --judge_model gpt-4o \
    --openai_api_key_file openai_api_key.txt

# Custom rate limit (default 15 requests per minute)
python evaluation/judge_responses.py \
    --answer_files outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx \
    --requests_per_minute 10
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--answer_files` | *required* | One or more Phase 1 answer Excel files |
| `--judge_model` | `gemini-2.5-flash` | Judge model name (Gemini or OpenAI) |
| `--gemini_api_key_file` | `gemini_api_key.txt` | Path to Gemini API key file |
| `--openai_api_key_file` | `openai_api_key.txt` | Path to OpenAI API key file |
| `--requests_per_minute` | `15` | Rate limit for API calls |
| `--output_dir` | `outputs/evaluations/judged` | Output directory |

### What happens

1. Detects provider (Gemini or OpenAI) from the judge model name
2. Reads the appropriate API key file
3. For each answer Excel file:
   - Reads the Answers sheet + model name from Metadata
   - Sends each row to the judge with a structured JSON prompt
   - Parses `correctness`, `completeness`, `clarity`, `conciseness` scores (1–5) + `justification`
   - Computes `overall = correctness×0.4 + completeness×0.3 + clarity×0.2 + conciseness×0.1`
   - Applies rate limiting (sleep between calls) + exponential backoff on errors
   - Saves `outputs/evaluations/judged/hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx`
4. Supports **resume**: if a partially-judged file exists, picks up where it left off

> **Note**: The judge model name and benchmark tag are embedded in the filename
> so you can easily compare scores from different judge LLMs and benchmark
> versions (e.g. `hpn_judged_..._v4_general_skills_by_gemini-2.5-flash.xlsx`
> vs `hpn_judged_..._v4_general_skills_by_gpt-4o.xlsx`).

### Output

`hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx` with three sheets:

| Sheet | Contents |
|-------|----------|
| **Metadata** | Model name, judge model, timestamp, API call count, RPM setting |
| **Judged** | All answer columns + correctness, completeness, clarity, conciseness, overall, justification (colour-coded) |
| **Summary** | Average scores by category and difficulty |

Colour coding in the **Judged** sheet:
- 🟢 Green (≥ 4.0) — Strong
- 🟡 Yellow (2.5–3.9) — Moderate
- 🔴 Red (< 2.5) — Weak

---

## Phase 3 — Comparative Report

**Script**: `evaluation/benchmark_report.py`

Reads two or more Phase 2 judged Excel files and produces a side-by-side
comparison.

### Usage

```bash
python evaluation/benchmark_report.py \
    --judged_files outputs/evaluations/judged/hpn_judged_Llama-3.1-8B-base-instruct_v4_general_skills_by_gemini-2.5-flash.xlsx \
                   outputs/evaluations/judged/hpn_judged_Llama-3.1-8B-trained-new-instruct_v4_general_skills_by_gemini-2.5-flash.xlsx
```

The judge model name and benchmark tag are automatically extracted from the
judged files' metadata and embedded in the report filenames.

### Output — Excel (`hpn_benchmark_report_<bmk_tag>_by_<judge>.xlsx`)

| Sheet | Contents |
|-------|----------|
| **Comparison** | Per-question scores for every model side-by-side (colour-coded) |
| **Overall Summary** | Aggregate mean scores per model (Correctness, Completeness, Clarity, Conciseness, Overall) |
| **By Category** | Mean overall score per category per model |
| **By Difficulty** | Mean overall score per difficulty per model |
| **Head-to-Head** | Per-question Δ (Model A − Model B), Win / Tie / Loss counts (only for 2 models) |

### Output — Markdown (`HPN_BENCHMARK_REPORT_<bmk_tag>_by_<judge>.md`)

A narrative report containing:
- Executive summary (best model, strongest/weakest categories)
- Overall scores table
- Category breakdown table + analysis
- Difficulty breakdown table + analysis
- Head-to-head summary (wins/ties/losses)
- Per-dimension analysis (Correctness, Completeness, Clarity, Conciseness)
- Scoring rubric appendix

---

## Filtering Judged Results

**Script**: `evaluation/filter_judged.py`

Reads all judged Excel files and produces filtered copies with two sheets:
**Good Answers** (overall > 4.0) and **Bad Answers** (overall < 1.3).
Useful for quickly identifying strong and weak model responses for analysis.

### Usage

```bash
# Default — scans all judged files
python evaluation/filter_judged.py

# Custom thresholds
python evaluation/filter_judged.py \
    --good_threshold 4.0 \
    --bad_threshold 1.3

# Custom directories
python evaluation/filter_judged.py \
    --judged_dir outputs/evaluations/judged \
    --output_dir outputs/evaluations/filtered
```

### Output

Creates filtered files in `outputs/evaluations/filtered/`, one per input
judged file:

```
outputs/evaluations/filtered/
└── filtered_hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx
    ├── Sheet: Good Answers  (overall > 4.0, green header)
    └── Sheet: Bad Answers   (overall < 1.3, red header)
```

---

## Output File Reference

```
outputs/evaluations/
├── answers/                                                ← Phase 1
│   └── hpn_answers_<model>_<bmk_tag>.xlsx
├── judged/                                                 ← Phase 2
│   └── hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx
├── filtered/                                               ← Filter tool
│   └── filtered_hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx
└── reports/                                                ← Phase 3
    ├── hpn_benchmark_report_<bmk_tag>_by_<judge>.xlsx
    └── HPN_BENCHMARK_REPORT_<bmk_tag>_by_<judge>.md
```

| File | Phase | Description |
|------|-------|-------------|
| `answers/hpn_answers_<model>_<bmk_tag>.xlsx` | 1 | Raw generated answers |
| `judged/hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx` | 2 | Scored answers with justifications |
| `filtered/filtered_hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx` | — | Good/bad filtered subsets |
| `reports/hpn_benchmark_report_<bmk_tag>_by_<judge>.xlsx` | 3 | Multi-model comparison workbook |
| `reports/HPN_BENCHMARK_REPORT_<bmk_tag>_by_<judge>.md` | 3 | Narrative Markdown report |

> **Why include the judge name and benchmark tag?** You may want to compare
> the same model answers judged by different LLMs (e.g. `gemini-2.5-flash` vs
> `gpt-4o`) or run different benchmark versions.  The filename convention keeps
> all variants side-by-side.

---

## Scoring Rubric

Each dimension uses a 1–5 integer scale. The **overall** score is a weighted
average: `correctness×0.4 + completeness×0.3 + clarity×0.2 + conciseness×0.1`.

### Correctness (weight: 40%)

| Score | Meaning |
|-------|---------|
| 5 | Fully correct, matches reference in all key technical claims |
| 4 | Mostly correct, minor inaccuracies |
| 3 | Partially correct, some errors or omissions |
| 2 | Mostly wrong, a few correct fragments |
| 1 | Fundamentally wrong or missing key facts |

### Completeness (weight: 30%)

| Score | Meaning |
|-------|---------|
| 5 | Covers >80% of key concepts from the reference |
| 4 | Covers 60–80% |
| 3 | Covers 40–60% |
| 2 | Covers 20–40% |
| 1 | Covers <20% of key concepts |

### Clarity (weight: 20%)

| Score | Meaning |
|-------|---------|
| 5 | Exceptionally clear and well-organized |
| 4 | Well-structured with minor clarity issues |
| 3 | Understandable but poorly structured |
| 2 | Mostly confusing with occasional clear points |
| 1 | Incoherent or very hard to follow |

### Conciseness (weight: 10%)

| Score | Meaning |
|-------|---------|
| 5 | Perfectly concise — every sentence adds value |
| 4 | Mostly focused, minimal padding |
| 3 | Acceptable length with some unnecessary content |
| 2 | Significant padding or irrelevant tangents |
| 1 | Extremely padded, repetitive, or off-topic |

---

## Troubleshooting

### `gemini_api_key.txt` not found
Create the file in the project root (same level as `requirements.txt`):
```bash
echo "YOUR_API_KEY_HERE" > gemini_api_key.txt
```

### `openai_api_key.txt` not found (when using OpenAI judge)
Create the file in the project root:
```bash
echo "$OPENAI_API_KEY" > openai_api_key.txt
```

### 429 Resource Exhausted / Rate Limit
The judge script has built-in rate limiting (default 15 RPM) and exponential
backoff.  If you still hit limits, lower the RPM:
```bash
python evaluation/judge_responses.py --answer_files ... --requests_per_minute 5
```

### Judge returns unparseable responses
The script tries `json.loads()` first, then falls back to regex extraction.
If both fail the question is scored `0.0` with the raw response stored in the
rationale column.  Rerunning with `--resume` will skip already-judged rows.

### CUDA Out of Memory during Phase 1
The models are loaded with `device_map="auto"` across 2 GPUs.  If you run out
of VRAM, ensure no other processes are using the GPUs:
```bash
nvidia-smi   # check usage
```

### Model continues generating after its answer (self-asks follow-ups)
This is prevented by the stop_strings mechanism added to Phase 1. If you
see this with older answer files, re-run Phase 1 to regenerate answers.

### Partially completed judge run
Phase 2 supports resume.  Simply rerun the same command — it reads the existing
judged file and picks up from the first un-judged row.
