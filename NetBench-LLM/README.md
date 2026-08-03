# LLM Continual Pre-training & Instruction Fine-tuning Pipeline

Pre-train **BASE** Llama, Qwen3.5, and Gemma 3/4 models on a custom research corpus and instruction-fine-tune them — while keeping the original models unchanged at every stage.

## 🎯 Key Features

- **Separate Models**: Original base models stay unchanged; trained / instruct models are written to dedicated directories
- **Token from File**: HuggingFace authentication via `hf_token.txt`
- **Multi-GPU (device_map="auto")**: Automatically distributes across 2× RTX A6000 (96 GB total)
- **Two Training Pipelines**:
  - *Continual Pre-training* – domain-adapt a base model on your JSON research corpus
  - *Instruction Fine-tuning* – teach a model to follow instructions using an Orca + Dolly dataset mix
- **Three-Phase HPN Q&A Benchmark**:
  - *Phase 1* – generate answers from local models **or API models** (GPT-4o, Gemini 2.5-Pro) with resume support
  - *Phase 2* – LLM-as-Judge scoring via Gemini or OpenAI with rate limiting and exponential backoff
  - *Phase 3* – comparative Excel + Markdown reports across any number of models
- **LM Adaptation Report**: measures CPT quality — HPN domain perplexity, general LM perplexity, and forgetting gap
- **Full 8-Variant Pipeline** (`scripts/run_pipeline.sh`): orchestrates all training + evaluation variants (S1–S8) for a single model with hardware auto-detection

## 📁 Directory Structure

```
LLM-Training/
├── hf_token.txt                    ← YOUR HUGGINGFACE TOKEN
│
├── models/                         ← ALL MODEL WEIGHTS
│   ├── base/                       ← Downloaded BASE models (unchanged)
│   │   ├── Llama-3.2-1B-base/
│   │   ├── Llama-3.1-8B-base/
│   │   ├── Qwen3.5-2B-Base/
│   │   ├── Qwen3.5-4B-Base/
│   │   ├── gemma-3-4b/
│   │   ├── gemma-4-e4b/            ← Gemma 4 E4B base (KV-shared)
│   │   └── gemma-4-e2b/            ← Gemma 4 E2B base (KV-shared)
│   ├── pretrained/                 ← Domain-adapted models
│   │   ├── Llama-3.2-1B-trained-new/
│   │   ├── Llama-3.1-8B-trained-new/
│   │   ├── Qwen3.5-2B-trained-new/
│   │   ├── Qwen3.5-4B-trained-new/
│   │   └── gemma-3-4b-trained-new/
│   ├── instruct-official/          ← Downloaded official instruct models (run_pipeline.sh)
│   ├── instruction/                ← Instruction-tuned models (our own SFT output)
│   │   ├── Llama-3.1-8B-base-instruct/
│   │   ├── Llama-3.1-8B-trained-new-instruct/
│   │   └── gemma-3-4b-instruct/
│   ├── lora/                       ← Raw LoRA adapters (PEFT, not standalone)
│   ├── lora-merged/                ← Adapters merged back into full weights
│   └── profiled/                   ← Models saved by profile_training.py
│
├── data/                           ← ALL TRAINING & EVALUATION DATA
│   ├── raw/                        ← Raw JSON corpora
│   │   ├── research_corpus_new.json  (753 docs — default corpus)
│   │   └── research_corpus_v3.json   (alternative corpus variant)
│   ├── processed/                  ← Tokenized pre-training data
│   │   ├── llama-3.2-1b/
│   │   ├── llama-3.1-8b/
│   │   ├── qwen3.5-2b/
│   │   ├── qwen3.5-4b/
│   │   ├── gemma-3-4b/
│   │   └── lora/                   ← LoRA-specific packed datasets
│   ├── instruction/                ← Tokenized instruction data
│   │   ├── llama-3.1-8b-base/
│   │   └── llama-3.1-8b-trained-new/
│   └── prompts/                    ← Evaluation prompts
│       ├── hpn_qa_benchmark.json           ← Original benchmark (9 categories)
│       ├── hpn_qa_benchmark_v4_general_skills.json  ← v4 benchmark (10 categories, default)
│       └── hpn_benchmark_v5.0_all.jsonl    ← v5 extended benchmark
│
├── outputs/                        ← ALL GENERATED OUTPUTS
│   ├── evaluations/                ← Evaluation results
│   │   ├── answers/                ← Phase 1 — model-generated answers
│   │   ├── judged/                 ← Phase 2 — LLM-as-Judge scored answers
│   │   ├── filtered/               ← Filtered good/bad answer subsets
│   │   ├── reports/                ← Phase 3 — comparative benchmark reports
│   │   └── adaptation_reports/     ← LM adaptation reports (CPT quality + forgetting)
│   ├── profiling_results/          ← JSON profiling data (auto-created)
│   └── profiling_reports/          ← Generated reports (auto-created)
│       ├── cumulative/             ← Comparison Excel + comparison plots
│       ├── per_run/<name>/         ← Per-run Excel + per-run plots
│       └── PROFILING_REPORT.md     ← Lightweight Markdown with image links
│
├── utils/                          ← Utility helpers
│   ├── setup_token.py              ← HF auth helper
│   ├── download_models.py          ← Download base models from HF
│   ├── model_utils.py              ← Model-family detection, stop tokens, LoRA targets
│   ├── check_system.py             ← Verify GPU / CUDA / disk
│   └── count_tokens.py             ← Token statistics
│
├── training/                       ← Training pipeline
│   ├── prepare_data.py             ← Tokenize corpus for pre-training
│   ├── pretrain_transformers.py    ← Continual pre-training
│   ├── prepare_instruction_data.py ← Download & tokenize Orca+Dolly
│   └── instruction_finetune.py     ← Instruction fine-tuning
│
├── evaluation/                     ← Model evaluation
│   ├── evaluate_model.py           ← Evaluate a single model
│   ├── compare_models.py           ← Compare base vs trained
│   ├── hpn_qa_benchmark.py         ← Phase 1: generate answers (local models + GPT-4o / Gemini API; resume support)
│   ├── judge_responses.py          ← Phase 2: LLM-as-Judge scoring via Gemini or OpenAI API
│   ├── benchmark_report.py         ← Phase 3: comparative report (Excel + Markdown)
│   ├── filter_judged.py            ← Filter judged results into good/bad subsets
│   └── lm_adaptation_report.py     ← LM adaptation: HPN perplexity, general perplexity, forgetting gap
│
├── inference/                      ← Interactive inference
│   ├── chat_interactive.py         ← Interactive BASE model completions
│   └── test_instruction_model.py   ← Test instruction-tuned models
│
├── profiling/                      ← Profiling pipeline (metrics & reports)
│   ├── profile_training.py         ← Profile pre-training or instruction FT
│   ├── profile_inference.py        ← Benchmark GPU / CPU inference speed
│   ├── generate_profile_report.py  ← Orchestrator: Markdown + Excel + plots
│   ├── report_plots.py             ← Matplotlib figure generation
│   └── report_excel.py             ← openpyxl Excel workbook generation
│
├── Shell Scripts
│   ├── chat.sh                     ← Launch base / trained model chat
│   └── test_instruct.sh            ← Launch instruction model testing
│
├── scripts/
│   ├── run_pipeline.sh             ← Full 8-variant evaluation pipeline (S1–S8)
│
├── docs/                           ← All documentation guides
│   ├── PRETRAINING_GUIDE.md        ← Continual pre-training documentation
│   ├── INSTRUCTION_FINETUNING_GUIDE.md ← Instruction fine-tuning documentation
│   ├── LORA_PIPELINE.md            ← LoRA adapter training documentation
│   ├── PROFILING_GUIDE.md          ← Full profiling documentation
│   ├── CHAT_GUIDE.md              ← Interactive testing documentation
│   ├── QA_BENCHMARK_GUIDE.md      ← HPN Q&A Benchmark pipeline guide
│   └── frontend/                   ← Frontend development documentation
│       ├── phases-a-b.md           ← Core infra + system tab
│       ├── phases-c-d.md           ← Chat + benchmark tabs
│       └── phases-e-f.md           ← Profiling tab + polish
│
├── requirements.txt
└── .gitignore
```

> **Note**: All training uses `device_map="auto"` (multi-GPU) because the
> system CUDA toolkit (11.8) differs from PyTorch CUDA (12.8), which
> prevents DeepSpeed from compiling CPU-offload kernels.

## � Documentation

All guides live in the [`docs/`](docs/) directory:

| Guide | Description |
|-------|-------------|
| [Continual Pre-training](docs/PRETRAINING_GUIDE.md) | Domain-adaptive pre-training on custom research corpus |
| [Instruction Fine-Tuning](docs/INSTRUCTION_FINETUNING_GUIDE.md) | Teaching models to follow instructions (Orca + Dolly) |
| [LoRA Adapter Training](docs/LORA_PIPELINE.md) | Parameter-efficient LoRA fine-tuning on HPN corpus |
| [Profiling Pipeline](docs/PROFILING_GUIDE.md) | GPU/memory/throughput profiling & report generation |
| [Interactive Testing](docs/CHAT_GUIDE.md) | Chat with base, trained, and instruction-tuned models |
| [HPN Q&A Benchmark](docs/QA_BENCHMARK_GUIDE.md) | Answer generation, LLM-as-Judge scoring (Gemini / OpenAI), comparative reports |

> **Full Pipeline Script**: `scripts/run_pipeline.sh` orchestrates all 8 training/evaluation variants (S1–S8) for a single model. Set the model variables at the top of the script for your hardware before running.

## �🚀 Quick Start

### 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Edit `hf_token.txt` and paste your HuggingFace token (get one at <https://huggingface.co/settings/tokens>).

### 2. System Check

```bash
python utils/check_system.py
```

### 3. Download Base Models

```bash
python utils/download_models.py --model 1b          # Llama-3.2-1B
python utils/download_models.py --model 8b          # Llama-3.1-8B
python utils/download_models.py --model qwen-2b     # Qwen3.5-2B-Base
python utils/download_models.py --model qwen-4b     # Qwen3.5-4B-Base
python utils/download_models.py --model gemma-4b    # Gemma-3-4B (Gemma 3)
python utils/download_models.py --model gemma-e4b   # Gemma-4-E4B (Gemma 4, KV-shared)
python utils/download_models.py --model gemma-e2b   # Gemma-4-E2B (Gemma 4, KV-shared)
python utils/download_models.py --model gemma-e4b-it # Gemma-4-E4B-IT (official instruct)
```

### 4. Continual Pre-training

```bash
# ── Llama shortcut ────────────────────────────────────
python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name meta-llama/Llama-3.1-8B \
    --output_dir data/processed/llama-3.1-8b

python training/pretrain_transformers.py \
    --model 8b \
    --data_dir data/processed/llama-3.1-8b \
    --output_dir models/pretrained/Llama-3.1-8B-trained-new

# ── Qwen shortcut ─────────────────────────────────────
python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name Qwen/Qwen3.5-2B-Base \
    --output_dir data/processed/qwen3.5-2b

python training/pretrain_transformers.py \
    --model qwen-2b \
    --data_dir data/processed/qwen3.5-2b \
    --output_dir models/pretrained/Qwen3.5-2B-trained-new

# ── Gemma shortcut ────────────────────────────────────
python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name google/gemma-3-4b-pt \
    --output_dir data/processed/gemma-3-4b

python training/pretrain_transformers.py \
    --model gemma-4b \
    --data_dir data/processed/gemma-3-4b \
    --output_dir models/pretrained/gemma-3-4b-trained-new

# ── Escape hatch (any model, any path) ────────────────
python training/pretrain_transformers.py \
    --model_path models/base/Qwen3.5-2B-Base \
    --data_dir   data/processed/qwen3.5-2b \
    --output_dir models/pretrained/Qwen3.5-2B-trained-new
```

### 5. Instruction Fine-tuning

```bash
# ── Llama ──────────────────────────────────────────────
python training/prepare_instruction_data.py \
    --model_path models/base/Llama-3.1-8B-base \
    --output_dir data/instruction/llama-3.1-8b-base \
    --max_samples 5000

python training/instruction_finetune.py \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/instruction/Llama-3.1-8B-base-instruct

# ── Qwen3.5 (ChatML template auto-selected) ───────────
python training/prepare_instruction_data.py \
    --model_path models/pretrained/Qwen3.5-2B-trained-new \
    --output_dir data/instruction/qwen3.5-2b \
    --max_samples 5000

python training/instruction_finetune.py \
    --model_path models/pretrained/Qwen3.5-2B-trained-new \
    --data_dir data/instruction/qwen3.5-2b \
    --output_dir models/instruction/Qwen3.5-2B-trained-new-instruct

# ── Gemma 3 (Gemma template auto-selected) ────────────
python training/prepare_instruction_data.py \
    --model_path models/pretrained/gemma-3-4b-trained-new \
    --output_dir data/instruction/gemma-3-4b \
    --max_samples 5000

python training/instruction_finetune.py \
    --model_path models/pretrained/gemma-3-4b-trained-new \
    --data_dir data/instruction/gemma-3-4b \
    --output_dir models/instruction/gemma-3-4b-instruct
```

### 6. Evaluate & Compare

```bash
# Single evaluation
python evaluation/evaluate_model.py -m models/pretrained/Llama-3.1-8B-trained-new

# Side-by-side base vs trained comparison
python evaluation/compare_models.py --model 8b
```

### 7. Interactive Testing

```bash
# BASE model completions (menu-driven)
./chat.sh

# Instruction model testing (menu-driven)
./test_instruct.sh

# Or directly:
python inference/chat_interactive.py --model_name models/base/Llama-3.1-8B-base
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct
```

### 8. Profiling

See [`docs/PROFILING_GUIDE.md`](docs/PROFILING_GUIDE.md) for full details.

```bash
# Profile a quick pre-training run (auto output_dir & run_name)
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.2-1B-base \
    --data_dir data/processed/llama-3.2-1b \
    --max_steps 50

# Auto-tokenize from raw JSON corpus
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.2-1B-base \
    --data_dir data/raw/research_corpus_new.json \
    --max_steps 50

# Profile instruction fine-tuning (auto-download Orca+Dolly)
python profiling/profile_training.py instruct \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir auto \
    --max_steps 50

# Benchmark inference on all discovered models (GPU)
# (includes models/profiled/ automatically)
python profiling/profile_inference.py --all

# Also benchmark on CPU
python profiling/profile_inference.py --model_path models/base/Llama-3.2-1B-base --cpu

# List available profiling data
python profiling/generate_profile_report.py --list

# Generate the unified Markdown report
python profiling/generate_profile_report.py
# → outputs/profiling_reports/PROFILING_REPORT.md  (Markdown + image links)
# → outputs/profiling_reports/cumulative/          (comparison Excel + plots)
# → outputs/profiling_reports/per_run/<name>/      (per-run Excel + plots)

# Generate report filtered to a specific model
python profiling/generate_profile_report.py --filter 8B

# Only cumulative comparison (skip per-run)
python profiling/generate_profile_report.py --cumulative-only
```

### 9. HPN Q&A Benchmark

See [`docs/QA_BENCHMARK_GUIDE.md`](docs/QA_BENCHMARK_GUIDE.md) for full details.

The HPN Q&A Benchmark evaluates models on 90 domain-specific questions
(10 categories × 3 difficulty levels). Phase 1 supports **local HuggingFace
models and API models** (GPT-4o, Gemini 2.5-Pro) with resume on failure.
An LLM-as-Judge (Gemini or OpenAI) scores each answer on **Correctness**,
**Completeness**, **Clarity**, and **Conciseness** (1–5 scale).

```bash
# ── Phase 1 — Generate answers ──────────────────────────────────────

# Local instruction-tuned model (Llama)
python evaluation/hpn_qa_benchmark.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct

# Local instruction-tuned model (Qwen — template auto-detected)
python evaluation/hpn_qa_benchmark.py \
    --model_path models/instruction/Qwen3.5-2B-trained-new-instruct

# GPT-4o via OpenAI API
python evaluation/hpn_qa_benchmark.py \
    --model_path gpt-4o \
    --openai_api_key_file openai_api_key.txt

# Gemini 2.5-Pro via Google API
python evaluation/hpn_qa_benchmark.py \
    --model_path gemini-2.5-pro \
    --gemini_api_key_file gemini_api_key.txt

# ── Phase 2 — Judge responses ────────────────────────────────────────

# Judge local models + GPT-4o answers with Gemini (cross-judge Report A)
python evaluation/judge_responses.py \
    --answer_files \
        outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx \
        outputs/evaluations/answers/hpn_answers_gpt-4o_v4_general_skills.xlsx \
    --judge_model gemini-2.5-pro \
    --gemini_api_key_file gemini_api_key.txt

# Judge local models + Gemini answers with GPT-4o (cross-judge Report B)
python evaluation/judge_responses.py \
    --answer_files \
        outputs/evaluations/answers/hpn_answers_Llama-3.1-8B-base-instruct_v4_general_skills.xlsx \
        outputs/evaluations/answers/hpn_answers_gemini-2.5-pro_v4_general_skills.xlsx \
    --judge_model gpt-4o \
    --openai_api_key_file openai_api_key.txt

# ── Phase 3 — Comparative report ────────────────────────────────────

python evaluation/benchmark_report.py \
    --judged_files \
        outputs/evaluations/judged/hpn_judged_Llama-3.1-8B-base-instruct_v4_general_skills_by_gemini-2.5-pro.xlsx \
        outputs/evaluations/judged/hpn_judged_gpt-4o_v4_general_skills_by_gemini-2.5-pro.xlsx

# ── Filter judged results into good/bad subsets ──────────────────────
python evaluation/filter_judged.py
```

> **Cross-judge rule**: scores are only directly comparable when the judge is
> the same. Do not include a model's own answers in a run where it acts as
> judge (self-judging bias).

**Output directory layout**:
```
outputs/evaluations/
├── answers/                                    ← Phase 1
│   └── hpn_answers_<model>_<bmk_tag>.xlsx     (local or API model name)
├── judged/                                     ← Phase 2
│   └── hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx
├── filtered/                                   ← Filtered good/bad subsets
│   └── filtered_hpn_judged_<model>_<bmk_tag>_by_<judge>.xlsx
├── reports/                                    ← Phase 3
│   ├── hpn_benchmark_report_<bmk_tag>_by_<judge>.xlsx
│   └── HPN_BENCHMARK_REPORT_<bmk_tag>_by_<judge>.md
└── adaptation_reports/                         ← LM Adaptation (CPT quality)
    ├── lm_adaptation_<base>_vs_<trained>.json
    └── lm_adaptation_<base>_vs_<trained>.xlsx
```

### 10. LM Adaptation Report

Measures CPT quality and catastrophic forgetting by comparing perplexity on the
HPN domain corpus and WikiText-2 before/after pre-training.

```bash
python evaluation/lm_adaptation_report.py \
    --base_model_path    models/base/Qwen3.5-2B-Base \
    --trained_model_path models/pretrained/Qwen3.5-2B-trained-new
```

| Metric | Description |
|--------|-------------|
| HPN Perplexity (base) | Domain perplexity before CPT |
| HPN Perplexity (trained) | Domain perplexity after CPT (should decrease) |
| General Perplexity (base) | WikiText-2 perplexity before CPT |
| General Perplexity (trained) | WikiText-2 perplexity after CPT |
| **Forgetting Gap** | `GeneralLossAfter − GeneralLossBefore` (positive = forgetting) |

Output: `outputs/evaluations/adaptation_reports/lm_adaptation_*.{json,xlsx}`



An interactive dashboard wrapping every pipeline stage in a browser UI,
with live log streaming, model selection dropdowns, and inline charts.

### Launch

```bash
# From the project root (venv must be active):
python frontend/app.py
# → http://localhost:7860

# Custom host / port / public share link:
python frontend/app.py --host 0.0.0.0 --port 7860 --share
```

### Features

| Tab | Sub-tabs | What it does |
|-----|----------|--------------|
| ⚙️ **System** | GPU Stats, Model Loader | Real-time GPU & RAM gauges; load/unload any model from `models/`; VRAM usage table |
| 🏗️ **Training** | Prepare CPT Data, Continual Pretrain, Prepare Instruction Data, Instruction FT, Outputs | Full-weight training pipeline for dataset prep, domain-adaptive pretraining, instruction-data prep, instruction fine-tuning, and artifact browsing |
| 💬 **Chat** | Single Chat, Compare | Stream responses from any loaded model; stop strings for Instruct mode; token stats; side-by-side model comparison |
| 📊 **Benchmark** | Phase 1 Generate, Phase 2 Judge, Phase 3 Report, Results | Full HPN Q&A benchmark pipeline; Gemini or OpenAI as judge; v4 benchmark (10 categories); comparative Markdown + Excel reports |
| 📈 **Profiling** | Training Profiler, Inference Profiler, Report Generator, Results Viewer | Profile training throughput & VRAM; benchmark inference latency (tok/s, TTFT); generate Excel + PNG reports; inline bar charts |
| 🔌 **LoRA Training** | Prepare Data, Train LoRA, Merge Adapter, Adapters | Parameter-efficient LoRA / QLoRA pipeline with live logs and merged-model output browsing |

### Architecture

```
frontend/
├── app.py                   ← Entry point; Gradio Blocks + launch()
├── core/
│   ├── model_manager.py     ← Thread-safe model registry (load/unload/VRAM)
│   └── job_runner.py        ← Background thread runner with live log capture
└── tabs/
    ├── system_tab.py        ← Phase B
    ├── training_tab.py      ← Full CPT + instruction-FT pipeline
    ├── chat_tab.py          ← Phase C
    ├── benchmark_tab.py     ← Phase D
    ├── profiling_tab.py     ← Phase E
    └── lora_tab.py          ← LoRA pipeline
```

---

## ⚠️ BASE Models vs Instruction Models

**BASE models** complete text — they do **not** follow instructions:

```python
# ❌ WRONG (instruction-style)
"What is GridFTP?"

# ✅ CORRECT (completion-style)
"In distributed computing, GridFTP is a protocol that"
```

**Instruction-tuned Llama models** follow the Open-Orca chat prompt format:

```
### System:
You are a helpful assistant.

### User:
Explain what GridFTP is and why it matters for large-scale data transfer.

### Assistant:
```

**Instruction-tuned Qwen3.5 models** follow the ChatML prompt format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain what GridFTP is and why it matters for large-scale data transfer.<|im_end|>
<|im_start|>assistant
```

**Instruction-tuned Gemma 3 models** follow the Gemma turn format (system folded into first user turn):

```
<start_of_turn>user
You are a helpful assistant.

Explain what GridFTP is and why it matters for large-scale data transfer.<end_of_turn>
<start_of_turn>model
```

The correct template is selected automatically based on the model path.

## Supported Models

| Key | HuggingFace ID | Family | Parameters | Notes |
|-----|---------------|--------|------------|-------|
| `1b` | `meta-llama/Llama-3.2-1B` | Llama | 1.2B | |
| `8b` | `meta-llama/Llama-3.1-8B` | Llama | 8B | |
| `qwen-2b` | `Qwen/Qwen3.5-2B-Base` | Qwen | 2B | |
| `qwen-4b` | `Qwen/Qwen3.5-4B-Base` | Qwen | 4B | |
| `gemma-4b` | `google/gemma-3-4b-pt` | Gemma 3 | 4B | |
| `gemma-e4b` | `google/gemma-4-E4B` | Gemma 4 | ~8.6B | KV-sharing; no gradient checkpointing |
| `gemma-e2b` | `google/gemma-4-E2B` | Gemma 4 | ~2.6B | KV-sharing; no gradient checkpointing |
| `gemma-e4b-it` | `google/gemma-4-E4B-it` | Gemma 4 | ~8.6B | Official instruct variant |

## 🔧 Requirements

- 2× NVIDIA RTX A6000 (or equivalent, ≥ 48 GB VRAM per GPU)
- Python 3.10+
- HuggingFace account with Llama and Gemma access (Qwen models are open-weight, no gating)
