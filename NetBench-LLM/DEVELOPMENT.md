# NetBench-LLM — development guide

Continual pre-training and instruction fine-tuning pipeline for Llama, Qwen3.5, and Gemma 3/4 models on a custom research corpus. All trained outputs are written to separate directories; originals are always preserved.

## Architecture

| Folder | Purpose |
|---|---|
| `training/` | Data prep + training: CPT, IFT, LoRA FT |
| `evaluation/` | 3-phase HPN benchmark + LM adaptation report + simple model eval |
| `inference/` | Interactive chat and instruction model tester |
| `profiling/` | GPU/memory/throughput profiling + reports |
| `frontend/` | Gradio dashboard (chat, training, benchmark, LoRA, profiling) |
| `utils/` | Shared helpers: HF auth, model download, `model_utils.py`, system check |
| `data/` | `raw/` → `processed/` (CPT) or `instruction/` (IFT/LoRA) → `prompts/` (benchmarks) |
| `models/` | All weights — see model stages below |
| `outputs/` | All generated artifacts: answers, judged results, reports, profiling |

Detailed guides live in `docs/` — always check there before implementing a new workflow.

## Model Stages

Models flow through stages into separate directories. **Never modify `models/base/`**.

```
models/base/             ← Downloaded from HF, read-only
models/pretrained/       ← After continual pre-training (CPT)
models/instruction/      ← After instruction fine-tuning (IFT, our own SFT)
models/instruct-official/ ← Downloaded official instruct variants (run_pipeline.sh S1/S2/S3/S5/S6/S8)
models/lora/             ← Raw LoRA adapters (PEFT, not standalone)
models/lora-merged/      ← Adapters merged back into full weights
models/profiled/         ← Checkpoints saved during profiling runs
```

Naming pattern: `{Family}-{Size}-{corpus_variant}-{stage}[-merged]`  
Example: `Qwen3.5-2B-trained-new-instruct`

## End-to-End Agentic Workflow

Follow these steps in order. See [docs/PRETRAINING_GUIDE.md](docs/PRETRAINING_GUIDE.md), [docs/INSTRUCTION_FINETUNING_GUIDE.md](docs/INSTRUCTION_FINETUNING_GUIDE.md), and [docs/LORA_PIPELINE.md](docs/LORA_PIPELINE.md) for full details.

### 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python utils/check_system.py          # verify GPU/CUDA/disk
# Add auth tokens:
echo "YOUR_HF_TOKEN" > hf_token.txt
echo "YOUR_GEMINI_KEY" > gemini_api_key.txt
echo "YOUR_OPENAI_KEY" > openai_api_key.txt
```

### 2. Download Base Model
```bash
# Shortcut keys: 1b | 8b | qwen-2b | qwen-4b | gemma-4b | gemma-e4b | gemma-e2b | gemma-e4b-it
# gemma-e4b / gemma-e2b = Gemma 4 (KV-shared, gradient checkpointing disabled in training)
python utils/download_models.py --model qwen-2b
```

### 3. Continual Pre-training (CPT)
```bash
python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name Qwen/Qwen3.5-2B-Base \
    --output_dir data/processed/qwen3.5-2b

python training/pretrain_transformers.py \
    --model qwen-2b \
    --data_dir data/processed/qwen3.5-2b
# → models/pretrained/Qwen3.5-2B-trained-new
```
> **Gotcha**: `pretrain_transformers.py` takes `--model <shortcut_key>` (e.g. `qwen-2b`), not a path. All other scripts take explicit directory paths.

### 4. Instruction Fine-tuning (IFT)
```bash
python training/prepare_instruction_data.py \
    --model_path models/pretrained/Qwen3.5-2B-trained-new \
    --output_dir data/instruction/qwen3.5-2b-trained-new

python training/instruction_finetune.py \
    --model_path models/pretrained/Qwen3.5-2B-trained-new \
    --data_dir   data/instruction/qwen3.5-2b-trained-new \
    --output_dir models/instruction/Qwen3.5-2B-trained-new-instruct
```

### 4b. LoRA Alternative (for <24 GB GPU)
```bash
python training/lora_finetune.py \
    --model_path models/base/Qwen3.5-2B-Base \
    --data_dir   data/processed/lora/qwen3.5-2b-v3 \
    --data_mode  corpus \
    --quantize_4bit

python training/merge_lora_adapter.py \
    --adapter_path models/lora/Qwen3.5-2B-Base-lora
# Reads base model path automatically from adapter_config.json
```
See [docs/LORA_PIPELINE.md](docs/LORA_PIPELINE.md).

### 5. Test Interactively
```bash
./test_instruct.sh                           # menu of instruction models
./test_instruct.sh models/instruction/...   # direct path
```

### 6. HPN Q&A Benchmark (3 phases)
See [docs/QA_BENCHMARK_GUIDE.md](docs/QA_BENCHMARK_GUIDE.md).

```bash
# Phase 1 — generate answers (local or API models)
python evaluation/hpn_qa_benchmark.py \
    --model_path models/instruction/Qwen3.5-2B-trained-new-instruct
# API: --model_path gpt-4o --openai_api_key_file openai_api_key.txt
# → outputs/evaluations/answers/hpn_answers_{model}.xlsx

# Phase 2 — LLM-as-Judge scoring
python evaluation/judge_responses.py \
    --answer_files outputs/evaluations/answers/hpn_answers_*.xlsx
# → outputs/evaluations/judged/hpn_judged_{model}_by_{judge}.xlsx

# Phase 3 — comparative report
python evaluation/benchmark_report.py \
    --judged_files outputs/evaluations/judged/hpn_judged_*.xlsx
# → outputs/evaluations/reports/HPN_BENCHMARK_REPORT_*.md + .xlsx

# Optional: filter judged results into good/bad subsets
python evaluation/filter_judged.py
```

### 7. LM Adaptation Report
Measures CPT quality: HPN domain perplexity, WikiText-2 general perplexity, and forgetting gap.

```bash
python evaluation/lm_adaptation_report.py \
    --base_model_path    models/base/Qwen3.5-2B-Base \
    --trained_model_path models/pretrained/Qwen3.5-2B-trained-new
# → outputs/evaluations/adaptation_reports/lm_adaptation_*.{json,xlsx}
```

### 8. Full 8-Variant Pipeline
For running all training and evaluation variants (S1–S8) in sequence:

```bash
# 1. Set the model variables at the top of run_pipeline.sh for your hardware
#    Fill in MODEL_NAME, MODEL_SHORTCUT, hardware params in scripts/run_pipeline.sh
# 2. Run the full pipeline:
bash scripts/run_pipeline.sh 2>&1 | tee outputs/pipeline_run.log
```

The variables to set are grouped at the top of the script and guarded by a PLACEHOLDER check, so it refuses to run half-configured.

## Key Conventions

- **All config is CLI args** — no YAML/JSON config files for training.
- **HF auth via file** — tokens are read from `hf_token.txt`, not env vars.
- **Datasets saved with `save_to_disk()`** — loaded with `load_from_disk()`, not HF Hub.
- **Token packing** — CPT data is packed into 2048-token blocks with EOS between documents.
- **Instruction labels** — prompt/padding tokens are masked with `-100`; only assistant responses are supervised.
- **Multi-GPU** — scripts use `device_map="auto"` for 2× RTX A6000 (96 GB total).
- **Resume support** — `hpn_qa_benchmark.py` resumes interrupted benchmark runs automatically.

## Profiling
```bash
python profiling/profile_training.py
python profiling/profile_inference.py
python profiling/generate_profile_report.py
```
See [docs/PROFILING_GUIDE.md](docs/PROFILING_GUIDE.md).

## Frontend (Gradio)
```bash
python frontend/app.py
```
See [docs/FRONTEND_GUIDE.md](docs/FRONTEND_GUIDE.md).

## Available Documentation

| Guide | Topic |
|---|---|
| [docs/PRETRAINING_GUIDE.md](docs/PRETRAINING_GUIDE.md) | CPT pipeline, hyperparameters |
| [docs/INSTRUCTION_FINETUNING_GUIDE.md](docs/INSTRUCTION_FINETUNING_GUIDE.md) | IFT pipeline, data formats |
| [docs/LORA_PIPELINE.md](docs/LORA_PIPELINE.md) | LoRA / QLoRA adapter training and merging |
| [docs/QA_BENCHMARK_GUIDE.md](docs/QA_BENCHMARK_GUIDE.md) | HPN 3-phase benchmark |
| [docs/PROFILING_GUIDE.md](docs/PROFILING_GUIDE.md) | GPU profiling and reports |
| [docs/FRONTEND_GUIDE.md](docs/FRONTEND_GUIDE.md) | Gradio dashboard |
| [docs/CHAT_GUIDE.md](docs/CHAT_GUIDE.md) | Interactive inference |
