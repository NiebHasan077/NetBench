# Profiling Pipeline — User Guide

A three-script pipeline for measuring **training performance**, **GPU/CPU inference speed**, and producing a unified **Markdown report**.

> `profile_training.py` **saves the trained model** to `--output_dir` so that
> downstream profiling (inference benchmarks, etc.) can run on it.
>
> Both `--output_dir` and `--run_name` are **optional** — sensible defaults are
> generated automatically if omitted.
>
> If `--data_dir` does **not** point to a pre-tokenized HuggingFace dataset, the
> script **auto-tokenizes** the data before training.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [profile_training.py](#1-profile_trainingpy)
4. [profile_inference.py](#2-profile_inferencepy)
5. [generate_profile_report.py](#3-generate_profile_reportpy)
6. [Metric Glossary](#metric-glossary)
7. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

```
profile_training.py pretrain  ─►  outputs/profiling_results/training_<run>.json
                              ─►  <output_dir>/  (saved model + tokenizer)
profile_training.py instruct  ─►  outputs/profiling_results/training_<run>.json
                              ─►  <output_dir>/  (saved model + tokenizer)
profile_inference.py          ─►  outputs/profiling_results/inference_<run>.json
                                      │
generate_profile_report.py  ──────────┘─►  outputs/profiling_reports/
   ├── report_plots.py (matplotlib)        ├── PROFILING_REPORT.md
   └── report_excel.py (openpyxl)          ├── cumulative/
                                           │   ├── cumulative_report.xlsx
                                           │   ├── loss_comparison.png
                                           │   ├── training_throughput.png
                                           │   ├── memory_comparison.png
                                           │   ├── inference_comparison.png
                                           │   └── gpu_vs_cpu.png
                                           └── per_run/<name>/
                                               ├── <name>.xlsx
                                               ├── loss_curve.png
                                               ├── gpu_utilization.png
                                               ├── memory_usage.png
                                               ├── inference_throughput.png
                                               └── inference_scaling.png
```

- All JSON files are saved into `outputs/profiling_results/`.
- Trained models are saved to the `--output_dir` you specify (recommended: `models/profiled/<name>`).
- The report generator reads **every** JSON in that directory (or a filtered subset) and produces **Excel workbooks**, **PNG plots**, and a **lightweight Markdown report** with embedded image links.
- Each JSON embeds the **model path**, **run name**, **mode**, **dataset info**, and **saved model path** so the report can show multi-model comparisons.

---

## Prerequisites

All dependencies are already in `requirements.txt`:

| Package       | Why                                      |
|---------------|------------------------------------------|
| `torch`       | Model loading, CUDA memory queries       |
| `transformers` | Model & tokenizer loading, Trainer      |
| `psutil`      | CPU / RAM metrics                        |
| `pynvml`      | GPU metrics (optional fallback to nvidia-smi) |
| `matplotlib`  | PNG plot generation (loss curves, GPU charts) |
| `openpyxl`    | Excel workbook generation (formatted tables) |

Make sure the virtual environment is activated:

```bash
source .venv/bin/activate
```

---

## 1. `profile_training.py`

Two subcommands: **`pretrain`** and **`instruct`**. Each mirrors the hyperparameters of the real training script (`pretrain_transformers.py` or `instruction_finetune.py`).

**Saves the trained model** to the `--output_dir` you specify. Inference profiling can then benchmark the saved model directly.

### Quick start

```bash
# ── Phase 1: Profile continual pre-training ────────────────────
# Explicit (all args)
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.2-1B-base \
    --data_dir data/processed/llama-3.2-1b \
    --output_dir models/profiled/Llama-3.2-1B-pretrain-quick \
    --run_name pretrain_1b_quick \
    --max_steps 50

# Minimal (auto output_dir & run_name)
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.2-1B-base \
    --data_dir data/processed/llama-3.2-1b \
    --max_steps 50
# → output_dir = models/profiled/Llama-3.2-1B-base-pretrain
# → run_name   = pretrain_Llama-3.2-1B-base_20250101_120000

# Auto-tokenize from raw JSON corpus
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.2-1B-base \
    --data_dir data/raw/research_corpus_new.json \
    --max_steps 50

# Full pre-training profiling on 8B
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/processed/llama-3.1-8b \
    --output_dir models/profiled/Llama-3.1-8B-pretrain \
    --run_name pretrain_8b

# ── Phase 2: Profile instruction fine-tuning ───────────────────
python profiling/profile_training.py instruct \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/profiled/Llama-3.1-8B-instruct \
    --run_name instruct_8b_base \
    --max_steps 50

# Auto-tokenize instruction data (downloads Orca + Dolly)
python profiling/profile_training.py instruct \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir auto \
    --max_steps 50

# Instruct on the domain-adapted model
python profiling/profile_training.py instruct \
    --model_path models/pretrained/Llama-3.1-8B-trained-new \
    --data_dir data/instruction/llama-3.1-8b-trained-new \
    --output_dir models/profiled/Llama-3.1-8B-trained-instruct \
    --run_name instruct_8b_trained
```

### Arguments (shared between `pretrain` and `instruct`)

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | *required* | Path to the base (or trained) model directory |
| `--data_dir` | *required* | Tokenized dataset dir, raw JSON corpus, or `auto` (instruct: downloads Orca+Dolly) |
| `--output_dir` | auto | Directory to save the trained model. Default: `models/profiled/<model_name>-<mode>` |
| `--run_name` | auto | A label for this profiling run. Default: `<mode>_<model_name>_<timestamp>` |
| `--num_epochs` | `3` | Number of training epochs |
| `--batch_size` | `2` | Per-device batch size |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `--learning_rate` | `2e-5` | Learning rate |
| `--warmup_ratio` | `0.1` | Warmup ratio |
| `--weight_decay` | `0.01` | Weight decay |
| `--max_grad_norm` | `1.0` | Max gradient norm |
| `--max_steps` | — | Limit total steps (useful for quick profiling) |
| `--logging_steps` | `5` | Log loss every N steps |
| `--eval_steps` | `100` | Run eval every N steps |
| `--gpu_poll_interval` | `5.0` | Seconds between GPU monitor samples |

### What it records

- **System snapshot**: hostname, OS, Python/PyTorch/CUDA versions, CPU model & RAM, per-GPU name/memory
- **Config**: model path, dtype, batch size, effective batch size, learning rate, scheduler, optimizer
- **Dataset info**: type (pretrain_corpus / instruction), source file/datasets, train/eval sample counts, token counts, sequence length statistics (min/mean/median/p95/max)
- **Per-step loss curve**: step, epoch, loss, learning rate, GPU allocated memory
- **Timing**: total training wall time, per-step average, per-epoch wall times, model save time
- **Throughput**: samples/second, tokens/second
- **Memory**: peak GPU VRAM (aggregate & per-GPU), CPU RAM before/after
- **GPU monitor**: per-GPU time-series (utilization %, memory used, temperature, power draw) with avg/min/max aggregates
- **Saved model path**: absolute path to the saved model directory

### Auto-tokenization

If `--data_dir` does **not** point to a pre-tokenized HuggingFace dataset (i.e. no `dataset_dict.json`), the script automatically tokenizes the data before training:

| Mode | `--data_dir` value | What happens |
|------|--------------------|--------------|
| `pretrain` | Path to a `.json` corpus file | Tokenizes the JSON corpus (same logic as `prepare_data.py`) and saves to `<stem>_tokenized_<model>/` |
| `instruct` | `auto` (or any non-tokenized path) | Downloads **Orca + Dolly** from HuggingFace Hub, tokenizes, and saves to `data/instruction/<model>-auto/` |

If the tokenized output already exists from a previous run, it is **reused** without re-tokenizing.

### Output

```
outputs/profiling_results/training_<run_name>.json   # metrics report
<output_dir>/                                # saved model + tokenizer
```

> The trained model and tokenizer are saved to `--output_dir` at the end of
> training.  `profile_inference.py --all` automatically discovers models in
> `models/profiled/` alongside `models/base/`, `models/pretrained/`, and
> `models/instruction/`.

---

## 2. `profile_inference.py`

Benchmarks a model's generation speed on GPU and/or CPU.

### Quick start

```bash
# Benchmark one model on GPU
python profiling/profile_inference.py \
    --model_path models/pretrained/Llama-3.1-8B-trained-new

# Benchmark one model on CPU
python profiling/profile_inference.py \
    --model_path models/base/Llama-3.2-1B-base \
    --cpu

# Benchmark ALL discovered models on GPU
python profiling/profile_inference.py --all

# Benchmark ALL discovered models on GPU + CPU
python profiling/profile_inference.py --all --cpu
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | — | Path to a single model to benchmark |
| `--all` | `false` | Discover and benchmark every model in `models/base/`, `models/pretrained/`, `models/instruction/`, `models/profiled/` |
| `--cpu` | `false` | Also benchmark on CPU (in addition to GPU) |
| `--max_new_tokens` | `128` | Default generation length |
| `--num_runs` | `3` | Repetitions per prompt (results are averaged) |
| `--warmup_runs` | `2` | Warm-up runs (discarded) before measuring |
| `--run_name` | auto | Label for this profiling run |
| `--prompts_file` | — | Custom JSON prompts file |

### What it records

- **Model load**: load time, GPU VRAM after loading
- **Per-prompt benchmark** (5 built-in prompts of varying length & style):
  - Total generation time, tokens/second, ms/token
  - Time-to-first-token (TTFT)
  - Peak GPU VRAM during generation
  - Averaged over `num_runs` runs
- **Output-length scaling** (same prompt, `max_new_tokens` = 32, 64, 128, 256, 512)
- **Input-length scaling** (growing input context, fixed output)
- **Summary**: aggregate averages across all prompts

### Output

```
outputs/profiling_results/inference_<run_name>.json           # single-model
outputs/profiling_results/inference_comparison_<device>.json   # --all mode
```

---

## 3. `generate_profile_report.py`

Reads **all** JSONs from `outputs/profiling_results/` and produces:
- **Per-run directories** — Excel workbooks + PNG plots for each training/inference run
- **Cumulative directory** — comparison Excel + comparison plots across all runs
- **Lightweight Markdown** — `PROFILING_REPORT.md` with embedded image links

Uses two helper modules:
- `report_plots.py` — all matplotlib figure generation
- `report_excel.py` — all openpyxl Excel workbook generation

### Which models get reported?

**All of them.** Every `training_*.json` and `inference_*.json` file in the results directory is included. Each JSON embeds its own `run_name`, `model_path`, and `mode`, so the report shows one row per run per model — it's a multi-model comparison by design.

To control what's in the report:

| Want to… | Do this |
|----------|---------|
| See what's available | `python profiling/generate_profile_report.py --list` |
| Report only 8B models | `python profiling/generate_profile_report.py --filter 8B` |
| Report only pre-training | `python profiling/generate_profile_report.py --filter pretrain` |
| Report a specific run | `python profiling/generate_profile_report.py --filter pretrain_1b_quick` |
| Only cumulative comparison | `python profiling/generate_profile_report.py --cumulative-only` |
| Start fresh | Delete unwanted JSONs from `outputs/profiling_results/` |

### Quick start

```bash
# List what's available
python profiling/generate_profile_report.py --list

# Generate full report (Markdown + Excel + plots)
python profiling/generate_profile_report.py

# Generate filtered report
python profiling/generate_profile_report.py --filter 8B

# Only cumulative comparison (skip per-run)
python profiling/generate_profile_report.py --cumulative-only

# Custom output directory
python profiling/generate_profile_report.py --output_dir my_reports
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `profiling_reports` | Output directory for all reports |
| `--results_dir` | `profiling_results` | Where to find JSON files |
| `--list` | — | List available JSONs and exit (no report) |
| `--filter` | — | Only include reports matching this substring (case-insensitive, matches run_name/model_name/model_path/mode) |
| `--cumulative-only` | — | Only generate cumulative comparison (skip per-run reports) |

### Output structure

```
outputs/profiling_reports/
├── PROFILING_REPORT.md              ← Lightweight Markdown with embedded images
├── cumulative/
│   ├── cumulative_report.xlsx       ← Training Summary + Inference Summary + GPU vs CPU sheets
│   ├── loss_comparison.png          ← Overlaid loss curves from all training runs
│   ├── training_throughput.png      ← Samples/s and tokens/s bar chart
│   ├── memory_comparison.png        ← Peak VRAM bar chart
│   ├── inference_comparison.png     ← Avg tok/s bar chart
│   └── gpu_vs_cpu.png              ← Side-by-side GPU vs CPU bars
└── per_run/
    ├── <training_run_name>/
    │   ├── <run_name>.xlsx          ← Config, Dataset, Loss Curve, GPU Monitor, Memory sheets
    │   ├── loss_curve.png           ← Loss + LR dual-axis plot
    │   ├── gpu_utilization.png      ← Util %, VRAM, temperature timeseries
    │   └── memory_usage.png         ← Peak VRAM per GPU + CPU delta
    └── <model_name>_<device>/
        ├── <run_name>.xlsx          ← Config, Prompt Benchmarks, Scaling, Summary sheets
        ├── inference_throughput.png  ← Tok/s per prompt bar chart
        └── inference_scaling.png    ← Output + input length scaling plots
```

### Per-run Excel workbooks

**Training run** sheets:
| Sheet | Contents |
|-------|----------|
| Config | All hyperparameters, system info, timing |
| Dataset | Data source, sample counts, token counts, sequence length stats |
| Loss Curve | Every logged step: step, epoch, loss, LR, GPU alloc |
| GPU Monitor | nvidia-smi timeseries: util %, VRAM, temperature, power |
| Memory | Peak VRAM per GPU, CPU RAM before/after |

**Inference run** sheets:
| Sheet | Contents |
|-------|----------|
| Config | Model path, device, system info, load time |
| Prompt Benchmarks | Per-prompt: avg time, tok/s, ms/tok, TTFT, peak VRAM |
| Output Scaling | Throughput vs max_new_tokens |
| Input Scaling | Throughput vs input context length |
| Summary | Aggregate averages |

### Cumulative Excel workbook

| Sheet | Contents |
|-------|----------|
| Training Summary | One row per training run: model, mode, throughput, loss, memory |
| Inference Summary | One row per inference run: model, device, tok/s, TTFT, memory |
| GPU vs CPU | Side-by-side comparison for models benchmarked on both devices |

### Markdown report sections

1. **Report Inventory** — which JSON files are included (runs + models)
2. **Training Summary** — one-row-per-run table
3. **Hardware** — CPU, RAM, GPUs
4. **Training Configurations** — batch size, LR, optimizer, scheduler
5. **Dataset Info** — type, source, samples, tokens, sequence length stats
6. **GPU Utilization** — per-GPU avg/max for utilization, VRAM, temperature, power
7. **Memory** — peak GPU and CPU memory
8. **Loss Curve** — embedded PNG plots (with Excel link for full data)
9. **Inference Results** — per-prompt generation speed (GPU and CPU)
10. **Overall Inference Summary** — aggregate metrics per model per device
11. **Output-Length Scaling** — throughput vs generation length
12. **Input-Length Scaling** — throughput vs context length
13. **GPU vs CPU Comparison** — speedup table for models benchmarked on both
14. **Cumulative Comparison** — embedded comparison plots + Excel links

---

## Metric Glossary

| Metric | Unit | Meaning |
|--------|------|---------|
| **Tokens/s** | tok/s | Output tokens generated per second |
| **ms/token** | ms | Milliseconds per output token (= 1000 / tokens_per_s) |
| **TTFT** | seconds | Time-to-first-token — latency before the first output token appears |
| **Samples/s** | samples/s | Training samples processed per second |
| **Peak GPU VRAM** | MB | Maximum GPU memory used during the run |
| **Utilization %** | % | Average GPU compute utilization (from nvidia-smi) |
| **Power** | W | Average GPU board power draw |
| **Final loss** | — | Training loss at the last logged step |
| **Effective batch** | — | `per_device_batch × grad_accum × num_gpus` |
| **Seq Len (p95)** | tokens | 95th percentile of non-padding token count per sample |

---

## Tips & Troubleshooting

### Gemma 4 models (E4B / E2B)

Gemma 4 models (`gemma-e4b`, `gemma-e2b`) use **KV-layer sharing** and are handled automatically:

- `gradient_checkpointing` is **disabled** (incompatible with KV-shared architecture)
- `use_cache` is kept **enabled** (required for correct KV-sharing)
- `token_type_ids` / `mm_token_type_ids` are injected as zeros by the collator

No extra flags are needed — `profile_training.py` detects KV-sharing from the model config.

```bash
# Profile Gemma 4 E4B pre-training
python profiling/profile_training.py pretrain \
    --model_path models/base/gemma-4-e4b \
    --data_dir data/processed/gemma-4-e4b \
    --max_steps 50
```

### Quick profiling run

Use `--max_steps 50` with `profile_training.py` to get fast results without running a full epoch.

### Saved models & inference profiling

`profile_training.py` saves the trained model to `--output_dir`. To benchmark it with inference profiling:

```bash
# Direct path
python profiling/profile_inference.py --model_path models/profiled/Llama-3.2-1B-pretrain-quick

# Or discover all (includes models/profiled/ automatically)
python profiling/profile_inference.py --all
```

### Minimal GPU memory

The profiling scripts use bfloat16 by default. If you run out of VRAM, reduce `--batch_size` to 1 and/or add `--max_steps 20`.

### nvidia-smi not found

The GPU monitor falls back to `torch.cuda.*` when `nvidia-smi` is unavailable. Utilization percentage and power draw will be missing in that case.

### Empty report sections

If a section says "No data found", it means no matching JSON files exist in `outputs/profiling_results/`. Run the relevant profiling script first.

### Too many runs in the report

Use `--filter` to narrow down, or delete unwanted JSONs from `outputs/profiling_results/`.

### Re-running and overwriting

Each profiling run generates a uniquely-named JSON (run_name). If you re-run with the same `--run_name`, the file **will be overwritten**. Use unique run names for comparison.

### Running on CPU

CPU inference on an 8B model will be **very slow** (expect < 1 tok/s). Start with the 1B model for CPU benchmarks:

```bash
python profiling/profile_inference.py --model_path models/base/Llama-3.2-1B-base --cpu
```

---

## Example Workflow

```bash
# 1. Profile pre-training (quick 50-step test, auto output_dir & run_name)
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.2-1B-base \
    --data_dir data/processed/llama-3.2-1b \
    --max_steps 50

# 1b. Or auto-tokenize from raw JSON + explicit names
python profiling/profile_training.py pretrain \
    --model_path models/base/Llama-3.2-1B-base \
    --data_dir data/raw/research_corpus_new.json \
    --output_dir models/profiled/Llama-3.2-1B-pretrain-quick \
    --run_name pretrain_1b_quick \
    --max_steps 50

# 2. Profile instruction fine-tuning (auto-download Orca+Dolly)
python profiling/profile_training.py instruct \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir auto \
    --max_steps 50

# 2b. Or use an existing tokenized instruction dataset
python profiling/profile_training.py instruct \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/profiled/Llama-3.1-8B-instruct-quick \
    --run_name instruct_8b_quick \
    --max_steps 50

# 3. Benchmark inference on all models (GPU)
#    (discovers models/base/, models/pretrained/, models/instruction/, models/profiled/)
python profiling/profile_inference.py --all

# 4. Also benchmark 1B on CPU
python profiling/profile_inference.py \
    --model_path models/base/Llama-3.2-1B-base --cpu

# 5. Benchmark the profiled model specifically
python profiling/profile_inference.py \
    --model_path models/profiled/Llama-3.2-1B-pretrain-quick

# 6. See what's available
python profiling/generate_profile_report.py --list

# 7. Generate the unified report
python profiling/generate_profile_report.py

# 8. Or generate a report for just the 1B model
python profiling/generate_profile_report.py --filter 1B --output report_1b.md
```
