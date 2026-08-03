# LoRA Adapter Training Guide

Efficiently embed HPN domain knowledge into an instruction-following model using
**LoRA (Low-Rank Adaptation)** — a parameter-efficient alternative to the full
Continual Pre-Training → Instruction Fine-Tuning pipeline.

> **Prerequisites** — An instruction-tuned model in `models/instruction/` or
> `models/instruct-official/` (e.g. `Llama-3.1-8B-base-instruct`).  The HPN
> research corpus at `data/raw/research_corpus_new.json`.  Libraries
> `peft>=0.10.0` and `trl>=0.8.0` installed (`pip install peft trl`).

---

## Table of Contents

1. [What the Merge Step Does (and what it does NOT do)](#1-what-the-merge-step-does)
2. [Why LoRA Instead of Full CPT + SFT?](#2-why-lora-instead-of-full-cpt--sft)
3. [Architecture Overview](#3-architecture-overview)
4. [Pipeline — Step by Step](#4-pipeline--step-by-step)
5. [Script Reference](#5-script-reference)
6. [Configuration Reference (2×48 GB GPU)](#6-configuration-reference-248-gb-gpu)
7. [QLoRA vs Full bfloat16](#7-qlora-vs-full-bfloat16)
8. [Evaluation Integration](#8-evaluation-integration)
9. [Experiment Matrix](#9-experiment-matrix)
10. [Tips & Best Practices](#10-tips--best-practices)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What the Merge Step Does

### Short answer
The merge step writes a **new model** to `models/lora-merged/`.
**The original base model is never touched.**

### What `merge_and_unload()` does mathematically

During LoRA training, the weight update is represented as two small matrices:

```
W_original   (frozen, e.g. 4096 × 4096  ≈ 128 MB per layer)
  +
B × A        (trained, e.g. 4096 × 64 × 64 × 4096 ≈ 4 MB per layer)
  =
W_effective  (what the model actually uses at inference time)
```

`merge_and_unload()` permanently adds `B × A` into `W_original` and returns a
normal `AutoModelForCausalLM` with no PEFT dependency.  This merged result is
saved as a fresh file:

```
models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged/
    config.json
    model.safetensors        ← new file; base model untouched
    tokenizer_config.json
    ...
```

### What is NOT modified

| Path | Status |
|---|---|
| `models/base/Llama-3.1-8B-base/` | **Untouched** — raw downloaded weights |
| `models/instruction/Llama-3.1-8B-base-instruct/` | **Untouched** — source instruct model |
| `models/lora/Llama-3.1-8B-base-instruct-lora/` | **Untouched** — adapter weights |
| `models/lora-merged/...-merged/` | **New file** created by merge |

You can merge the same adapter multiple times (e.g., to try different output
paths) without any risk.

---

## 2. Why LoRA Instead of Full CPT + SFT?

The existing CPT → SFT pipeline was found to **under-perform the base model**
due to several compounding issues:

| Root cause | CPT+SFT | LoRA |
|---|---|---|
| Catastrophic forgetting from general SFT (Orca+Dolly erasing CPT) | Yes — SFT overwrites CPT domain weights | No — base instruct weights frozen |
| Corpus too small for full weight update (1.44 M tokens) | Fatal for 8B | Efficient — small adapters learn from small corpus |
| High learning rate corrupting general capabilities | Yes (LR=2e-5) | Not applicable (base weights frozen) |
| `eval_steps=500` bug (eval never ran) | Present | Fixed — dynamic eval_steps |
| `DataCollatorForLanguageModeling` overwriting labels | Present | Same collator, but CLM-packed data has no padding so no masking issue |

LoRA is structurally immune to catastrophic forgetting because the frozen base
model retains 100% of its instruction-following capability; only the small
adapter learns the domain.

---

## 3. Architecture Overview

```
                          EXISTING PIPELINE (unchanged)
  ┌──────────────────────────────────────────────────────────────────┐
  │  base ──► CPT ──► pretrained ──► SFT (Orca+Dolly) ──► instruct  │
  └──────────────────────────────────────────────────────────────────┘

                          NEW LoRA PIPELINE (loosely coupled)
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  instruct  ──── LoRA training ────►  adapter                    │
  │  (frozen)       (HPN corpus)        models/lora/                │
  │                                          │                       │
  │                                          ▼                       │
  │                                    merge_and_unload()            │
  │                                          │                       │
  │                                          ▼                       │
  │                                merged model                      │
  │                                models/lora-merged/               │
  │                                          │                       │
  │                                          ▼                       │
  │                          existing evaluation pipeline            │
  │                          (hpn_qa_benchmark.py unchanged)         │
  └──────────────────────────────────────────────────────────────────┘
```

**Directory layout produced by the LoRA pipeline:**

```
models/
  lora/                                     ← adapter weights only
    Llama-3.1-8B-base-instruct-lora/
      adapter_config.json                   ← LoRA configuration
      adapter_model.safetensors             ← trained matrices (~100 MB for r=64)
      tokenizer_config.json
      checkpoints/
        checkpoint-NNN/                     ← intermediate adapter snapshots

  lora-merged/                              ← full-weight merged models
    Llama-3.1-8B-base-instruct-lora-merged/
      config.json
      model.safetensors                     ← ~16 GB, same size as base
      tokenizer_config.json
```

---

## 4. Pipeline — Step by Step

### Step 1 — Install new dependencies

```bash
pip install peft>=0.10.0 trl>=0.8.0
```

### Step 2 — Prepare data

Tokenizes `data/raw/research_corpus_new.json` into packed 2048-token blocks.
Uses the instruct model's tokenizer for consistency.

```bash
# 8B model
python training/prepare_lora_data.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
    --output_dir data/processed/lora/llama-3.1-8b

# 1B model
python training/prepare_lora_data.py \
    --model_path models/instruction/Llama-3.2-1B-base-instruct \
    --output_dir data/processed/lora/llama-3.2-1b
```

**Shortcut — if CPT data for this model already exists:**
```bash
python training/prepare_lora_data.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
    --output_dir data/processed/lora/llama-3.1-8b \
    --reuse_cpt_data data/processed/llama-3.1-8b
```

### Step 3 — Train LoRA adapter

```bash
# Primary experiment: 8B base-instruct + LoRA (full bfloat16, 2×48GB)
python training/lora_finetune.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
    --data_dir   data/processed/lora/llama-3.1-8b

# Primary experiment: 1B base-instruct + LoRA
python training/lora_finetune.py \
    --model_path models/instruction/Llama-3.2-1B-base-instruct \
    --data_dir   data/processed/lora/llama-3.2-1b

# Ablation: trained-new-instruct + LoRA (CPT+LoRA stacking test)
python training/lora_finetune.py \
    --model_path models/instruction/Llama-3.1-8B-trained-new-instruct \
    --data_dir   data/processed/lora/llama-3.1-8b
```

Output: `models/lora/Llama-3.1-8B-base-instruct-lora/`

### Step 4 — Merge adapter

```bash
# Auto-detects base model from adapter_config.json
python training/merge_lora_adapter.py \
    --adapter_path models/lora/Llama-3.1-8B-base-instruct-lora

# Explicit paths
python training/merge_lora_adapter.py \
    --adapter_path    models/lora/Llama-3.1-8B-base-instruct-lora \
    --base_model_path models/instruction/Llama-3.1-8B-base-instruct \
    --output_dir      models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged
```

Output: `models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged/`

### Step 5 — Evaluate

Use the **existing** benchmark pipeline unchanged:

```bash
python evaluation/hpn_qa_benchmark.py \
    --model_path models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged \
    --benchmark  data/prompts/hpn_qa_benchmark_v2_general_skills.json \
    --output_dir outputs/evaluations/answers
```

Or use the **📊 Benchmark** tab in the dashboard — select the merged model
from the model dropdown in Phase 1.

---

## 5. Script Reference

### `training/prepare_lora_data.py`

Prepares the HPN research corpus for CLM LoRA training.

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Local model path for tokenizer |
| `--corpus_file` | `data/raw/research_corpus_new.json` | Input corpus |
| `--output_dir` | `data/processed/lora/` | Arrow dataset output |
| `--max_length` | `2048` | Block size (tokens) |
| `--val_ratio` | `0.05` | Validation split fraction |
| `--reuse_cpt_data` | None | Reuse existing processed dataset; skips tokenization |
| `--seed` | `42` | Random seed |

---

### `training/lora_finetune.py`

Trains LoRA adapter weights. Base model is never modified.

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Source instruct model |
| `--data_dir` | `data/processed/lora/` | Processed Arrow dataset |
| `--data_mode` | `corpus` | `corpus` — CLM domain adaptation (CPT-style); `instruct` — supervised instruction fine-tuning (SFT-style) |
| `--output_dir` | auto-derived | Adapter save path (`models/lora/<name>-lora/`) |
| `--quantize_4bit` | `False` | Use 4-bit NF4 QLoRA (for GPUs < 24 GB) |
| `--lora_rank` | `64` | LoRA rank r |
| `--lora_alpha` | `128` | LoRA alpha (scaling = alpha/r = 2.0) |
| `--lora_dropout` | `0.05` | Dropout on adapter layers |
| `--target_modules` | auto-detected | Comma-separated LoRA target module names (default: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`) |
| `--epochs` | `3` | Training epochs |
| `--batch_size` | `4` | Per-device batch |
| `--grad_accum` | `4` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_ratio` | `0.1` | LR warmup fraction |
| `--bf16` | `True` | bfloat16 training |
| `--gradient_checkpointing` | `True` | Recompute activations (saves VRAM) |

**`--data_mode corpus`** (default): packed CLM training on the raw corpus — same objective as CPT.  Use with `prepare_lora_data.py` output.

**`--data_mode instruct`**: supervised fine-tuning on instruction-response pairs.  Use with an instruction dataset in JSONL format (e.g. `data/Instruct-FTD/` or Orca+Dolly).  Only assistant tokens are supervised (prompt is masked with -100).

---

### `training/merge_lora_adapter.py`

Merges adapter into base model weights.  Produces a standard full-weight model.

| Argument | Default | Description |
|---|---|---|
| `--adapter_path` | required | Directory with `adapter_config.json` |
| `--base_model_path` | auto from `adapter_config.json` | Base model used during training |
| `--output_dir` | `models/lora-merged/<name>-merged/` | Output path |

---

## 6. Configuration Reference (2×48 GB GPU)

### Recommended settings for RTX A6000 (48 GB × 2)

```
Hardware:          2× RTX A6000, 48 GB VRAM each (96 GB total)
8B model size:     ~16 GB in bfloat16
Memory per GPU:    ~20–25 GB with batch=4 + grad_checkpointing

LoRA config (8B):
  rank (r):               64       ← 4× more capacity than default 16
  alpha:                  128      ← scaling factor = alpha/r = 2.0
  target_modules:         q_proj, k_proj, v_proj, o_proj,
                          gate_proj, up_proj, down_proj
  dropout:                0.05
  trainable parameters:  ~115 M (1.4% of 8B)

Training config:
  precision:              bfloat16  (no quantization needed)
  optimizer:              adamw_torch
  batch_size:             4 per device
  grad_accum:             4 steps
  effective batch:        16
  learning_rate:          2e-4
  scheduler:              cosine
  warmup_ratio:           0.1
  gradient_checkpointing: True (use_reentrant=False)
  epochs:                 3
  eval_steps:             dynamic (total_steps // 10, min 10)
```

### For lower-VRAM GPUs (< 24 GB single GPU)

Pass `--quantize_4bit`:

```
8B model in 4-bit NF4: ~5–6 GB
LoRA adapters:          ~1 GB
Total:                  ~6–7 GB at batch=2

LoRA config:
  rank (r):       16   (reduce to limit adapter VRAM)
  alpha:          32

Training config:
  optimizer:      paged_adamw_8bit (automatically selected with --quantize_4bit)
  batch_size:     2
  grad_accum:     8
  effective batch: 16
```

CLI:
```bash
python training/lora_finetune.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
    --data_dir   data/processed/lora/llama-3.1-8b \
    --quantize_4bit \
    --lora_rank 16 --lora_alpha 32 \
    --batch_size 2 --grad_accum 8
```

---

## 7. QLoRA vs Full bfloat16

| | Full bfloat16 (default) | QLoRA 4-bit (`--quantize_4bit`) |
|---|---|---|
| **VRAM (8B)** | ~16 GB | ~6 GB |
| **Training quality** | Higher — no quantization noise on weights | Slight quality loss from 4-bit rounding |
| **Convergence** | Typically faster | Same LR works, slightly slower |
| **Optimizer** | `adamw_torch` | `paged_adamw_8bit` |
| **Merge accuracy** | Full precision throughout | Base model reloaded in bfloat16 for merge |
| **Use case** | 2×48 GB (this project) | Single 24 GB or smaller GPU |

**Recommendation for this project:** Full bfloat16.  With 96 GB total VRAM
there is no reason to use 4-bit quantization — it only adds noise.

---

## 8. Evaluation Integration

The merged model (in `models/lora-merged/`) is a standard HuggingFace
`AutoModelForCausalLM` with no PEFT dependency.  It integrates directly
with every existing evaluation and inference script.

### Via CLI

```bash
# Benchmark (answer generation)
python evaluation/hpn_qa_benchmark.py \
    --model_path models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged \
    --benchmark  data/prompts/hpn_qa_benchmark_v4_general_skills.json \
    --temperature 0.0

# Interactive chat
python inference/chat_interactive.py \
    --model_path models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged
```

### Via Dashboard

1. Open **📊 Benchmark** tab → **Phase 1**
2. Rescan models — the merged model appears under `models/lora-merged/`
3. Select it alongside `Llama-3.1-8B-base-instruct` for direct comparison
4. Use benchmark v2 (`hpn_qa_benchmark_v2_general_skills.json`) for a fair
   comparison — it tests transferable HPN reasoning, not paper-specific recall

---

## 9. Experiment Matrix

Suggested evaluation table for the paper:

| Model | Precision | Benchmark | Score |
|---|---|---|---|
| `1B-base-instruct` (baseline) | bfloat16 | v2 | 2.18 |
| `1B-trained-new-instruct` | bfloat16 | v2 | 2.16 |
| **`1B-base-instruct + LoRA`** (primary) | bfloat16 | v2 | ? |
| `8B-base-instruct` (baseline) | bfloat16 | v2 | 2.48 |
| `8B-trained-new-instruct` | bfloat16 | v2 | 2.26 |
| **`8B-base-instruct + LoRA`** (primary) | bfloat16 | v2 | ? |
| `8B-trained-new-instruct + LoRA` (ablation) | bfloat16 | v2 | ? |
| `Qwen3.5-2B-trained-new-instruct` | bfloat16 | v4 | ? |
| **`Qwen3.5-2B-trained-new-instruct + LoRA`** | bfloat16 | v4 | ? |
| `Qwen3.5-4B-trained-new-instruct` | bfloat16 | v4 | ? |
| **`Qwen3.5-4B-trained-new-instruct + LoRA`** | bfloat16 | v4 | ? |

> Use **v2 benchmark** (`hpn_qa_benchmark_v2_general_skills.json`) for all
> comparisons — it tests transferable HPN reasoning skills and does not
> penalise models for missing papers (AutoMDT, GDM, FastBioDL) that were
> absent from the training corpus.
>
> Use `temperature=0, do_sample=False` for deterministic, reproducible results.

---

## 10. Tips & Best Practices

### Rank and alpha selection

- `alpha / r` controls the effective magnitude of the LoRA update.  Keep
  this ratio at `2.0` unless you have a specific reason to change it.
- Higher rank = more capacity but more parameters.  For a 1.44 M token
  corpus, `r=64` is appropriate.  If you expand the corpus to 100 M tokens,
  consider `r=128`.

### Learning rate

`2e-4` is the standard LoRA LR and works well for CLM domain adaptation.
Do not use `2e-5` (the full fine-tuning LR used in the existing scripts) —
it is 10× too low for LoRA adapters and will result in very slow convergence.

### Multiple epochs vs one epoch

With only ~700 documents (1.44 M tokens), 3 epochs is reasonable.  If loss
is still decreasing at epoch 3, try 5 epochs.  Monitor eval_loss in
TensorBoard (`tensorboard --logdir models/lora/`).

### Eval_steps

The script computes `eval_steps = max(10, total_steps // 10)` dynamically.
This fixes the bug in the original CPT script where `eval_steps=500` with
132 total steps meant evaluation never ran.

### Target modules

All 7 linear projection layers are targeted by default:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
This covers both attention and FFN and works for both LLaMA 3.x and Qwen3.5
(they share identical projection layer names).

To override for a model family with different layer names, use `--target_modules`:

```bash
python training/lora_finetune.py \
    --model_path models/instruction/Qwen3.5-2B-trained-new-instruct \
    --data_dir   data/processed/lora/qwen3.5-2b \
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### Adapter size vs merge accuracy

A rank-64 adapter for a 7-allocation per layer × 32 layers produces about
115 M trainable parameters stored in ~230 MB (bfloat16).  After
`merge_and_unload()`, the merged model is the same size as the base model.

---

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'peft'`

```bash
pip install peft>=0.10.0 trl>=0.8.0
```

### `RuntimeError: Trying to backward through the graph a second time`

This happens when `use_reentrant=True` (PyTorch default) is combined with
PEFT LoRA and gradient checkpointing.  The script sets
`gradient_checkpointing_kwargs={"use_reentrant": False}` which should prevent
this.  If you are using a custom training setup, add this flag.

### `KeyError: 'validation'` when reusing CPT data

The CPT dataset (`prepare_data.py`) may have saved the split as `test` instead
of `validation`.  The `--reuse_cpt_data` path in `prepare_lora_data.py`
automatically remaps `test` → `validation`.

### VRAM OOM during 4-bit merge

`merge_and_unload()` temporarily holds both the quantized model (dequantized
to bfloat16) and the adapter in memory.  For an 8B model this peaks at ~35 GB.
If you OOM, reduce GPU utilization before running the merge:
```bash
# Free the model loaded in the dashboard System tab first, then:
python training/merge_lora_adapter.py --adapter_path ...
```

### Adapter training loss not decreasing

1. Check `eval_loss` in TensorBoard — ensure eval_steps are actually firing.
2. Verify the data was prepared correctly:
   ```bash
   python utils/count_tokens.py --data_dir data/processed/lora/llama-3.1-8b
   ```
3. Increase `--epochs` to 5 — with 1.44 M tokens, the model may need more passes.
4. Confirm the dataset actually reached `models/lora/` with non-zero
   `adapter_model.safetensors` size.
