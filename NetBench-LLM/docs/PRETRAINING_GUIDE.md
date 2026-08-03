# Continual Pre-training Guide

> For instruction fine-tuning, see
> [INSTRUCTION_FINETUNING_GUIDE.md](INSTRUCTION_FINETUNING_GUIDE.md).

## Table of Contents
1. [Understanding Continual Pre-training](#understanding-continual-pre-training)
2. [Understanding Instruction Fine-tuning](#understanding-instruction-fine-tuning)
3. [BASE Models vs Instruct Models](#base-models-vs-instruct-models)
4. [Hardware Setup](#hardware-setup)
5. [Pre-training Workflow](#pre-training-workflow)
6. [Instruction Fine-tuning Workflow](#instruction-fine-tuning-workflow)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Troubleshooting](#troubleshooting)

---

## Understanding Continual Pre-training

### What is Continual Pre-training?

Continual pre-training (domain-adaptive pre-training) continues the causal
language modelling objective on domain-specific text.  Unlike fine-tuning:

| Aspect | Pre-training | Fine-tuning |
|--------|-------------|-------------|
| **Objective** | Next-token prediction (causal LM) | Task-specific |
| **Data Format** | Raw text (JSON corpus) | Instruction / response pairs |
| **Goal** | Inject domain knowledge | Teach specific behaviour |
| **Learning Rate** | Lower (1e-5 – 5e-5) | Similar or slightly higher |

### Our Setup

- Corpus: `data/raw/research_corpus_new.json` — **753 documents, ~9.5 M tokens** (or `research_corpus_v3.json` for the v3 variant)
- Tokenised data: `data/processed/llama-3.1-8b/`, `data/processed/llama-3.2-1b/`, `data/processed/gemma-3-4b/`, etc.
- Max sequence length: **2 048 tokens**

---

## Understanding Instruction Fine-tuning

After continual pre-training, the model "knows" the domain but still only
completes text.  Instruction fine-tuning teaches it to **follow instructions**
in an Open-Orca chat-style prompt format.

### Dataset

We use a **mixed** instruction dataset downloaded from HuggingFace:

| Source | Samples | Strength |
|--------|---------|----------|
| Open-Orca/OpenOrca | 10 000 | Reasoning, structured output (GPT-4 distillation) |
| databricks/databricks-dolly-15k | 15 000 | Diverse Q&A, creative writing, explanations |

For practical training we trim to **5 000 samples** (configurable via `--max_samples`).

### Prompt Template

The Open-Orca chat template is used (NOT the Llama 3.1 chat template,
because the base model has `chat_template=None`):

```
### System:
{system_prompt or context}

### User:
{question}

### Assistant:
{response}
```

---

## BASE Models vs Instruct Models

### Critical Understanding

This pipeline starts from **BASE** models:

- `meta-llama/Llama-3.2-1B` (BASE)
- `meta-llama/Llama-3.1-8B` (BASE)
- `Qwen/Qwen3.5-2B-Base` (BASE)
- `Qwen/Qwen3.5-4B-Base` (BASE)
- `google/gemma-3-4b-pt` (BASE, Gemma 3)
- `google/gemma-4-E4B` (BASE, Gemma 4 — KV-shared, no gradient checkpointing)
- `google/gemma-4-E2B` (BASE, Gemma 4 — KV-shared, no gradient checkpointing)

NOT instruct/chat variants:
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

### How to Prompt BASE Models

```python
# ❌ WRONG – instruction-style
"What is GridFTP?"

# ✅ CORRECT – completion-style
"In the context of grid computing, GridFTP provides"
```

Use `chat_interactive.py` or `./chat.sh` for interactive base-model testing.

### How to Prompt Instruction-Tuned Models

**Llama models** (Open-Orca template):

```
### System:
You are a helpful assistant.

### User:
Explain what GridFTP is and why it matters for large-scale data transfer.

### Assistant:
```

**Qwen3.5 models** (ChatML template):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain what GridFTP is and why it matters for large-scale data transfer.<|im_end|>
<|im_start|>assistant
```

**Gemma 3 models** (Gemma turn template — system prompt folded into first user turn):

```
<start_of_turn>user
You are a helpful assistant.

Explain what GridFTP is and why it matters for large-scale data transfer.<end_of_turn>
<start_of_turn>model
```

The correct template is selected automatically based on the model path.
Use `test_instruction_model.py` or `./test_instruct.sh` for interactive
instruction-model testing.

---

## Hardware Setup

| Resource | Specification |
|----------|---------------|
| GPUs | 2× NVIDIA RTX A6000 |
| VRAM per GPU | 48 GB |
| Total VRAM | 96 GB |
| System CUDA | 11.8 |
| PyTorch CUDA | 12.8 |

> **Important**: Because the system CUDA toolkit (11.8) doesn't match
> the PyTorch CUDA build (12.8), **DeepSpeed cannot compile its CPU
> offload kernels**.  All training therefore uses `device_map="auto"`,
> which distributes the model across both GPUs automatically.

---

## Pre-training Workflow

### Llama Pipeline (shortcut)

```bash
# 1. Download base model
python utils/download_models.py --model 8b

# 2. Prepare tokenised data
python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name meta-llama/Llama-3.1-8B \
    --output_dir data/processed/llama-3.1-8b

# 3. Run continual pre-training
python training/pretrain_transformers.py \
    --model 8b \
    --data_dir data/processed/llama-3.1-8b \
    --output_dir models/pretrained/Llama-3.1-8B-trained-new

# 4. Evaluate trained model
python evaluation/evaluate_model.py -m models/pretrained/Llama-3.1-8B-trained-new

# 5. Compare base vs trained
python evaluation/compare_models.py --model 8b
```

### Qwen3.5 Pipeline (shortcut)

```bash
# 1. Download base model
python utils/download_models.py --model qwen-2b

# 2. Prepare tokenised data
python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name Qwen/Qwen3.5-2B-Base \
    --output_dir data/processed/qwen3.5-2b

# 3. Run continual pre-training
python training/pretrain_transformers.py \
    --model qwen-2b \
    --data_dir data/processed/qwen3.5-2b \
    --output_dir models/pretrained/Qwen3.5-2B-trained-new
```

### Gemma 3 Pipeline (shortcut)

```bash
# 1. Download base model
python utils/download_models.py --model gemma-4b

# 2. Prepare tokenised data
python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name google/gemma-3-4b-pt \
    --output_dir data/processed/gemma-3-4b

# 3. Run continual pre-training
python training/pretrain_transformers.py \
    --model gemma-4b \
    --data_dir data/processed/gemma-3-4b \
    --output_dir models/pretrained/gemma-3-4b-trained-new
```

### Gemma 4 Pipeline (KV-shared models)

Gemma 4 models (`gemma-e4b`, `gemma-e2b`) use KV-layer sharing.  The training
script automatically detects this and **disables gradient checkpointing** while
keeping `use_cache=True` (required for correct KV-sharing).

```bash
# Gemma 4 E4B (~8.6B params)
python utils/download_models.py --model gemma-e4b

python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name google/gemma-4-E4B \
    --output_dir data/processed/gemma-4-e4b

python training/pretrain_transformers.py \
    --model gemma-e4b \
    --data_dir data/processed/gemma-4-e4b \
    --output_dir models/pretrained/gemma-4-e4b-trained-new

# Gemma 4 E2B (~2.6B params)
python utils/download_models.py --model gemma-e2b

python training/prepare_data.py \
    --input_file data/raw/research_corpus_new.json \
    --model_name google/gemma-4-E2B \
    --output_dir data/processed/gemma-4-e2b

python training/pretrain_transformers.py \
    --model gemma-e2b \
    --data_dir data/processed/gemma-4-e2b \
    --output_dir models/pretrained/gemma-4-e2b-trained-new
```

> **Gemma 4 note**: `token_type_ids` and `mm_token_type_ids` are injected
> automatically by the training collator.  No extra flags are needed.

### Escape Hatch — `--model_path` for Any Model

When `--model_path` is provided, the shortcut dict (`--model 1b/8b/qwen-2b/qwen-4b/gemma-4b/gemma-e4b/gemma-e2b`)
is bypassed entirely.  This allows continual pre-training for **any** model
without modifying the registry:

```bash
python training/pretrain_transformers.py \
    --model_path models/base/Qwen3.5-2B-Base \
    --data_dir   data/processed/qwen3.5-2b \
    --output_dir models/pretrained/Qwen3.5-2B-trained-new
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | — | Direct path to any base model directory. Bypasses `--model`. |
| `--data_dir` | auto-detected | Directory containing the tokenised dataset. |
| `--output_dir` | auto-derived | Where to save the trained model. |

> When `--model_path` is used, `--data_dir` is **required** (no auto-detection).
> When `--model` is used, the data directory is auto-detected from a set of
> known paths.

### Estimated Training Time

| Model | Epochs | Approx. Time |
|-------|--------|---------------|
| Llama-3.2-1B | 3 | ~3–4 min |
| Llama-3.1-8B | 3 | ~80–90 min |
| Qwen3.5-2B | 3 | ~5–8 min |
| Qwen3.5-4B | 3 | ~15–25 min |
| Gemma-3-4B | 3 | ~15–25 min |
| Gemma-4-E2B | 3 | ~10–15 min |
| Gemma-4-E4B | 3 | ~25–40 min |

*(Times on 2× RTX A6000 with `device_map="auto"`.)*

---

## Instruction Fine-tuning Workflow

### Complete Pipeline

```bash
# 1. Prepare instruction dataset (downloads from HuggingFace)
python training/prepare_instruction_data.py \
    --model_path models/base/Llama-3.1-8B-base \
    --output_dir data/instruction/llama-3.1-8b-base \
    --max_samples 5000

# 2. Fine-tune base model
python training/instruction_finetune.py \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/instruction/Llama-3.1-8B-base-instruct

# 3. (Optional) Fine-tune the domain-trained model too
python training/prepare_instruction_data.py \
    --model_path models/pretrained/Llama-3.1-8B-trained-new \
    --output_dir data/instruction/llama-3.1-8b-trained-new \
    --max_samples 5000

python training/instruction_finetune.py \
    --model_path models/pretrained/Llama-3.1-8B-trained-new \
    --data_dir data/instruction/llama-3.1-8b-trained-new \
    --output_dir models/instruction/Llama-3.1-8B-trained-new-instruct

# 4. Test interactively
./test_instruct.sh
```

### Estimated Fine-tuning Time (5 000 samples, 3 epochs)

| Source Model | Approx. Time |
|--------------|---------------|
| 8B base | ~98 min |
| 8B trained-new | ~98 min |

---

### Evaluation Strategy

See [`docs/QA_BENCHMARK_GUIDE.md`](QA_BENCHMARK_GUIDE.md) for the full HPN Q&A Benchmark.

### LM Adaptation Report

After CPT, measure domain perplexity improvement and general-capability forgetting:

```bash
python evaluation/lm_adaptation_report.py \
    --base_model_path    models/base/Qwen3.5-2B-Base \
    --trained_model_path models/pretrained/Qwen3.5-2B-trained-new
# → outputs/evaluations/adaptation_reports/lm_adaptation_*.{json,xlsx}
```

The report measures:
- **HPN domain perplexity** before/after CPT (should decrease)
- **WikiText-2 general perplexity** before/after CPT (forgetting check)
- **Forgetting Gap** = `GeneralLossAfter − GeneralLossBefore` (positive = some forgetting)

### Completion-Based Evaluation (Base / Trained Models)

```bash
# Evaluate base model
python evaluation/evaluate_model.py -m models/base/Llama-3.1-8B-base

# Evaluate trained model
python evaluation/evaluate_model.py -m models/pretrained/Llama-3.1-8B-trained-new

# Side-by-side comparison
python evaluation/compare_models.py --model 8b
```

### What to Look For

**Before training**: Generic completions — doesn't know specific terms
(GridFTP, NWS, data-aware scheduling, etc.)

**After training**: More accurate domain completions, correct use of
technical terminology, references to specific systems/papers.

### Instruction Model Testing

```bash
# Interactive REPL
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct

# Single instruction
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
    --instruction "Explain what GridFTP is."

# Compare both instruction models
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
                 models/instruction/Llama-3.1-8B-trained-new-instruct \
    --instruction "What is the Network Weather Service?"
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--per_device_train_batch_size 1`
- Increase gradient accumulation steps
- Ensure `gradient_checkpointing` is enabled (default: `True`)

### Model Download Issues
```bash
huggingface-cli login
# Also accept the model license at https://huggingface.co/meta-llama
```

### DeepSpeed Errors
DeepSpeed CPU-offload kernels require the **system CUDA toolkit** version
to match PyTorch's CUDA version.  If they differ (as in our setup:
system 11.8 vs PyTorch 12.8), use `device_map="auto"` instead
and do **not** pass `--deepspeed`.

### Transformers 5.0 Breaking Changes
- `evaluation_strategy` → `eval_strategy`
- These have already been updated in the codebase.

> **Note**: The correct parameter for `AutoModelForCausalLM.from_pretrained()`
> is `torch_dtype` (not `dtype`).  This has always been the case.
