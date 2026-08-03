# Instruction Fine-Tuning Guide

Teach a domain-adapted (or vanilla base) Llama, Qwen, or Gemma 3/4 model to **follow
instructions** using a chat prompt template (Open-Orca for Llama, ChatML for
Qwen, Gemma turns for Gemma) and a curated mix of public instruction datasets.

> **Prerequisites** — A downloaded base model (e.g. `models/base/Llama-3.1-8B-base`,
> `models/base/Qwen3.5-2B-Base`, `models/base/gemma-3-4b`, or
> `models/base/gemma-4-e4b`) or a continually pre-trained model
> (e.g. `models/pretrained/Llama-3.1-8B-trained-new`,
> `models/pretrained/Qwen3.5-2B-trained-new`,
> `models/pretrained/gemma-3-4b-trained-new`, or
> `models/pretrained/gemma-4-e4b-trained-new`).
> See [PRETRAINING_GUIDE.md](PRETRAINING_GUIDE.md) for the pre-training
> pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Instruction Fine-Tuning?](#why-instruction-fine-tuning)
3. [Dataset](#dataset)
4. [Prompt Template — Open-Orca](#prompt-template--open-orca)
5. [Pipeline — Step by Step](#pipeline--step-by-step)
6. [Script Reference](#script-reference)
7. [Training Hyperparameters](#training-hyperparameters)
8. [Testing & Evaluation](#testing--evaluation)
9. [Comparing Models](#comparing-models)
10. [Tips & Best Practices](#tips--best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BASE Model              ──►  Continual Pre-training  ──►  TRAINED Model │
│  (meta-llama/Llama-3.1-8B)    (data/raw/research_corpus_new.json)                │
│                                                                          │
│  BASE Model ─────────────┐                                               │
│  or TRAINED Model ───────┤──►  Instruction Fine-tuning  ──► INSTRUCT    │
│                           │    (Orca + Dolly mix)            Model       │
└──────────────────────────────────────────────────────────────────────────┘
```

Two instruction-tuned variants are produced per model family:

| Source Model | Output Model | Purpose |
|-------------|--------------|---------|
| `models/base/Llama-3.1-8B-base` | `models/instruction/Llama-3.1-8B-base-instruct` | General instruction following |
| `models/pretrained/Llama-3.1-8B-trained-new` | `models/instruction/Llama-3.1-8B-trained-new-instruct` | Domain-aware instruction following |
| `models/base/Qwen3.5-2B-Base` | `models/instruction/Qwen3.5-2B-Base-instruct` | General instruction following (Qwen) |
| `models/pretrained/Qwen3.5-2B-trained-new` | `models/instruction/Qwen3.5-2B-trained-new-instruct` | Domain-aware instruction following (Qwen) |
| `models/base/gemma-3-4b` | `models/instruction/gemma-3-4b-base-instruct` | General instruction following (Gemma 3) |
| `models/pretrained/gemma-3-4b-trained-new` | `models/instruction/gemma-3-4b-instruct` | Domain-aware instruction following (Gemma 3) |
| `models/base/gemma-4-e4b` | `models/instruction/gemma-4-e4b-instruct` | General instruction following (Gemma 4 E4B) |
| `models/pretrained/gemma-4-e4b-trained-new` | `models/instruction/gemma-4-e4b-trained-instruct` | Domain-aware instruction following (Gemma 4 E4B) |
| `models/base/gemma-4-e2b` | `models/instruction/gemma-4-e2b-instruct` | General instruction following (Gemma 4 E2B) |

The domain-trained variants are the more interesting ones: they combine domain
knowledge (from pre-training on the HPC/networking research corpus) with the
ability to follow user instructions.

---

## Why Instruction Fine-Tuning?

After continual pre-training the model *knows* the domain but still only
**completes text**.  It cannot answer questions, follow instructions, or
produce structured output.

| Capability | Base Model | Trained Model | Instruct Model |
|-----------|-----------|---------------|----------------|
| Domain knowledge | ❌ | ✅ | ✅ |
| Follow instructions | ❌ | ❌ | ✅ |
| Answer questions | ❌ | ❌ | ✅ |
| Structured output | ❌ | ❌ | ✅ |

---

## Dataset

We use a **mixed** instruction dataset downloaded automatically from
HuggingFace:

| Source | Samples | Strength |
|--------|---------|----------|
| [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | 10,000 | Reasoning, structured output (GPT-4 distillation) |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 15,000 | Diverse Q&A, creative writing, explanations |

**Total**: ~25,000 raw samples. For practical training we default to
**5,000 samples** (configurable via `--max_samples`).

### Why this mix?

1. **Orca** — teaches the model to reason step-by-step, follow complex
   prompts, and produce structured output (JSON, lists, comparisons)
2. **Dolly** — provides Q&A diversity across many categories (technical
   explanations, creative writing, summarisation, classification)
3. Together they give a well-rounded instruction-following capability
   without requiring domain-specific labelled data

### Schema normalisation

Both datasets are normalised to a unified `(system, question, response)`
schema before formatting:

| Field | Orca source | Dolly source |
|-------|-------------|-------------|
| `system` | `system_prompt` | `context` |
| `question` | `question` | `instruction` |
| `response` | `response` | `response` |

---

## Prompt Templates

The correct template is selected **automatically** based on the model family
(detected from the model path). You do not need to specify the template
manually.

### Llama Models — Open-Orca Template

#### With system prompt / context

```
### System:
{system_prompt or context}

### User:
{question or instruction}

### Assistant:
{response}<|end_of_text|>
```

#### Without system prompt

```
### System:
You are a helpful assistant.

### User:
{question or instruction}

### Assistant:
{response}<|end_of_text|>
```

#### Why Open-Orca (not Llama's built-in chat template)?

The Llama base models (`meta-llama/Llama-3.1-8B`) have `chat_template=None`.
They were not instruction-tuned by Meta.  The Open-Orca template provides
a simple, reliable chat structure that:

- Works with any base model (no special tokens required)
- Is easy to parse at inference time
- Is widely used in the open-source community

### Qwen Models — ChatML Template

#### With system prompt / context

```
<|im_start|>system
{system_prompt or context}<|im_end|>
<|im_start|>user
{question or instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

#### Without system prompt

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question or instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

#### Why ChatML for Qwen?

Qwen3.5 models natively use the ChatML format with `<|im_start|>` and
`<|im_end|>` special tokens.  Using this native format ensures the model's
tokenizer handles the template tokens correctly and produces the best results.

### Gemma 3 Models — Gemma Turn Template

Gemma 3 does not have a dedicated system turn.  The system prompt is folded
into the first user turn.

#### With system prompt / context

```
<start_of_turn>user
{system_prompt or context}

{question or instruction}<end_of_turn>
<start_of_turn>model
{response}<end_of_turn>
```

#### Without system prompt

```
<start_of_turn>user
{question or instruction}<end_of_turn>
<start_of_turn>model
{response}<end_of_turn>
```

#### Why this template for Gemma?

Gemma 3 models use `<start_of_turn>` / `<end_of_turn>` as their native turn
delimiters.  Since there is no system role in Gemma's format, the system
prompt is prepended to the first user message.

### EOS tokens and stop strings

> **EOS token in training data**: Each training sample ends with the model's
> EOS token appended after the assistant's response.  This teaches the model to
> **stop generating** after answering, rather than continuing with
> self-generated follow-up questions.

| Model Family | EOS Token | Inference Stop Strings |
|-------------|-----------|----------------------|
| Llama | `<\|end_of_text\|>` | `### User:`, `### System:`, `\n### ` |
| Qwen | `<\|im_end\|>` | `<\|im_start\|>`, `<\|im_end\|>` |
| Gemma | `<end_of_turn>` | `<start_of_turn>`, `<end_of_turn>` |

---

## Pipeline — Step by Step

### Quick start — Llama (copy-paste)

```bash
source .venv/bin/activate

# ── Step 1: Prepare instruction dataset ─────────────────────────
python training/prepare_instruction_data.py \
    --model_path models/base/Llama-3.1-8B-base \
    --output_dir data/instruction/llama-3.1-8b-base \
    --max_samples 5000

python training/prepare_instruction_data.py \
    --model_path models/pretrained/Llama-3.1-8B-trained-new \
    --output_dir data/instruction/llama-3.1-8b-trained-new \
    --max_samples 5000

# ── Step 2: Run instruction fine-tuning ─────────────────────────
python training/instruction_finetune.py \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/instruction/Llama-3.1-8B-base-instruct

python training/instruction_finetune.py \
    --model_path models/pretrained/Llama-3.1-8B-trained-new \
    --data_dir data/instruction/llama-3.1-8b-trained-new \
    --output_dir models/instruction/Llama-3.1-8B-trained-new-instruct

# ── Step 3: Test the instruction models ─────────────────────────
./test_instruct.sh
```

### Quick start — Qwen3.5 (copy-paste)

```bash
source .venv/bin/activate

# ── Step 1: Prepare instruction dataset (ChatML template auto-selected) ──
python training/prepare_instruction_data.py \
    --model_path models/pretrained/Qwen3.5-2B-trained-new \
    --output_dir data/instruction/qwen3.5-2b \
    --max_samples 5000

# ── Step 2: Run instruction fine-tuning ─────────────────────────
python training/instruction_finetune.py \
    --model_path models/pretrained/Qwen3.5-2B-trained-new \
    --data_dir data/instruction/qwen3.5-2b \
    --output_dir models/instruction/Qwen3.5-2B-trained-new-instruct

# ── Step 3: Test the instruction model ──────────────────────────
python inference/test_instruction_model.py \
    --model_path models/instruction/Qwen3.5-2B-trained-new-instruct
```

### Quick start — Gemma 3 (copy-paste)

```bash
source .venv/bin/activate

# ── Step 1: Prepare instruction dataset (Gemma template auto-selected) ──
python training/prepare_instruction_data.py \
    --model_path models/pretrained/gemma-3-4b-trained-new \
    --output_dir data/instruction/gemma-3-4b \
    --max_samples 5000

# ── Step 2: Run instruction fine-tuning ─────────────────────────
python training/instruction_finetune.py \
    --model_path models/pretrained/gemma-3-4b-trained-new \
    --data_dir data/instruction/gemma-3-4b \
    --output_dir models/instruction/gemma-3-4b-instruct

# ── Step 3: Test the instruction model ──────────────────────────
python inference/test_instruction_model.py \
    --model_path models/instruction/gemma-3-4b-instruct
```

### Quick start — Gemma 4 E4B / E2B (copy-paste)

Gemma 4 models use the same Gemma turn template as Gemma 3.  Gradient
checkpointing is automatically disabled (KV-shared architecture).

```bash
source .venv/bin/activate

# ── Step 1: Prepare instruction dataset ──
python training/prepare_instruction_data.py \
    --model_path models/pretrained/gemma-4-e4b-trained-new \
    --output_dir data/instruction/gemma-4-e4b \
    --max_samples 5000

# ── Step 2: Run instruction fine-tuning ──
python training/instruction_finetune.py \
    --model_path models/pretrained/gemma-4-e4b-trained-new \
    --data_dir data/instruction/gemma-4-e4b \
    --output_dir models/instruction/gemma-4-e4b-instruct

# ── Step 3: Test ──
python inference/test_instruction_model.py \
    --model_path models/instruction/gemma-4-e4b-instruct
```

### Detailed walkthrough

#### Step 1 — Prepare Instruction Data

```bash
python training/prepare_instruction_data.py \
    --model_path models/base/Llama-3.1-8B-base \
    --output_dir data/instruction/llama-3.1-8b-base \
    --max_samples 5000 \
    --max_length 512 \
    --val_ratio 0.05 \
    --seed 42
```

What this does:

1. Downloads **Open-Orca** (10k) and **Dolly** (15k) from HuggingFace
2. Normalises both datasets to `(system, question, response)` schema
3. Shuffles and trims to `--max_samples` (default: all ~25k)
4. Formats every sample using the Open-Orca chat template
5. Tokenizes with the target model's tokenizer
6. Splits into train (95%) and validation (5%)
7. Saves the tokenized Arrow dataset + tokenizer to `--output_dir`

Output structure:

```
data/instruction/llama-3.1-8b-base/
├── train/              ← Tokenized training split
├── validation/         ← Tokenized validation split
├── tokenizer/          ← Copy of the tokenizer
└── dataset_dict.json   ← HuggingFace dataset metadata
```

#### Step 2 — Instruction Fine-Tuning

```bash
python training/instruction_finetune.py \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/instruction/Llama-3.1-8B-base-instruct \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5
```

What this does:

1. Loads the source model with `device_map="auto"` (distributed across 2× GPUs)
2. Enables gradient checkpointing + bfloat16 mixed precision
3. Trains using HuggingFace `Trainer` with cosine LR schedule
4. Evaluates every 100 steps on the validation split
5. Saves checkpoints every 200 steps (keeps last 2)
6. Saves the final instruction-tuned model + tokenizer to `--output_dir`

> **Important**: The source model is **never modified**. A completely new
> model is written to the output directory.

#### Step 3 — Test Interactively

```bash
# Menu-driven launcher
./test_instruct.sh

# Or call the Python script directly
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct
```

---

## Script Reference

### `prepare_instruction_data.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | *required* | Path to the model whose tokenizer to use |
| `--output_dir` | *required* | Where to save the tokenized dataset |
| `--max_length` | `512` | Maximum token length per sample |
| `--max_samples` | all (~25k) | Limit the number of samples |
| `--val_ratio` | `0.05` | Fraction held out for validation |
| `--seed` | `42` | Random seed for reproducibility |

### `instruction_finetune.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | *required* | Path to source model (base or trained) |
| `--data_dir` | *required* | Tokenized dataset from `prepare_instruction_data.py` |
| `--output_dir` | *required* | Where to save the instruction-tuned model |
| `--num_epochs` | `3` | Number of training epochs |
| `--batch_size` | `2` | Per-device batch size |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `--learning_rate` | `2e-5` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_ratio` | `0.1` | Warmup ratio |
| `--max_grad_norm` | `1.0` | Maximum gradient norm for clipping |
| `--bf16` | `True` | Use bfloat16 mixed precision |
| `--gradient_checkpointing` | `True` | Enable gradient checkpointing |
| `--logging_steps` | `10` | Log every N steps |
| `--save_steps` | `200` | Checkpoint every N steps |
| `--eval_steps` | `100` | Evaluate every N steps |
| `--save_total_limit` | `2` | Max checkpoints to keep |

### `test_instruction_model.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | *required* | One or more model paths |
| `--instruction` | — | Single-shot instruction (omit for interactive REPL) |
| `--max_new_tokens` | `256` | Maximum generation length |
| `--temperature` | `0.7` | Sampling temperature |

### `test_instruct.sh`

Menu-driven shell wrapper around `test_instruction_model.py`.  Discovers all
models in `models/instruction/` and offers a numbered selection menu, plus a
"Compare ALL" option.

---

## Training Hyperparameters

### Default configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | Standard for instruction FT at this data scale |
| Per-device batch size | 2 | Fits 8B model on 48 GB GPU with checkpointing |
| Gradient accumulation | 8 | Effective batch size = 2 × 8 × 2 GPUs = **32** |
| Learning rate | 2e-5 | Conservative; prevents catastrophic forgetting |
| LR scheduler | Cosine | Smooth decay to near-zero |
| Warmup ratio | 0.1 | 10% of total steps |
| Weight decay | 0.01 | Light regularisation |
| Max grad norm | 1.0 | Gradient clipping for stability |
| Precision | bfloat16 | Saves VRAM, native on A6000 |
| Optimizer | AdamW (PyTorch) | Standard choice |

### Effective batch size calculation

```
effective_batch = per_device_batch × grad_accum × num_gpus
                = 2 × 8 × 2
                = 32 samples per optimiser step
```

### Estimated training time (5,000 samples, 3 epochs)

| Source Model | Approx. Time | Hardware |
|--------------|--------------|----------|
| Llama-3.1-8B base | ~98 min | 2× RTX A6000 |
| Llama-3.1-8B trained-new | ~98 min | 2× RTX A6000 |

---

## Testing & Evaluation

### Interactive REPL

```bash
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct
```

**REPL commands:**

| Command | Action |
|---------|--------|
| `:q` / `:quit` | Exit |
| `:temp 0.5` | Change temperature |
| `:tokens 512` | Change max new tokens |
| `:help` | Show help |

**Multiline input:**

```
📝  Instruction: <<<
... You are a network tuning assistant.
... Predict the optimal number of threads for this transfer:
... source: ANL, destination: NERSC, file size: 10 GB
... >>>
```

### Single-shot mode

```bash
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-trained-new-instruct \
    --instruction "Explain what GridFTP is and why it matters for large-scale data transfer."
```

### Sample test instructions for domain evaluation

```
Explain what GridFTP is and why it matters for large-scale data transfer.
What is the Network Weather Service and how does it predict network performance?
Describe data-aware scheduling in distributed systems.
How does parallel stream transfer improve throughput over wide-area networks?
Compare ECMP and adaptive routing for LLM training cluster networks.
What is the dual-ToR design in HPN and why does it improve fault tolerance?
Explain how ZeRO partitioning reduces memory consumption during LLM training.
```

---

## Comparing Models

### Side-by-side comparison (same prompt, multiple models)

```bash
python inference/test_instruction_model.py \
    --model_path models/instruction/Llama-3.1-8B-base-instruct \
                 models/instruction/Llama-3.1-8B-trained-new-instruct \
    --instruction "What is the Network Weather Service?"
```

### What to look for

| Aspect | Base → Instruct | Trained-New → Instruct |
|--------|----------------|----------------------|
| General Q&A | ✅ Good | ✅ Good |
| Domain terms (GridFTP, NWS, ECMP) | ❌ Vague / hallucinated | ✅ Accurate |
| Technical depth | Shallow | Detailed, cites concepts from corpus |
| Structured output | ✅ Follows format | ✅ Follows format |

The **trained-new → instruct** model should demonstrate noticeably better
domain-specific knowledge because it was first pre-trained on the research
corpus before being instruction-tuned.

---

## Tips & Best Practices

### Data quality matters more than quantity

5,000 well-formatted samples typically outperform 25,000 noisy ones.  Start
with `--max_samples 5000` and increase only if evaluation shows the model
needs more diversity.

### Tokenizer consistency

Always use the **same tokenizer** for data preparation and training.  The
`prepare_instruction_data.py` script ensures this by loading the tokenizer
from the target model path.

### Don't skip validation

The 5% validation split enables early detection of overfitting.  Watch the
eval loss in the training logs — if it starts increasing while train loss
keeps dropping, you're overfitting.

### Prompt format must match at inference

The inference prompt must **exactly match** the training template.  The
`test_instruction_model.py` script handles this automatically — it detects
the model family and selects the correct template (Open-Orca for Llama,
ChatML for Qwen).

### Stop strings prevent run-on generation

At inference time, model-family-aware stop strings are applied to halt
generation if the model tries to continue past its answer:
- **Llama**: `### User:`, `### System:`, `\n### `
- **Qwen**: `<|im_start|>`, `<|im_end|>`
- **Gemma**: `<start_of_turn>`, `<end_of_turn>`

Combined with the EOS token in training data, this ensures clean single-answer
output.

### Resume from checkpoint

If training is interrupted, you can resume from the latest checkpoint:

```bash
python training/instruction_finetune.py \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/instruction/Llama-3.1-8B-base-instruct
# The Trainer will automatically detect checkpoints in output_dir/checkpoints/
```

### Profiling instruction fine-tuning

Use `profile_training.py instruct` instead of `instruction_finetune.py` to
capture detailed GPU/memory/throughput metrics alongside training:

```bash
python profiling/profile_training.py instruct \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --max_steps 50
```

See [PROFILING_GUIDE.md](PROFILING_GUIDE.md) for full profiling documentation.

---

## Troubleshooting

### CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix**: Reduce batch size and/or increase gradient accumulation:

```bash
python training/instruction_finetune.py \
    --model_path models/base/Llama-3.1-8B-base \
    --data_dir data/instruction/llama-3.1-8b-base \
    --output_dir models/instruction/Llama-3.1-8B-base-instruct \
    --batch_size 1 \
    --gradient_accumulation_steps 16
```

### "Data not found" error

```
Data not found: data/instruction/llama-3.1-8b-base
Run prepare_instruction_data.py first.
```

**Fix**: Run the data preparation step first:

```bash
python training/prepare_instruction_data.py \
    --model_path models/base/Llama-3.1-8B-base \
    --output_dir data/instruction/llama-3.1-8b-base
```

### Model generates garbage / repeated tokens

This usually means the prompt template at inference doesn't match the training
template.  Ensure you use `test_instruction_model.py` (which embeds the
correct Open-Orca template) rather than raw `model.generate()`.

### Model doesn't use domain knowledge

If the instruct model doesn't show domain-specific knowledge, you're likely
testing the **base → instruct** variant.  Use the **trained-new → instruct**
variant instead — it was pre-trained on the research corpus first.

### Eval loss spikes

Small spikes are normal.  Sustained increase means overfitting — reduce
`--num_epochs` to 2 or increase `--max_samples`.

### HuggingFace download errors

The data preparation script downloads from HuggingFace Hub.  Ensure you're
authenticated:

```bash
python utils/setup_token.py
# or
huggingface-cli login
```

### Transformers 5.0 breaking changes

These have already been fixed in the codebase:

- `evaluation_strategy` → `eval_strategy`

> **Note**: The correct parameter for `AutoModelForCausalLM.from_pretrained()`
> is `torch_dtype` (not `dtype`).  This has always been the case.
