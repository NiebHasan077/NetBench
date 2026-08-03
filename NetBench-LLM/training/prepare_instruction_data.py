#!/usr/bin/env python3
"""
Download and prepare mixed instruction dataset (Orca + Dolly) from HuggingFace
for instruction fine-tuning Llama 3.1 8B models.

Strategy:
  - Open-Orca/OpenOrca (10k samples): Complex reasoning, structured outputs, JSON formatting
  - Databricks Dolly (15k samples): Diverse Q&A, creative writing, technical explanations

Both datasets are normalized to a unified schema (system, question, response)
and formatted using the Open-Orca chat template:

    ### System:
    {system_prompt or context}

    ### User:
    {question or instruction}

    ### Assistant:
    {response}

This combination provides:
  1. Strong reasoning capabilities (from Orca's GPT-4 distillation)
  2. Structured output format adherence (JSON, specific formats)
  3. Diverse Q&A coverage (technical questions, explanations)
  4. Concise, task-focused responses

Total: ~25k high-quality instruction samples optimized for:
  - Technical Q&A (e.g., "What is pipelining in data transfer?")
  - Complex prompts with specific output formats (e.g., JSON responses)
  - Step-by-step reasoning tasks

Usage:
    # Prepare for 8B base model
    python prepare_instruction_data.py \
        --model_path models/base/Llama-3.1-8B-base \
        --output_dir data/instruction/llama-3.1-8b-base-mixed

    # Prepare for 8B trained-new model
    python prepare_instruction_data.py \
        --model_path models/pretrained/Llama-3.1-8B-trained-new \
        --output_dir data/instruction/llama-3.1-8b-trained-new-mixed

    # Use fewer samples for quick testing
    python prepare_instruction_data.py \
        --model_path models/base/Llama-3.1-8B-base \
        --output_dir data/instruction/llama-3.1-8b-base-mixed \
        --max_samples 5000
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth
from utils.model_utils import detect_model_family, get_eos_token_str, fix_tokenizer_padding

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─── Chat templates ────────────────────────────────────────────────────────────
#
# Open-Orca template (Llama 3.x models):
#   ### System: / ### User: / ### Assistant:
#   Ends with the model's EOS token so the model learns to stop after the
#   assistant turn.  Without EOS the model continues in base-model style —
#   self-asking follow-up questions and repeating content.
#
# ChatML template (Qwen3.5 models):
#   <|im_start|>system / <|im_start|>user / <|im_start|>assistant
#   Ends with <|im_end|> which is Qwen's native stop token in ChatML format.
#
# Gemma template (Gemma 3 models):
#   <start_of_turn>user / <start_of_turn>model
#   Ends with <end_of_turn> which is Gemma's native stop token.
#   Gemma 3 folds the system prompt into the first user turn.

# ── Llama / Open-Orca templates ──────────────────────────────────────────────
ORCA_TEMPLATE_WITH_SYSTEM = (
    "### System:\n{system}\n\n"
    "### User:\n{question}\n\n"
    "### Assistant:\n{response}{eos}"
)

ORCA_TEMPLATE_NO_SYSTEM = (
    "### System:\nYou are a helpful assistant.\n\n"
    "### User:\n{question}\n\n"
    "### Assistant:\n{response}{eos}"
)

# ── Qwen / ChatML templates ──────────────────────────────────────────────────
CHATML_TEMPLATE_WITH_SYSTEM = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n{response}<|im_end|>"
)

CHATML_TEMPLATE_NO_SYSTEM = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n{response}<|im_end|>"
)

# ── Gemma templates ─────────────────────────────────────────────────────
# Gemma 3 folds the system prompt into the first user turn.
GEMMA_TEMPLATE_WITH_SYSTEM = (
    "<start_of_turn>user\n{system}\n\n{question}<end_of_turn>\n"
    "<start_of_turn>model\n{response}<end_of_turn>"
)

GEMMA_TEMPLATE_NO_SYSTEM = (
    "<start_of_turn>user\n{question}<end_of_turn>\n"
    "<start_of_turn>model\n{response}<end_of_turn>"
)


def format_sample(sample: dict, model_family: str = "llama", eos: str = "<|end_of_text|>") -> str:
    """
    Format a single sample into the appropriate chat template.

    For Llama models: Open-Orca template (### System / ### User / ### Assistant)
    For Qwen models:  ChatML template (<|im_start|> / <|im_end|>)
    For Gemma models: Gemma template (<start_of_turn> / <end_of_turn>)

    Args:
        sample:       dict with keys 'system', 'question', 'response'
        model_family: 'llama', 'qwen', or 'gemma' (from detect_model_family)
        eos:          EOS token string appended at the end (Llama only; Qwen
                      uses <|im_end|> and Gemma uses <end_of_turn> inline)
    """
    question = sample.get("question", "").strip()
    system = sample.get("system", "").strip()
    response = sample.get("response", "").strip()

    if not question or not response:
        return ""

    if model_family == "qwen":
        if system:
            return CHATML_TEMPLATE_WITH_SYSTEM.format(
                system=system, question=question, response=response,
            )
        else:
            return CHATML_TEMPLATE_NO_SYSTEM.format(
                question=question, response=response,
            )
    elif model_family == "gemma":
        if system:
            return GEMMA_TEMPLATE_WITH_SYSTEM.format(
                system=system, question=question, response=response,
            )
        else:
            return GEMMA_TEMPLATE_NO_SYSTEM.format(
                question=question, response=response,
            )
    else:
        if system:
            return ORCA_TEMPLATE_WITH_SYSTEM.format(
                system=system, question=question, response=response, eos=eos,
            )
        else:
            return ORCA_TEMPLATE_NO_SYSTEM.format(
                question=question, response=response, eos=eos,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed instruction dataset (Orca + Dolly) for Llama fine-tuning"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model whose tokenizer to use "
             "(e.g. models/base/Llama-3.1-8B-base)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the tokenized dataset",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with pre-split JSONL files (train.jsonl + validation.jsonl). "
             "Skips HuggingFace download when provided.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum token length per sample (default: 2048)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples (default: use all ~25k from Orca+Dolly mix)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # ── authenticate (needed if tokenizer fetches from hub) ────────────
    setup_hf_auth()

    # ── validate paths ─────────────────────────────────────────────────
    if not Path(args.model_path).exists():
        logger.error(f"Model not found at: {args.model_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── detect model family to select the right template and EOS token ────
    model_family = detect_model_family(args.model_path)
    logger.info(f"Detected model family: {model_family}")

    # ── load tokenizer from the target model ───────────────────────────
    logger.info(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    fix_tokenizer_padding(tokenizer, model_family)

    eos_str = get_eos_token_str(tokenizer, model_family)
    logger.info(f"Using EOS/stop token: {eos_str!r}")

    # ── load dataset ───────────────────────────────────────────────────────
    _has_presplit = False

    if args.input_dir:
        # ── local JSONL path (pre-curated dataset with train/val split) ──
        train_jsonl = os.path.join(args.input_dir, "train.jsonl")
        val_jsonl = os.path.join(args.input_dir, "validation.jsonl")
        if not Path(train_jsonl).exists():
            logger.error(f"train.jsonl not found in {args.input_dir}")
            sys.exit(1)
        if not Path(val_jsonl).exists():
            logger.error(f"validation.jsonl not found in {args.input_dir}")
            sys.exit(1)
        logger.info(f"Loading local JSONL dataset from {args.input_dir}...")
        _train_ds = load_dataset("json", data_files=train_jsonl, split="train")
        _val_ds = load_dataset("json", data_files=val_jsonl, split="train")
        logger.info(f"Loaded {len(_train_ds):,} train + {len(_val_ds):,} validation samples")
        _has_presplit = True
    else:
        # ── HuggingFace download path (Orca + Dolly mix) ───────────────
        logger.info("Downloading mixed dataset (Orca + Dolly) from HuggingFace...")
        from datasets import concatenate_datasets

        logger.info("  Loading Open-Orca/OpenOrca (reasoning subset)...")
        orca_dataset = load_dataset("Open-Orca/OpenOrca", split="train[:10000]")
        logger.info("  Loading databricks/databricks-dolly-15k (Q&A diversity)...")
        dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

        def normalize_orca(sample):
            return {
                "system": (sample.get("system_prompt", "") or "")[:500],
                "question": sample.get("question", ""),
                "response": sample.get("response", ""),
                "category": "orca_reasoning",
            }

        def normalize_dolly(sample):
            context = (sample.get("context", "") or "").strip()
            return {
                "system": context,
                "question": sample.get("instruction", ""),
                "response": sample.get("response", ""),
                "category": sample.get("category", "dolly"),
            }

        orca_normalized = orca_dataset.map(normalize_orca, remove_columns=orca_dataset.column_names)
        dolly_normalized = dolly_dataset.map(normalize_dolly, remove_columns=dolly_dataset.column_names)

        dataset = concatenate_datasets([orca_normalized, dolly_normalized])
        dataset = dataset.shuffle(seed=args.seed)
        logger.info(f"Combined: {len(dataset)} samples (Orca: {len(orca_dataset)}, Dolly: {len(dolly_dataset)})")

        if args.max_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(
                range(min(args.max_samples, len(dataset)))
            )
            logger.info(f"Trimmed to {len(dataset)} samples")

    # ── format every sample into the chat template ─────────────────────
    _template_names = {"qwen": "ChatML", "gemma": "Gemma", "llama": "Open-Orca"}
    logger.info(f"Formatting samples with {_template_names.get(model_family, 'Open-Orca')} template...")

    def add_formatted_text(sample):
        formatted = format_sample(sample, model_family=model_family, eos=eos_str)
        if formatted:  # Only keep non-empty samples
            sample["text"] = formatted
            return sample
        return None

    # ── tokenize (no padding — dynamic padding handled at train time) ──
    logger.info(f"Tokenizing (max_length={args.max_length}, dynamic padding)...")

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_attention_mask=True,
        )
        tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        return tokens

    if _has_presplit:
        # ── pre-split JSONL: format + tokenize each split separately ───
        _train_ds = _train_ds.map(add_formatted_text, desc="Formatting train")
        _train_ds = _train_ds.filter(lambda x: x.get("text") is not None)
        _val_ds = _val_ds.map(add_formatted_text, desc="Formatting validation")
        _val_ds = _val_ds.filter(lambda x: x.get("text") is not None)

        train_tok = _train_ds.map(
            tokenize, batched=True,
            remove_columns=_train_ds.column_names,
            desc="Tokenizing train", num_proc=4,
        )
        val_tok = _val_ds.map(
            tokenize, batched=True,
            remove_columns=_val_ds.column_names,
            desc="Tokenizing validation", num_proc=4,
        )
        split = DatasetDict({"train": train_tok, "validation": val_tok})
    else:
        # ── HF path: format, tokenize, then split ─────────────────────
        dataset = dataset.map(add_formatted_text, desc="Formatting")
        dataset = dataset.filter(lambda x: x.get("text") is not None)
        tokenized = dataset.map(
            tokenize, batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing", num_proc=4,
        )
        split = tokenized.train_test_split(
            test_size=args.val_ratio, seed=args.seed
        )
        split["validation"] = split.pop("test")

    # ── token statistics ───────────────────────────────────────────────
    train_tokens = sum(len(ids) for ids in split["train"]["input_ids"])
    val_tokens = sum(len(ids) for ids in split["validation"]["input_ids"])

    logger.info("=" * 60)
    logger.info("Dataset Statistics:")
    logger.info("=" * 60)
    logger.info(f"Train samples:      {len(split['train']):,}")
    logger.info(f"Validation samples: {len(split['validation']):,}")
    logger.info(f"Train tokens:       {train_tokens:,}")
    logger.info(f"Validation tokens:  {val_tokens:,}")
    logger.info(f"Total tokens:       {train_tokens + val_tokens:,}")
    logger.info(f"Max length:         {args.max_length}")
    logger.info(f"Tokenizer:          {args.model_path}")
    logger.info("=" * 60)

    # ── save tokenizer alongside the data ──────────────────────────────
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)

    # ── save Arrow dataset ─────────────────────────────────────────────
    logger.info(f"Saving tokenized dataset to: {args.output_dir}")
    split.save_to_disk(args.output_dir)

    logger.info("")
    logger.info("✅ Instruction dataset ready!")
    logger.info(f"   {args.output_dir}/")
    logger.info(f"   ├── train/")
    logger.info(f"   ├── validation/")
    logger.info(f"   └── tokenizer/")
    logger.info("")


if __name__ == "__main__":
    main()
