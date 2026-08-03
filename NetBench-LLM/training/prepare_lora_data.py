#!/usr/bin/env python3
"""
Prepare the HPN research corpus for LoRA adapter training.

Reads data/raw/research_corpus_new.json and tokenizes the raw text into
fixed-length blocks (default 2048 tokens) using the causal-LM packing
strategy — identical to the pre-training data pipeline.  The resulting
Arrow dataset is saved to data/processed/lora/ and can be used directly
by training/lora_finetune.py.

If an existing CPT-processed dataset is available and you want to reuse
it, pass --reuse_cpt_data <path> to skip re-tokenization.

Usage:
    # Prepare from scratch (tokenizer from a local instruct model)
    python training/prepare_lora_data.py \\
        --model_path models/instruction/Llama-3.1-8B-base-instruct \\
        --output_dir data/processed/lora/llama-3.1-8b

    # Reuse an existing CPT processed dataset
    python training/prepare_lora_data.py \\
        --model_path models/instruction/Llama-3.1-8B-base-instruct \\
        --output_dir data/processed/lora/llama-3.1-8b \\
        --reuse_cpt_data data/processed/llama-3.1-8b
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CORPUS = str(_ROOT / "data" / "raw" / "research_corpus_new.json")
_DEFAULT_OUTPUT = str(_ROOT / "data" / "processed" / "lora")


def load_corpus(file_path: str) -> list[str]:
    """Load and extract text strings from a research corpus JSON or JSONL file."""
    logger.info(f"Loading corpus from: {file_path}")
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
            if isinstance(data, dict):
                for key in ("documents", "records", "data", "questions"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    data = [data]
    texts = []
    for item in tqdm(data, desc="Extracting texts"):
        if isinstance(item, dict) and "text" in item:
            t = item["text"].strip()
            if t:
                texts.append(t)
        elif isinstance(item, dict) and "content" in item:
            t = str(item["content"]).strip()
            if t:
                texts.append(t)
        elif isinstance(item, str) and item.strip():
            texts.append(item.strip())
    logger.info(f"Extracted {len(texts):,} non-empty text documents")
    return texts


def tokenize_and_pack(
    texts: list[str],
    tokenizer,
    max_length: int,
    val_ratio: float,
    seed: int,
) -> DatasetDict:
    """
    Tokenize texts and pack them into fixed-length blocks for CLM training.

    Strategy:
      1. Tokenize each document (no padding, truncation off)
      2. Append EOS token to every document — teaches the model that a sequence
         ends at EOS so it doesn't continue across document boundaries or
         repeat itself during generation
      3. Concatenate all token ids
      4. Slice into non-overlapping blocks of max_length tokens
      5. Create labels = input_ids (standard CLM objective)
    """
    dataset = Dataset.from_dict({"text": texts})

    # 95 / 5 split before tokenization to avoid cross-document contamination
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)

    eos_id = tokenizer.eos_token_id

    def _tokenize(batch):
        result = tokenizer(
            batch["text"],
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )
        # Append EOS after every document so packed blocks contain natural
        # document-boundary stop signals.  Without this the model sees one
        # continuous token stream and never learns to stop.
        result["input_ids"] = [ids + [eos_id] for ids in result["input_ids"]]
        result["attention_mask"] = [mask + [1] for mask in result["attention_mask"]]
        return result

    def _group_texts(examples):
        """Concatenate and slice into max_length blocks."""
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concat[list(examples.keys())[0]])
        total = (total // max_length) * max_length  # drop remainder
        result = {
            k: [v[i : i + max_length] for i in range(0, total, max_length)]
            for k, v in concat.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Tokenizing (max_length={max_length}, no padding, packing)...")
    tokenized = split.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=4,
    )

    logger.info("Packing into fixed-length blocks...")
    packed = tokenized.map(
        _group_texts,
        batched=True,
        desc="Packing",
    )

    return DatasetDict(
        train=packed["train"],
        validation=packed["test"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HPN research corpus for LoRA adapter training"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path to model whose tokenizer to use "
             "(e.g. models/instruction/Llama-3.1-8B-base-instruct)",
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        default=_DEFAULT_CORPUS,
        help=f"Path to research corpus JSON (default: {_DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_DEFAULT_OUTPUT,
        help=f"Directory to save tokenized Arrow dataset (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Block size in tokens (default: 2048)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction held out for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--reuse_cpt_data",
        type=str,
        default=None,
        metavar="PATH",
        help="If provided AND the path exists, skip tokenization and copy "
             "the processed dataset from this path instead.",
    )
    args = parser.parse_args()

    setup_hf_auth()

    # ── validate inputs ────────────────────────────────────────────────
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── fast path: reuse existing CPT dataset ─────────────────────────
    if args.reuse_cpt_data and Path(args.reuse_cpt_data).exists():
        logger.info(f"Reusing existing processed dataset from: {args.reuse_cpt_data}")
        logger.warning(
            "Ensure the block size and tokenizer match your current settings "
            f"(expected max_length={args.max_length}, "
            f"model={args.model_path})."
        )
        dataset = load_from_disk(args.reuse_cpt_data)
        # CPT datasets may use 'test' instead of 'validation' as the split key.
        if "test" in dataset and "validation" not in dataset:
            dataset = dataset.rename_column("test", "validation") if hasattr(dataset, "rename_column") else \
                DatasetDict(train=dataset["train"], validation=dataset["test"])
        logger.info(f"Loaded — train: {len(dataset['train']):,}, "
                    f"val: {len(dataset['validation']):,} samples")
        logger.info(f"Saving to: {args.output_dir}")
        dataset.save_to_disk(args.output_dir)
        logger.info("✅ Dataset ready (reused CPT data).")
        return

    # ── validate corpus file (only needed when tokenizing from scratch) ─
    if not Path(args.corpus_file).exists():
        logger.error(f"Corpus file not found: {args.corpus_file}")
        sys.exit(1)

    # ── load tokenizer ─────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── load & prepare texts ───────────────────────────────────────────
    texts = load_corpus(args.corpus_file)
    if not texts:
        logger.error("No texts found in corpus. Exiting.")
        sys.exit(1)

    # ── tokenize + pack ────────────────────────────────────────────────
    dataset = tokenize_and_pack(
        texts=texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # ── statistics ─────────────────────────────────────────────────────
    train_tokens = len(dataset["train"]) * args.max_length
    val_tokens   = len(dataset["validation"]) * args.max_length

    logger.info("=" * 60)
    logger.info("Dataset Statistics:")
    logger.info("=" * 60)
    logger.info(f"Train samples:      {len(dataset['train']):,}")
    logger.info(f"Validation samples: {len(dataset['validation']):,}")
    logger.info(f"Train tokens:       {train_tokens:,}")
    logger.info(f"Validation tokens:  {val_tokens:,}")
    logger.info(f"Total tokens:       {train_tokens + val_tokens:,}")
    logger.info(f"Block size:         {args.max_length}")
    logger.info(f"Tokenizer:          {args.model_path}")
    logger.info("=" * 60)

    # ── save ───────────────────────────────────────────────────────────
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)

    logger.info(f"Saving tokenized dataset to: {args.output_dir}")
    dataset.save_to_disk(args.output_dir)

    logger.info("")
    logger.info("✅ LoRA data preparation complete!")
    logger.info(f"   {args.output_dir}/")
    logger.info(f"   ├── train/")
    logger.info(f"   ├── validation/")
    logger.info(f"   └── tokenizer/")
    logger.info("")
    logger.info("Next step:")
    logger.info(f"  python training/lora_finetune.py \\")
    logger.info(f"    --model_path {args.model_path} \\")
    logger.info(f"    --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
