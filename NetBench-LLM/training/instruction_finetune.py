#!/usr/bin/env python3
"""
Instruction Fine-Tuning Script for Llama Models.
Trains a model to follow Open-Orca chat-style instructions using HuggingFace Trainer.

Takes a tokenized instruction dataset (from prepare_instruction_data.py)
and fine-tunes a local Llama model.  Keeps the source model unchanged
and writes the fine-tuned weights to a new directory.

Usage:
    # Fine-tune 8B base model
    python instruction_finetune.py \
        --model_path models/base/Llama-3.1-8B-base \
        --data_dir data/instruction/llama-3.1-8b-base \
        --output_dir models/instruction/Llama-3.1-8B-base-instruct

    # Fine-tune 8B trained-new model
    python instruction_finetune.py \
        --model_path models/pretrained/Llama-3.1-8B-trained-new \
        --data_dir data/instruction/llama-3.1-8b-trained-new \
        --output_dir models/instruction/Llama-3.1-8B-trained-new-instruct
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth
from utils.model_utils import detect_model_family, fix_tokenizer_padding, get_gemma_token_type_key

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MemoryCallback(TrainerCallback):
    """Log GPU memory every 50 steps."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                resv = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(
                    f"GPU {i} – Allocated: {alloc:.2f} GB, Reserved: {resv:.2f} GB"
                )


class InstructionDataCollator:
    """
    Dynamic-padding collator for instruction fine-tuning.
    Preserves pre-computed labels from the dataset and pads per-batch
    instead of to a global max_length.  Labels = -100 for padding positions
    so they are ignored by the cross-entropy loss.
    """

    def __init__(self, tokenizer, model_family="llama", gemma_tt_key="token_type_ids"):
        self.pad_token_id = tokenizer.pad_token_id
        self.model_family = model_family
        self.gemma_tt_key = gemma_tt_key

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f["attention_mask"])
            lab = list(f.get("labels", ids[:]))
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.model_family == "gemma":
            batch[self.gemma_tt_key] = torch.zeros_like(batch["input_ids"])
        return batch


def parse_args():
    p = argparse.ArgumentParser(
        description="Instruction fine-tune a Llama model"
    )

    # Required paths
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to source model "
             "(e.g. models/base/Llama-3.1-8B-base or "
             "models/pretrained/Llama-3.1-8B-trained-new)",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with tokenized instruction dataset "
             "(output of prepare_instruction_data.py)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the instruction-tuned model",
    )

    # Training hyper-parameters
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Optimisation flags
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument(
        "--gradient_checkpointing", action="store_true", default=True
    )
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer name passed to TrainingArguments. Use 'paged_adamw_8bit' "
             "or 'adamw_bnb_8bit' (requires bitsandbytes) to cut optimizer-state "
             "memory by ~4x — essential for full-weight SFT of 9B+ models on "
             "48 GB GPUs without DeepSpeed CPU offload.",
    )

    # Logging / saving
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from a HuggingFace Trainer checkpoint directory.",
    )
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
    )

    return p.parse_args()


def main():
    args = parse_args()
    setup_hf_auth()

    # ── validate paths ─────────────────────────────────────────────────
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    if not Path(args.data_dir).exists():
        logger.error(f"Data not found: {args.data_dir}")
        logger.error("Run training/prepare_instruction_data.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INSTRUCTION FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Source model:  {args.model_path}")
    logger.info(f"Data dir:      {args.data_dir}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Epochs:        {args.num_epochs}")
    logger.info(f"Batch size:    {args.batch_size}")
    logger.info(f"Grad accum:    {args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)

    # ── detect model family ────────────────────────────────────────────
    model_family = detect_model_family(args.model_path)
    logger.info(f"Detected model family: {model_family}")

    # ── load tokenizer ─────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    fix_tokenizer_padding(tokenizer, model_family)

    # ── load model ─────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model_path}...")

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if args.deepspeed is None and args.local_rank == -1:
        model_kwargs["device_map"] = "auto"

    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        logger.info("Flash Attention not available, using SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, **model_kwargs
    )

    # Disable KV-cache (incompatible with gradient checkpointing during training).
    # Set on config after loading because some architectures (e.g. Gemma 3) do not
    # accept use_cache as a constructor kwarg.
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ── load tokenized dataset ─────────────────────────────────────────
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset = load_from_disk(args.data_dir)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    logger.info(f"Train samples:      {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(eval_dataset):,}")

    # ── auto-clamp eval/save steps ─────────────────────────────────────
    _num_gpus = max(1, torch.cuda.device_count())
    _steps_per_epoch = max(1, len(train_dataset) // (
        args.batch_size * _num_gpus * args.gradient_accumulation_steps
    ))
    if args.eval_steps > _steps_per_epoch:
        args.eval_steps = max(1, _steps_per_epoch)
        logger.warning(
            f"eval_steps clamped to {args.eval_steps} (1x per epoch) "
            f"to ensure evaluation runs"
        )
    if args.save_steps > _steps_per_epoch:
        args.save_steps = max(1, _steps_per_epoch)
        logger.warning(
            f"save_steps clamped to {args.save_steps} (1x per epoch)"
        )
    if args.save_steps % args.eval_steps != 0:
        old_save_steps = args.save_steps
        args.save_steps = args.eval_steps
        logger.warning(
            "save_steps adjusted from %s to %s because "
            "load_best_model_at_end=True requires save_steps to be a "
            "round multiple of eval_steps",
            old_save_steps,
            args.save_steps,
        )

    # ── data collator ──────────────────────────────────────────────────────
    _gemma_tt_key = get_gemma_token_type_key(model) if model_family == "gemma" else "token_type_ids"
    data_collator = InstructionDataCollator(
        tokenizer=tokenizer, model_family=model_family,
        gemma_tt_key=_gemma_tt_key,
    )
    if model_family == "gemma":
        logger.info(f"Gemma model detected — {_gemma_tt_key} will be injected by collator")

    # ── training arguments ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=max(1, int(args.warmup_ratio * (len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) + 1) * args.num_epochs)),
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if getattr(model, "hf_device_map", None):
        training_args._n_gpu = 1
        logger.warning(
            "Model is sharded via device_map='auto' — disabling DataParallel. "
            "For better multi-GPU utilisation, launch with torchrun/DeepSpeed."
        )

    # ── trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[MemoryCallback()],
    )

    # ── train ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("🚀 Starting instruction fine-tuning...")
    logger.info("")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── save model and tokenizer ───────────────────────────────────────
    logger.info(f"Saving instruction-tuned model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── metrics ────────────────────────────────────────────────────────
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ INSTRUCTION FINE-TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Source model (UNCHANGED): {args.model_path}")
    logger.info(f"Instruction-tuned model:  {args.output_dir}")
    logger.info(f"Training loss:            {metrics['train_loss']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
