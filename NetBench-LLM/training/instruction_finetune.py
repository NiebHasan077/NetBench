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
import math
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


class FiniteMetricsCallback(TrainerCallback):
    """Stop immediately if training/eval metrics become non-finite."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        for key in ("loss", "eval_loss", "grad_norm"):
            if key not in logs:
                continue
            try:
                value = float(logs[key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {key}={logs[key]} at step {state.global_step}. "
                    "Stop this run and restart from the source model, not from "
                    "the current SFT checkpoint."
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


def validate_supervised_labels(dataset, split_name: str) -> None:
    """Ensure the tokenized SFT split contains assistant labels to learn from."""
    supervised_counts = []
    zero_label_rows = []
    for idx, sample in enumerate(dataset):
        labels = sample.get("labels")
        if labels is None:
            raise ValueError(f"{split_name} sample {idx} has no labels column")
        count = sum(1 for label in labels if label != -100)
        supervised_counts.append(count)
        if count == 0:
            zero_label_rows.append(idx)

    if zero_label_rows:
        preview = ", ".join(str(i) for i in zero_label_rows[:10])
        raise ValueError(
            f"{split_name} has {len(zero_label_rows)} sample(s) with no "
            f"supervised assistant tokens. First bad indices: {preview}. "
            "Regenerate the instruction dataset."
        )

    avg = sum(supervised_counts) / max(1, len(supervised_counts))
    logger.info(
        "%s supervised labels/token counts: min=%d avg=%.1f max=%d",
        split_name,
        min(supervised_counts),
        avg,
        max(supervised_counts),
    )


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
        help=(
            "Optimizer name passed to TrainingArguments. For 27B full SFT, "
            "paged_adamw_8bit is a practical first choice when bitsandbytes "
            "is installed."
        ),
    )

    # Logging / saving
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument(
        "--save_final_only",
        action="store_true",
        help=(
            "Disable intermediate Trainer checkpoints and save only the final "
            "model. Leave this off for resumable long runs."
        ),
    )
    p.add_argument(
        "--no_load_best_model_at_end",
        action="store_true",
        help=(
            "Do not reload the best eval checkpoint at the end of training. "
            "This keeps checkpoint retention predictable for large full-model runs."
        ),
    )
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
    logger.info(f"Optimizer:     {args.optim}")
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
        if model_family == "gemma":
            # Gemma SFT uses dynamically padded batches.  On this stack SDPA can
            # produce non-finite values in masked attention rows; eager is
            # slower but stable for the small HPN SFT set.
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Flash Attention not available, using eager attention for Gemma SFT stability")
        else:
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("Flash Attention not available, using SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, **model_kwargs
    )

    # Gemma 3's GQA attention path and Gemma 4's KV-shared layers are unstable
    # under gradient-checkpoint recomputation: PyTorch sees different tensor
    # metadata on backward and raises CheckpointError.  CPT already uses the
    # same guard; keep SFT aligned so reruns resume cleanly.
    _text_cfg = getattr(model.config, "text_config", model.config)
    _has_kv_sharing = getattr(_text_cfg, "num_kv_shared_layers", 0) > 0
    _model_type = getattr(model.config, "model_type", "").lower()
    _skip_gc = _has_kv_sharing or "gemma" in _model_type or model_family == "gemma"

    if _has_kv_sharing:
        model.config.use_cache = True
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True
        logger.info(
            f"Model has {_text_cfg.num_kv_shared_layers} KV-shared layers — "
            "keeping use_cache=True, disabling gradient checkpointing"
        )
    else:
        # Standard models: disable KV-cache during training.  Set on config
        # after loading because some architectures do not accept use_cache as a
        # constructor kwarg.
        model.config.use_cache = False

    if args.gradient_checkpointing and not _skip_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled (use_reentrant=False)")
    elif args.gradient_checkpointing and _skip_gc and not _has_kv_sharing:
        logger.info(
            "Gradient checkpointing DISABLED — Gemma attention recomputation "
            "can change tensor metadata. VRAM is ample; GC not needed."
        )

    # ── load tokenized dataset ─────────────────────────────────────────
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset = load_from_disk(args.data_dir)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    logger.info(f"Train samples:      {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(eval_dataset):,}")
    validate_supervised_labels(train_dataset, "train")
    validate_supervised_labels(eval_dataset, "validation")

    # ── auto-clamp eval/save steps ─────────────────────────────────────
    # device_map="auto" shards one training process across visible GPUs; it is
    # not data parallelism, so checkpoint cadence must not divide by GPU count.
    _world_size = max(1, int(os.environ.get("WORLD_SIZE", "1") or "1"))
    _num_train_workers = _world_size if (
        args.deepspeed is not None or args.local_rank != -1 or _world_size > 1
    ) else 1
    _steps_per_epoch = max(1, len(train_dataset) // (
        args.batch_size * _num_train_workers * args.gradient_accumulation_steps
    ))
    load_best_model_at_end = not (
        args.save_final_only or args.no_load_best_model_at_end
    )
    if args.eval_steps > _steps_per_epoch:
        args.eval_steps = max(1, _steps_per_epoch)
        logger.warning(
            f"eval_steps clamped to {args.eval_steps} (1x per epoch) "
            f"to ensure evaluation runs"
        )
    if args.save_final_only:
        logger.warning(
            "Intermediate checkpoints disabled; only the final model will be saved."
        )
    else:
        if args.save_steps > _steps_per_epoch:
            args.save_steps = max(1, _steps_per_epoch)
            logger.warning(
                f"save_steps clamped to {args.save_steps} (1x per epoch)"
            )
        if load_best_model_at_end and args.save_steps % args.eval_steps != 0:
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
        save_strategy="no" if args.save_final_only else "steps",
        bf16=args.bf16,
        logging_nan_inf_filter=False,
        gradient_checkpointing=args.gradient_checkpointing and not _skip_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if (args.gradient_checkpointing and not _skip_gc) else None,
        use_cache=_has_kv_sharing,
        optim=args.optim,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # When loaded with device_map="auto" in a single-process run, Trainer would
    # otherwise wrap it in DataParallel on multi-GPU nodes and gather logits on
    # GPU 0.  Keep that mode only for non-distributed fallback launches.
    if getattr(model, "hf_device_map", None):
        training_args._n_gpu = 1
        logger.warning(
            "Model loaded with device_map='auto' — disabling Trainer DataParallel. "
            "For better multi-GPU utilisation, launch with torchrun + DeepSpeed."
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
    if trainer.model_accepts_loss_kwargs:
        trainer.model_accepts_loss_kwargs = False
        logger.info(
            "Overrode model_accepts_loss_kwargs → False "
            "(model forward has **kwargs but does not scale loss by num_items_in_batch)"
        )
    trainer.add_callback(FiniteMetricsCallback())

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
