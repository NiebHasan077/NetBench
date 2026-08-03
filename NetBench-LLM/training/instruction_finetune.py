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
    """Log this process' CUDA memory every 50 steps."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            device = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            resv = torch.cuda.memory_reserved(device) / 1024**3
            rank = os.environ.get("RANK", "0")
            logger.info(
                f"Rank {rank} GPU {device} - Allocated: {alloc:.2f} GB, Reserved: {resv:.2f} GB"
            )


class NonFiniteMetricCallback(TrainerCallback):
    """Abort training as soon as loss/grad metrics become NaN or Inf."""

    _WATCHED = ("loss", "eval_loss", "grad_norm")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        for key in self._WATCHED:
            if key not in logs:
                continue
            try:
                value = float(logs[key])
            except (TypeError, ValueError):
                continue
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError(
                    f"Non-finite {key}={logs[key]} at step {state.global_step}. "
                    "Stop this run and restart from a clean checkpoint with a stable precision mode."
                )


def distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_final_model(trainer, tokenizer, output_dir):
    """Save final weights without forcing DeepSpeed through get_state_dict()."""
    os.makedirs(output_dir, exist_ok=True)

    if getattr(trainer, "is_deepspeed_enabled", False):
        engine = getattr(trainer, "deepspeed", None)
        if engine is None:
            raise RuntimeError("DeepSpeed save requested, but trainer.deepspeed is not available")

        logger.info(
            "DeepSpeed detected; saving 16-bit model with "
            "DeepSpeedEngine.save_16bit_model to avoid final-save OOM"
        )
        saved = engine.save_16bit_model(output_dir, "pytorch_model.bin")
        if not saved:
            raise RuntimeError(
                "DeepSpeed did not save 16-bit weights. If using ZeRO-3, enable "
                "stage3_gather_16bit_weights_on_model_save or recover with zero_to_fp32.py."
            )

        distributed_barrier()

        if trainer.is_world_process_zero():
            model_for_metadata = getattr(engine, "module", trainer.model)
            model_for_metadata.config.save_pretrained(output_dir)
            generation_config = getattr(model_for_metadata, "generation_config", None)
            if generation_config is not None:
                generation_config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(trainer.args, os.path.join(output_dir, "training_args.bin"))

        distributed_barrier()
        return

    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)


def resolve_precision(args) -> tuple[str, torch.dtype, bool, bool]:
    """Return (precision_name, model_dtype, trainer_bf16, trainer_fp16)."""
    precision = args.precision
    if args.bf16:
        precision = "bf16"
    elif args.fp16_full:
        precision = "fp16_full"
    elif args.no_fp16:
        precision = "fp32"
    elif args.fp16:
        precision = "fp16_amp"

    if precision == "auto":
        precision = (
            "bf16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "fp16_amp"
        )

    if precision == "bf16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 requested, but this GPU/runtime does not report BF16 support.")
        return precision, torch.bfloat16, True, False
    if precision == "fp16_amp":
        return precision, torch.float32, False, True
    if precision == "fp16_full":
        logger.warning(
            "Using full FP16 trainable weights without AMP GradScaler. "
            "This is memory-efficient but can produce NaN/Inf during full-weight training."
        )
        return precision, torch.float16, False, False
    if precision == "fp32":
        return precision, torch.float32, False, False
    raise ValueError(f"Unknown precision mode: {precision}")


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
    p.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16_amp", "fp16_full", "fp32"],
        default="auto",
        help=(
            "Training precision. auto chooses bf16 when supported, otherwise "
            "fp16_amp (fp32 weights + AMP GradScaler). fp16_full is memory-light "
            "but unsafe for full-weight training and may produce NaNs."
        ),
    )
    p.add_argument("--bf16", action="store_true", default=False,
                   help="Deprecated alias for --precision bf16.")
    p.add_argument("--fp16", action="store_true", default=False,
                   help="Deprecated alias for --precision fp16_amp.")
    p.add_argument("--fp16_full", action="store_true", default=False,
                   help="Load/train full model weights in fp16 without AMP scaling. Unsafe; use only if necessary.")
    p.add_argument("--no_fp16", action="store_true", default=False,
                   help="Deprecated alias for --precision fp32.")
    p.add_argument(
        "--gradient_checkpointing", action="store_true", default=True
    )

    # Logging / saving
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit",
                   help="HF Trainer optimizer. paged_adamw_8bit pages optimizer "
                        "states to CPU pinned memory, enabling full SFT of 8B+ "
                        "models on <64 GB GPUs without DeepSpeed.")
    p.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of DataLoader worker processes. Keep 0 on older kernels "
            "to reduce multiprocessing hang risk."
        ),
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from a HuggingFace Trainer checkpoint directory.",
    )
    p.add_argument(
        "--final_save_only",
        action="store_true",
        help=(
            "Disable mid-training eval/checkpoint saves and write only the "
            "final model. Useful for DeepSpeed full fine-tuning on shared "
            "nodes where optimizer checkpointing can trigger host OOM kills."
        ),
    )
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
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

    precision, model_dtype, trainer_bf16, trainer_fp16 = resolve_precision(args)
    logger.info(
        "Precision mode: %s (model dtype=%s, Trainer bf16=%s, fp16=%s)",
        precision,
        model_dtype,
        trainer_bf16,
        trainer_fp16,
    )

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }

    # Single-process fallback uses device_map="auto" to fit full training on
    # smaller cards. Under torchrun/DDP each process owns one full model replica,
    # so device_map must be disabled or every rank tries to shard across all
    # visible GPUs and autograd becomes brittle.
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
    if args.final_save_only:
        logger.warning(
            "final_save_only enabled: disabling mid-training eval/checkpoint "
            "saves and saving only the final model"
        )
    elif args.eval_steps > _steps_per_epoch:
        args.eval_steps = max(1, _steps_per_epoch)
        logger.warning(
            f"eval_steps clamped to {args.eval_steps} (1x per epoch) "
            f"to ensure evaluation runs"
        )
    if not args.final_save_only and args.save_steps > _steps_per_epoch:
        args.save_steps = max(1, _steps_per_epoch)
        logger.warning(
            f"save_steps clamped to {args.save_steps} (1x per epoch)"
        )
    if not args.final_save_only and args.save_steps % args.eval_steps != 0:
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
        eval_strategy="no" if args.final_save_only else "steps",
        save_strategy="no" if args.final_save_only else "steps",
        bf16=trainer_bf16,
        fp16=trainer_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        load_best_model_at_end=not args.final_save_only,
        metric_for_best_model=None if args.final_save_only else "eval_loss",
        greater_is_better=None if args.final_save_only else False,
    )

    # When the model is already sharded across GPUs by device_map="auto",
    # Trainer must not wrap it in DataParallel. DataParallel gathers on GPU 0
    # and can immediately OOM during backward.
    if getattr(model, "hf_device_map", None):
        training_args._n_gpu = 1
        logger.warning(
            "Model is sharded via device_map='auto' — disabling DataParallel. "
            "For better multi-GPU utilisation, use ZeRO/FSDP or a torchrun "
            "profile that can fit a full trainable replica per GPU."
        )

    # ── trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[MemoryCallback(), NonFiniteMetricCallback()],
    )

    # ── train ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("🚀 Starting instruction fine-tuning...")
    logger.info("")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── save model and tokenizer ───────────────────────────────────────
    logger.info(f"Saving instruction-tuned model to {args.output_dir}...")
    save_final_model(trainer, tokenizer, args.output_dir)

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
