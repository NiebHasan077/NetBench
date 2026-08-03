#!/usr/bin/env python3
"""
Pre-training Script for LLaMA Models using Transformers
Trains from LOCAL base model and saves to separate trained directory.
Keeps original model unchanged.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_from_disk
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth
from utils.model_utils import detect_model_family, get_gemma_token_type_key

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "1b": {
        "base_dir": "models/base/Llama-3.2-1B-base",
        "trained_dir": "models/pretrained/Llama-3.2-1B-trained-new",
        "hf_name": "meta-llama/Llama-3.2-1B"
    },
    "8b": {
        "base_dir": "models/base/Llama-3.1-8B-base",
        "trained_dir": "models/pretrained/Llama-3.1-8B-trained-new",
        "hf_name": "meta-llama/Llama-3.1-8B"
    },
    "qwen-2b": {
        "base_dir": "models/base/Qwen3.5-2B-Base",
        "trained_dir": "models/pretrained/Qwen3.5-2B-trained-new",
        "hf_name": "Qwen/Qwen3.5-2B-Base"
    },
    "qwen-4b": {
        "base_dir": "models/base/Qwen3.5-4B-Base",
        "trained_dir": "models/pretrained/Qwen3.5-4B-trained-new",
        "hf_name": "Qwen/Qwen3.5-4B-Base"
    },
    "gemma-4b": {
        "base_dir": "models/base/gemma-3-4b",
        "trained_dir": "models/pretrained/gemma-3-4b-trained-new",
        "hf_name": "google/gemma-3-4b-pt"
    },
    "gemma-e4b": {
        "base_dir": "models/base/gemma-4-e4b",
        "trained_dir": "models/pretrained/gemma-4-e4b-trained-new",
        "hf_name": "google/gemma-4-E4B"
    },
    "gemma-e2b": {
        "base_dir": "models/base/gemma-4-e2b",
        "trained_dir": "models/pretrained/gemma-4-e2b-trained-new",
        "hf_name": "google/gemma-4-E2B"
    },
}


class MemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage"""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train LLaMA models")
    
    # Model selection — use --model for shortcut keys OR --model_path for
    # direct paths to any base model directory (escape hatch).
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["1b", "8b", "qwen-2b", "qwen-4b", "gemma-4b", "gemma-e4b", "gemma-e2b"],
        default=None,
        help="Model shortcut: 1b, 8b, qwen-2b, qwen-4b, gemma-4b, gemma-e4b, gemma-e2b. "
             "Ignored when --model_path is provided."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Direct path to any base model directory. "
             "Bypasses the --model shortcut dict entirely."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing processed dataset (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save trained model (uses default if not specified)"
    )
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps used for LR warmup (converted to warmup_steps)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Optimization
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer name passed to TrainingArguments. Use 'paged_adamw_8bit' or "
             "'adamw_bnb_8bit' (requires bitsandbytes) to cut optimizer-state memory "
             "by ~4x — essential for large models like Gemma 4 E4B on 48 GB GPUs. "
             "(default: adamw_torch)",
    )
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    
    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", -1)))
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ── resolve base_dir and trained_dir ────────────────────────────
    if args.model_path:
        # Escape hatch: use --model_path directly, bypass shortcut dict
        base_dir = args.model_path
        if args.output_dir is None:
            # Derive a sensible default: models/pretrained/<dir-name>-trained-new
            base_name = Path(base_dir).name
            trained_dir = f"models/pretrained/{base_name}-trained-new"
        else:
            trained_dir = args.output_dir
    elif args.model:
        model_config = MODELS[args.model]
        base_dir = model_config["base_dir"]
        trained_dir = args.output_dir if args.output_dir else model_config["trained_dir"]
    else:
        # Default to 1b for backward compat
        model_config = MODELS["1b"]
        base_dir = model_config["base_dir"]
        trained_dir = args.output_dir if args.output_dir else model_config["trained_dir"]
    
    # Check if base model exists locally
    if not Path(base_dir).exists():
        logger.error(f"Base model not found at: {base_dir}")
        if args.model:
            logger.error("Please run: python utils/download_models.py --model " + args.model)
        sys.exit(1)
    
    # Auto-detect data directory
    if args.data_dir is None:
        if args.model:
            _model_slug = {
                "1b":       "llama-3.2-1b",
                "8b":       "llama-3.1-8b",
                "qwen-2b":  "qwen3.5-2b",
                "qwen-4b":  "qwen3.5-4b",
                "gemma-4b": "gemma-3-4b",
                "gemma-e4b": "gemma-4-e4b",
                "gemma-e2b": "gemma-4-e2b",
            }[args.model]
        else:
            # Derive slug from base_dir name
            _model_slug = Path(base_dir).name.lower()
        possible_dirs = [
            f"data/processed/{_model_slug}",
            f"processed_data_new/{_model_slug}",
            f"processed_data/{_model_slug}",
            "data/processed",
            "processed_data_new",
            "processed_data",
            "tokenized_data",
        ]
        for d in possible_dirs:
            if Path(d).exists():
                args.data_dir = d
                break
        if args.data_dir is None:
            logger.error("No processed data found. Please run training/prepare_data.py first.")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(trained_dir, exist_ok=True)
    checkpoint_dir = os.path.join(trained_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Training Configuration:")
    logger.info("="*60)
    logger.info(f"Base model: {base_dir}")
    logger.info(f"Output (trained): {trained_dir}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info("="*60)
    
    # Load model from LOCAL directory
    logger.info(f"Loading model from: {base_dir}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if args.bf16 else torch.float32,
    }
    
    # Try flash attention if available
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        logger.info("Flash Attention not available, using default SDPA attention")
        model_kwargs["attn_implementation"] = "sdpa"
    
    # Don't use device_map with DeepSpeed - DeepSpeed handles device placement
    if args.deepspeed is None and args.local_rank == -1:
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,  # Load from LOCAL directory
        **model_kwargs
    )

    # ── KV-shared layer detection (Gemma 4 E2B/E4B) ─────────────────
    # Gemma 4 E2B/E4B share KV states across layers via an internal
    # DynamicCache.  use_cache=False prevents cache creation, breaking
    # KV-shared layers → garbage logits (HF issue #45242).
    # Gradient checkpointing is also incompatible: the mutable DynamicCache
    # accumulates duplicate entries during GC recomputation.
    _text_cfg = getattr(model.config, 'text_config', model.config)
    _has_kv_sharing = getattr(_text_cfg, 'num_kv_shared_layers', 0) > 0

    if _has_kv_sharing:
        model.config.use_cache = True
        if hasattr(model.config, 'text_config'):
            model.config.text_config.use_cache = True
        logger.info(
            f"Model has {_text_cfg.num_kv_shared_layers} KV-shared layers — "
            f"keeping use_cache=True, disabling gradient checkpointing"
        )
    else:
        # Standard models: disable KV-cache (incompatible with gradient
        # checkpointing during training).  Set on config after loading because
        # some architectures do not accept use_cache as a constructor kwarg.
        model.config.use_cache = False

    # Enable gradient checkpointing (skip for KV-shared models — the mutable
    # DynamicCache accumulates duplicate KV entries during GC recomputation).
    if args.gradient_checkpointing and not _has_kv_sharing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Load tokenizer from LOCAL directory
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = load_from_disk(args.data_dir)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")

    # ── auto-clamp eval/save steps ─────────────────────────────────────
    _num_gpus = max(1, torch.cuda.device_count())
    _steps_per_epoch = max(1, len(train_dataset) // (
        args.per_device_train_batch_size * _num_gpus * args.gradient_accumulation_steps
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

    # Data collator
    _base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Gemma models are multimodal architectures that require a token-type
    # tensor to distinguish text from image tokens.  For text-only pretraining
    # all values are 0 (= text).  The field name differs between generations:
    #   Gemma 3 → token_type_ids    Gemma 4 → mm_token_type_ids
    model_family = detect_model_family(base_dir)
    if model_family == "gemma":
        _tt_key = get_gemma_token_type_key(model)
        def data_collator(features):
            batch = _base_collator(features)
            batch[_tt_key] = torch.zeros_like(batch["input_ids"])
            return batch
        logger.info(f"Gemma model detected — injecting {_tt_key} into batches")
    else:
        data_collator = _base_collator
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=max(1, int(args.warmup_ratio * (len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) + 1) * args.num_train_epochs)),
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing and not _has_kv_sharing,
        optim=args.optim,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        use_cache=_has_kv_sharing,
    )
    
    # When the model was loaded with device_map="auto" (pipeline-parallel across
    # GPUs), Trainer must NOT wrap it in DataParallel — that would try to gather
    # all outputs onto GPU 0 and OOM.  Override _n_gpu so Trainer skips the DP wrap.
    if getattr(model, "hf_device_map", None):
        training_args._n_gpu = 1
        logger.warning(
            "Model is sharded via device_map='auto' — disabling DataParallel. "
            "For better multi-GPU utilisation, launch with: "
            "torchrun --nproc_per_node=NUM_GPUS training/pretrain_transformers.py ..."
        )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[MemoryCallback()],
    )

    # Some model forward methods have **kwargs, which makes the Trainer
    # infer model_accepts_loss_kwargs=True.  That flag tells the Trainer
    # the model internally scales the loss by num_items_in_batch across
    # the gradient-accumulation window — but the model does NOT do this.
    # Override to False so the Trainer normalises the loss correctly.
    if trainer.model_accepts_loss_kwargs:
        trainer.model_accepts_loss_kwargs = False
        logger.info(
            "Overrode model_accepts_loss_kwargs → False "
            "(model forward has **kwargs but does not scale loss by num_items_in_batch)"
        )
    
    # Train
    logger.info("Starting pre-training...")
    train_result = trainer.train()
    
    # Save final trained model to trained_dir (separate from base)
    logger.info(f"Saving trained model to: {trained_dir}")
    trainer.save_model(trained_dir)
    tokenizer.save_pretrained(trained_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("="*60)
    logger.info("Pre-training completed!")
    logger.info("="*60)
    logger.info(f"✅ Original base model: {base_dir} (UNCHANGED)")
    logger.info(f"✅ Trained model: {trained_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
