#!/usr/bin/env python3
"""
LoRA Adapter Fine-Tuning Script.

Supports two training modes via --data_mode:

  corpus (default) — Domain adaptation via CLM on the research corpus.
    - Input dataset: packed 2048-token blocks from prepare_lora_data.py
    - Objective: next-token prediction on all tokens
    - Base model can be a raw base model OR an instruct model
    - Collator: DataCollatorForLanguageModeling (all tokens are valid targets)

  instruct — Instruction fine-tuning with chat-template supervision.
    - Input dataset: tokenized instruction data from prepare_instruction_data.py
    - Objective: next-token prediction on assistant response tokens only
    - Labels = -100 on padding/prompt tokens are preserved from the dataset
    - Collator: InstructionDataCollator (dynamic padding, preserves -100 labels)

Precision modes:
  Full bfloat16 (default, recommended for 2xRTX A6000 48GB each):
    Model loaded in bfloat16.  LoRA weights trained in full precision.
  QLoRA 4-bit (--quantize_4bit, for GPUs < 24 GB):
    Model loaded in 4-bit NF4 via bitsandbytes.

The source model is NEVER modified — adapter is saved separately.

Usage:
    # ── Corpus LoRA on a BASE model (domain adaptation) ──────────
    python training/lora_finetune.py \\
        --model_path models/base/Qwen3.5-2B-Base \\
        --data_dir   data/processed/lora/qwen3.5-2b-v3 \\
        --data_mode  corpus

    # ── Corpus LoRA on an instruct model (additional domain knowledge) ──
    python training/lora_finetune.py \\
        --model_path models/instruction/Qwen3.5-2B-v3-hpn-instruct \\
        --data_dir   data/processed/lora/qwen3.5-2b-v3 \\
        --data_mode  corpus

    # ── Instruction LoRA (HPN fine-tuning) ───────────────────────
    python training/lora_finetune.py \\
        --model_path models/lora-merged/Qwen3.5-2B-base-lora-corpus-merged \\
        --data_dir   data/instruction/qwen3.5-2b-v3-hpn \\
        --data_mode  instruct

    # ── QLoRA on a single small GPU (< 24 GB) ────────────────────
    python training/lora_finetune.py \\
        --model_path models/base/Qwen3.5-2B-Base \\
        --data_dir   data/processed/lora/qwen3.5-2b-v3 \\
        --quantize_4bit \\
        --batch_size 2 --grad_accum 8 --lora_rank 16
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
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth
from utils.model_utils import detect_model_family, get_lora_target_modules, fix_tokenizer_padding, get_gemma_token_type_key, is_gemma4

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


class InstructionDataCollator:
    """
    Dynamic-padding collator for instruction LoRA fine-tuning.
    Preserves pre-computed labels (including -100 for padding/prompt tokens)
    from the dataset and pads per-batch rather than to a global max_length.
    This prevents DataCollatorForLanguageModeling from overwriting label masks.
    """

    def __init__(self, pad_token_id: int, model_family: str = "llama",
                 gemma_tt_key: str = "token_type_ids"):
        self.pad_token_id = pad_token_id
        self.model_family = model_family
        self.gemma_tt_key = gemma_tt_key

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids  = list(f["input_ids"])
            mask = list(f["attention_mask"])
            lab  = list(f.get("labels", ids[:]))
            pad_len = max_len - len(ids)
            input_ids.append(ids  + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0]                * pad_len)
            labels.append(lab  + [-100]              * pad_len)
        batch = {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }
        if self.model_family == "gemma":
            batch[self.gemma_tt_key] = torch.zeros_like(batch["input_ids"])
        return batch


class MemoryCallback(TrainerCallback):
    """Log GPU memory every 50 steps."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                resv  = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(
                    f"GPU {i} – Allocated: {alloc:.2f} GB, Reserved: {resv:.2f} GB"
                )


class BestAdapterSaveCallback(TrainerCallback):
    """
    Save the LoRA adapter to <output_dir>/best_adapter/ whenever eval_loss
    improves.  This provides a safe fallback if training is interrupted before
    the final model.save_pretrained() call completes.

    Because load_best_model_at_end=False is required with PEFT (the Trainer's
    built-in reload from checkpoint does not support PEFT adapter-only saves in
    all versions), we implement best-checkpoint tracking manually here.
    """

    def __init__(self, output_dir: str):
        self._best_loss = float("inf")
        self._best_dir  = os.path.join(output_dir, "best_adapter")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        if eval_loss < self._best_loss:
            self._best_loss = eval_loss
            model = kwargs.get("model")
            if model is not None:
                logger.info(
                    f"New best eval_loss={eval_loss:.4f} — saving adapter to "
                    f"{self._best_dir}"
                )
                model.save_pretrained(self._best_dir)
                logger.info(f"Best adapter saved: {self._best_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a LoRA adapter on an instruct model using the HPN corpus"
    )

    # ── paths ──────────────────────────────────────────────────────────
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to source model — can be a raw base model OR an instruct model",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Arrow dataset directory.  For corpus mode: output of prepare_lora_data.py.  "
             "For instruct mode: output of prepare_instruction_data.py.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Adapter save path (default: models/lora/<model-name>-lora/)",
    )
    p.add_argument(
        "--data_mode",
        type=str,
        choices=["corpus", "instruct"],
        default="corpus",
        help="Training mode: 'corpus' = CLM on packed research text (default); "
             "'instruct' = instruction fine-tuning with preserved -100 label masks "
             "and dynamic padding.",
    )

    # ── precision mode ─────────────────────────────────────────────────
    p.add_argument(
        "--quantize_4bit",
        action="store_true",
        default=False,
        help="Use 4-bit NF4 QLoRA quantization (for GPUs < 24 GB).  "
             "Default: off — full bfloat16 recommended for 2x48GB RTX A6000.",
    )

    # ── LoRA hyperparameters ───────────────────────────────────────────
    # Defaults tuned for 2x RTX A6000 (48 GB each):
    #   rank=64 gives 4x more representational capacity than rank=16
    #   alpha=128 maintains the standard alpha/r = 2.0 scaling factor
    p.add_argument("--lora_rank",    type=int,   default=64,
                   help="LoRA rank r (default: 64 for 2x48GB)")
    p.add_argument("--lora_alpha",   type=int,   default=128,
                   help="LoRA alpha (default: 128; scaling = alpha/r = 2.0)")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout (default: 0.05)")
    p.add_argument(
        "--target_modules",
        type=str,
        default=None,
        help="Comma-separated LoRA target module names "
             "(default: auto-detected per model family — "
             "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)",
    )

    # ── training hyperparameters tuned for 2x48GB ─────────────────────
    # batch_size=4 × grad_accum=4 → effective batch 16
    # For 8B bfloat16: model ~16 GB + activations ~8 GB ≈ 24 GB / GPU with grad checkpointing
    # Well within 48 GB; increase batch_size to 8 if you want more throughput.
    p.add_argument("--epochs",        type=int,   default=5,
                   help="Training epochs (default: 5; eval_loss was still declining at epoch 3)")
    p.add_argument("--batch_size",    type=int,   default=4,
                   help="Per-device TRAIN batch size (default: 4 for 2x48GB)")
    p.add_argument("--eval_batch_size", type=int, default=2,
                   help="Per-device EVAL batch size (default: 2).  "
                        "Kept smaller than --batch_size because eval runs without "
                        "gradient checkpointing, so full layer activations must fit "
                        "in VRAM.  Using the same value as --batch_size caused a "
                        "100× eval slowdown (CUDA managed-memory paging) near end "
                        "of epoch 3 on the 8B model.")
    p.add_argument("--grad_accum",    type=int,   default=4,
                   help="Gradient accumulation steps (default: 4; effective batch = 16)")
    p.add_argument("--lr",            type=float, default=2e-4,
                   help="Learning rate (default: 2e-4, standard for LoRA)")
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # ── flags ──────────────────────────────────────────────────────────
    p.add_argument("--bf16",                   action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--logging_steps",          type=int, default=10)
    p.add_argument("--save_total_limit",       type=int, default=3)
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from a Trainer checkpoint directory "
             "(e.g. models/lora/<name>-lora/checkpoints/checkpoint-747).  "
             "Use this to recover from an interrupted run.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    setup_hf_auth()

    # ── resolve paths ─────────────────────────────────────────────────
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    if args.data_dir is None:
        args.data_dir = str(_ROOT / "data" / "processed" / "lora")

    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Run training/prepare_lora_data.py first.")
        sys.exit(1)

    model_name = Path(args.model_path).name
    if args.output_dir is None:
        args.output_dir = str(_ROOT / "models" / "lora" / f"{model_name}-lora")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    precision_label = "4-bit NF4 QLoRA" if args.quantize_4bit else "bfloat16 (full precision)"

    logger.info("=" * 60)
    logger.info("LORA ADAPTER FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Source model (frozen): {args.model_path}")
    logger.info(f"Data dir:              {args.data_dir}")
    logger.info(f"Adapter output:        {args.output_dir}")
    logger.info(f"Precision mode:        {precision_label}")
    logger.info(f"Data mode:             {args.data_mode}")
    logger.info(f"LoRA rank / alpha:     {args.lora_rank} / {args.lora_alpha}")
    logger.info(f"Epochs:                {args.epochs}")
    logger.info(f"Batch / grad_accum:    {args.batch_size} / {args.grad_accum}  "
                f"(effective batch = {args.batch_size * args.grad_accum})")
    logger.info(f"Learning rate:         {args.lr}")
    logger.info("=" * 60)

    # ── detect model family ────────────────────────────────────────────
    model_family = detect_model_family(args.model_path)
    if args.target_modules:
        lora_target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    else:
        lora_target_modules = get_lora_target_modules(model_family, model_path=args.model_path)
    logger.info(f"Detected model family:  {model_family}")
    logger.info(f"LoRA target modules:    {lora_target_modules}")

    # ── load tokenizer ─────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    fix_tokenizer_padding(tokenizer, model_family)

    # ── attention implementation ───────────────────────────────────────
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        if model_family == "gemma":
            # Gemma instruction batches are dynamically padded.  On this stack
            # SDPA can produce non-finite values in masked attention rows; eager
            # is slower but stable for HPN LoRA/SFT.
            attn_impl = "eager"
            logger.info("Flash Attention not available — using eager attention for Gemma stability")
        else:
            attn_impl = "sdpa"
            logger.info("Flash Attention not available — using SDPA")

    # ── load model ────────────────────────────────────────────────────
    if args.quantize_4bit:
        # ── QLoRA path: 4-bit NF4 (for GPUs < 24 GB) ─────────────────
        logger.info("Loading model in 4-bit NF4 (QLoRA mode)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 GB/B params
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        # prepare_model_for_kbit_training:
        #   1. Casts non-quantized layers (LayerNorm, lm_head) back to float32
        #      so they remain stable during training.
        #   2. Enables input_require_grads so gradients flow through the
        #      frozen 4-bit layers into the trainable LoRA adapters.
        # This call is ONLY correct for quantized (kbit) models.
        model = prepare_model_for_kbit_training(model)
        optimizer = "paged_adamw_8bit"  # memory-efficient for quantized training
    else:
        # ── Full bfloat16 path: recommended for 2x RTX A6000 (48 GB each) ──
        # 8B model: ~16 GB; leaves ample room for batch size 4 + activations.
        logger.info("Loading model in bfloat16 (full precision)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        # No prepare_model_for_kbit_training needed for full-precision models.
        # enable_input_require_grads ensures gradients flow into adapter layers.
        model.enable_input_require_grads()
        optimizer = "adamw_torch_fused"   # fused variant: fuses param update into a single kernel

    # ── Gemma checkpoint/cache handling ───────────────────────────────
    # Gemma 3's GQA attention path and Gemma 4's KV-shared layers are unstable
    # under gradient-checkpoint recomputation. Gemma 4 E2B/E4B also share KV
    # states across layers via DynamicCache (num_kv_shared_layers=18/20).
    # Three mechanisms in HF Transformers break Gemma 4 during training when
    # use_cache=False (HF issue #45242):
    #   1. Gemma4TextModel.forward skips DynamicCache construction
    #   2. @merge_with_config_defaults forces use_cache=False when GC=True
    #   3. GradientCheckpointingLayer strips past_key_values per layer
    # Fix: keep use_cache=True and disable gradient checkpointing.
    # LoRA's small trainable footprint (<1% of params) keeps VRAM
    # manageable; use --batch_size 1 with higher --grad_accum if needed.
    _text_cfg = getattr(model.config, 'text_config', model.config)
    _has_kv_sharing = getattr(_text_cfg, 'num_kv_shared_layers', 0) > 0
    _model_type = getattr(model.config, "model_type", "").lower()
    _skip_gc = _has_kv_sharing or "gemma" in _model_type or model_family == "gemma"

    if _has_kv_sharing:
        model.config.use_cache = True
        if hasattr(model.config, 'text_config'):
            model.config.text_config.use_cache = True
        logger.info(
            f"Model has {_text_cfg.num_kv_shared_layers} KV-shared layers — "
            "keeping use_cache=True, will disable gradient checkpointing"
        )
    else:
        # Standard models: disable KV-cache (incompatible with GC).
        # Set on config after loading; some architectures do not accept
        # use_cache as a constructor kwarg.
        model.config.use_cache = False
        if _skip_gc:
            logger.info(
                "Gradient checkpointing will be disabled — Gemma attention "
                "recomputation can change tensor metadata. VRAM is ample; GC not needed."
            )

    # ── Gemma 4: freeze vision / audio towers for text-only training ───
    # Gemma 4 is multimodal (vision + audio + text).  For text-only LoRA
    # training we freeze everything outside the text decoder so that:
    #   1. No gradient buffers are allocated for ~3 B non-text parameters
    #   2. No accidental weight updates to vision/audio projections
    # Ref: https://huggingface.co/docs/google-cloud/examples/vertex-ai-notebooks-fine-tune-gemma-4
    if model_family == "gemma" and is_gemma4(args.model_path):
        frozen_count = 0
        for name, param in model.named_parameters():
            if not name.startswith("model.language_model"):
                param.requires_grad = False
                frozen_count += 1
        logger.info(
            f"Gemma 4: froze {frozen_count} non-language-model parameters "
            f"(vision/audio towers & embeddings)"
        )

    # ── wrap with LoRA ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── load dataset ───────────────────────────────────────────────────
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset       = load_from_disk(args.data_dir)
    train_dataset = dataset["train"]
    eval_dataset  = dataset["validation"]

    logger.info(f"Train samples:      {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(eval_dataset):,}")

    # ── dynamic eval / save steps ──────────────────────────────────────
    # eval_steps is set dynamically to avoid the CPT bug where eval_steps=500
    # with total_steps=132 meant evaluation never triggered.
    steps_per_epoch = max(
        1,
        len(train_dataset) // (args.batch_size * args.grad_accum),
    )
    total_steps      = steps_per_epoch * args.epochs
    checkpoint_steps = max(10, total_steps // 10)

    logger.info(f"Steps per epoch:    {steps_per_epoch}")
    logger.info(f"Total steps:        {total_steps}")
    logger.info(f"Checkpoint / eval:  every {checkpoint_steps} steps")

    # ── data collator ──────────────────────────────────────────────────
    _gemma_tt_key = get_gemma_token_type_key(model) if model_family == "gemma" else "token_type_ids"
    if args.data_mode == "instruct":
        # Preserve -100 label masks baked by prepare_instruction_data.py.
        # Dynamic padding per batch — no wasted compute on padding tokens.
        data_collator = InstructionDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            model_family=model_family,
            gemma_tt_key=_gemma_tt_key,
        )
        logger.info("Using InstructionDataCollator (dynamic padding, preserved -100 labels)")
    else:
        # corpus mode: packed text blocks, all tokens are valid CLM targets.
        # DataCollatorForLanguageModeling is safe here because prepare_lora_data.py
        # produces no padding — blocks are always exactly max_length tokens.
        _base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        if model_family == "gemma":
            def data_collator(features):
                batch = _base_collator(features)
                batch[_gemma_tt_key] = torch.zeros_like(batch["input_ids"])
                return batch
            logger.info(f"Using DataCollatorForLanguageModeling + {_gemma_tt_key} injection (Gemma corpus)")
        else:
            data_collator = _base_collator
            logger.info("Using DataCollatorForLanguageModeling (packed corpus, CLM objective)")

    # ── training arguments ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        # IMPORTANT: eval_batch_size must be smaller than train batch_size.
        # Eval runs WITHOUT gradient checkpointing (full layer activations stay
        # in VRAM).  Using the same value as --batch_size caused a 100× eval
        # slowdown (39 min vs 23 sec) due to CUDA managed-memory paging to
        # system RAM on the 8B model near end of epoch 3.
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=max(1, int(args.warmup_ratio * (len(train_dataset) // (args.batch_size * args.grad_accum) + 1) * args.epochs)),
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=checkpoint_steps,
        eval_steps=checkpoint_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing and not _skip_gc,
        # use_reentrant=False is required when combining PEFT LoRA with gradient
        # checkpointing.  Without it, PyTorch's reentrant checkpoint may attempt
        # to recompute the frozen base-model backward pass and trigger a
        # "Trying to backward through the graph a second time" error.
        gradient_checkpointing_kwargs={"use_reentrant": False} if (args.gradient_checkpointing and not _skip_gc) else None,
        # For models with KV-shared layers (Gemma 4 E2B/E4B), use_cache
        # must stay True so the internal DynamicCache is constructed for
        # KV sharing.  TrainingArguments defaults use_cache=False, which
        # would override our model.config setting.
        use_cache=_has_kv_sharing,
        # eval_accumulation_steps: accumulate eval outputs in chunks before
        # moving to CPU.  Prevents OOM from building a giant logit tensor when
        # eval_dataset is large.
        eval_accumulation_steps=4,
        # prediction_loss_only: skip storing logits during eval — we only need
        # the scalar eval_loss, not per-token predictions.  Saves significant
        # VRAM and speeds up evaluation.
        prediction_loss_only=True,
        optim=optimizer,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        # load_best_model_at_end=False:
        # HuggingFace Trainer's best-checkpoint reload logic is incompatible
        # with PEFT in many versions — checkpoints contain only adapter weights,
        # not the full model, so the Trainer's from_pretrained reload fails.
        # BestAdapterSaveCallback handles best-checkpoint tracking manually.
        load_best_model_at_end=False,
    )

    # This script always loads the model with device_map="auto".  In a
    # single-process run, Trainer would otherwise wrap the PEFT model in
    # DataParallel on multi-GPU nodes.  That replicates the model and gathers
    # logits on GPU 0; for Gemma this immediately OOMs when the loss casts
    # logits to float32.  Keep the process in device_map mode unless the user
    # explicitly launched distributed training.
    if torch.cuda.device_count() > 1 and int(os.environ.get("LOCAL_RANK", "-1")) == -1:
        training_args._n_gpu = 1
        logger.warning(
            "Model loaded with device_map='auto' — disabling Trainer DataParallel. "
            "For better multi-GPU utilisation, launch with: "
            "torchrun --nproc_per_node=NUM_GPUS training/lora_finetune.py ..."
        )

    # ── trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[MemoryCallback(), BestAdapterSaveCallback(args.output_dir)],
    )

    # Gemma4ForConditionalGeneration.forward has **kwargs, which makes the
    # Trainer infer model_accepts_loss_kwargs=True.  That flag tells the
    # Trainer the model internally scales the loss by num_items_in_batch
    # across the gradient-accumulation window — but the model does NOT do
    # this (it uses CrossEntropyLoss(reduction='mean') per micro-batch).
    # As a result the logged loss is inflated by grad_accum and gradients
    # are scaled grad_accum× too large.  Override to False so the Trainer
    # normalises the loss correctly.
    if trainer.model_accepts_loss_kwargs:
        trainer.model_accepts_loss_kwargs = False
        logger.info(
            "Overrode model_accepts_loss_kwargs → False "
            "(model forward has **kwargs but does not scale loss by num_items_in_batch)"
        )

    # ── train ──────────────────────────────────────────────────────────
    # Suppress a cosmetic stream-mismatch warning that fires every backward pass
    # when device_map="auto" (pipeline parallelism) is combined with gradient
    # checkpointing.  Correctness is unaffected; PyTorch just adds a sync.
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    logger.info("")
    logger.info("Starting LoRA adapter training...")
    logger.info("")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── save adapter and tokenizer ─────────────────────────────────────
    # model.save_pretrained() on a PeftModel saves ONLY the adapter weights
    # (adapter_config.json + adapter_model.safetensors).
    # The base model at args.model_path is NEVER modified.
    logger.info(f"Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── metrics ────────────────────────────────────────────────────────
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("")
    logger.info("=" * 60)
    logger.info("LORA TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Source model (UNCHANGED): {args.model_path}")
    logger.info(f"LoRA adapter saved to:    {args.output_dir}")
    logger.info(f"  adapter_config.json       (LoRA config)")
    logger.info(f"  adapter_model.safetensors (trained weights)")
    logger.info(f"Best adapter checkpoint:  {args.output_dir}/best_adapter/")
    logger.info(f"  (lowest eval_loss snapshot — use if final save was interrupted)")
    logger.info(f"Training loss:            {metrics['train_loss']:.4f}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step — merge adapter into base model for evaluation:")
    logger.info(f"  python training/merge_lora_adapter.py \\")
    logger.info(f"    --adapter_path {args.output_dir} \\")
    logger.info(f"    --base_model_path {args.model_path}")
    logger.info("")
    logger.info("To resume a crashed run from the last checkpoint:")
    logger.info(f"  python training/lora_finetune.py \\")
    logger.info(f"    --model_path {args.model_path} \\")
    logger.info(f"    --data_dir {args.data_dir} \\")
    logger.info(f"    --output_dir {args.output_dir} \\")
    logger.info(f"    --resume_from_checkpoint {checkpoint_dir}/checkpoint-<N>")


if __name__ == "__main__":
    main()
