#!/usr/bin/env python3
"""
Merge a LoRA adapter back into the base model weights.

Produces a full-weight model saved to models/lora-merged/ that can be
used directly with the existing evaluation pipeline (evaluation/
hpn_qa_benchmark.py) without any PEFT dependencies.

The base model is loaded in bfloat16 (full precision, not 4-bit) for
merging so that the merged weights are clean.

Usage:
    # Auto-detect base model from adapter_config.json
    python training/merge_lora_adapter.py \\
        --adapter_path  models/lora/Llama-3.1-8B-base-instruct-lora

    # Explicit base model path
    python training/merge_lora_adapter.py \\
        --adapter_path    models/lora/Llama-3.1-8B-base-instruct-lora \\
        --base_model_path models/instruction/Llama-3.1-8B-base-instruct \\
        --output_dir      models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth
from utils.model_utils import detect_model_family, fix_tokenizer_padding

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


def _read_adapter_base(adapter_path: str) -> Optional[str]:
    """Try to read base_model_name_or_path from adapter_config.json."""
    cfg = Path(adapter_path) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        with open(cfg) as f:
            data = json.load(f)
        return data.get("base_model_name_or_path")
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model for evaluation"
    )
    p.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter directory "
             "(contains adapter_config.json and adapter_model.safetensors)",
    )
    p.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to the base model used during LoRA training.  "
             "If not specified, auto-detected from adapter_config.json.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the merged model "
             "(default: models/lora-merged/<adapter-dir-name>-merged/)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    setup_hf_auth()

    # ── resolve adapter path ───────────────────────────────────────────
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error(f"Adapter not found: {adapter_path}")
        sys.exit(1)
    if not (adapter_path / "adapter_config.json").exists():
        logger.error(f"adapter_config.json missing in: {adapter_path}")
        sys.exit(1)

    # ── resolve base model path ────────────────────────────────────────
    base_model_path = args.base_model_path
    if base_model_path is None:
        base_model_path = _read_adapter_base(str(adapter_path))
        if base_model_path:
            logger.info(f"Auto-detected base model: {base_model_path}")
        else:
            logger.error(
                "Could not auto-detect base model from adapter_config.json.  "
                "Provide --base_model_path explicitly."
            )
            sys.exit(1)

    if not Path(base_model_path).exists():
        logger.error(f"Base model not found: {base_model_path}")
        sys.exit(1)

    # ── resolve output path ────────────────────────────────────────────
    if args.output_dir is None:
        adapter_name = adapter_path.name
        args.output_dir = str(
            _ROOT / "models" / "lora-merged" / f"{adapter_name}-merged"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LORA ADAPTER MERGE")
    logger.info("=" * 60)
    logger.info(f"Base model:  {base_model_path}")
    logger.info(f"Adapter:     {adapter_path}")
    logger.info(f"Output:      {args.output_dir}")
    logger.info("=" * 60)

    # ── detect model family ────────────────────────────────────────────
    model_family = detect_model_family(base_model_path)
    logger.info(f"Detected model family: {model_family}")

    # ── load tokenizer ─────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )
    fix_tokenizer_padding(tokenizer, model_family)

    # ── load base model in bfloat16 (full precision for clean merge) ───
    logger.info(f"Loading base model in bfloat16 (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # ── load adapter and merge ──────────────────────────────────────────
    logger.info(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    logger.info("Merging adapter weights into base model (merge_and_unload)...")
    model = model.merge_and_unload()

    # ── save merged model ──────────────────────────────────────────────
    logger.info(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Merged model: {args.output_dir}")
    logger.info("The merged model is a full-weight model with no PEFT dependency.")
    logger.info("It can be used directly with evaluation/hpn_qa_benchmark.py:")
    logger.info("")
    logger.info(f"  python evaluation/hpn_qa_benchmark.py \\")
    logger.info(f"    --model_path {args.output_dir} \\")
    logger.info(f"    --benchmark  data/prompts/hpn_qa_benchmark_v2_general_skills.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
