#!/usr/bin/env python3
"""
Download and save base models locally.
Keeps original models separate from trained versions.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth

# Model configurations
MODELS = {
    "1b": {
        "name": "meta-llama/Llama-3.2-1B",
        "local_dir": "models/base/Llama-3.2-1B-base"
    },
    "8b": {
        "name": "meta-llama/Llama-3.1-8B",
        "local_dir": "models/base/Llama-3.1-8B-base"
    },
    "qwen-2b": {
        "name": "Qwen/Qwen3.5-2B-Base",
        "local_dir": "models/base/Qwen3.5-2B-Base"
    },
    "qwen-4b": {
        "name": "Qwen/Qwen3.5-4B-Base",
        "local_dir": "models/base/Qwen3.5-4B-Base"
    },
    "gemma-4b": {
        "name": "google/gemma-3-4b-pt",
        "local_dir": "models/base/gemma-3-4b"
    },
    "gemma-e4b": {
        "name": "google/gemma-4-E4B",
        "local_dir": "models/base/gemma-4-e4b"
    },
    "gemma-e2b": {
        "name": "google/gemma-4-E2B",
        "local_dir": "models/base/gemma-4-e2b"
    },
    "gemma-e4b-it": {
    "name": "google/gemma-4-E4B-it",
    "local_dir": "models/instruction/gemma-4-e4b-it"
    },

}


def download_model(model_key: str, force: bool = False):
    """Download and save a model locally."""
    
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")
    
    config = MODELS[model_key]
    model_name = config["name"]
    local_dir = config["local_dir"]
    
    # Check if already downloaded
    if Path(local_dir).exists() and not force:
        print(f"✅ Model already exists at: {local_dir}")
        print("   Use --force to re-download")
        return local_dir
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Saving to: {local_dir}")
    print(f"{'='*60}\n")
    
    # Create directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Download tokenizer
    print("📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(local_dir)
    print(f"   ✅ Tokenizer saved")
    
    # Download model
    print("\n📥 Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.save_pretrained(local_dir, safe_serialization=True)
    print(f"   ✅ Model saved")
    
    print(f"\n{'='*60}")
    print(f"✅ Download complete: {local_dir}")
    print(f"{'='*60}\n")
    
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download base models locally")
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        default="1b",
        help=f"Model to download: {', '.join(MODELS.keys())}, or all"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if exists"
    )
    
    args = parser.parse_args()
    
    # Setup authentication
    print("🔐 Setting up HuggingFace authentication...")
    setup_hf_auth()
    
    # Download models
    if args.model == "all":
        for key in MODELS:
            download_model(key, args.force)
    else:
        download_model(args.model, args.force)
    
    print("\n📁 Model Directory Structure:")
    print("   models/base/")
    print("   ├── Llama-3.2-1B-base/          <- Original Llama 1B (unchanged)")
    print("   ├── Llama-3.1-8B-base/          <- Original Llama 8B (unchanged)")
    print("   ├── Qwen3.5-2B-Base/            <- Original Qwen3.5 2B (unchanged)")
    print("   ├── Qwen3.5-4B-Base/            <- Original Qwen3.5 4B (unchanged)")
    print("   └── gemma-3-4b/                 <- Original Gemma 3 4B (unchanged)")
    print("\n   After pre-training:")
    print("   models/pretrained/")
    print("   ├── Llama-3.2-1B-trained-new/   <- Domain-adapted Llama 1B")
    print("   ├── Llama-3.1-8B-trained-new/   <- Domain-adapted Llama 8B")
    print("   ├── Qwen3.5-2B-trained-new/     <- Domain-adapted Qwen3.5 2B")
    print("   ├── Qwen3.5-4B-trained-new/     <- Domain-adapted Qwen3.5 4B")
    print("   └── gemma-3-4b-trained-new/     <- Domain-adapted Gemma 3 4B")
    print("\n   After instruction fine-tuning:")
    print("   models/instruction/")
    print("   ├── Llama-3.1-8B-base-instruct/        <- Instruct from base")
    print("   ├── Llama-3.1-8B-trained-new-instruct/  <- Instruct from trained")
    print("   └── gemma-3-4b-instruct/                <- Instruct from Gemma 3 4B")


if __name__ == "__main__":
    main()
