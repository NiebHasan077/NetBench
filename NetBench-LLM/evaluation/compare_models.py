#!/usr/bin/env python3
"""
Compare Base vs Trained Models
Loads both models and compares their outputs on the same prompts.
"""

import json
import argparse
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Model directories
MODELS = {
    "1b": {
        "base": "models/base/Llama-3.2-1B-base",
        "trained": "models/pretrained/Llama-3.2-1B-trained-new"
    },
    "8b": {
        "base": "models/base/Llama-3.1-8B-base",
        "trained": "models/pretrained/Llama-3.1-8B-trained-new"
    }
}

PROMPT_TEMPLATES = {
    "academic": "In the field of distributed computing and high-performance data transfer, {topic} refers to",
    "continuation": "{topic} is a technology that",
    "factual": "The main purpose of {topic} is to",
}


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
    """Generate completion."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_topics(file_path: str) -> list:
    """Load topics from JSON."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("topics", [])
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare base vs trained models")
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["1b", "8b"],
        default="1b",
        help="Model size to compare"
    )
    parser.add_argument(
        "--prompts_file", "-p",
        type=str,
        default="data/prompts/test_prompts.json",
        help="JSON file with topics"
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        default=None,
        help="Output JSON file"
    )
    parser.add_argument(
        "--template", "-t",
        type=str,
        default="academic",
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Prompt template"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Max tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Get model paths
    config = MODELS[args.model]
    base_path = config["base"]
    trained_path = config["trained"]
    
    # Check models exist
    if not Path(base_path).exists():
        print(f"❌ Base model not found: {base_path}")
        print(f"   Run: python utils/download_models.py --model {args.model}")
        sys.exit(1)
    
    if not Path(trained_path).exists():
        print(f"❌ Trained model not found: {trained_path}")
        print(f"   Run training first with: python training/pretrain_transformers.py --model {args.model}")
        sys.exit(1)
    
    # Output file
    if args.output_file is None:
        args.output_file = f"outputs/evaluations/comparison_{args.model}.json"
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load topics
    topics = load_topics(args.prompts_file)
    print(f"Loaded {len(topics)} topics")
    
    # Load BASE model
    print("\n" + "="*60)
    print("Loading BASE model (original, unchanged)")
    print("="*60)
    base_model, base_tokenizer = load_model(base_path)
    
    # Generate base responses
    print("\nGenerating BASE model responses...")
    base_results = []
    for topic in tqdm(topics):
        prompt = PROMPT_TEMPLATES[args.template].format(topic=topic)
        response = generate(base_model, base_tokenizer, prompt, args.max_tokens)
        base_results.append({"topic": topic, "prompt": prompt, "response": response})
    
    # Free memory
    del base_model, base_tokenizer
    torch.cuda.empty_cache()
    
    # Load TRAINED model
    print("\n" + "="*60)
    print("Loading TRAINED model (trained on your corpus)")
    print("="*60)
    trained_model, trained_tokenizer = load_model(trained_path)
    
    # Generate trained responses
    print("\nGenerating TRAINED model responses...")
    trained_results = []
    for topic in tqdm(topics):
        prompt = PROMPT_TEMPLATES[args.template].format(topic=topic)
        response = generate(trained_model, trained_tokenizer, prompt, args.max_tokens)
        trained_results.append({"topic": topic, "prompt": prompt, "response": response})
    
    # Free memory
    del trained_model, trained_tokenizer
    torch.cuda.empty_cache()
    
    # Compare and print results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    comparisons = []
    for base, trained in zip(base_results, trained_results):
        print(f"\n📝 Topic: {base['topic']}")
        print(f"   Prompt: {base['prompt']}")
        print(f"\n   🔵 BASE MODEL:")
        print(f"      {base['response'][:300]}...")
        print(f"\n   🟢 TRAINED MODEL:")
        print(f"      {trained['response'][:300]}...")
        print("-"*70)
        
        comparisons.append({
            "topic": base["topic"],
            "prompt": base["prompt"],
            "base_response": base["response"],
            "trained_response": trained["response"],
        })
    
    # Save results
    output_data = {
        "model_size": args.model,
        "base_model": base_path,
        "trained_model": trained_path,
        "timestamp": datetime.now().isoformat(),
        "template": args.template,
        "comparisons": comparisons
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Comparison saved to: {args.output_file}")
    print("\nSummary:")
    print(f"   Base model: {base_path}")
    print(f"   Trained model: {trained_path}")
    print(f"   Topics compared: {len(topics)}")


if __name__ == "__main__":
    main()
