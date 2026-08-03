#!/usr/bin/env python3
"""
Model Evaluation Script for BASE LLMs
Evaluates a single model using completion-style prompts.
Use compare_models.py to compare base vs trained models.
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelEvaluator:
    """Evaluator for BASE language models using completion-style prompts."""
    
    PROMPT_TEMPLATES = {
        "academic": "In the field of distributed computing and high-performance data transfer, {topic} refers to",
        "continuation": "{topic} is a technology that",
        "factual": "The main purpose of {topic} is to",
        "technical": "When discussing grid computing infrastructure, {topic} provides",
    }
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        
        self.model.eval()
        print(f"Model loaded on device")
        
    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion for the given prompt."""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        generated = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def evaluate_topic(self, topic: str, template_name: str = "academic", **kwargs) -> dict:
        """Evaluate model on a single topic."""
        
        template = self.PROMPT_TEMPLATES.get(template_name, self.PROMPT_TEMPLATES["academic"])
        prompt = template.format(topic=topic)
        response = self.generate_completion(prompt, **kwargs)
        
        return {
            "topic": topic,
            "template": template_name,
            "prompt": prompt,
            "response": response,
        }


def load_topics(prompts_file: str) -> list:
    """Load topics from JSON file."""
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("topics", data.get("prompts", []))
    return []


def main():
    parser = argparse.ArgumentParser(description="Evaluate a BASE LLM")
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        required=True,
        help="Path to local model directory"
    )
    parser.add_argument(
        "--prompts_file", "-p",
        type=str,
        default="test_prompts.json",
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
        choices=["academic", "continuation", "factual", "technical"],
        help="Prompt template"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Max tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    # Set output file
    if args.output_file is None:
        model_name = Path(args.model_path).name
        args.output_file = f"outputs/evaluations/{model_name}_eval.json"
    
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load topics
    topics = load_topics(args.prompts_file)
    print(f"Loaded {len(topics)} topics")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path)
    evaluator.load_model()
    
    # Run evaluation
    results = []
    for topic in tqdm(topics, desc="Evaluating"):
        result = evaluator.evaluate_topic(
            topic, 
            args.template,
            max_new_tokens=args.max_tokens
        )
        results.append(result)
        print(f"\n📝 {topic}")
        print(f"   {result['response'][:150]}...")
    
    # Save results
    output_data = {
        "model": args.model_path,
        "timestamp": datetime.now().isoformat(),
        "template": args.template,
        "results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
