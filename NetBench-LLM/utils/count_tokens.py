#!/usr/bin/env python3
"""
Count tokens in datasets - supports both processed and raw formats
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm


def count_tokens_processed_dataset(data_dir):
    """Count tokens in already processed dataset (Arrow format)"""
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return None
    
    print(f"\n{'='*70}")
    print(f"📊 Analyzing PROCESSED dataset: {data_dir.name}")
    print(f"{'='*70}\n")
    
    try:
        # Load dataset
        print("Loading dataset...")
        dataset = load_from_disk(str(data_dir))
        
        # Calculate tokens
        print("Counting tokens in training set...")
        train_tokens = sum(len(sample['input_ids']) for sample in tqdm(dataset['train']))
        
        print("Counting tokens in validation set...")
        val_tokens = sum(len(sample['input_ids']) for sample in tqdm(dataset['validation']))
        
        total_tokens = train_tokens + val_tokens
        
        train_samples = len(dataset['train'])
        val_samples = len(dataset['validation'])
        total_samples = train_samples + val_samples
        
        # Print statistics
        print("\n" + "="*70)
        print("📈 TOKEN STATISTICS")
        print("="*70)
        print(f"{'Metric':<30} {'Count':>20}")
        print("-"*70)
        print(f"{'Train samples':<30} {train_samples:>20,}")
        print(f"{'Validation samples':<30} {val_samples:>20,}")
        print(f"{'Total samples':<30} {total_samples:>20,}")
        print("-"*70)
        print(f"{'Train tokens':<30} {train_tokens:>20,}")
        print(f"{'Validation tokens':<30} {val_tokens:>20,}")
        print(f"{'TOTAL TOKENS':<30} {total_tokens:>20,}")
        print("-"*70)
        print(f"{'Average tokens/sample':<30} {total_tokens/total_samples:>20,.1f}")
        print("="*70)
        
        # Show sample details
        if train_samples > 0:
            sample = dataset['train'][0]
            print(f"\n📝 Sample Details (first training sample):")
            print(f"   Input IDs length:     {len(sample['input_ids']):>10,}")
            print(f"   Labels length:        {len(sample['labels']):>10,}")
            print(f"   Attention mask length:{len(sample['attention_mask']):>10,}")
        
        print()
        
        return {
            'train_tokens': train_tokens,
            'val_tokens': val_tokens,
            'total_tokens': total_tokens,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'total_samples': total_samples,
        }
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def count_tokens_raw_file(filepath, tokenizer_name="meta-llama/Llama-3.2-1B"):
    """Count tokens in raw JSON file"""
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ Error: File not found: {filepath}")
        return None
    
    print(f"\n{'='*70}")
    print(f"📊 Analyzing RAW file: {filepath.name}")
    print(f"{'='*70}\n")
    
    try:
        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Load JSON data
        print(f"Loading JSON data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} documents")
        
        # Extract texts
        texts = []
        for item in data:
            if isinstance(item, dict) and 'text' in item:
                text = item['text'].strip()
                if text:
                    texts.append(text)
            elif isinstance(item, str):
                texts.append(item.strip())
        
        print(f"Found {len(texts)} valid text documents")
        
        # Count tokens
        print("Tokenizing and counting...")
        total_tokens = 0
        token_counts = []
        
        for text in tqdm(texts, desc="Processing documents"):
            tokens = tokenizer.encode(text, add_special_tokens=True)
            num_tokens = len(tokens)
            token_counts.append(num_tokens)
            total_tokens += num_tokens
        
        # Statistics
        avg_tokens = total_tokens / len(texts) if texts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        
        # Print results
        print("\n" + "="*70)
        print("📈 TOKEN STATISTICS")
        print("="*70)
        print(f"{'Metric':<30} {'Count':>20}")
        print("-"*70)
        print(f"{'Total documents':<30} {len(texts):>20,}")
        print(f"{'TOTAL TOKENS':<30} {total_tokens:>20,}")
        print("-"*70)
        print(f"{'Average tokens/document':<30} {avg_tokens:>20,.1f}")
        print(f"{'Min tokens/document':<30} {min_tokens:>20,}")
        print(f"{'Max tokens/document':<30} {max_tokens:>20,}")
        print("-"*70)
        print(f"{'Tokenizer':<30} {tokenizer_name:>20}")
        print(f"{'Vocab size':<30} {len(tokenizer):>20,}")
        print("="*70)
        print()
        
        return {
            'total_tokens': total_tokens,
            'total_documents': len(texts),
            'avg_tokens': avg_tokens,
            'min_tokens': min_tokens,
            'max_tokens': max_tokens,
        }
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None


def count_all_datasets():
    """Count tokens in all available datasets"""
    
    print("\n" + "🔍 " + "="*68)
    print("  SCANNING ALL DATASETS IN PROJECT")
    print("="*70 + "\n")
    
    results = {}
    
    # Check processed datasets
    processed_dirs = [
        "data/processed/llama-3.2-1b",
        "data/processed/llama-3.1-8b",
        "processed_data/llama-3.2-1b",
        "processed_data/llama-3.1-8b",
        "processed_data_new/llama-3.2-1b",
        "processed_data_new/llama-3.1-8b",
    ]
    
    for data_dir in processed_dirs:
        if Path(data_dir).exists():
            result = count_tokens_processed_dataset(data_dir)
            if result:
                results[data_dir] = result
    
    # Check raw files
    raw_files = [
        "data/raw/research_corpus.json",
        "data/raw/research_corpus_new.json",
        "research_corpus.json",
        "research_corpus_new.json",
    ]
    
    for filepath in raw_files:
        if Path(filepath).exists():
            result = count_tokens_raw_file(filepath)
            if result:
                results[filepath] = result
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("📊 SUMMARY OF ALL DATASETS")
        print("="*70)
        for name, data in results.items():
            if 'total_tokens' in data:
                print(f"\n{name}:")
                print(f"  Total tokens: {data['total_tokens']:,}")
        print("="*70 + "\n")
    else:
        print("⚠️  No datasets found in project directory\n")


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count tokens in processed dataset
  python count_tokens.py --processed data/processed/llama-3.2-1b
  
  # Count tokens in raw JSON file
  python count_tokens.py --raw data/raw/research_corpus_new.json --tokenizer meta-llama/Llama-3.2-1B
  
  # Scan and count all datasets
  python count_tokens.py --all
        """
    )
    
    parser.add_argument(
        "--processed",
        type=str,
        help="Path to processed dataset directory (Arrow format)"
    )
    parser.add_argument(
        "--raw",
        type=str,
        help="Path to raw JSON file"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer to use for raw files (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Count tokens in all available datasets"
    )
    
    args = parser.parse_args()
    
    if args.all:
        count_all_datasets()
    elif args.processed:
        count_tokens_processed_dataset(args.processed)
    elif args.raw:
        count_tokens_raw_file(args.raw, args.tokenizer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
