#!/usr/bin/env python3
"""
Data Preparation Script for Pre-training
Converts research_corpus.json into tokenized format for LLaMA pre-training
"""

import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

def load_research_corpus(file_path):
    """Load research corpus from JSON file"""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} documents")
    return data

def prepare_texts(data):
    """Extract text from research corpus"""
    texts = []
    for item in tqdm(data, desc="Extracting texts"):
        if isinstance(item, dict) and 'text' in item:
            text = item['text'].strip()
            if text:  # Only add non-empty texts
                texts.append(text)
        elif isinstance(item, str):
            texts.append(item.strip())
    
    print(f"Prepared {len(texts)} text documents")
    return texts

def tokenize_and_save(texts, model_name, output_dir, max_length=2048, test_size=0.05):
    """Tokenize texts and save to disk"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # EOS token ID used to separate documents inside packed blocks.
    # This teaches the model that a sequence ends at EOS and the next token
    # belongs to a different document — preventing cross-document repetition.
    eos_token_id = tokenizer.eos_token_id

    # Create dataset
    print("Creating dataset...")
    dataset = Dataset.from_dict({"text": texts})

    # Split into train/validation
    print(f"Splitting dataset (test_size={test_size})...")
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)

    # Tokenization function — no truncation so full documents are preserved
    # before packing.  EOS is appended to every document so the model learns
    # document boundaries and stops generating at the right place.
    def tokenize_function(examples):
        """Tokenize texts for causal language modeling"""
        result = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )
        # Append EOS token to every document so packed blocks contain natural
        # stop signals between documents.
        result["input_ids"] = [
            ids + [eos_token_id] for ids in result["input_ids"]
        ]
        result["attention_mask"] = [
            mask + [1] for mask in result["attention_mask"]
        ]
        return result

    # Group texts into chunks for efficient training
    def group_texts(examples, block_size=max_length):
        """Group texts into fixed-length blocks"""
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Drop the small remainder
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        # Create labels (same as input_ids for causal LM)
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize datasets
    print("Tokenizing train dataset...")
    tokenized_train = split_dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=split_dataset["train"].column_names,
        desc="Tokenizing train dataset",
        num_proc=4,
    )

    print("Tokenizing validation dataset...")
    tokenized_val = split_dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=split_dataset["test"].column_names,
        desc="Tokenizing validation dataset",
        num_proc=4,
    )

    # Group into blocks
    print("Grouping texts into blocks...")
    lm_train = tokenized_train.map(
        lambda examples: group_texts(examples, max_length),
        batched=True,
        desc="Grouping train texts",
    )

    lm_val = tokenized_val.map(
        lambda examples: group_texts(examples, max_length),
        batched=True,
        desc="Grouping validation texts",
    )
    
    # Create final dataset dict
    final_dataset = DatasetDict({
        "train": lm_train,
        "validation": lm_val
    })
    
    # Save to disk
    print(f"Saving processed dataset to {output_dir}...")
    final_dataset.save_to_disk(output_dir)
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    print(f"Train samples: {len(final_dataset['train'])}")
    print(f"Validation samples: {len(final_dataset['validation'])}")
    print(f"Max sequence length: {max_length}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print("="*50)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    return final_dataset

def main():
    parser = argparse.ArgumentParser(description="Prepare research corpus for pre-training")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/raw/research_corpus_new.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name for tokenizer (meta-llama/Llama-3.2-1B or meta-llama/Llama-3.1-8B)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        help="Validation split size"
    )
    
    args = parser.parse_args()
    
    # Load data
    data = load_research_corpus(args.input_file)
    
    # Prepare texts
    texts = prepare_texts(data)
    
    if len(texts) == 0:
        print("ERROR: No texts found in the dataset!")
        return
    
    # Tokenize and save
    dataset = tokenize_and_save(
        texts=texts,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        test_size=args.test_size
    )
    
    print("\n✅ Data preparation complete!")
    print(f"\nTo start pre-training, run:")
    print(f"  python training/pretrain_transformers.py --data_dir {args.output_dir}")

if __name__ == "__main__":
    main()
