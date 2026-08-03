#!/usr/bin/env python3
"""Count the size of the shared HPN research corpus used for continual pre-training.

The corpus is a JSON array of records ``[{"text": ...}, ...]`` (gitignored; see
DATA_PROVENANCE.md). Reports record / character / word counts and a tokenizer-
referenced token count (OpenAI ``cl100k_base`` BPE, a stable reference tokenizer).
The number quoted in the paper comes from this script.

Usage:
  analysis/.venv/bin/python analysis/count_corpus_tokens.py [path/to/research_corpus_v3.json]
"""
from __future__ import annotations
import argparse
import json
import sys

import tiktoken

DEFAULT = "NetBench-LLM/data/raw/research_corpus_v3.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", nargs="?", default=DEFAULT, help=f"corpus JSON (default: {DEFAULT})")
    ap.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    args = ap.parse_args()

    try:
        with open(args.corpus, encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        sys.exit(f"corpus not found: {args.corpus}\n(it is gitignored; pass the absolute path)")

    texts = [str(r.get("text", "")) for r in records]
    enc = tiktoken.get_encoding(args.encoding)
    n_tokens = sum(len(t) for t in enc.encode_ordinary_batch(texts))
    n_chars = sum(len(t) for t in texts)
    n_words = sum(len(t.split()) for t in texts)

    print(f"corpus     : {args.corpus}")
    print(f"records    : {len(records):,}")
    print(f"characters : {n_chars:,}")
    print(f"words      : {n_words:,}")
    print(f"tokens     : {n_tokens:,}  ({args.encoding})")
    print(f"           ~ {n_tokens/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
