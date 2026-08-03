#!/usr/bin/env python3
"""Count the size of the instruction fine-tuning (SFT) dataset used by Instruct-FTD.

Each split is JSONL with one example per line; the trained text is the
``system`` + ``question`` + ``response`` fields (other fields are metadata).
Reports per-split example and token counts (OpenAI ``cl100k_base`` BPE, the same
reference tokenizer used for the corpus). The numbers quoted in the paper come
from this script.

Usage:
  analysis/.venv/bin/python analysis/count_sft_tokens.py [path/to/v3_run_plus_json]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import tiktoken

DEFAULT = "NetBench-LLM/data/Instruct-FTD/v3_run_plus_json"
FIELDS = ("system", "question", "response")


def count_split(path: str, enc) -> tuple[int, int]:
    n_ex = n_tok = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = "\n".join(str(rec.get(k, "")) for k in FIELDS)
            n_ex += 1
            n_tok += len(enc.encode_ordinary(text))
    return n_ex, n_tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", nargs="?", default=DEFAULT, help=f"SFT dir (default: {DEFAULT})")
    ap.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    tot_ex = tot_tok = 0
    for split in ("train", "validation"):
        path = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            sys.exit(f"not found: {path}\n(it is gitignored; pass the absolute dir)")
        ex, tok = count_split(path, enc)
        tot_ex += ex
        tot_tok += tok
        print(f"{split:10s}: {ex:>6,} examples   {tok:>12,} tokens")
    print(f"{'total':10s}: {tot_ex:>6,} examples   {tot_tok:>12,} tokens  ({args.encoding})")
    print(f"           ~ {tot_tok/1e6:.1f}M tokens, {tot_ex:,} examples")


if __name__ == "__main__":
    main()
