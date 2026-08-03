"""Stage 01 — passage chunking.

Splits each paper's text into sliding-window passages (1000 tokens, 100 overlap
by default) with stable IDs of the form `paper_<NNNN>__p<NNN>`. The output is
the ground truth for evidence-quote substring checks downstream.

Usage:
    .venv/bin/python -m src.stage_01_chunk            # chunk full corpus
    .venv/bin/python -m src.stage_01_chunk --limit 5  # quick smoke
    .venv/bin/python -m src.stage_01_chunk --resume   # skip papers already in output

Output (JSONL): {paper_id, passage_id, text}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterator

import tiktoken
from tqdm import tqdm

from src.config import load_config
from src.io_utils import append_jsonl, read_jsonl
from src.schema import Passage

logger = logging.getLogger("stage_01")


def load_corpus(path: Path) -> list[str]:
    """Read corpus JSON and return a list of paper texts.

    Accepts either a list of strings or a list of dicts with a `text` key
    (the shape used by research_corpus_v3.json).
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list at {path}, got {type(data).__name__}")

    texts: list[str] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and "text" in item:
            t = item["text"]
            if not isinstance(t, str):
                raise ValueError(f"paper {i}: 'text' is not a string")
            texts.append(t)
        else:
            raise ValueError(
                f"paper {i}: expected str or dict with 'text', got {type(item).__name__}"
            )
    return texts


def chunk_paper(
    paper_idx: int,
    text: str,
    encoder: tiktoken.Encoding,
    window: int,
    overlap: int,
    paper_pad: int,
    chunk_pad: int = 3,
) -> Iterator[Passage]:
    """Yield Passage objects for one paper using a sliding token window."""
    if not text or not text.strip():
        return
    tokens = encoder.encode(text, disallowed_special=())
    if not tokens:
        return

    stride = window - overlap
    if stride <= 0:
        raise ValueError(f"window ({window}) must be greater than overlap ({overlap})")

    paper_id = f"paper_{paper_idx:0{paper_pad}d}"
    chunk_idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + window, len(tokens))
        passage_text = encoder.decode(tokens[start:end])
        yield Passage(
            paper_id=paper_id,
            passage_id=f"{paper_id}__p{chunk_idx:0{chunk_pad}d}",
            text=passage_text,
        )
        chunk_idx += 1
        if end >= len(tokens):
            break
        start += stride


def existing_paper_ids(output_path: Path) -> set[str]:
    return {rec["paper_id"] for rec in read_jsonl(output_path) if "paper_id" in rec}


def existing_passage_ids(output_path: Path) -> set[str]:
    return {rec["passage_id"] for rec in read_jsonl(output_path) if "passage_id" in rec}


def spot_check(output_path: Path, n: int = 5, max_chars: int = 240) -> None:
    """Print n random passage previews so a human can eyeball-verify them."""
    records = list(read_jsonl(output_path))
    if not records:
        logger.warning("output empty; nothing to spot-check")
        return
    sample = random.sample(records, min(n, len(records)))
    print("\n--- spot-check (random passages) ---")
    for rec in sample:
        preview = rec["text"][:max_chars].replace("\n", " ")
        suffix = "..." if len(rec["text"]) > max_chars else ""
        print(f"[{rec['passage_id']}] {preview}{suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--input", default=None, help="Path to corpus JSON (default: from config.yaml)")
    parser.add_argument("--output", default=None, help="Path to passages JSONL (default: data/passages.jsonl)")
    parser.add_argument("--resume", action="store_true", help="Skip papers already present in output")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N papers (debug)")
    parser.add_argument("--spot-check", type=int, default=5, help="Print N random passages at end (0 to disable)")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed for spot-check (default: from config.yaml)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    in_path = Path(args.input) if args.input else cfg.resolve(cfg.paths.corpus)
    out_path = Path(args.output) if args.output else cfg.resolve(cfg.paths.data_dir) / "passages.jsonl"
    if not in_path.exists():
        logger.error("corpus not found: %s", in_path)
        return 1

    if not args.resume and out_path.exists():
        logger.info("clearing existing output: %s", out_path)
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_paper_ids = existing_paper_ids(out_path) if args.resume else set()
    seen_passage_ids: set[str] = existing_passage_ids(out_path) if args.resume else set()
    if args.resume:
        logger.info(
            "resume: %d paper_ids and %d passage_ids already in output",
            len(skip_paper_ids), len(seen_passage_ids),
        )

    logger.info("loading corpus: %s", in_path)
    texts = load_corpus(in_path)
    logger.info("loaded %d papers", len(texts))

    if args.limit:
        texts = texts[: args.limit]
        logger.info("--limit applied: processing %d papers", len(texts))

    paper_pad = max(4, len(str(len(texts) - 1))) if texts else 4
    encoder = tiktoken.get_encoding("cl100k_base")
    window = cfg.chunking.window_tokens
    overlap = cfg.chunking.overlap_tokens

    n_papers_chunked = 0
    n_passages_written = 0
    n_skipped = 0
    n_empty: list[int] = []

    for paper_idx, text in enumerate(tqdm(texts, desc="chunking", unit="paper")):
        paper_id = f"paper_{paper_idx:0{paper_pad}d}"
        if paper_id in skip_paper_ids:
            n_skipped += 1
            continue

        produced = 0
        for passage in chunk_paper(paper_idx, text, encoder, window, overlap, paper_pad=paper_pad):
            if passage.passage_id in seen_passage_ids:
                logger.error("duplicate passage_id detected: %s", passage.passage_id)
                return 2
            seen_passage_ids.add(passage.passage_id)
            append_jsonl(out_path, passage.model_dump())
            n_passages_written += 1
            produced += 1

        if produced > 0:
            n_papers_chunked += 1
        else:
            n_empty.append(paper_idx)
            logger.warning("paper %d (%s) produced 0 passages (empty/whitespace text)",
                           paper_idx, paper_id)

    logger.info(
        "done: %d papers chunked, %d passages written, %d skipped (resume), %d empty",
        n_papers_chunked, n_passages_written, n_skipped, len(n_empty),
    )

    # Coverage check (Phase 1 exit criterion)
    if n_empty:
        logger.warning("EMPTY PAPERS: %s", n_empty[:20])

    if args.spot_check > 0:
        seed = args.seed if args.seed is not None else cfg.seed
        random.seed(seed)
        spot_check(out_path, n=args.spot_check)

    return 0


if __name__ == "__main__":
    sys.exit(main())
