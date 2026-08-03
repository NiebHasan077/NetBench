#!/usr/bin/env python3
"""CLI for Phase 3 evidence chunk generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from instruct_ftd.chunking import (  # noqa: E402
    ChunkProfile,
    build_chunks_for_profile,
    build_rag_bundles,
    read_normalized_papers,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build short and RAG chunk artifacts.")
    parser.add_argument(
        "--normalized_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/normalized/normalized_papers.jsonl"),
        help="Input normalized papers JSONL.",
    )
    parser.add_argument(
        "--short_chunks_out",
        default=str(PROJECT_ROOT / "data/intermediate/chunks/short_chunks.jsonl"),
        help="Output JSONL for short HPN chunks.",
    )
    parser.add_argument(
        "--rag_chunks_out",
        default=str(PROJECT_ROOT / "data/intermediate/chunks/rag_chunks.jsonl"),
        help="Output JSONL for RAG chunks.",
    )
    parser.add_argument(
        "--rag_bundles_out",
        default=str(PROJECT_ROOT / "data/intermediate/chunks/rag_prompt_bundles.jsonl"),
        help="Output JSONL for RAG prompt bundle candidates.",
    )
    parser.add_argument(
        "--report_path",
        default=str(PROJECT_ROOT / "reports/phase3_chunking_report.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    normalized_jsonl = Path(args.normalized_jsonl)
    if not normalized_jsonl.exists():
        raise SystemExit(
            f"ERROR: normalized papers file not found: {normalized_jsonl}\n"
            "Run normalize_corpus.py first or pass --normalized_jsonl."
        )

    print(f"Input:   {normalized_jsonl}")
    print(f"Outputs: {args.short_chunks_out}")
    print(f"         {args.rag_chunks_out}")
    print(f"         {args.rag_bundles_out}")

    t_start = time.monotonic()
    papers = read_normalized_papers(normalized_jsonl)
    print(f"Loaded {len(papers)} papers — building chunks...")

    short_profile = ChunkProfile(
        name="short",
        target_tokens=220,
        overlap_tokens=40,
        min_tokens=60,
    )
    rag_profile = ChunkProfile(
        name="rag",
        target_tokens=900,
        overlap_tokens=120,
        min_tokens=140,
    )

    short_chunks = build_chunks_for_profile(papers, short_profile)
    rag_chunks = build_chunks_for_profile(papers, rag_profile)
    rag_bundles = build_rag_bundles(rag_chunks)

    write_jsonl(Path(args.short_chunks_out), [chunk.to_dict() for chunk in short_chunks])
    write_jsonl(Path(args.rag_chunks_out), [chunk.to_dict() for chunk in rag_chunks])
    write_jsonl(Path(args.rag_bundles_out), [bundle.to_dict() for bundle in rag_bundles])

    report = {
        "paper_count": len(papers),
        "short_chunk_count": len(short_chunks),
        "rag_chunk_count": len(rag_chunks),
        "rag_bundle_count": len(rag_bundles),
        "avg_short_chunks_per_paper": round(len(short_chunks) / len(papers), 2) if papers else 0.0,
        "avg_rag_chunks_per_paper": round(len(rag_chunks) / len(papers), 2) if papers else 0.0,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = time.monotonic() - t_start
    print(f"Done in {elapsed:.1f}s  |  short_chunks={len(short_chunks)}  rag_chunks={len(rag_chunks)}  rag_bundles={len(rag_bundles)}")
    print(f"Short chunks written to: {args.short_chunks_out}")
    print(f"RAG chunks written to: {args.rag_chunks_out}")
    print(f"RAG bundles written to: {args.rag_bundles_out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
