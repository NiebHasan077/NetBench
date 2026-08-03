#!/usr/bin/env python3
"""CLI for Phase 6 full candidate generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from instruct_ftd.candidate_generation import (  # noqa: E402
    default_run_dir,
    run_candidate_generation,
)
from instruct_ftd.chunking import read_normalized_papers  # noqa: E402
from instruct_ftd.pilot import load_chunks, load_rag_bundles  # noqa: E402
from instruct_ftd.prompt_families import FAMILIES, family_specs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 full candidate generation against local Ollama.")
    parser.add_argument(
        "--model",
        default="gemma4:26b",
        help="Local Ollama teacher model name.",
    )
    parser.add_argument(
        "--normalized_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/normalized/normalized_papers.jsonl"),
        help="Input normalized paper JSONL.",
    )
    parser.add_argument(
        "--short_chunks_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/chunks/short_chunks.jsonl"),
        help="Input short chunk JSONL.",
    )
    parser.add_argument(
        "--rag_chunks_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/chunks/rag_chunks.jsonl"),
        help="Input RAG chunk JSONL.",
    )
    parser.add_argument(
        "--rag_bundles_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/chunks/rag_prompt_bundles.jsonl"),
        help="Input RAG bundle JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory for this run. Defaults to a timestamped candidates run dir.",
    )
    parser.add_argument(
        "--paper_limit",
        type=int,
        default=0,
        help="Maximum number of papers to process. Use 0 for all eligible papers.",
    )
    parser.add_argument(
        "--min_parse_confidence",
        type=float,
        default=0.45,
        help="Minimum parse confidence for full generation.",
    )
    parser.add_argument(
        "--min_keyword_score",
        type=int,
        default=1,
        help="Minimum HPN keyword score required for paper selection.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional family override applied to every selected paper. If omitted, use the default per-paper 4-5 example plan.",
    )
    parser.add_argument(
        "--include_unanswerable_every",
        type=int,
        default=5,
        help="When using the default plan, add one rag_unanswerable request every N papers. Use 0 to disable.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Ollama sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Ollama top-p value.",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=700,
        help="Maximum generated tokens for each teacher response.",
    )
    parser.add_argument(
        "--base_url",
        default="http://127.0.0.1:11434",
        help="Local Ollama base URL.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Plan requests and write raw rows without contacting Ollama.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous candidate-generation run in the same output directory using checkpoint files.",
    )
    parser.add_argument(
        "--family_specs_out",
        default=str(PROJECT_ROOT / "reports/phase6_prompt_families.json"),
        help="Where to write the prompt-family registry snapshot for this phase.",
    )
    parser.add_argument(
        "--list_families",
        action="store_true",
        help="Print all available prompt family names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_families:
        print("Available prompt families:")
        for name in sorted(FAMILIES):
            print(f"  {name}")
        raise SystemExit(0)

    if args.families:
        unknown = [name for name in args.families if name not in FAMILIES]
        if unknown:
            raise SystemExit(f"Unknown prompt families: {', '.join(unknown)}")

    # Validate input files before spending time loading anything
    input_files = {
        "--normalized_jsonl": Path(args.normalized_jsonl),
        "--short_chunks_jsonl": Path(args.short_chunks_jsonl),
        "--rag_chunks_jsonl": Path(args.rag_chunks_jsonl),
        "--rag_bundles_jsonl": Path(args.rag_bundles_jsonl),
    }
    for flag, path in input_files.items():
        if not path.exists():
            raise SystemExit(f"ERROR: input file not found: {path}\nPass {flag} to specify its location.")

    output_dir = Path(args.output_dir) if args.output_dir else default_run_dir(PROJECT_ROOT)
    paper_limit = None if args.paper_limit <= 0 else args.paper_limit

    print(f"Phase 6 candidate generation")
    print(f"  model:         {args.model}")
    print(f"  paper_limit:   {paper_limit if paper_limit is not None else 'all'}")
    print(f"  families:      {', '.join(args.families) if args.families else 'default plan'}")
    print(f"  dry_run:       {args.dry_run}")
    print(f"  resume:        {args.resume}")
    print(f"  output_dir:    {output_dir}")

    papers = read_normalized_papers(Path(args.normalized_jsonl))
    short_chunks = load_chunks(Path(args.short_chunks_jsonl))
    rag_chunks = load_chunks(Path(args.rag_chunks_jsonl))
    rag_bundles = load_rag_bundles(Path(args.rag_bundles_jsonl))

    manifest = run_candidate_generation(
        model=args.model,
        output_dir=output_dir,
        papers=papers,
        short_chunks=short_chunks,
        rag_chunks=rag_chunks,
        rag_bundles=rag_bundles,
        paper_limit=paper_limit,
        min_parse_confidence=args.min_parse_confidence,
        min_keyword_score=args.min_keyword_score,
        families_override=args.families,
        include_unanswerable_every=args.include_unanswerable_every,
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=args.num_predict,
        base_url=args.base_url,
        dry_run=args.dry_run,
        resume=args.resume,
    )

    family_specs_path = Path(args.family_specs_out)
    family_specs_path.parent.mkdir(parents=True, exist_ok=True)
    family_specs_path.write_text(json.dumps(family_specs(), indent=2), encoding="utf-8")

    print(f"Candidate run directory: {output_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
