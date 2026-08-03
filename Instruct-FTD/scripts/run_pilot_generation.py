#!/usr/bin/env python3
"""CLI for Phase 5 pilot generation with local Ollama."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from instruct_ftd.chunking import read_normalized_papers  # noqa: E402
from instruct_ftd.pilot import (  # noqa: E402
    default_run_dir,
    load_chunks,
    load_rag_bundles,
    run_pilot,
)
from instruct_ftd.prompt_families import FAMILIES, family_specs  # noqa: E402


DEFAULT_FAMILIES = [
    "hpn_fact_qa",
    "hpn_concept_explanation",
    "hpn_comparison",
    "hpn_diagnosis",
    "rag_grounded_qa",
    "rag_unanswerable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pilot synthetic generation pass against local Ollama.")
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
        help="Output directory for this pilot run. Defaults to a timestamped run dir.",
    )
    parser.add_argument(
        "--paper_limit",
        type=int,
        default=4,
        help="Number of high-quality papers to include in the pilot run.",
    )
    parser.add_argument(
        "--min_parse_confidence",
        type=float,
        default=0.45,
        help="Minimum parse confidence for pilot paper selection.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=DEFAULT_FAMILIES,
        help="Prompt families to execute. Defaults to a representative pilot subset.",
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
        default=500,
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
        help="Plan prompts and write manifest/raw rows without contacting Ollama.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous pilot run in the same output directory using checkpoint files.",
    )
    parser.add_argument(
        "--family_specs_out",
        default=str(PROJECT_ROOT / "reports/phase4_prompt_families.json"),
        help="Where to write the machine-readable family registry snapshot.",
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

    print(f"Phase 5 pilot generation")
    print(f"  model:         {args.model}")
    print(f"  paper_limit:   {args.paper_limit}")
    print(f"  families:      {', '.join(args.families)}")
    print(f"  dry_run:       {args.dry_run}")
    print(f"  resume:        {args.resume}")
    print(f"  output_dir:    {output_dir}")

    papers = read_normalized_papers(Path(args.normalized_jsonl))
    short_chunks = load_chunks(Path(args.short_chunks_jsonl))
    rag_chunks = load_chunks(Path(args.rag_chunks_jsonl))
    rag_bundles = load_rag_bundles(Path(args.rag_bundles_jsonl))

    manifest = run_pilot(
        model=args.model,
        output_dir=output_dir,
        papers=papers,
        short_chunks=short_chunks,
        rag_chunks=rag_chunks,
        rag_bundles=rag_bundles,
        family_names=args.families,
        paper_limit=args.paper_limit,
        min_parse_confidence=args.min_parse_confidence,
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

    print(f"Pilot run directory: {output_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
