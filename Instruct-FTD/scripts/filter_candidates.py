#!/usr/bin/env python3
"""CLI for Phase 7 filtering, judging, and deduplication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from instruct_ftd.filtering import default_run_dir, run_filtering  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 7 filtering over candidate examples.")
    parser.add_argument(
        "--input_hpn",
        default="",
        help="Input HPN candidate JSONL (e.g. data/intermediate/candidates/<run>/candidate_hpn.jsonl).",
    )
    parser.add_argument(
        "--input_rag",
        default="",
        help="Input RAG candidate JSONL (e.g. data/intermediate/candidates/<run>/candidate_rag.jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory for filtered artifacts. Defaults to a timestamped filtered run dir.",
    )
    parser.add_argument(
        "--judge_model",
        default="",
        help="Optional local Ollama model for post-dedup judging. Leave empty for heuristic-only filtering.",
    )
    parser.add_argument(
        "--base_url",
        default="http://127.0.0.1:11434",
        help="Local Ollama base URL for optional judging.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a judged filtering run in the same output directory using saved judge results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_hpn:
        raise SystemExit(
            "ERROR: --input_hpn is required.\n"
            "Example: data/intermediate/candidates/<run>/candidate_hpn.jsonl"
        )
    if not args.input_rag:
        raise SystemExit(
            "ERROR: --input_rag is required.\n"
            "Example: data/intermediate/candidates/<run>/candidate_rag.jsonl"
        )

    input_hpn = Path(args.input_hpn)
    input_rag = Path(args.input_rag)

    if not input_hpn.exists():
        raise SystemExit(f"ERROR: HPN candidate file not found: {input_hpn}")
    if not input_rag.exists():
        raise SystemExit(f"ERROR: RAG candidate file not found: {input_rag}")

    output_dir = Path(args.output_dir) if args.output_dir else default_run_dir(PROJECT_ROOT)
    report = run_filtering(
        input_hpn_path=input_hpn,
        input_rag_path=input_rag,
        output_dir=output_dir,
        judge_model=args.judge_model or None,
        base_url=args.base_url,
        resume=args.resume,
    )
    print(f"Filtered output directory: {output_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
