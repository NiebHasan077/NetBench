#!/usr/bin/env python3
"""CLI for Phase 2 corpus normalization."""

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

from instruct_ftd.normalize import (  # noqa: E402
    build_normalization_report,
    normalize_corpus,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize the Instruct-FTD OCR corpus.")
    parser.add_argument(
        "--corpus_path",
        default=str(PROJECT_ROOT.parent / "NetBench-LLM" / "data" / "raw" / "research_corpus_new.json"),
        help="Path to the local OCR-flattened research corpus JSON; the corpus is not distributed in Git.",
    )
    parser.add_argument(
        "--pdfs_dir",
        default=str(PROJECT_ROOT / "pdfs"),
        help="Directory containing source PDFs for best-effort mapping.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/normalized/normalized_papers.jsonl"),
        help="Output JSONL path for normalized papers.",
    )
    parser.add_argument(
        "--report_path",
        default=str(PROJECT_ROOT / "reports/phase2_normalization_report.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_path = Path(args.corpus_path)
    pdfs_dir = Path(args.pdfs_dir)
    output_jsonl = Path(args.output_jsonl)
    report_path = Path(args.report_path)

    if not corpus_path.exists():
        raise SystemExit(f"ERROR: corpus file not found: {corpus_path}\nPass --corpus_path to specify its location.")

    print(f"Corpus:  {corpus_path}")
    print(f"PDFs:    {pdfs_dir}")
    print(f"Output:  {output_jsonl}")
    print(f"Report:  {report_path}")

    t_start = time.monotonic()
    papers = normalize_corpus(corpus_path=corpus_path, pdfs_dir=pdfs_dir)
    print(f"Normalizing {len(papers)} papers...")
    write_jsonl(output_jsonl, [paper.to_dict() for paper in papers])

    report = build_normalization_report(papers)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = time.monotonic() - t_start
    print(f"Done in {elapsed:.1f}s  |  papers={len(papers)}")
    print(f"Normalized papers written to: {output_jsonl}")
    print(f"Normalization report written to: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

