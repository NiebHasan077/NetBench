#!/usr/bin/env python3
"""CLI for Phase 8 generic anchor import."""

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

from instruct_ftd.anchors import (  # noqa: E402
    build_anchor_report,
    import_generic_anchors,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and normalize generic OpenOrca/Dolly anchor data.")
    parser.add_argument(
        "--orca_local",
        default="",
        help="Optional local JSON/JSONL file containing OpenOrca-like rows.",
    )
    parser.add_argument(
        "--dolly_local",
        default="",
        help="Optional local JSON/JSONL file containing Dolly-like rows.",
    )
    parser.add_argument(
        "--orca_hf_split",
        default="train[:10000]",
        help="HF split expression for OpenOrca when not using --orca_local.",
    )
    parser.add_argument(
        "--dolly_hf_split",
        default="train",
        help="HF split expression for Dolly when not using --dolly_local.",
    )
    parser.add_argument(
        "--max_orca_samples",
        type=int,
        default=10000,
        help="Maximum OpenOrca samples to keep after normalization.",
    )
    parser.add_argument(
        "--max_dolly_samples",
        type=int,
        default=15000,
        help="Maximum Dolly samples to keep after normalization.",
    )
    parser.add_argument(
        "--min_response_words",
        type=int,
        default=20,
        help="Minimum response length for a retained generic anchor example.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/anchors/generic_anchor.jsonl"),
        help="Output JSONL path for normalized generic anchors.",
    )
    parser.add_argument(
        "--report_path",
        default=str(PROJECT_ROOT / "reports/phase8_anchor_stats.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Phase 8 generic anchor import")
    if args.orca_local:
        print(f"  orca source: {args.orca_local} (local)")
    else:
        print(f"  orca source: HuggingFace split='{args.orca_hf_split}' max={args.max_orca_samples}")
    if args.dolly_local:
        print(f"  dolly source: {args.dolly_local} (local)")
    else:
        print(f"  dolly source: HuggingFace split='{args.dolly_hf_split}' max={args.max_dolly_samples}")
    print(f"  output:      {args.output_jsonl}")

    t_start = time.monotonic()
    records = import_generic_anchors(
        orca_local=Path(args.orca_local) if args.orca_local else None,
        dolly_local=Path(args.dolly_local) if args.dolly_local else None,
        orca_hf_split=args.orca_hf_split,
        dolly_hf_split=args.dolly_hf_split,
        max_orca_samples=args.max_orca_samples,
        max_dolly_samples=args.max_dolly_samples,
        min_response_words=args.min_response_words,
    )

    output_jsonl = Path(args.output_jsonl)
    report_path = Path(args.report_path)

    write_jsonl(output_jsonl, [record.to_dict() for record in records])

    report = build_anchor_report(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = time.monotonic() - t_start
    print(f"Done in {elapsed:.1f}s  |  records={len(records)}")
    print(f"Generic anchors written to: {output_jsonl}")
    print(f"Anchor report written to: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

