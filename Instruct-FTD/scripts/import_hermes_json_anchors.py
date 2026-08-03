#!/usr/bin/env python3
"""Import Hermes JSON-mode examples as generic JSON anchors."""

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

from instruct_ftd.json_anchors import (  # noqa: E402
    DEFAULT_HERMES_JSON_CONFIGS,
    HERMES_DATASET_NAME,
    load_hermes_json_anchors,
    load_local_hermes_json_anchors,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import NousResearch Hermes JSON-mode rows as generic_json anchors."
    )
    parser.add_argument(
        "--local_jsonl",
        default="",
        help="Optional local JSON/JSONL file with Hermes-shaped rows. When set, Hugging Face is not used.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_HERMES_JSON_CONFIGS),
        help="Hermes dataset configs to load from Hugging Face.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face split to load.",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=166,
        help="Number of valid JSON anchors to write.",
    )
    parser.add_argument(
        "--max_samples_per_config",
        type=int,
        default=0,
        help="Optional cap before normalization for each config. Use 0 for no cap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic selection seed.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=str(PROJECT_ROOT / "data/intermediate/anchors/hermes_json_anchor.jsonl"),
        help="Output JSONL path for normalized Hermes JSON anchors.",
    )
    parser.add_argument(
        "--report_path",
        default=str(PROJECT_ROOT / "reports/phase8_hermes_json_anchor_stats.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = args.max_samples_per_config or None

    print("Hermes JSON anchor import")
    if args.local_jsonl:
        print(f"  source:       {args.local_jsonl} (local)")
    else:
        print(f"  source:       {HERMES_DATASET_NAME}")
        print(f"  configs:      {', '.join(args.configs)}")
        print(f"  split:        {args.split}")
    print(f"  target_count: {args.target_count}")
    print(f"  output:       {args.output_jsonl}")

    t_start = time.monotonic()
    if args.local_jsonl:
        rows = load_local_hermes_json_anchors(
            Path(args.local_jsonl),
            target_count=args.target_count,
            seed=args.seed,
        )
    else:
        rows = load_hermes_json_anchors(
            configs=args.configs,
            split=args.split,
            target_count=args.target_count,
            max_samples_per_config=max_samples,
            seed=args.seed,
        )

    output_jsonl = Path(args.output_jsonl)
    report_path = Path(args.report_path)
    write_jsonl(output_jsonl, rows)

    report = {
        "source": args.local_jsonl or HERMES_DATASET_NAME,
        "configs": [] if args.local_jsonl else args.configs,
        "split": "" if args.local_jsonl else args.split,
        "target_count": args.target_count,
        "record_count": len(rows),
        "category_counts": {"generic_json": len(rows)},
        "task_type_counts": {"json_response": len(rows)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = time.monotonic() - t_start
    print(f"Done in {elapsed:.1f}s  |  records={len(rows)}")
    print(f"Hermes JSON anchors written to: {output_jsonl}")
    print(f"Anchor report written to: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
