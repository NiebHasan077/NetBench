#!/usr/bin/env python3
"""CLI for Phase 9 final mixing and splitting."""

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

from instruct_ftd.mixing import mix_and_split  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final train/validation datasets from filtered HPN/RAG and generic anchors.")
    parser.add_argument(
        "--filtered_hpn",
        default="",
        help="Filtered HPN JSONL input (e.g. data/intermediate/filtered/<run>/filtered_hpn.jsonl).",
    )
    parser.add_argument(
        "--filtered_rag",
        default="",
        help="Filtered RAG JSONL input (e.g. data/intermediate/filtered/<run>/filtered_rag.jsonl).",
    )
    parser.add_argument(
        "--generic_anchors",
        default=str(PROJECT_ROOT / "data/intermediate/anchors/generic_anchor.jsonl"),
        help="Generic anchor JSONL input.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "data/final"),
        help="Output directory for train/validation JSONL and mix report.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed.",
    )
    parser.add_argument(
        "--strict_mix",
        action="store_true",
        help="Require all three source splits and enforce 60/20/20 exactly up to feasible total.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.filtered_hpn:
        raise SystemExit(
            "ERROR: --filtered_hpn is required.\n"
            "Example: data/intermediate/filtered/<run>/filtered_hpn.jsonl"
        )
    if not args.filtered_rag:
        raise SystemExit(
            "ERROR: --filtered_rag is required.\n"
            "Example: data/intermediate/filtered/<run>/filtered_rag.jsonl"
        )

    hpn_path = Path(args.filtered_hpn)
    rag_path = Path(args.filtered_rag)
    anchors_path = Path(args.generic_anchors)

    if not hpn_path.exists():
        raise SystemExit(f"ERROR: filtered HPN file not found: {hpn_path}")
    if not rag_path.exists():
        raise SystemExit(f"ERROR: filtered RAG file not found: {rag_path}")
    if not anchors_path.exists():
        print(f"WARNING: generic anchors file not found: {anchors_path} — continuing without anchors", file=sys.stderr)

    print(f"Phase 9 final mixing")
    print(f"  filtered_hpn:     {hpn_path}")
    print(f"  filtered_rag:     {rag_path}")
    print(f"  generic_anchors:  {anchors_path}")
    print(f"  output_dir:       {args.output_dir}")
    print(f"  val_ratio:        {args.val_ratio}  seed={args.seed}  strict_mix={args.strict_mix}")

    t_start = time.monotonic()
    filtered_hpn = _read_jsonl(hpn_path)
    filtered_rag = _read_jsonl(rag_path)
    generic_anchors = _read_jsonl(anchors_path)

    print(f"Loaded hpn={len(filtered_hpn)}  rag={len(filtered_rag)}  anchors={len(generic_anchors)}")

    report = mix_and_split(
        filtered_hpn=filtered_hpn,
        filtered_rag=filtered_rag,
        generic_anchors=generic_anchors,
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict_mix=args.strict_mix,
    )

    elapsed = time.monotonic() - t_start
    print(f"Done in {elapsed:.1f}s")
    print(f"Final dataset directory: {args.output_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

