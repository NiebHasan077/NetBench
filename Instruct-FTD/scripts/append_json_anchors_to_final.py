#!/usr/bin/env python3
"""Append validated JSON-response anchors to an existing final dataset."""

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

from instruct_ftd.json_anchors import append_json_anchors_to_final  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append generic_json anchors to existing train/validation JSONL files."
    )
    parser.add_argument(
        "--input_train",
        required=True,
        help="Existing final train.jsonl.",
    )
    parser.add_argument(
        "--input_validation",
        required=True,
        help="Existing final validation.jsonl.",
    )
    parser.add_argument(
        "--json_anchors",
        required=True,
        help="Validated generic_json anchor JSONL file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=166,
        help="Number of JSON anchors to append.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the augmented train/validation files.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of appended JSON anchors assigned to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic selection and shuffle seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output train/validation/report files.",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"ERROR: {label} not found: {path}")


def _check_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / "train.jsonl",
        output_dir / "validation.jsonl",
        output_dir / "json_append_report.json",
    ]
    if not overwrite:
        present = [path for path in existing if path.exists()]
        if present:
            paths = "\n".join(f"  - {path}" for path in present)
            raise SystemExit(
                "ERROR: output files already exist. Use --overwrite to replace them:\n"
                f"{paths}"
            )


def main() -> None:
    args = parse_args()

    input_train = Path(args.input_train)
    input_validation = Path(args.input_validation)
    json_anchors = Path(args.json_anchors)
    output_dir = Path(args.output_dir)

    _require_file(input_train, "input train file")
    _require_file(input_validation, "input validation file")
    _require_file(json_anchors, "JSON anchor file")
    _check_output_dir(output_dir, args.overwrite)

    print("Append JSON anchors to final dataset")
    print(f"  input_train:      {input_train}")
    print(f"  input_validation: {input_validation}")
    print(f"  json_anchors:     {json_anchors}")
    print(f"  count:            {args.count}")
    print(f"  val_ratio:        {args.val_ratio}")
    print(f"  output_dir:       {output_dir}")

    t_start = time.monotonic()
    report = append_json_anchors_to_final(
        input_train=input_train,
        input_validation=input_validation,
        json_anchors=json_anchors,
        output_dir=output_dir,
        count=args.count,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    elapsed = time.monotonic() - t_start
    print(f"Done in {elapsed:.1f}s")
    print(f"Augmented final dataset directory: {output_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
