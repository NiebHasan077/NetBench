"""
build_index.py
--------------
Scans all PDF files in the paperbase, extracts their titles, and saves a
normalized title → file-path index to `title_index.json`.

When a title collision is detected, three strong-confidence checks are run
immediately:

  1. page_count  — both PDFs have the same page count
  2. file_size   — sizes are within 5 % of each other
  3. text_sim    — Jaccard word-token similarity of first-page body text ≥ 0.80

All three pass  → STRONG duplicate: the new file is excluded from the index
                  and recorded in duplicate_candidates.json as confirmed.
Fewer than three → WEAK candidate: first occurrence is kept in the index,
                  and the pair is written to duplicate_candidates.json
                  for manual review via verify_duplicates.py.

All logic lives in index_manager.py; this module is a thin CLI wrapper.

Usage:
    python build_index.py [--root <path>] [--output <path>]

Defaults:
    --root    : grandparent of this script (New-Networking-Papers/)
    --output  : title_index.json in the same directory as this script
"""

import argparse
import sys
from pathlib import Path

# Support both `python build_index.py` (direct) and `python -m paperbase.build_index`
try:
    from .index_manager import IndexManager
except ImportError:
    from index_manager import IndexManager  # type: ignore[no-redef]


def build_index(root: Path, output: Path) -> None:
    pdf_files = sorted(root.rglob("*.pdf"))
    total = len(pdf_files)
    print(f"Found {total} PDF(s) under '{root}'")

    im = IndexManager()   # always start fresh for a full rebuild
    stats = {"added": 0, "strong_dup": 0, "weak_dup": 0, "failed": 0}

    for i, pdf in enumerate(pdf_files, 1):
        result = im.add_pdf(pdf, root=root)

        if result.status == "added":
            stats["added"] += 1
            print(f"  [{i}/{total}] OK    {pdf.name}  [{result.source}]")

        elif result.status == "strong_dup":
            stats["strong_dup"] += 1
            try:
                rel = str(pdf.relative_to(root.parent))
            except ValueError:
                rel = str(pdf)
            print(f"  [{i}/{total}] STRONG '{result.raw_title[:60]}'")
            print(f"            keep      : {result.existing_path}")
            print(f"            excluded  : {rel}")

        elif result.status == "weak_dup":
            stats["weak_dup"] += 1
            try:
                rel = str(pdf.relative_to(root.parent))
            except ValueError:
                rel = str(pdf)
            print(f"  [{i}/{total}] WEAK?  '{result.raw_title[:60]}'")
            print(f"            existing  : {result.existing_path}")
            print(f"            new       : {rel}")

        else:  # extract_failed
            stats["failed"] += 1
            print(f"  [{i}/{total}] SKIP  {pdf.name}  ({result.source})")

    im.save(output)
    candidates_path = output.parent / "duplicate_candidates.json"
    print(f"\nIndex saved to '{output}'")
    print(f"Duplicate candidates saved to '{candidates_path}'")
    print(f"  Indexed         : {stats['added']}")
    print(f"  STRONG excluded : {stats['strong_dup']}")
    print(f"  WEAK candidates : {stats['weak_dup']}")
    print(f"  Extract failed  : {stats['failed']}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Build a title index from a PDF paperbase.")
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir.parent.parent,
        help=(
            "Root directory to scan recursively for PDFs "
            "(default: grandparent of this script, i.e. New-Networking-Papers/)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "title_index.json",
        help="Output JSON file path (default: title_index.json next to this script).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Error: '{root}' is not a directory.")

    build_index(root, args.output)


if __name__ == "__main__":
    main()
