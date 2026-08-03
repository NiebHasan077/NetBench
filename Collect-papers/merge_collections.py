#!/usr/bin/env python3
"""
merge_collections.py — Merge multiple PDF collections into one, with dedup.

Scans one or more source directories for PDFs, builds a unified title index,
detects duplicates (STRONG/WEAK) using the same 3-signal system as
paperbase/build_index.py, and copies unique papers into a target directory.

After merging, use the standard paperbase tools for review and cleanup:

  1. paperbase/verify_duplicates.py  — re-verify WEAK candidates
  2. paperbase/delete_duplicates.py  — remove confirmed duplicates

Usage:
    # Preview (dry-run): see what would be copied and what's a duplicate
    python merge_collections.py \\
        --source /path/to/manual-papers \\
        --source /path/to/hpn/papers \\
        --target /path/to/merged-collection \\
        --dry-run

    # Perform the merge
    python merge_collections.py \\
        --source /path/to/manual-papers \\
        --source /path/to/hpn/papers \\
        --target /path/to/merged-collection

    # Merge and save the index inside the target directory
    python merge_collections.py \\
        --source CoNEXT SIGCOMM NSDI INFOCOM \\
        --source downloads/hpn/papers \\
        --target /path/to/merged \\
        --index /path/to/merged/title_index.json
"""

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

# Allow running as `python merge_collections.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperbase.index_manager import IndexManager, normalize, extract_title


def _safe_copy(src: Path, dest_dir: Path) -> Path:
    """Copy src into dest_dir, appending a numeric suffix on name collision."""
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
        return dest
    stem, suffix = src.stem, src.suffix
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(src, dest)
    return dest


def merge(
    sources: list[Path],
    target: Path,
    index_path: Path,
    dry_run: bool,
) -> None:
    """Merge PDFs from *sources* into *target* with title-based dedup."""
    # Discover all PDFs across all source directories
    all_pdfs: list[tuple[Path, str]] = []  # (abs_path, source_label)
    for src_dir in sources:
        pdfs = sorted(src_dir.rglob("*.pdf"))
        label = src_dir.name
        for pdf in pdfs:
            all_pdfs.append((pdf, label))

    total = len(all_pdfs)
    print(f"Found {total} PDF(s) across {len(sources)} source(s)")
    for src_dir in sources:
        count = sum(1 for p, _ in all_pdfs if p.is_relative_to(src_dir))
        print(f"  {src_dir.name:30s} : {count}")
    print()

    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)

    im = IndexManager()
    stats = {
        "added": 0,
        "strong_dup": 0,
        "weak_dup": 0,
        "extract_failed": 0,
        "copied": 0,
    }

    for i, (pdf_path, label) in enumerate(all_pdfs, 1):
        result = im.add_pdf(pdf_path, root=target)

        if result.status == "added":
            stats["added"] += 1
            tag = f"[{label}]"
            if dry_run:
                print(f"  [{i}/{total}] NEW   {tag:20s} {result.raw_title[:70]}")
            else:
                dest = _safe_copy(pdf_path, target)
                # Update the stored path to be relative to target.parent
                try:
                    rel = str(dest.relative_to(target.parent))
                except ValueError:
                    rel = str(dest)
                im._index[result.norm_title]["path"] = rel
                stats["copied"] += 1
                print(f"  [{i}/{total}] COPY  {tag:20s} {result.raw_title[:70]}")

        elif result.status == "strong_dup":
            stats["strong_dup"] += 1
            print(
                f"  [{i}/{total}] DUP   [{label}]"
                f"  '{result.raw_title[:55]}'"
                f"  (exists: {result.existing_path})"
            )

        elif result.status == "weak_dup":
            stats["weak_dup"] += 1
            print(
                f"  [{i}/{total}] WEAK? [{label}]"
                f"  '{result.raw_title[:55]}'"
                f"  (exists: {result.existing_path})"
            )

        else:  # extract_failed
            stats["extract_failed"] += 1
            if dry_run:
                print(f"  [{i}/{total}] SKIP  [{label}]  {pdf_path.name}  ({result.source})")
            else:
                # Copy even if title extraction failed — let the user decide
                _safe_copy(pdf_path, target)
                stats["copied"] += 1
                print(f"  [{i}/{total}] COPY? [{label}]  {pdf_path.name}  (title: {result.source})")

    if not dry_run:
        im.save(index_path)
        candidates_path = index_path.parent / "duplicate_candidates.json"
        print(f"\n  Index saved      : {index_path}")
        if candidates_path.exists():
            print(f"  Dup candidates   : {candidates_path}")

    # Summary
    mode = "DRY RUN" if dry_run else "MERGE"
    print(f"\n{'=' * 60}")
    print(f"{mode} Summary")
    print(f"  Total PDFs scanned     : {total}")
    print(f"  Unique papers          : {stats['added']}")
    print(f"  STRONG duplicates      : {stats['strong_dup']}")
    print(f"  WEAK candidates        : {stats['weak_dup']}")
    print(f"  Title-extract failed   : {stats['extract_failed']}")
    if not dry_run:
        print(f"  Files copied to target : {stats['copied']}")
    print(f"{'=' * 60}")

    if stats["weak_dup"] > 0:
        print(
            "\nReview WEAK candidates with:\n"
            f"  .venv/bin/python paperbase/verify_duplicates.py "
            f"--candidates {index_path.parent / 'duplicate_candidates.json'}"
        )
    if stats["strong_dup"] > 0 or stats["weak_dup"] > 0:
        print(
            "\nAfter verification, remove confirmed duplicates with:\n"
            f"  .venv/bin/python paperbase/delete_duplicates.py "
            f"--confirmed {index_path.parent / 'confirmed_duplicates.json'} "
            f"--root {target.parent} --confirm"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple PDF collections into one, with duplicate detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Preview the merge
              python merge_collections.py \\
                  --source ../CoNEXT --source ../SIGCOMM \\
                  --source downloads/hpn/papers \\
                  --target ../merged \\
                  --dry-run

              # Perform the merge
              python merge_collections.py \\
                  --source ../CoNEXT --source ../SIGCOMM \\
                  --source downloads/hpn/papers \\
                  --target ../merged
        """),
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        dest="sources",
        help="Source directory containing PDFs (repeatable). "
             "Scanned recursively.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Target directory to copy unique PDFs into.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to save the title_index.json "
             "(default: <target>/title_index.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only — no files written.",
    )
    args = parser.parse_args()

    # Resolve all paths
    sources = [s.resolve() for s in args.sources]
    target = args.target.resolve()
    index_path = (
        args.index.resolve() if args.index else target / "title_index.json"
    )

    # Validate source directories
    for src in sources:
        if not src.is_dir():
            sys.exit(f"Error: source directory not found: {src}")

    merge(sources, target, index_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
