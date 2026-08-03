"""
delete_duplicates.py
--------------------
Reads `confirmed_duplicates.json` (produced by verify_duplicates.py) and
removes the duplicate copy of each pair.

Safety features
---------------
  • Dry-run by default — prints exactly what would be deleted, touches nothing.
  • Requires --confirm to perform actual deletions.
  • --confidence controls the minimum confidence level to act on:
        STRONG  (default) — only pairs where all 3 checks passed
        LIKELY            — include 2-check pairs as well
  • Before deleting, re-checks that the target file still exists on disk.
  • Keeps a deletion log in `deletion_log.json`.

Which copy is deleted?
-----------------------
The `duplicate` path (the second occurrence found during indexing) is always
the one targeted for deletion.  The `existing` path (first occurrence, the
one stored in the title index) is kept untouched.

Usage:
    # Preview only
    python delete_duplicates.py

    # Delete STRONG duplicates
    python delete_duplicates.py --confirm

    # Delete STRONG + LIKELY duplicates
    python delete_duplicates.py --confidence LIKELY --confirm

    # Operate on specific confidence and preview
    python delete_duplicates.py --confidence LIKELY
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


_CONFIDENCE_RANK = {"STRONG": 2, "LIKELY": 1, "WEAK": 0}


def _resolve(rel_path: str, root: Path) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _load_confirmed(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def delete_duplicates(
    confirmed_path: Path,
    root: Path,
    min_confidence: str,
    dry_run: bool,
    log_path: Path,
) -> None:
    all_pairs = _load_confirmed(confirmed_path)
    min_rank  = _CONFIDENCE_RANK[min_confidence]

    eligible = [
        p for p in all_pairs
        if _CONFIDENCE_RANK.get(p.get("confidence", "WEAK"), 0) >= min_rank
    ]
    skipped_conf = len(all_pairs) - len(eligible)

    print(f"Loaded {len(all_pairs)} confirmed pair(s) from '{confirmed_path}'")
    print(f"  Min confidence filter : {min_confidence}")
    print(f"  Eligible for deletion : {len(eligible)}")
    print(f"  Skipped (below threshold) : {skipped_conf}")
    print()

    if not eligible:
        print("Nothing to delete.")
        return

    mode = "DRY-RUN" if dry_run else "LIVE DELETE"
    print(f"{'=' * 60}")
    print(f"Mode: {mode}")
    print(f"{'=' * 60}\n")

    log_entries: list[dict] = []
    deleted = 0
    errors  = 0

    for i, pair in enumerate(eligible, 1):
        raw_title     = pair.get("raw_title", "?")
        existing_rel  = pair["existing"]
        duplicate_rel = pair["duplicate"]
        confidence    = pair.get("confidence", "?")

        duplicate_abs = _resolve(duplicate_rel, root)

        print(f"[{i}/{len(eligible)}] {confidence}  '{raw_title[:70]}'")
        print(f"  keep   : {existing_rel}")
        print(f"  delete : {duplicate_rel}")

        if not duplicate_abs.exists():
            print(f"  SKIP — file not found on disk: {duplicate_abs}")
            errors += 1
            print()
            continue

        if dry_run:
            size_kb = duplicate_abs.stat().st_size / 1024
            print(f"  [DRY-RUN] would delete  ({size_kb:.1f} KB)")
            log_entries.append({
                "action":     "dry-run",
                "confidence": confidence,
                "raw_title":  raw_title,
                "kept":       existing_rel,
                "deleted":    duplicate_rel,
                "timestamp":  datetime.now().isoformat(),
            })
        else:
            try:
                size_kb = duplicate_abs.stat().st_size / 1024
                duplicate_abs.unlink()
                print(f"  DELETED  ({size_kb:.1f} KB)")
                deleted += 1
                log_entries.append({
                    "action":     "deleted",
                    "confidence": confidence,
                    "raw_title":  raw_title,
                    "kept":       existing_rel,
                    "deleted":    duplicate_rel,
                    "timestamp":  datetime.now().isoformat(),
                })
            except OSError as exc:
                print(f"  ERROR — could not delete: {exc}")
                errors += 1
                log_entries.append({
                    "action":     "error",
                    "confidence": confidence,
                    "raw_title":  raw_title,
                    "kept":       existing_rel,
                    "deleted":    duplicate_rel,
                    "error":      str(exc),
                    "timestamp":  datetime.now().isoformat(),
                })
        print()

    # Append to log (preserve history across runs)
    existing_log: list[dict] = []
    if log_path.exists():
        try:
            existing_log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    log_path.write_text(
        json.dumps(existing_log + log_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Final summary
    print("=" * 60)
    print("Summary")
    if dry_run:
        print(f"  [DRY-RUN] Files that would be deleted : {len(eligible) - errors}")
        print(f"  Errors (not found on disk)            : {errors}")
        print()
        print("Run with --confirm to perform actual deletions.")
    else:
        print(f"  Deleted  : {deleted}")
        print(f"  Errors   : {errors}")
    print(f"\nDeletion log saved to '{log_path}'")
    if not dry_run and deleted > 0:
        print("\nIMPORTANT: Re-run build_index.py to refresh title_index.json.")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Delete confirmed duplicate PDFs from the paperbase."
    )
    parser.add_argument(
        "--confirmed",
        type=Path,
        default=script_dir / "confirmed_duplicates.json",
        help="Input confirmed duplicates JSON (default: confirmed_duplicates.json).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir.parent.parent.parent,
        help="Root directory that paths in the JSON are relative to "
             "(default: 3 levels up from this script, i.e. Downloads/).",
    )
    parser.add_argument(
        "--confidence",
        choices=["STRONG", "LIKELY"],
        default="STRONG",
        help="Minimum confidence level to delete (default: STRONG).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Actually perform deletions. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=script_dir / "deletion_log.json",
        help="Path to append the deletion log (default: deletion_log.json).",
    )
    args = parser.parse_args()

    if not args.confirmed.exists():
        sys.exit(
            f"Confirmed duplicates file not found: '{args.confirmed}'\n"
            "Run verify_duplicates.py first."
        )

    delete_duplicates(
        confirmed_path=args.confirmed,
        root=args.root.resolve(),
        min_confidence=args.confidence,
        dry_run=not args.confirm,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
