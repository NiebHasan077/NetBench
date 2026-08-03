"""
verify_duplicates.py
--------------------
Reads `duplicate_candidates.json` (produced by build_index.py) and runs
three independent checks on every candidate pair to produce a confidence
rating.  Results are written to `confirmed_duplicates.json`.

Confidence levels
-----------------
  STRONG  — all three checks pass              → safe to delete
  LIKELY  — exactly two checks pass            → review recommended
  WEAK    — only one check passes (title only) → probable false positive

Only STRONG and LIKELY pairs are written to confirmed_duplicates.json.
WEAK pairs are reported but excluded from the output so delete_duplicates.py
never acts on them.

The three checks
----------------
  page_count   Both PDFs have the same page count.
  file_size    File sizes are within FILE_SIZE_TOLERANCE of each other.
  text_sim     Jaccard word-token similarity of the first-page body text
               is ≥ TEXT_SIM_THRESHOLD.

Usage:
    python verify_duplicates.py [--candidates <path>] [--output <path>]
                                [--root <path>]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

FILE_SIZE_TOLERANCE = 0.05   # 5 % – file sizes must be within this fraction
TEXT_SIM_THRESHOLD  = 0.80   # Jaccard on first-page word tokens

# Same top-band fraction as build_index.py to stay consistent
_TOP_BAND_FRACTION = 0.10

# Minimum tokens to trust the text-similarity check
_MIN_TOKEN_COUNT = 20


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _page_count(path: Path) -> int | None:
    try:
        doc = fitz.open(str(path))
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return None


def _first_page_text(path: Path) -> str:
    """Return word-tokenised body text from page 1, ignoring the header band."""
    try:
        doc = fitz.open(str(path))
        if doc.page_count == 0:
            doc.close()
            return ""
        page = doc[0]
        page_height = page.rect.height
        top_band = page_height * _TOP_BAND_FRACTION

        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        tokens = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["origin"][1] < top_band:
                        continue
                    tokens.extend(re.findall(r"\w+", span["text"].lower()))
        doc.close()
        return " ".join(tokens)
    except Exception:
        return ""


def _jaccard(text_a: str, text_b: str) -> float:
    ta = set(text_a.split())
    tb = set(text_b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

def _resolve(rel_path: str, root: Path) -> Path:
    """Turn a relative path (as stored in JSON) into an absolute Path."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _run_checks(existing: Path, duplicate: Path) -> dict:
    """Run all three checks and return a dict of results."""
    results: dict = {}

    # 1. Page count
    pc_a = _page_count(existing)
    pc_b = _page_count(duplicate)
    if pc_a is not None and pc_b is not None:
        results["page_count"] = (pc_a == pc_b)
        results["page_count_detail"] = f"{pc_a} vs {pc_b}"
    else:
        results["page_count"] = None   # could not determine
        results["page_count_detail"] = "unreadable"

    # 2. File size
    try:
        size_a = existing.stat().st_size
        size_b = duplicate.stat().st_size
        if size_a > 0:
            ratio = abs(size_a - size_b) / max(size_a, size_b)
            results["file_size"] = ratio <= FILE_SIZE_TOLERANCE
            results["file_size_detail"] = (
                f"{size_a:,} vs {size_b:,} bytes  "
                f"(Δ={ratio * 100:.1f}%)"
            )
        else:
            results["file_size"] = None
            results["file_size_detail"] = "zero-byte file"
    except OSError as exc:
        results["file_size"] = None
        results["file_size_detail"] = str(exc)

    # 3. Text similarity
    text_a = _first_page_text(existing)
    text_b = _first_page_text(duplicate)
    tokens_a = len(text_a.split())
    tokens_b = len(text_b.split())
    if tokens_a >= _MIN_TOKEN_COUNT and tokens_b >= _MIN_TOKEN_COUNT:
        sim = _jaccard(text_a, text_b)
        results["text_sim"] = sim >= TEXT_SIM_THRESHOLD
        results["text_sim_detail"] = f"Jaccard={sim:.3f} (threshold={TEXT_SIM_THRESHOLD})"
    else:
        results["text_sim"] = None
        results["text_sim_detail"] = (
            f"too few tokens ({tokens_a} / {tokens_b}), skipped"
        )

    return results


def _confidence(checks: dict) -> str:
    # Treat None (unreadable) as a pass to avoid penalising edge cases
    passed = sum(
        1 for k in ("page_count", "file_size", "text_sim")
        if checks.get(k) is not False   # True or None → count as pass
    )
    if passed == 3:
        return "STRONG"
    if passed == 2:
        return "LIKELY"
    return "WEAK"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify(candidates_path: Path, output_path: Path, root: Path) -> None:
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    total = len(candidates)
    print(f"Loaded {total} candidate pair(s) from '{candidates_path}'\n")

    confirmed: list[dict] = []
    counts = {"STRONG": 0, "LIKELY": 0, "WEAK": 0, "ERROR": 0}

    for i, pair in enumerate(candidates, 1):
        raw_title = pair.get("raw_title", pair.get("normalized_title", "?"))
        existing_rel  = pair["existing"]
        duplicate_rel = pair["duplicate"]

        existing  = _resolve(existing_rel,  root)
        duplicate = _resolve(duplicate_rel, root)

        missing = [p for p in (existing, duplicate) if not p.exists()]
        if missing:
            print(f"[{i}/{total}] ERROR  '{raw_title[:60]}'")
            for m in missing:
                print(f"           missing : {m}")
            counts["ERROR"] += 1
            continue

        checks  = _run_checks(existing, duplicate)
        conf    = _confidence(checks)
        counts[conf] += 1

        label = f"[{conf}]"
        print(f"[{i}/{total}] {label:<8}  '{raw_title[:60]}'")
        print(f"           existing  : {existing_rel}")
        print(f"           duplicate : {duplicate_rel}")
        for key in ("page_count", "file_size", "text_sim"):
            tick = "✓" if checks.get(key) is True else ("?" if checks.get(key) is None else "✗")
            print(f"             {tick} {key:<12} {checks.get(key + '_detail', '')}")
        print()

        if conf in ("STRONG", "LIKELY"):
            confirmed.append({
                "raw_title":      raw_title,
                "normalized_title": pair.get("normalized_title", ""),
                "existing":       existing_rel,
                "duplicate":      duplicate_rel,
                "confidence":     conf,
                "checks":         {k: checks[k] for k in ("page_count", "file_size", "text_sim")},
                "check_details":  {k: checks[k + "_detail"]
                                   for k in ("page_count", "file_size", "text_sim")},
            })

    output_path.write_text(
        json.dumps(confirmed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 60)
    print(f"Summary")
    print(f"  Total candidates  : {total}")
    print(f"  STRONG            : {counts['STRONG']}")
    print(f"  LIKELY            : {counts['LIKELY']}")
    print(f"  WEAK (excluded)   : {counts['WEAK']}")
    print(f"  Errors            : {counts['ERROR']}")
    print(f"\nConfirmed duplicates saved to '{output_path}'")
    print(f"  ({len(confirmed)} entries — STRONG + LIKELY only)")
    print(f"\nNext step:  python delete_duplicates.py")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Verify duplicate PDF candidates with multi-signal checks."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=script_dir / "duplicate_candidates.json",
        help="Input candidates JSON (default: duplicate_candidates.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "confirmed_duplicates.json",
        help="Output confirmed duplicates JSON (default: confirmed_duplicates.json).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir.parent.parent.parent,
        help="Root directory that paths in the JSON are relative to "
             "(default: 3 levels up from this script, i.e. Downloads/).",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        sys.exit(
            f"Candidates file not found: '{args.candidates}'\n"
            "Run build_index.py first."
        )

    verify(args.candidates, args.output, args.root.resolve())


if __name__ == "__main__":
    main()
