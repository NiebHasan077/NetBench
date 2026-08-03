"""
check_paper.py
--------------
Check whether one or more PDF files already exist in the title index built
by build_index.py.

Usage:
    python check_paper.py <paper1.pdf> [<paper2.pdf> ...]
    python check_paper.py --dir <folder_with_new_pdfs>

Options:
    --index   Path to title_index.json  (default: title_index.json next to this script)
    --dir     Check all PDFs in a directory instead of individual files
"""

import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

# Support both `python check_paper.py` (direct) and `python -m paperbase.check_paper`
try:
    from .index_manager import extract_title, normalize
except ImportError:
    from index_manager import extract_title, normalize  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word tokens (0.0 – 1.0)."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

EXACT_MATCH     = "EXACT"
FUZZY_MATCH     = "FUZZY"    # token overlap ≥ threshold but not exact
NO_MATCH        = "NOT FOUND"
EXTRACT_FAILED  = "EXTRACT FAILED"

FUZZY_THRESHOLD = 0.85       # tune as needed


def check_one(pdf_path: Path, index: dict) -> dict:
    """Return a result dict for a single PDF."""
    raw_title, source = extract_title(pdf_path)

    if not raw_title:
        return {
            "file": str(pdf_path),
            "status": EXTRACT_FAILED,
            "detail": source,
        }

    key = normalize(raw_title)

    # 1. Exact (normalized) match
    if key in index:
        entry = index[key]
        return {
            "file": str(pdf_path),
            "status": EXACT_MATCH,
            "extracted_title": raw_title,
            "matched_title": entry["raw_title"],
            "matched_path": entry["path"],
        }

    # 2. Fuzzy match (guards against minor OCR / spacing differences)
    best_score, best_key = 0.0, None
    for idx_key in index:
        score = _token_overlap(key, idx_key)
        if score > best_score:
            best_score, best_key = score, idx_key

    if best_score >= FUZZY_THRESHOLD and best_key:
        entry = index[best_key]
        return {
            "file": str(pdf_path),
            "status": FUZZY_MATCH,
            "similarity": round(best_score, 3),
            "extracted_title": raw_title,
            "matched_title": entry["raw_title"],
            "matched_path": entry["path"],
        }

    return {
        "file": str(pdf_path),
        "status": NO_MATCH,
        "extracted_title": raw_title,
    }


def _print_result(result: dict) -> None:
    status = result["status"]
    sep = "-" * 60

    if status == EXACT_MATCH:
        print(f"[{status}]  {Path(result['file']).name}")
        print(f"  Title  : {result['extracted_title']}")
        print(f"  Found  : {result['matched_path']}")

    elif status == FUZZY_MATCH:
        print(f"[{status}]  {Path(result['file']).name}  (similarity={result['similarity']})")
        print(f"  New    : {result['extracted_title']}")
        print(f"  Stored : {result['matched_title']}")
        print(f"  Path   : {result['matched_path']}")

    elif status == NO_MATCH:
        print(f"[{status}]  {Path(result['file']).name}")
        print(f"  Title  : {result['extracted_title']}")

    else:  # EXTRACT_FAILED
        print(f"[{status}]  {Path(result['file']).name}")
        print(f"  Reason : {result['detail']}")

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Check if PDF papers already exist in the title index."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="One or more PDF file paths to check.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Check all PDFs inside this directory (non-recursive).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=script_dir / "title_index.json",
        help="Path to title_index.json (default: next to this script).",
    )
    args = parser.parse_args()

    # Resolve PDF list
    # Reconstruct paths that were split by the shell due to unquoted spaces.
    # Greedy longest-match: try joining consecutive tokens until a real file is found.
    def _reconstruct_paths(tokens: list[Path]) -> list[Path]:
        strs = [str(t) for t in tokens]
        result: list[Path] = []
        i = 0
        while i < len(strs):
            matched = False
            for j in range(len(strs), i, -1):
                candidate = Path(" ".join(strs[i:j]))
                if candidate.exists():
                    result.append(candidate)
                    i = j
                    matched = True
                    break
            if not matched:
                result.append(Path(strs[i]))  # keep as-is; will fail gracefully later
                i += 1
        return result

    pdf_list: list[Path] = _reconstruct_paths(list(args.pdfs))
    if args.dir:
        d = args.dir.resolve()
        if not d.is_dir():
            sys.exit(f"Error: '{d}' is not a directory.")
        pdf_list += sorted(d.glob("*.pdf"))

    if not pdf_list:
        sys.exit("No PDF files specified. Use positional arguments or --dir.")

    # Load index
    if not args.index.exists():
        sys.exit(
            f"Index not found at '{args.index}'.\n"
            "Run build_index.py first to create it."
        )
    index: dict = json.loads(args.index.read_text(encoding="utf-8"))
    print(f"Loaded index with {len(index)} entries from '{args.index}'\n")

    # Check each PDF
    counts = {EXACT_MATCH: 0, FUZZY_MATCH: 0, NO_MATCH: 0, EXTRACT_FAILED: 0}
    for pdf in pdf_list:
        result = check_one(pdf.resolve(), index)
        _print_result(result)
        counts[result["status"]] += 1

    # Summary
    print("Summary")
    print(f"  Total checked  : {len(pdf_list)}")
    print(f"  Exact matches  : {counts[EXACT_MATCH]}")
    print(f"  Fuzzy matches  : {counts[FUZZY_MATCH]}")
    print(f"  Not found      : {counts[NO_MATCH]}")
    print(f"  Extract failed : {counts[EXTRACT_FAILED]}")


if __name__ == "__main__":
    main()
