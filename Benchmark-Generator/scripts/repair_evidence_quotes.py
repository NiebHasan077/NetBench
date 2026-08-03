"""Repair paraphrased evidence quotes in synthesis and adversarial candidate files.

gemma4:26b frequently paraphrases instead of copying verbatim text despite
explicit prompting. This script replaces non-verbatim quotes with the best
matching verbatim excerpt found in the actual passage text.

Algorithm:
  For each evidence item where quote.lower() not in passage.lower():
    1. Slide overlapping word windows across the passage.
    2. Score each window by Jaccard(quote_words, window_words).
    3. Take the best window (≥ MIN_OVERLAP threshold) as replacement.
    4. If no window beats the threshold, leave quote unchanged
       (item will still fail 6.2 and be dropped in validation).

Usage:
    .venv/bin/python scripts/repair_evidence_quotes.py
    .venv/bin/python scripts/repair_evidence_quotes.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Minimum Jaccard overlap between quote words and window words to accept repair
MIN_OVERLAP = 0.25
# Window sizes (in words) to try when searching
WINDOW_SIZES = [25, 40, 60, 80]
# Maximum length (chars) for a repaired quote
MAX_QUOTE_CHARS = 240


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _best_verbatim_excerpt(quote: str, passage: str) -> tuple[str, float]:
    """Return (best_excerpt, score) where score is Jaccard similarity."""
    quote_words = set(_tokenize(quote))
    if not quote_words:
        return "", 0.0

    passage_words = _tokenize(passage)
    best_score = 0.0
    best_start_word = 0
    best_end_word = 0

    for w_size in WINDOW_SIZES:
        for i in range(max(1, len(passage_words) - w_size + 1)):
            window = passage_words[i : i + w_size]
            window_set = set(window)
            intersection = len(quote_words & window_set)
            union = len(quote_words | window_set)
            score = intersection / union if union else 0.0
            if score > best_score:
                best_score = score
                best_start_word = i
                best_end_word = i + w_size

    if best_score < MIN_OVERLAP:
        return "", best_score

    # Reconstruct original text span from word positions
    # Re-tokenize with spans to find character positions
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\b\w+\b", passage)]
    if best_end_word > len(word_spans):
        best_end_word = len(word_spans)
    if best_start_word >= len(word_spans):
        return "", 0.0

    char_start = word_spans[best_start_word][0]
    char_end = word_spans[best_end_word - 1][1]
    excerpt = passage[char_start:char_end].strip()

    # Trim to MAX_QUOTE_CHARS at a sentence or word boundary
    if len(excerpt) > MAX_QUOTE_CHARS:
        excerpt = excerpt[:MAX_QUOTE_CHARS]
        # Trim to last full word
        last_space = excerpt.rfind(" ")
        if last_space > MAX_QUOTE_CHARS // 2:
            excerpt = excerpt[:last_space]

    return excerpt, best_score


def repair_file(
    candidates_path: Path,
    passages: dict[str, str],
    dry_run: bool = False,
) -> dict:
    """Repair quotes in one candidates file. Returns stats dict."""
    records = []
    with candidates_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    stats = {
        "total_items": len(records),
        "items_with_bad_quotes": 0,
        "quotes_repaired": 0,
        "quotes_unrepaired": 0,
    }

    for rec in records:
        item_had_bad = False
        for ev in rec.get("evidence", []):
            pid = ev.get("passage_id", "")
            quote = ev.get("quote", "")
            passage = passages.get(pid, "")
            if not passage or not quote:
                continue
            if quote.lower() in passage.lower():
                continue  # already verbatim — nothing to do

            # Quote is not verbatim; try to find best replacement
            item_had_bad = True
            excerpt, score = _best_verbatim_excerpt(quote, passage)
            if excerpt:
                stats["quotes_repaired"] += 1
                if not dry_run:
                    ev["quote"] = excerpt
                print(
                    f"  REPAIRED (score={score:.2f}) {rec['id']} [{pid}]\n"
                    f"    orig:    {quote[:80]}...\n"
                    f"    repaired:{excerpt[:80]}..."
                )
            else:
                stats["quotes_unrepaired"] += 1
                print(
                    f"  UNREPAIRABLE (score={score:.2f}) {rec['id']} [{pid}]\n"
                    f"    quote: {quote[:80]}..."
                )

        if item_had_bad:
            stats["items_with_bad_quotes"] += 1

    if not dry_run:
        # Backup original
        backup = candidates_path.with_suffix(".jsonl.bak")
        shutil.copy2(candidates_path, backup)
        print(f"  backup saved: {backup}")

        # Write repaired version
        with candidates_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  repaired file saved: {candidates_path}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair paraphrased evidence quotes.")
    parser.add_argument("--dry-run", action="store_true", help="Report repairs without writing.")
    args = parser.parse_args()

    # Load passages
    passages_path = PROJECT_ROOT / "data" / "passages.jsonl"
    if not passages_path.exists():
        print(f"ERROR: passages not found at {passages_path}")
        raise SystemExit(1)

    print("Loading passages …")
    passages: dict[str, str] = {}
    with passages_path.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            passages[p["passage_id"]] = p["text"]
    print(f"  loaded {len(passages)} passages")

    targets = [
        PROJECT_ROOT / "data" / "synthesis_candidates.jsonl",
        PROJECT_ROOT / "data" / "adversarial_candidates.jsonl",
    ]

    for path in targets:
        if not path.exists():
            print(f"SKIP: {path.name} not found")
            continue
        print(f"\n=== {path.name} ===")
        stats = repair_file(path, passages, dry_run=args.dry_run)
        print(f"\n  Summary:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    if args.dry_run:
        print("\nDRY RUN — no files were modified.")


if __name__ == "__main__":
    main()
