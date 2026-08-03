#!/usr/bin/env python3
"""Merge filled human-rating workbook(s) into one tidy CSV.

Accepts one or more filled copies of the blinded workbook (a single
``HPN_human_judging_v1_filled_<initials>.xlsx`` or several partial saves,
e.g. one per 5-question batch). Their union must cover each of the 125
response codes exactly once.

Every row is validated before it is trusted:
  - the four dimension scores are whole numbers 1-5;
  - the answer text is byte-identical (modulo surrounding whitespace) to the
    blinded original, which catches accidental row edits or reordering;
  - response codes match sampling_key.csv exactly (no duplicates, no gaps).

Outputs:
  - outputs/human_scores_long.csv - one row per response with the rater's
    four dimension scores, the weighted composite ``overall`` (the judge's
    instructed weights, so the human composite is defined identically to the
    judge's), and the un-blinded system identity from sampling_key.csv.
  - outputs/HPN_human_judging_v1_filled.xlsx - the blank workbook with all
    125 merged scores filled in (single consolidated sheet).

Usage:
    python merge_filled_ratings.py                      # outputs/filled/*.xlsx
    python merge_filled_ratings.py --filled outputs/HPN_human_judging_v1_filled_ab.xlsx
"""

import argparse
import csv
import glob
import re
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
BLANK = HERE / "outputs/HPN_human_judging_v1.xlsx"
KEY = HERE / "outputs/sampling_key.csv"

DIMENSIONS = ["correctness", "completeness", "clarity", "conciseness"]
WEIGHTS = {"correctness": 0.4, "completeness": 0.3, "clarity": 0.2, "conciseness": 0.1}
RID_RE = re.compile(r"^Q\d{2}-[a-e]$")


def _norm(text: str) -> str:
    """Normalize answer text for the integrity comparison. A LibreOffice
    re-save stores cell newlines as literal ``&#10;`` entities; tolerate that
    (and CR/LF variants) while still failing on any real word-level edit."""
    return text.replace("&#10;", "\n").replace("\r\n", "\n").replace("\r", "\n").strip()


def read_scoring_rows(path: Path) -> dict[str, dict]:
    """{response_id: {dim scores..., note}} for every response row in one workbook."""
    wb = load_workbook(path, read_only=True, data_only=True)
    if "Scoring" not in wb.sheetnames:
        raise SystemExit(f"❌ {path.name}: no 'Scoring' sheet")
    out = {}
    for row in wb["Scoring"].iter_rows(min_row=2):
        rid = row[0].value
        if not (isinstance(rid, str) and RID_RE.match(rid)):
            continue
        if rid in out:
            raise SystemExit(f"❌ {path.name}: duplicate response id {rid}")
        rec = {"answer_text": str(row[1].value or "")}
        for dim, cell in zip(DIMENSIONS, row[2:6]):
            v = cell.value
            if not (isinstance(v, (int, float)) and float(v).is_integer() and 1 <= v <= 5):
                raise SystemExit(f"❌ {path.name}: {rid} {dim} = {v!r} (want integer 1-5)")
            rec[dim] = int(v)
        out[rid] = rec
    wb.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filled", nargs="+", default=[str(HERE / "outputs/filled/*.xlsx")],
        help="Filled workbook path(s) or glob(s); union must cover all 125 responses.")
    parser.add_argument("--rater", default="human",
                        help="Rater label recorded in the output CSV.")
    parser.add_argument("--out", default=str(HERE / "outputs/human_scores_long.csv"))
    args = parser.parse_args()

    paths = sorted({p for pat in args.filled for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"❌ no filled workbooks match {args.filled}")

    # blinded original: reference answer text per response id
    original = {}
    wb = load_workbook(BLANK, read_only=True)
    for row in wb["Scoring"].iter_rows(min_row=2):
        rid = row[0].value
        if isinstance(rid, str) and RID_RE.match(rid):
            original[rid] = str(row[1].value or "")
    wb.close()

    merged: dict[str, dict] = {}
    for p in paths:
        rows = read_scoring_rows(Path(p))
        for rid, rec in rows.items():
            if rid in merged:
                raise SystemExit(f"❌ {rid} scored in more than one workbook ({Path(p).name})")
            if _norm(rec["answer_text"]) != _norm(original[rid]):
                raise SystemExit(f"❌ {Path(p).name}: answer text at {rid} differs from "
                                 "the blinded original - row edited or reordered?")
            merged[rid] = rec
        print(f"   {Path(p).name}: {len(rows)} responses")

    with open(KEY, encoding="utf-8") as fh:
        key = {r["response_id"]: r for r in csv.DictReader(fh)}
    missing = sorted(set(key) - set(merged))
    extra = sorted(set(merged) - set(key))
    if missing or extra:
        raise SystemExit(f"❌ coverage mismatch: missing={missing[:5]}... extra={extra[:5]}")

    fields = ["rater", "response_id", "question_no", "question_id", "base_model",
              "p_id", "setting", "variant_run", "category", "difficulty",
              *DIMENSIONS, "overall"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rid in sorted(merged):
            k, rec = key[rid], merged[rid]
            overall = sum(WEIGHTS[d] * rec[d] for d in DIMENSIONS)
            w.writerow({"rater": args.rater, "response_id": rid,
                        **{f: k[f] for f in ["question_no", "question_id", "base_model",
                                             "p_id", "setting", "variant_run",
                                             "category", "difficulty"]},
                        **{d: rec[d] for d in DIMENSIONS},
                        "overall": round(overall, 2)})

    # single consolidated workbook: the blank with every score filled in
    wb = load_workbook(BLANK)
    for row in wb["Scoring"].iter_rows(min_row=2):
        rid = row[0].value
        if isinstance(rid, str) and RID_RE.match(rid):
            for col, dim in enumerate(DIMENSIONS, start=2):
                row[col].value = merged[rid][dim]
    filled_path = BLANK.with_name("HPN_human_judging_v1_filled.xlsx")
    wb.save(filled_path)

    print(f"✅ {len(merged)} responses from {len(paths)} workbook(s) -> {args.out}")
    print(f"   consolidated workbook -> {filled_path}")


if __name__ == "__main__":
    main()
