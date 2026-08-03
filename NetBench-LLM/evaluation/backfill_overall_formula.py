#!/usr/bin/env python3
"""Backfill an `overall_formula` column into released judged workbooks.

Appends, after the last existing column of the "Judged" sheet, the
deterministic rubric composite

    overall_formula = 0.4*correctness + 0.3*completeness
                      + 0.2*clarity + 0.1*conciseness

computed from the judge's integer dimension scores, and records the operation
in the workbook's Metadata sheet. Motivation and audit: the judge-reported
`overall` drifts from the instructed formula (docs/JUDGE_OVERALL_AUDIT.md);
the EACL paper reports the formula value as its primary metric, so the
released evidence should carry it explicitly.

Idempotent: workbooks that already have the column are skipped. All existing
cells, sheets, and styles are left untouched; the new cells copy the styling
of their row's `overall` cell, and the auto-filter range is extended by one
column.

Usage (from the repo root):
    python NetBench-LLM/evaluation/backfill_overall_formula.py \
        --by-model NetBench-LLM/outputs/by_model [--dry-run]
"""
from __future__ import annotations
import argparse
import glob
import os
from copy import copy
from datetime import date

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

DIMS = ["correctness", "completeness", "clarity", "conciseness"]
WEIGHTS = [0.4, 0.3, 0.2, 0.1]
NOTE_KEY = "overall_formula_note"
NOTE = ("column added post hoc by evaluation/backfill_overall_formula.py "
        "({}): 0.4*correctness + 0.3*completeness + 0.2*clarity "
        "+ 0.1*conciseness from this sheet's integer dimension scores; "
        "see docs/JUDGE_OVERALL_AUDIT.md")


def backfill(path: str, dry_run: bool = False) -> str:
    wb = load_workbook(path)
    ws = wb["Judged"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    if "overall_formula" in headers:
        wb.close()
        return "skip (already present)"
    missing = [d for d in DIMS if d not in headers]
    if missing:
        wb.close()
        return f"ERROR missing dims {missing}"
    if dry_run:
        wb.close()
        return "would add"

    dim_cols = [headers.index(d) + 1 for d in DIMS]
    ov_col = headers.index("overall") + 1
    new_col = len(headers) + 1

    # header cell, styled like its left neighbour
    src = ws.cell(row=1, column=len(headers))
    dst = ws.cell(row=1, column=new_col, value="overall_formula")
    dst.font, dst.fill = copy(src.font), copy(src.fill)
    dst.alignment, dst.border = copy(src.alignment), copy(src.border)

    n_filled = 0
    for r in range(2, ws.max_row + 1):
        if ws.cell(row=r, column=1).value is None:
            continue
        vals = [ws.cell(row=r, column=c).value for c in dim_cols]
        cell = ws.cell(row=r, column=new_col)
        ref = ws.cell(row=r, column=ov_col)
        cell.alignment, cell.border = copy(ref.alignment), copy(ref.border)
        if all(isinstance(v, (int, float)) for v in vals):
            cell.value = round(sum(w * v for w, v in zip(WEIGHTS, vals)), 2)
            n_filled += 1
    ws.column_dimensions[get_column_letter(new_col)].width = 16
    if ws.auto_filter.ref:
        first, last = ws.auto_filter.ref.split(":")
        row_end = "".join(ch for ch in last if ch.isdigit())
        ws.auto_filter.ref = f"{first}:{get_column_letter(new_col)}{row_end}"

    meta = wb["Metadata"]
    mrow = meta.max_row + 1
    for col, value in ((1, NOTE_KEY), (2, NOTE.format(date.today().isoformat()))):
        c = meta.cell(row=mrow, column=col, value=value)
        prev = meta.cell(row=mrow - 1, column=col)
        c.border = copy(prev.border)
        if col == 1:
            c.font = copy(prev.font)

    wb.save(path)
    wb.close()
    return f"added ({n_filled} rows)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-model", default="outputs/by_model",
                    help="by_model tree containing */evaluations/judged/*.xlsx")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.by_model, "*/evaluations/judged/*.xlsx")))
    if not paths:
        raise SystemExit(f"no judged workbooks under {args.by_model}")
    counts: dict[str, int] = {}
    for p in paths:
        res = backfill(p, dry_run=args.dry_run)
        counts[res.split(" (")[0]] = counts.get(res.split(" (")[0], 0) + 1
        print(f"{res:28s} {p}")
    print("\nsummary:", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
          f"({len(paths)} workbooks)")


if __name__ == "__main__":
    main()
