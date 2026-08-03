#!/usr/bin/env python3
"""
Filter Judged Excel Files
==========================

Reads all judged Excel files from outputs/evaluations/judged/ and
produces filtered copies in outputs/evaluations/filtered/ with two sheets:
    • "Good Answers"  — rows where overall score > 4
    • "Bad Answers"   — rows where overall score < 1.3

Thresholds:
    BAD_THRESHOLD  = 1.3   (overall < 1.3  → bad)
    GOOD_THRESHOLD = 4.0   (overall > 4.0  → good)

Does NOT modify any existing files.

Usage:
    python evaluation/filter_judged.py
    python evaluation/filter_judged.py --judged_dir outputs/evaluations/judged
    python evaluation/filter_judged.py --output_dir outputs/evaluations/filtered
    python evaluation/filter_judged.py --good_threshold 4.0 --bad_threshold 1.3
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_JUDGED_DIR  = "outputs/evaluations/judged"
DEFAULT_OUTPUT_DIR  = "outputs/evaluations/filtered"
DEFAULT_BAD_THRESHOLD  = 1.3
DEFAULT_GOOD_THRESHOLD = 4.0

# ═══════════════════════════════════════════════════════════════════════
# Styling (consistent with project style)
# ═══════════════════════════════════════════════════════════════════════

_HEADER_FONT   = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
_GOOD_FILL     = PatternFill(start_color="2E7D32", end_color="2E7D32", fill_type="solid")  # dark green
_BAD_FILL      = PatternFill(start_color="C62828", end_color="C62828", fill_type="solid")  # dark red
_HEADER_ALIGN  = Alignment(horizontal="center", vertical="center", wrap_text=True)
_WRAP_ALIGN    = Alignment(vertical="top", wrap_text=True)
_THIN_BORDER   = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
_ROW_GOOD_FILL = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")  # light green
_ROW_BAD_FILL  = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")  # light red
_SCORE_GOOD    = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
_SCORE_BAD     = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _style_header_row(ws, headers: list[str], fill: PatternFill):
    """Apply styled header row to the given worksheet."""
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font  = _HEADER_FONT
        cell.fill  = fill
        cell.alignment = _HEADER_ALIGN
        cell.border = _THIN_BORDER


def _auto_col_widths(ws, headers: list[str]):
    """Set reasonable column widths based on header names."""
    width_map = {
        "id":               8,
        "category":        22,
        "difficulty":      14,
        "question":        55,
        "reference_answer": 65,
        "model_answer":    65,
        "correctness":     14,
        "completeness":    14,
        "clarity":         12,
        "conciseness":     14,
        "overall":         12,
        "justification":   70,
        "source_paper":    35,
        "keywords":        35,
    }
    for col_idx, header in enumerate(headers, start=1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width_map.get(header, 20)


def _write_rows(ws, headers: list[str], rows: list[dict],
                row_bg_fill: PatternFill, score_fill: PatternFill,
                overall_col_idx: int):
    """Write data rows with styling to worksheet."""
    for row_idx, row_data in enumerate(rows, start=2):
        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=row_data.get(header))
            cell.alignment = _WRAP_ALIGN
            cell.border    = _THIN_BORDER
            cell.fill      = row_bg_fill

        # Highlight the overall score cell distinctly
        score_cell = ws.cell(row=row_idx, column=overall_col_idx)
        score_cell.fill = score_fill
        score_cell.font = Font(name="Calibri", bold=True, size=11)
        score_cell.alignment = Alignment(horizontal="center", vertical="center")


# ═══════════════════════════════════════════════════════════════════════
# Core: auto-detect score column + filter
# ═══════════════════════════════════════════════════════════════════════

def _detect_score_column(headers: list[str]) -> str | None:
    """
    Auto-detect the overall score column name.
    Tries 'overall' first, then falls back to any column that looks like
    a numeric score column (contains 'score' or 'overall' case-insensitively).
    """
    normalized = [h.lower().strip() if h else "" for h in headers]

    # Exact match first
    for h in headers:
        if h and h.lower().strip() == "overall":
            return h

    # Fuzzy: anything containing 'overall'
    for h in headers:
        if h and "overall" in h.lower():
            return h

    # Fuzzy: anything containing 'score'
    for h in headers:
        if h and "score" in h.lower():
            return h

    return None


def filter_judged_file(
    input_path: Path,
    output_path: Path,
    good_threshold: float,
    bad_threshold: float,
):
    """
    Read a judged Excel file, filter rows into good/bad buckets,
    and save to a new Excel file with two sheets.
    """
    wb_in = load_workbook(input_path, read_only=True, data_only=True)

    # Try 'Judged' sheet first; fall back to first sheet
    if "Judged" in wb_in.sheetnames:
        ws_in = wb_in["Judged"]
    else:
        ws_in = wb_in.active
        print(f"  ⚠ 'Judged' sheet not found — using sheet: '{ws_in.title}'")

    # Read all rows into memory
    all_rows = list(ws_in.iter_rows(values_only=True))
    if not all_rows:
        print(f"  ⚠ Sheet is empty — skipping.")
        wb_in.close()
        return

    headers = [str(h) if h is not None else "" for h in all_rows[0]]
    data_rows = all_rows[1:]

    # Auto-detect overall score column
    score_col_name = _detect_score_column(headers)
    if score_col_name is None:
        print(f"  ✗ Could not detect a score column in: {headers}")
        wb_in.close()
        return

    score_col_idx = headers.index(score_col_name)
    print(f"  → Detected score column: '{score_col_name}' (column {score_col_idx + 1})")

    # Convert rows to dicts
    dict_rows = [dict(zip(headers, row)) for row in data_rows]

    # Filter
    good_rows = []
    bad_rows  = []
    skipped   = 0

    for row in dict_rows:
        val = row.get(score_col_name)
        try:
            score = float(val)
        except (TypeError, ValueError):
            skipped += 1
            continue

        if score > good_threshold:
            good_rows.append(row)
        elif score < bad_threshold:
            bad_rows.append(row)

    wb_in.close()

    print(f"  → Total data rows : {len(dict_rows)}")
    print(f"  → Good (>{good_threshold}): {len(good_rows)} rows")
    print(f"  → Bad  (<{bad_threshold}):  {len(bad_rows)} rows")
    print(f"  → Skipped (no score): {skipped} rows")

    if not good_rows and not bad_rows:
        print(f"  ⚠ No rows matched either filter — output file not created.")
        return

    # ── Build output workbook ────────────────────────────────────────
    wb_out = Workbook()

    # Overall score column index for highlight
    overall_output_idx = headers.index(score_col_name) + 1

    # ── Good Answers sheet ──────────────────────────────────────────
    ws_good = wb_out.active
    ws_good.title = f"Good Answers (>{good_threshold})"
    _style_header_row(ws_good, headers, _GOOD_FILL)
    _auto_col_widths(ws_good, headers)
    ws_good.row_dimensions[1].height = 30
    if good_rows:
        _write_rows(ws_good, headers, good_rows, _ROW_GOOD_FILL, _SCORE_GOOD, overall_output_idx)
    else:
        ws_good.cell(row=2, column=1, value="No rows with overall score > " + str(good_threshold))

    # ── Bad Answers sheet ────────────────────────────────────────────
    ws_bad = wb_out.create_sheet(f"Bad Answers (<{bad_threshold})")
    _style_header_row(ws_bad, headers, _BAD_FILL)
    _auto_col_widths(ws_bad, headers)
    ws_bad.row_dimensions[1].height = 30
    if bad_rows:
        _write_rows(ws_bad, headers, bad_rows, _ROW_BAD_FILL, _SCORE_BAD, overall_output_idx)
    else:
        ws_bad.cell(row=2, column=1, value="No rows with overall score < " + str(bad_threshold))

    # ── Summary sheet ───────────────────────────────────────────────
    ws_summary = wb_out.create_sheet("Summary")
    summary_data = [
        ("Key", "Value"),
        ("source_file",          str(input_path.name)),
        ("score_column_used",    score_col_name),
        ("good_threshold",       f"> {good_threshold}"),
        ("bad_threshold",        f"< {bad_threshold}"),
        ("total_data_rows",      len(dict_rows)),
        ("good_rows_count",      len(good_rows)),
        ("bad_rows_count",       len(bad_rows)),
        ("skipped_rows",         skipped),
    ]
    for row_idx, (key, value) in enumerate(summary_data, start=1):
        ws_summary.cell(row=row_idx, column=1, value=key)
        ws_summary.cell(row=row_idx, column=2, value=str(value))
        if row_idx == 1:
            for col in (1, 2):
                c = ws_summary.cell(row=row_idx, column=col)
                c.font = _HEADER_FONT
                c.fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
                c.alignment = _HEADER_ALIGN
                c.border = _THIN_BORDER
        else:
            ws_summary.cell(row=row_idx, column=1).font = Font(name="Calibri", bold=True, size=11)
            ws_summary.cell(row=row_idx, column=1).border = _THIN_BORDER
            ws_summary.cell(row=row_idx, column=2).border = _THIN_BORDER

    ws_summary.column_dimensions["A"].width = 28
    ws_summary.column_dimensions["B"].width = 45

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb_out.save(output_path)
    print(f"  ✓ Saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Filter judged Excel files into good/bad answer sheets."
    )
    parser.add_argument(
        "--judged_dir",
        default=DEFAULT_JUDGED_DIR,
        help=f"Directory containing judged Excel files (default: {DEFAULT_JUDGED_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write filtered Excel files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--good_threshold",
        type=float,
        default=DEFAULT_GOOD_THRESHOLD,
        help=f"Overall score threshold for 'good' answers — rows with score > this (default: {DEFAULT_GOOD_THRESHOLD})",
    )
    parser.add_argument(
        "--bad_threshold",
        type=float,
        default=DEFAULT_BAD_THRESHOLD,
        help=f"Overall score threshold for 'bad' answers — rows with score < this (default: {DEFAULT_BAD_THRESHOLD})",
    )
    args = parser.parse_args()

    judged_dir = Path(args.judged_dir)
    output_dir = Path(args.output_dir)

    if not judged_dir.exists():
        print(f"✗ Judged directory does not exist: {judged_dir}")
        sys.exit(1)

    # Find all .xlsx files (skip temp lock files starting with .~lock.)
    xlsx_files = sorted([
        f for f in judged_dir.glob("*.xlsx")
        if not f.name.startswith(".~lock.")
    ])

    if not xlsx_files:
        print(f"✗ No .xlsx files found in: {judged_dir}")
        sys.exit(1)

    print(f"\nFilter Judged Excel Files")
    print(f"{'─' * 60}")
    print(f"  Judged dir    : {judged_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Good threshold: overall > {args.good_threshold}")
    print(f"  Bad threshold : overall < {args.bad_threshold}")
    print(f"  Files found   : {len(xlsx_files)}")
    print()

    for xlsx_file in xlsx_files:
        # Derive output filename: replace prefix 'hpn_judged_' → 'hpn_filtered_'
        out_name = xlsx_file.name
        if out_name.startswith("hpn_judged_"):
            out_name = "hpn_filtered_" + out_name[len("hpn_judged_"):]
        else:
            out_name = "filtered_" + out_name

        output_path = output_dir / out_name

        print(f"Processing: {xlsx_file.name}")
        filter_judged_file(
            input_path=xlsx_file,
            output_path=output_path,
            good_threshold=args.good_threshold,
            bad_threshold=args.bad_threshold,
        )
        print()

    print(f"Done. Filtered files saved to: {output_dir}")


if __name__ == "__main__":
    main()
