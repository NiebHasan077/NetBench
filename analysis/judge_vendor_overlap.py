#!/usr/bin/env python3
"""Vendor-overlap self-preference check between the two LLM judges.

Each judge shares a vendor with systems in the roster: GPT-5.1 (OpenAI) with
the GPT-4o baseline; Gemini-3.5-Flash (Google) with the Gemma models and
Gemini-2.5-Pro. This script slices the system-level leniency offset
(second-judge mean minus primary-judge mean, formula metric) by vendor tie and
records the frontier baselines' ordering under both judges. Cited in
Appendix G (judge validation).

Input : analysis/outputs/HPN_JUDGE_AGREEMENT_gpt-5.1_vs_gemini-3.5-flash_formula.xlsx
        (Leaderboard sheet)
Output: analysis/outputs/judge_vendor_overlap.csv
"""

import re
from pathlib import Path

import pandas as pd

A = Path(__file__).resolve().parent / "outputs"
WORKBOOK = A / "HPN_JUDGE_AGREEMENT_gpt-5.1_vs_gemini-3.5-flash_formula.xlsx"
OUT = A / "judge_vendor_overlap.csv"

APIS = ["claude-sonnet-4-6", "gemini-2.5-pro", "gpt-4o"]


def classify(model: str) -> str:
    if model == "gpt-4o":
        return "judge1_vendor (gpt-4o)"
    if re.search(r"gemma|gemini", model, re.IGNORECASE):
        return "judge2_vendor (gemma/gemini systems)"
    return "neutral (neither vendor)"


def main() -> None:
    lb = pd.read_excel(WORKBOOK, sheet_name="Leaderboard")
    lb["offset"] = lb["gemini-3.5-flash_mean"] - lb["gpt-5.1_mean"]
    lb["slice"] = lb["model"].map(classify)

    rows = []
    for name, grp in lb.groupby("slice"):
        rows.append({
            "kind": "slice", "name": name, "n": len(grp),
            "gpt-5.1_mean": round(grp["gpt-5.1_mean"].mean(), 4),
            "gemini-3.5-flash_mean": round(grp["gemini-3.5-flash_mean"].mean(), 4),
            "offset": round(grp["offset"].mean(), 4),
            "gpt-5.1_rank": "", "gemini-3.5-flash_rank": "",
        })
    rows.append({
        "kind": "slice", "name": "all systems", "n": len(lb),
        "gpt-5.1_mean": round(lb["gpt-5.1_mean"].mean(), 4),
        "gemini-3.5-flash_mean": round(lb["gemini-3.5-flash_mean"].mean(), 4),
        "offset": round(lb["offset"].mean(), 4),
        "gpt-5.1_rank": "", "gemini-3.5-flash_rank": "",
    })
    for api in APIS:
        r = lb.loc[lb["model"] == api].iloc[0]
        rows.append({
            "kind": "api", "name": api, "n": 1,
            "gpt-5.1_mean": round(r["gpt-5.1_mean"], 4),
            "gemini-3.5-flash_mean": round(r["gemini-3.5-flash_mean"], 4),
            "offset": round(r["offset"], 4),
            "gpt-5.1_rank": int(r["gpt-5.1_rank"]),
            "gemini-3.5-flash_rank": int(r["gemini-3.5-flash_rank"]),
        })

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
