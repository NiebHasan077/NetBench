#!/usr/bin/env python3
"""Sensitivity of the RQ6 practical-significance counts to the thresholds.

The practical rule (S5) is |delta| >= 0.25 AND |d_z| >= 0.2 AND BH-significant.
This sweeps a grid of alternative (delta, d_z) minima over the 72-contrast
primary family (BH verdicts held fixed) and counts how many BH-significant
contrasts would clear each pair, so the "30 of 54" headline can be read
against neighboring conventions rather than a single point.

Input : analysis/outputs/significance_formula.csv
Output: analysis/outputs/threshold_sensitivity.csv
Cited in Appendix F; justification discussion in S5.
"""

from pathlib import Path

import pandas as pd

A = Path(__file__).resolve().parent / "outputs"
DELTAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
DZS = [0.10, 0.20, 0.30, 0.50]


def main() -> None:
    sig = pd.read_csv(A / "significance_formula.csv")
    bh = sig[sig["bh_sig"]]
    rows = []
    for dmin in DELTAS:
        row = {"delta_min": dmin}
        for dzmin in DZS:
            row[f"dz_{dzmin:.1f}"] = int(
                ((bh["mean_diff"].abs() >= dmin) & (bh["cohens_dz"].abs() >= dzmin)).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(A / "threshold_sensitivity.csv", index=False)
    print(f"BH-significant total: {len(bh)} of {len(sig)}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
