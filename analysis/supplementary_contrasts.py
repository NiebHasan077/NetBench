#!/usr/bin/env python3
"""Supplementary paired contrasts OUTSIDE the pre-specified primary family.

P1<->P2 (Instruct vs LoRA SFT) and P2<->P3 (LoRA SFT vs full SFT) were not in
the pre-specified family (S5) but are needed to back the LoRA-vs-full-SFT
prose comparisons with uncertainty. Same protocol as the primary family
(Wilcoxon, seeded 10k bootstrap CI, Cohen's d_z, BH within this supplementary
family, same practical-significance rule), reusing aggregate_scores' code so
the machinery is identical. Reported in Appendix F, clearly labeled
supplementary.

Input : analysis/outputs/scores_long.csv (formula metric)
Output: analysis/outputs/significance_formula_supplementary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_scores import add_practical, run_contrasts  # noqa: E402

A = Path(__file__).resolve().parent / "outputs"

SUPP_CONTRASTS = [
    ("P1", "P2", "closed_book", "Instruct vs LoRA SFT"),
    ("P2", "P3", "closed_book", "LoRA SFT vs full SFT"),
]


def main() -> None:
    long = pd.read_csv(A / "scores_long.csv")
    allq = long[long["dimension"] == "overall_formula"].rename(
        columns={"score": "overall"}).copy()
    allq["provisional"] = False
    trained = sorted(allq.loc[allq["p_id"] != "BASE-API", "base_model"].unique())

    supp = pd.DataFrame(run_contrasts(allq, trained, SUPP_CONTRASTS))
    supp["p_bh"] = np.round(
        false_discovery_control(supp["p_raw"].to_numpy(), method="bh"), 5)
    supp["p_raw"] = supp["p_raw"].round(5)
    supp = add_practical(supp)
    supp["note"] = "supplementary contrast, outside the pre-specified primary family"
    cols = ["base_model", "size_group", "group", "contrast", "description", "n",
            "mean_a", "mean_b", "mean_diff", "median_diff", "ci95_lo", "ci95_hi",
            "p_raw", "p_bh", "bh_sig", "cohens_dz", "practically_sig", "note"]
    supp[cols].to_csv(A / "significance_formula_supplementary.csv", index=False)
    print(supp[["base_model", "contrast", "mean_diff", "ci95_lo", "ci95_hi",
                "p_bh", "bh_sig", "cohens_dz", "practically_sig"]].to_string(index=False))


if __name__ == "__main__":
    main()
