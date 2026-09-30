"""R6: test the frontier-parity claims instead of ranking point estimates.

Phase 12, answering the reviewer's "'frontier parity' is not established by
ranking point estimates". The paper says open-weight systems "rival", "tie" and
"surpass" the frontier APIs, and backs those words with Table 13, which is a
ranking of means. A ranking cannot establish equivalence, and "not
significantly different" is not "equivalent" -- absence of evidence again. The
sharpest instance is §7.5's "Qwen3.5-9B P6 tying Gemini-2.5-Pro exactly at
4.358", which is a coincidence of rounding presented as a result.

Every system answered the same 233 questions, so the comparison can be paired
and tested properly, with no rerun of anything:

  superiority   two-sided sign-flip permutation on the paired mean difference
                (this is what "surpasses" needs).
  equivalence   TOST at the paper's own practical margin of +/-0.25. Two
                one-sided permutation tests: reject "the gap is at most -0.25"
                AND reject "the gap is at least +0.25". Equivalent only if both
                reject. Reported alongside the 90% interval, which is the
                interval form of the same decision -- inside +/-0.25 means
                equivalent at alpha = 0.05.

Both resample by source-paper cluster (see `clusters.py`), so this inherits R1's
cluster-robust treatment rather than re-introducing the independence assumption
R1 was written to check.

A comparison can come out four ways, and three of them are not "parity":
superior, equivalent, inferior, or *inconclusive* -- the interval is too wide to
rule either out, which is the honest verdict when a study is underpowered for
the margin it chose. We report whichever it is.

    analysis/.venv/bin/python analysis/equivalence_tests.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clusters as C                                        # noqa: E402
from aggregate_scores import (FDR_Q, PRACTICAL_MIN_DELTA,    # noqa: E402
                              SIZE_GROUP, cohen_dz)

METRIC = "overall_formula"
MARGIN = PRACTICAL_MIN_DELTA          # +/-0.25, the paper's pre-registered bar
APIS = [m for m, g in SIZE_GROUP.items() if g == "api-baseline"]


def verdict(row) -> str:
    """The 2x2 that TOST actually produces, not a winner.

    Difference-from-zero and equivalence-within-the-margin are separate
    questions and all four combinations occur, so collapsing them to a ranking
    is what got the paper into this. It is the same statistical-versus-practical
    distinction §7.7 already makes; applying it here rather than declaring a
    winner is just consistency.
    """
    differs = row["p_bh_superiority"] <= FDR_Q
    equiv = row["p_tost"] <= FDR_Q
    direction = "higher" if row["mean_diff"] > 0 else "lower"
    if differs and equiv:
        return f"{direction}, but below the practical margin"
    if differs:
        return f"meaningfully {direction}"
    if equiv:
        return "equivalent"
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="analysis/outputs")
    ap.add_argument("--scores-long", default="analysis/outputs/scores_long.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.scores_long)
    d = d[d.dimension == METRIC]
    piv = d.pivot_table(index=["base_model", "question_id"], columns="p_id",
                        values="score")
    cl = C.question_clusters()

    # Each open-weight model's best variant, the same rule Table 13 uses.
    best: dict[str, str] = {}
    for model in sorted(piv.index.get_level_values(0).unique()):
        if SIZE_GROUP.get(model) == "api-baseline":
            continue
        means = piv.loc[model].mean()
        best[model] = str(means.idxmax())

    rows = []
    for model, variant in best.items():
        open_scores = piv.loc[model][variant].dropna()
        for api in APIS:
            api_scores = piv.loc[api]["BASE-API"].dropna()
            pair = pd.concat([open_scores.rename("open"),
                              api_scores.rename("api")], axis=1).dropna()
            diff = (pair["open"] - pair["api"]).to_numpy()
            cids = pair.index.map(cl).to_numpy()
            lo95, hi95 = C.cluster_bootstrap_ci(diff, cids, level=95.0)
            lo90, hi90 = C.cluster_bootstrap_ci(diff, cids, level=90.0)
            # TOST: shift by the margin and ask a directional question each way.
            p_lower = C.cluster_signflip_p(diff + MARGIN, cids, alternative="greater")
            p_upper = C.cluster_signflip_p(diff - MARGIN, cids, alternative="less")
            rows.append(dict(
                open_system=f"{model} ({variant})", api=api, n=len(pair),
                mean_open=round(float(pair["open"].mean()), 3),
                mean_api=round(float(pair["api"].mean()), 3),
                mean_diff=round(float(diff.mean()), 3),
                ci95_lo=round(lo95, 3), ci95_hi=round(hi95, 3),
                ci90_lo=round(lo90, 3), ci90_hi=round(hi90, 3),
                cohens_dz=round(cohen_dz(diff), 3),
                p_superiority=round(C.cluster_signflip_p(diff, cids), 5),
                p_tost_lower=round(p_lower, 5), p_tost_upper=round(p_upper, 5),
                p_tost=round(max(p_lower, p_upper), 5)))

    out = pd.DataFrame(rows)
    out["p_bh_superiority"] = np.round(
        false_discovery_control(out["p_superiority"].to_numpy(), method="bh"), 5)
    out["equivalent_90ci"] = (out["ci90_lo"] > -MARGIN) & (out["ci90_hi"] < MARGIN)
    out["verdict"] = out.apply(verdict, axis=1)
    out = out.sort_values(["open_system", "api"]).reset_index(drop=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "equivalence_frontier.csv")
    out.to_csv(path, index=False)

    print(f"margin +/-{MARGIN}; {len(out)} paired comparisons over n={out.n.iloc[0]} "
          f"questions, resampled by source-paper cluster\n")
    print(f"{'open-weight system':26s} {'API':18s} {'diff':>7s} {'90% CI':>17s}  verdict")
    for _, r in out.iterrows():
        print(f"{r.open_system:26s} {r.api:18s} {r.mean_diff:+7.3f} "
              f"[{r.ci90_lo:+.3f},{r.ci90_hi:+.3f}]  {r.verdict}")
    print()
    for v, n in out["verdict"].value_counts().items():
        print(f"  {n:2d}  {v}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
