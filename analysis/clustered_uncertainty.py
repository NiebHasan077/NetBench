"""R1: re-run the primary contrast family with cluster-robust uncertainty.

Phase 12, answering the reviewer's "the statistical unit of analysis is
questionable". The pre-registered analysis pairs by question and resamples
questions; questions are drawn from 171 source papers and several can share
one, so the effective sample is smaller than 233 and the intervals are
optimistic by some amount. This script measures that amount rather than
arguing about it.

For each of the primary (model x contrast) cells it recomputes the same mean
paired difference under two cluster-level resamplers from `clusters.py` -- a
cluster bootstrap CI and a cluster sign-flip permutation p -- then re-applies
Benjamini-Hochberg over the clustered p-values and reports which verdicts move.

This does NOT replace `significance_formula.csv`. The pre-registered analysis
stays exactly as published; a protocol fixed before the results were seen is
worth more than a protocol improved after, and swapping it out post hoc would
throw away the one thing pre-registration buys. This is reported alongside it
as a robustness check, which is what it is.

Reads only committed outputs (`scores_long.csv`, the benchmark JSONL) -- no
model, judge, or retrieval run is involved.

    analysis/.venv/bin/python analysis/clustered_uncertainty.py
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
from aggregate_scores import (DZ_MIN, FDR_Q, MIN_N,          # noqa: E402
                              PRACTICAL_MIN_DELTA, PRIMARY_CONTRASTS,
                              SIZE_GROUP, cohen_dz)

METRIC = "overall_formula"   # the jbd/eacl primary metric (see CLAUDE.md)


def load_pivot(scores_long: str) -> pd.DataFrame:
    d = pd.read_csv(scores_long)
    d = d[d.dimension == METRIC]
    return d.pivot_table(index=["base_model", "question_id"], columns="p_id",
                         values="score")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="analysis/outputs")
    ap.add_argument("--scores-long", default="analysis/outputs/scores_long.csv")
    ap.add_argument("--baseline", default="analysis/outputs/significance_formula.csv",
                    help="the pre-registered question-level result, for the delta report")
    args = ap.parse_args()

    cl = C.question_clusters()
    info = C.summary(cl)
    print(f"clusters: {info['n_clusters']} over {info['n_items']} items "
          f"(mean {info['mean_size']}, max {info['max_size']}); worst-case CI "
          f"inflation at ICC=1 is x{info['worst_case_ci_inflation']}")

    piv = load_pivot(args.scores_long)
    rows = []
    for model in sorted(piv.index.get_level_values(0).unique()):
        if SIZE_GROUP.get(model) == "api-baseline":
            continue                                   # APIs have no adaptation variants
        sub = piv.loc[model]
        for a, b, group, desc in PRIMARY_CONTRASTS:
            if a not in sub or b not in sub:
                continue
            pair = sub[[a, b]].dropna()
            if len(pair) < MIN_N:
                continue
            diff = (pair[b] - pair[a]).to_numpy()
            cids = pair.index.map(cl).to_numpy()
            lo, hi = C.cluster_bootstrap_ci(diff, cids)
            # The published p is Wilcoxon; ours is a sign-flip permutation.
            # Comparing those two directly would blame the clustering for a
            # change of test statistic, so run the permutation twice -- once
            # with every question its own cluster, once with the real clusters
            # -- and isolate the clustering effect as the difference.
            singleton = np.arange(len(diff))
            rows.append(dict(
                base_model=model, size_group=SIZE_GROUP.get(model), group=group,
                contrast=f"{a} vs {b}", description=desc, n=len(pair),
                n_clusters=int(len(set(cids))),
                mean_diff=round(float(diff.mean()), 3),
                ci95_lo_clustered=round(lo, 3), ci95_hi_clustered=round(hi, 3),
                p_perm_question=round(C.cluster_signflip_p(diff, singleton), 5),
                p_perm_clustered=round(C.cluster_signflip_p(diff, cids), 5),
                cohens_dz=round(cohen_dz(diff), 3)))

    out = pd.DataFrame(rows)
    for src, tag in (("p_perm_question", "perm_question"),
                     ("p_perm_clustered", "clustered")):
        out[f"p_bh_{tag}"] = np.round(
            false_discovery_control(out[src].to_numpy(), method="bh"), 5)
        out[f"bh_sig_{tag}"] = out[f"p_bh_{tag}"] <= FDR_Q
        out[f"practically_sig_{tag}"] = (
            (out["mean_diff"].abs() >= PRACTICAL_MIN_DELTA)
            & (out["cohens_dz"].abs() >= DZ_MIN) & out[f"bh_sig_{tag}"])

    # --- compare against the pre-registered question-level result ---
    base = pd.read_csv(args.baseline)[
        ["base_model", "contrast", "ci95_lo", "ci95_hi", "p_bh", "bh_sig",
         "practically_sig"]]
    m = out.merge(base, on=["base_model", "contrast"], how="left",
                  suffixes=("", "_question"))
    m["ci_width_question"] = (m["ci95_hi"] - m["ci95_lo"]).round(3)
    m["ci_width_clustered"] = (m["ci95_hi_clustered"] - m["ci95_lo_clustered"]).round(3)
    m["ci_inflation"] = (m["ci_width_clustered"] / m["ci_width_question"]).round(3)
    m["bh_sig_changed"] = m["bh_sig_clustered"] != m["bh_sig"]
    m["practically_sig_changed"] = m["practically_sig_clustered"] != m["practically_sig"]

    os.makedirs(args.out, exist_ok=True)
    m.to_csv(os.path.join(args.out, "significance_formula_clustered.csv"), index=False)
    pd.DataFrame(sorted(cl.items()), columns=["question_id", "cluster_id"]).to_csv(
        os.path.join(args.out, "question_clusters.csv"), index=False)

    print(f"\nmedian CI inflation: x{m['ci_inflation'].median():.3f}  "
          f"(max x{m['ci_inflation'].max():.3f}; worst case allowed by the design "
          f"is x{info['worst_case_ci_inflation']})")
    print("\nBH-significant of 72, decomposing the two changes:")
    print(f"   {int(m['bh_sig'].sum()):2d}  published        (Wilcoxon, questions independent)")
    print(f"   {int(m['bh_sig_perm_question'].sum()):2d}  permutation      (same independence, sign-flip test)"
          f"   <- cost of the test")
    print(f"   {int(m['bh_sig_clustered'].sum()):2d}  permutation      (clustered by source paper)"
          f"        <- cost of clustering")
    print(f"\npractically significant: {int(m['practically_sig'].sum())} published -> "
          f"{int(m['practically_sig_clustered'].sum())} clustered")
    ch = m[m["bh_sig_perm_question"] != m["bh_sig_clustered"]]
    if len(ch):
        print("\nlost to clustering alone (permutation held fixed):")
        for _, r in ch.iterrows():
            print(f"   {r.base_model:14s} {r.contrast:10s} d={r.mean_diff:+.3f} "
                  f"(p {r.p_bh_perm_question:.4f} -> {r.p_bh_clustered:.4f})"
                  f"{'   [PRACTICALLY SIGNIFICANT]' if r.practically_sig else ''}")
    else:
        print("\nno BH verdict changed under clustering alone.")
    if not m["practically_sig_changed"].any():
        print("no practically significant effect changed status.")
    print(f"\nwrote {args.out}/significance_formula_clustered.csv and question_clusters.csv")


if __name__ == "__main__":
    main()
