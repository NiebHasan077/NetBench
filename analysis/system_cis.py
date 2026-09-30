#!/usr/bin/env python3
"""Per-system 95% confidence intervals on the mean composite score.

The results tables report each system's mean ``overall_formula`` over the 233
scored questions with no spread. This adds one: a percentile bootstrap on the
mean that resamples whole source-paper clusters (``clusters.py``), so questions
drawn from the same paper move together. That is the conservative choice, and
the same resampler as the clustered robustness analysis.

Reads only ``scores_long.csv`` (from ``aggregate_scores.py``); no model, judge,
or retrieval run is involved.

    analysis/.venv/bin/python analysis/system_cis.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clusters import cluster_bootstrap_ci, question_clusters  # noqa: E402

SCORES = "analysis/outputs/scores_long.csv"
OUT = "analysis/outputs/system_cis.csv"
METRIC = "overall_formula"


def main() -> None:
    clusters = question_clusters()
    df = pd.read_csv(SCORES)
    df = df[df["dimension"] == METRIC]

    rows = []
    keys = ["size_group", "base_model", "setting", "p_id"]
    for key, g in df.groupby(keys, sort=True):
        g = g.sort_values("question_id")
        scores = g["score"].to_numpy(dtype=float)
        cl = np.array([clusters[q] for q in g["question_id"]])
        lo, hi = cluster_bootstrap_ci(scores, cl)
        rows.append({**dict(zip(keys, key)),
                     "n": len(scores),
                     "n_clusters": len(set(cl)),
                     "mean": round(float(scores.mean()), 3),
                     "ci95_lo": round(lo, 3),
                     "ci95_hi": round(hi, 3),
                     "half_width": round((hi - lo) / 2, 3)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    hw = out["half_width"]
    print(f"✅ {len(out)} systems -> {OUT}")
    print(f"   95% CI half-width: min {hw.min():.3f}, median {hw.median():.3f}, max {hw.max():.3f}")


if __name__ == "__main__":
    main()
