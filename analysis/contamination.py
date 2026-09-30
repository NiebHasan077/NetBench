"""R8: test for pre-training contamination without asserting a cutoff date.

Phase 12, answering the reviewer's "contamination analysis using paper dates and
model knowledge cutoffs". The obvious implementation -- look up each vendor's
published cutoff, split items on it, compare -- has a defect that would undo the
result: the cutoffs are facts we would be *asserting* rather than measuring,
they are vague, and vendors refresh checkpoints under one name. A reviewer can
reject the input and the analysis with it.

So the cutoff is never asserted. Two designs replace it.

**1. The open-book control.** For item i and model m take

        delta = closed-book (P1) - open-book (P6)

on the same question, the same rubric, the same model. This is how much the
model already knew relative to being handed the evidence. If exposure to a paper
during pre-training inflates closed-book answers, delta should be *larger* for
older papers, which are likelier to have been in the crawl. Differencing against
open book removes what would otherwise be the fatal confound -- newer papers
cover harder or less-settled topics -- because it is the same question on both
sides. The raw closed-book gradient is reported next to it precisely to show how
much of it is that confound.

**2. A discontinuity search instead of a date.** A cutoff, if it bites, shows up
as a *step*: items after it lose the exposure advantage. So sweep candidate
cutoff years and ask whether any produces one, with Benjamini-Hochberg over the
sweep. This asks the reviewer's question without needing to know the answer to
it, and reports which years were tested rather than which was believed.

**What neither design can fix, and the paper says so.** Publication year is not
training-data availability. These are open-access papers; an arXiv preprint can
precede its published date by two years, and reposting blurs it further. That
smearing is one-directional: it moves genuinely-exposed items into the "recent"
bin, which *weakens* any real gradient. So a null here is evidence of the weaker
kind -- consistent with no contamination, not proof of none -- and a step search
suffers more from the smearing than a gradient does. Reported as a bound, never
as a clean bill of health.

Resampling is by source paper (`clusters.py`), which matters more here than
anywhere else in the phase: items sharing a paper share its year exactly, so
question-level intervals on a year gradient would be badly optimistic.

Reads only committed outputs. Nothing is re-run.

    analysis/.venv/bin/python analysis/contamination.py
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clusters as C                                        # noqa: E402
from aggregate_scores import FDR_Q, SIZE_GROUP              # noqa: E402

METRIC = "overall_formula"
CLOSED, OPEN = "P1", "P6"          # instruct closed-book vs the same model + RAG
CUTOFFS = (2020, 2021, 2022, 2023, 2024, 2025)
MIN_BIN = 10                       # matches aggregate_scores.MIN_N


def item_years(benchmark: str, papers: str) -> pd.Series:
    """question_id -> newest source-paper year.

    Newest, not oldest: exposure is about whether the model could have seen the
    material, and a multi-paper item is only as "new" as its most recent source.
    """
    yr = pd.read_csv(papers).set_index("paper_id")["year"].to_dict()
    out = {}
    with open(benchmark, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("excluded_from_scoring"):
                continue
            ps = r["source_papers"]
            ps = ast.literal_eval(ps) if isinstance(ps, str) else ps or []
            ys = [yr[p] for p in ps if p in yr and yr[p] == yr[p]]
            if ys:
                out[r["id"]] = max(ys)
    return pd.Series(out, name="year")


def gradient(year: np.ndarray, val: np.ndarray, cl: np.ndarray) -> dict:
    """Spearman rho and an OLS slope per decade, both cluster-bootstrapped."""
    rho = spearmanr(year, val).statistic
    slope = float(np.polyfit(year, val, 1)[0]) * 10.0
    idx = C._groups(cl)
    rng = np.random.default_rng(C.SEED)
    k = len(idx)
    reps_r, reps_s = [], []
    for draw in rng.integers(0, k, size=(C.RESAMPLE_N, k)):
        sel = np.concatenate([idx[j] for j in draw])
        y, v = year[sel], val[sel]
        if len(np.unique(y)) < 2:
            continue
        reps_r.append(spearmanr(y, v).statistic)
        reps_s.append(np.polyfit(y, v, 1)[0] * 10.0)
    return dict(rho=round(float(rho), 3),
                rho_lo=round(float(np.percentile(reps_r, 2.5)), 3),
                rho_hi=round(float(np.percentile(reps_r, 97.5)), 3),
                slope_per_decade=round(slope, 3),
                slope_lo=round(float(np.percentile(reps_s, 2.5)), 3),
                slope_hi=round(float(np.percentile(reps_s, 97.5)), 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-long", default="analysis/outputs/scores_long.csv")
    ap.add_argument("--benchmark",
                    default="NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl")
    ap.add_argument("--papers", default="analysis/outputs/source_papers.csv")
    ap.add_argument("--out", default="analysis/outputs")
    args = ap.parse_args()

    years = item_years(args.benchmark, args.papers)
    d = pd.read_csv(args.scores_long)
    d = d[d.dimension == METRIC]
    piv = d.pivot_table(index=["base_model", "question_id"], columns="p_id",
                        values="score")
    models = [m for m in sorted(piv.index.get_level_values(0).unique())
              if SIZE_GROUP.get(m) != "api-baseline"]
    cl = C.question_clusters()

    # Per item: mean over models of closed-book, open-book, and their difference.
    # Averaging over models first keeps one row per question, which is the unit
    # the source-paper clustering is defined on.
    frames = []
    for m in models:
        s = piv.loc[m]
        if CLOSED not in s or OPEN not in s:
            continue
        frames.append(pd.DataFrame({"closed": s[CLOSED], "open": s[OPEN]}))
    per_item = sum(frames) / len(frames)
    per_item = per_item.join(years, how="inner").dropna()
    per_item["delta"] = per_item.closed - per_item.open
    q = per_item.index.to_numpy()
    cids = np.array([cl[i] for i in q])
    yr = per_item.year.to_numpy(dtype=float)

    print(f"{len(per_item)} scored items carry a source-paper year, over "
          f"{len(set(cids))} source-paper clusters, {len(frames)} models\n")
    print(per_item.groupby(pd.cut(per_item.year, [0, 2015, 2019, 2022, 2024, 2026],
                                  labels=["<=2015", "2016-19", "2020-22",
                                          "2023-24", "2025-26"]),
                           observed=True)
          .agg(n=("delta", "size"), closed=("closed", "mean"),
               open=("open", "mean"), delta=("delta", "mean")).round(3).to_string())

    rows = []
    print("\nyear gradient (95% CI resampled by source paper):")
    for label, val in (("closed-book only (confounded)", per_item.closed.to_numpy()),
                       ("open-book only (control)", per_item.open.to_numpy()),
                       ("closed - open (the test)", per_item.delta.to_numpy())):
        g = gradient(yr, val, cids)
        rows.append(dict(analysis="gradient", series=label, **g))
        print(f"  {label:30s} rho {g['rho']:+.3f} "
              f"[{g['rho_lo']:+.3f},{g['rho_hi']:+.3f}]   "
              f"slope/decade {g['slope_per_decade']:+.3f} "
              f"[{g['slope_lo']:+.3f},{g['slope_hi']:+.3f}]")

    print("\ndiscontinuity search -- mean(delta | year<=c) - mean(delta | year>c):")
    steps, ps = [], []
    for c in CUTOFFS:
        old, new = yr <= c, yr > c
        if min(old.sum(), new.sum()) < MIN_BIN:
            continue
        # Two independent groups of clusters, so bootstrap each side separately
        # and difference the replicates rather than resampling a paired vector.
        rng = np.random.default_rng(C.SEED)
        reps = []
        gi_o, gi_n = C._groups(cids[old]), C._groups(cids[new])
        vo, vn = per_item.delta.to_numpy()[old], per_item.delta.to_numpy()[new]
        for _ in range(C.RESAMPLE_N):
            a = np.concatenate([gi_o[j] for j in rng.integers(0, len(gi_o), len(gi_o))])
            b = np.concatenate([gi_n[j] for j in rng.integers(0, len(gi_n), len(gi_n))])
            reps.append(vo[a].mean() - vn[b].mean())
        reps = np.array(reps)
        obs = vo.mean() - vn.mean()
        p = 2 * min((reps <= 0).mean(), (reps >= 0).mean())
        steps.append(dict(analysis="discontinuity", cutoff=c,
                          n_before=int(old.sum()), n_after=int(new.sum()),
                          step=round(float(obs), 3),
                          step_lo=round(float(np.percentile(reps, 2.5)), 3),
                          step_hi=round(float(np.percentile(reps, 97.5)), 3),
                          p_raw=round(float(p), 4)))
        ps.append(p)
    for r, p_bh in zip(steps, false_discovery_control(np.clip(ps, 1e-9, 1.0))):
        r["p_bh"] = round(float(p_bh), 4)
        print(f"  cutoff {r['cutoff']}  n {r['n_before']:3d}/{r['n_after']:3d}  "
              f"step {r['step']:+.3f} [{r['step_lo']:+.3f},{r['step_hi']:+.3f}]  "
              f"p_BH {r['p_bh']:.3f}")
    rows += steps

    hits = [r for r in steps if r["p_bh"] <= FDR_Q]
    print(f"\ncandidate cutoffs showing a step at q={FDR_Q}: "
          f"{len(hits)} of {len(steps)}")
    print("A null here is the weaker kind of evidence: publication year is not\n"
          "training-data availability, and open-access preprints smear exposure\n"
          "earlier, which drains a real gradient rather than inventing one.")

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "contamination.csv"),
                              index=False)
    print(f"\nwrote {args.out}/contamination.csv")


if __name__ == "__main__":
    main()
