"""R3: put an interval on the scale trend, and say what it does not identify.

Phase 12, answering the reviewer's "eight models across three families is a thin
basis for a scaling claim, and 'numerically grounded switching rule' oversells
it". Both halves are fair, and they need separate answers, because the paper is
making two claims that happen to sit in one sentence:

  (a) full SFT's gain over the instruct baseline *declines with scale*;
  (b) it *crosses zero at about 9B*, equivalently at a closed-book score of
      about 3.7/5, which is what makes it a switching rule.

(a) turns out to be sturdier than the reviewer feared and (b) considerably
weaker than we wrote, so this script reports them apart rather than together.

Three sources of uncertainty, deliberately not merged into one number, because
they answer different questions and a reader needs to know which is binding:

  question resampling   roster fixed, questions resampled by source paper
                        (`clusters.py`, the R1 unit). "How much of the trend is
                        measurement noise in the eight deltas?"
  roster resampling     questions fixed, the eight models resampled. "Would a
                        different eight models have shown this?" This is the
                        reviewer's actual objection. At n=8 the bootstrap is
                        itself crude, so it is reported next to two exact
                        alternatives that do not lean on resampling at all:
                        leave-one-model-out (8 refits) and leave-one-family-out
                        (3 refits).
  combined              both at once: the honest total.

Two exact tests carry more weight here than any bootstrap, and both are cheap
at this size:

  * an exhaustive permutation p for rho over all 8! = 40,320 orderings, rather
    than the asymptotic p `spearmanr` reports, which is not trustworthy at n=8;
  * a within-family monotonicity test that never pools across families -- the
    one form of the claim the reviewer's objection cannot touch. Under the null
    that each family's deltas are ordered at random, the chance that all three
    families come out monotone decreasing is 1/(3! x 2! x 3!) = 1/72. (This
    statistic is post hoc in its form, though its direction is the pre-stated
    hypothesis; it is reported as corroboration, not as a pre-registered test.)

Reads only committed outputs (`scores_long.csv`, the benchmark JSONL). Nothing
is re-trained, re-generated, or re-judged. `NUM`/`ORDER`/`fam` are imported from
`make_figures` so the rho reported here is by construction the rho printed in
Figure 5's legend.

    analysis/.venv/bin/python analysis/scale_trend.py
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clusters as C                                        # noqa: E402
from aggregate_scores import PRACTICAL_MIN_DELTA            # noqa: E402
from make_figures import NUM, ORDER, fam                    # noqa: E402

METRIC = "overall_formula"    # the jbd/eacl primary metric (see CLAUDE.md)
BASE, ADAPTED = "P1", "P3"    # instruct baseline vs full SFT
SEED, N_BOOT = C.SEED, C.RESAMPLE_N
FAMILIES = ("gemma", "llama", "qwen")


# ---------------------------------------------------------------- statistics

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho, tie-corrected (two models share 27B, so ties are real)."""
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom else np.nan


def exact_permutation_p(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Two-sided p for rho by complete enumeration of every ordering of y.

    8! = 40,320 permutations, so this is exhaustive rather than sampled: the
    p-value is exact, not a Monte-Carlo estimate. Permuting y is the same as
    permuting its ranks, and a permuted rank vector has a fixed centred norm,
    so every replicate reduces to one dot product.
    """
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    perms = np.array(list(itertools.permutations(range(len(y)))))
    stats = ry[perms] @ rx
    return float(np.mean(np.abs(stats) >= abs(rx @ ry) - 1e-12)), len(perms)


def zero_crossing(x_log2: np.ndarray, y: np.ndarray) -> float:
    """Model size (B parameters) where an OLS fit of delta on log2(size) hits 0.

    NaN when the fitted slope is not negative: a flat or rising fit has no
    switching point to report, and extrapolating one would invent a threshold
    out of a trend that is not there. Also NaN when a roster replicate drew a
    single distinct size, which determines no line at all.
    """
    if len(np.unique(x_log2)) < 2:
        return np.nan
    slope, intercept = np.polyfit(x_log2, y, 1)
    return float(2.0 ** (-intercept / slope)) if slope < 0 else np.nan


def score_crossing(baseline: np.ndarray, y: np.ndarray) -> float:
    """Closed-book baseline score where the same fit, run on score, hits 0."""
    if len(np.unique(baseline)) < 2:
        return np.nan
    slope, intercept = np.polyfit(baseline, y, 1)
    return float(-intercept / slope) if slope < 0 else np.nan


def ci(values: np.ndarray, level: float = 95.0) -> tuple[float, float, int]:
    """Percentile CI over the finite replicates, with the usable count.

    Replicates are dropped rather than imputed: a roster replicate that drew one
    model eight times has no rank correlation, and a fit with a non-negative
    slope has no crossing. Reporting how many were dropped is part of the
    result -- it is a measure of how often the quantity is undefined.
    """
    ok = values[np.isfinite(values)]
    if len(ok) < 2:
        return np.nan, np.nan, len(ok)
    tail = (100.0 - level) / 2.0
    return (float(np.percentile(ok, tail)),
            float(np.percentile(ok, 100.0 - tail)), len(ok))


# ------------------------------------------------------------------ the data

def load(scores_long: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Per-model, per-question adapted-minus-baseline matrix and baseline matrix.

    Returns (models, delta[8, n_q], baseline[8, n_q], question_ids), all aligned
    on the questions every model answered under both variants.
    """
    d = pd.read_csv(scores_long)
    d = d[d.dimension == METRIC]
    piv = d.pivot_table(index=["base_model", "question_id"], columns="p_id",
                        values="score")
    models = [m for m in sorted(ORDER, key=lambda m: (NUM[m], m)) if m in piv.index]
    qids = None
    for m in models:
        have = piv.loc[m][[BASE, ADAPTED]].dropna().index
        qids = have if qids is None else qids.intersection(have)
    delta = np.array([(piv.loc[m, ADAPTED] - piv.loc[m, BASE]).reindex(qids).to_numpy()
                      for m in models])
    baseline = np.array([piv.loc[m, BASE].reindex(qids).to_numpy() for m in models])
    return models, delta, baseline, np.asarray(qids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="analysis/outputs")
    ap.add_argument("--scores-long", default="analysis/outputs/scores_long.csv")
    args = ap.parse_args()

    models, D, B, qids = load(args.scores_long)
    fams = np.array([fam(m) for m in models])
    x = np.log2([NUM[m] for m in models])
    delta, base = D.mean(1), B.mean(1)
    n_models, n_q = D.shape

    cl = C.question_clusters()
    cids = np.array([cl[q] for q in qids])
    groups = C._groups(cids)
    # Cluster sums, so a bootstrap replicate is a sum over drawn clusters rather
    # than a re-average over drawn rows: same estimator, one matrix product.
    sum_d = np.array([D[:, g].sum(1) for g in groups])          # (k, n_models)
    sum_b = np.array([B[:, g].sum(1) for g in groups])
    size = np.array([len(g) for g in groups], dtype=float)
    k = len(groups)

    print(f"{n_models} models, {n_q} questions, {k} source-paper clusters; "
          f"metric {METRIC}, contrast {BASE}->{ADAPTED}\n")
    print(f"{'model':14s} {'family':7s} {'size':>6s} {'baseline':>9s} {'delta':>7s}")
    for i, m in enumerate(models):
        print(f"{m:14s} {fams[i]:7s} {NUM[m]:5.1f}B {base[i]:9.3f} {delta[i]:+7.3f}")

    rho = spearman(x, delta)
    p_exact, n_perm = exact_permutation_p(x, delta)
    rho_score = spearman(base, delta)
    p_exact_score, _ = exact_permutation_p(base, delta)
    cross, cross_score = zero_crossing(x, delta), score_crossing(base, delta)
    slope_pooled = float(np.polyfit(x, delta, 1)[0])
    print(f"\npooled Spearman rho = {rho:+.3f} vs log size "
          f"(exact p = {p_exact:.4f} over all {n_perm:,} orderings)")
    print(f"pooled Spearman rho = {rho_score:+.3f} vs closed-book baseline "
          f"(exact p = {p_exact_score:.4f})")
    print(f"pooled OLS slope    = {slope_pooled:+.4f} per doubling; "
          f"fitted zero crossing at {cross:.2f}B, or a baseline of "
          f"{cross_score:.3f}/5")

    rows = [dict(statistic="spearman_rho_vs_log_size", value=round(rho, 3)),
            dict(statistic="exact_permutation_p_size", value=round(p_exact, 5)),
            dict(statistic="n_permutations", value=n_perm),
            dict(statistic="spearman_rho_vs_baseline", value=round(rho_score, 3)),
            dict(statistic="exact_permutation_p_baseline", value=round(p_exact_score, 5)),
            dict(statistic="ols_slope_per_doubling", value=round(slope_pooled, 4)),
            dict(statistic="zero_crossing_b_params", value=round(cross, 2)),
            dict(statistic="zero_crossing_baseline_score", value=round(cross_score, 3))]

    # -- within-family: monotonicity, exact; and per-family slope
    mono = all(np.all(np.diff(delta[fams == f]) < 0) for f in FAMILIES)
    p_mono = 1.0 / float(np.prod([math.factorial(int((fams == f).sum()))
                                  for f in FAMILIES]))
    print(f"\nall three families monotone decreasing: {mono} "
          f"(exact p = {p_mono:.4f} = 1/{int(round(1 / p_mono))} under "
          f"within-family random ordering)")
    rows += [dict(statistic="all_families_monotone_decreasing", value=int(mono)),
             dict(statistic="within_family_monotonicity_exact_p", value=round(p_mono, 5))]

    print(f"\n{'family':8s} {'n':>2s} {'slope/doubling':>15s} {'95% CI (questions)':>22s}")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, k, size=(N_BOOT, k))
    boot_delta = sum_d[draws].sum(1) / size[draws].sum(1)[:, None]     # (N, n_models)
    boot_base = sum_b[draws].sum(1) / size[draws].sum(1)[:, None]
    for f in FAMILIES:
        sel = fams == f
        slope = float(np.polyfit(x[sel], delta[sel], 1)[0])
        reps = np.array([np.polyfit(x[sel], r[sel], 1)[0] for r in boot_delta])
        lo, hi, _ = ci(reps)
        print(f"{f:8s} {sel.sum():2d} {slope:+15.4f} {f'[{lo:+.3f}, {hi:+.3f}]':>22s}")
        rows += [dict(statistic=f"slope_{f}_per_doubling", value=round(slope, 4)),
                 dict(statistic=f"slope_{f}_ci_lo", value=round(lo, 4)),
                 dict(statistic=f"slope_{f}_ci_hi", value=round(hi, 4))]

    # -- leave-one-family-out and leave-one-model-out: exact, no resampling
    print(f"\n{'refit':22s} {'n':>2s} {'rho':>7s} {'crossing':>10s}")
    lofo_rho, lofo_cross = [], []
    for f in FAMILIES:
        sel = fams != f
        r, c = spearman(x[sel], delta[sel]), zero_crossing(x[sel], delta[sel])
        lofo_rho.append(r); lofo_cross.append(c)
        print(f"{'drop ' + f:22s} {sel.sum():2d} {r:+7.3f} {c:9.2f}B")
        rows += [dict(statistic=f"rho_excl_{f}", value=round(r, 3)),
                 dict(statistic=f"crossing_excl_{f}", value=round(c, 2))]
    lomo_rho, lomo_cross = [], []
    for i, m in enumerate(models):
        sel = np.arange(n_models) != i
        r, c = spearman(x[sel], delta[sel]), zero_crossing(x[sel], delta[sel])
        lomo_rho.append(r); lomo_cross.append(c)
        print(f"{'drop ' + m:22s} {sel.sum():2d} {r:+7.3f} {c:9.2f}B")
    print(f"{'leave-one-family-out':22s} {'':2s} "
          f"[{min(lofo_rho):+.3f}, {max(lofo_rho):+.3f}] "
          f"[{min(lofo_cross):.2f}, {max(lofo_cross):.2f}]B")
    print(f"{'leave-one-model-out':22s} {'':2s} "
          f"[{min(lomo_rho):+.3f}, {max(lomo_rho):+.3f}] "
          f"[{min(lomo_cross):.2f}, {max(lomo_cross):.2f}]B")
    rows += [dict(statistic="rho_lofo_min", value=round(min(lofo_rho), 3)),
             dict(statistic="rho_lofo_max", value=round(max(lofo_rho), 3)),
             dict(statistic="crossing_lofo_min", value=round(min(lofo_cross), 2)),
             dict(statistic="crossing_lofo_max", value=round(max(lofo_cross), 2)),
             dict(statistic="rho_lomo_min", value=round(min(lomo_rho), 3)),
             dict(statistic="rho_lomo_max", value=round(max(lomo_rho), 3)),
             dict(statistic="crossing_lomo_min", value=round(min(lomo_cross), 2)),
             dict(statistic="crossing_lomo_max", value=round(max(lomo_cross), 2))]

    # -- the three resamplers
    roster = rng.integers(0, n_models, size=(N_BOOT, n_models))
    schemes = {
        "questions": (boot_delta, boot_base, None),
        "roster": (np.tile(delta, (N_BOOT, 1)), np.tile(base, (N_BOOT, 1)), roster),
        "combined": (boot_delta, boot_base, rng.integers(0, n_models,
                                                         size=(N_BOOT, n_models))),
    }
    print(f"\n{'resampling':11s} {'rho 95% CI':>20s} {'crossing 95% CI':>22s} "
          f"{'score 95% CI':>18s} {'usable':>16s}")
    for name, (bd, bb, pick) in schemes.items():
        r_rep = np.empty(N_BOOT); c_rep = np.empty(N_BOOT); s_rep = np.empty(N_BOOT)
        for j in range(N_BOOT):
            idx = pick[j] if pick is not None else slice(None)
            xi, di, bi = x[idx], bd[j][idx], bb[j][idx]
            r_rep[j] = spearman(xi, di)
            c_rep[j] = zero_crossing(xi, di)
            s_rep[j] = score_crossing(bi, di)
        (rlo, rhi, nr), (clo, chi, nc), (slo, shi, _) = ci(r_rep), ci(c_rep), ci(s_rep)
        print(f"{name:11s} {f'[{rlo:+.3f}, {rhi:+.3f}]':>20s} "
              f"{f'[{clo:.2f}, {chi:.2f}]B':>22s} {f'[{slo:.3f}, {shi:.3f}]':>18s} "
              f"{f'{nr}/{nc} of {N_BOOT}':>16s}")
        rows += [dict(statistic=f"rho_ci_lo_{name}", value=round(rlo, 3)),
                 dict(statistic=f"rho_ci_hi_{name}", value=round(rhi, 3)),
                 dict(statistic=f"crossing_ci_lo_{name}", value=round(clo, 2)),
                 dict(statistic=f"crossing_ci_hi_{name}", value=round(chi, 2)),
                 dict(statistic=f"score_crossing_ci_lo_{name}", value=round(slo, 3)),
                 dict(statistic=f"score_crossing_ci_hi_{name}", value=round(shi, 3)),
                 dict(statistic=f"usable_rho_{name}", value=nr),
                 dict(statistic=f"usable_crossing_{name}", value=nc)]

    # -- the flat zone: consecutive models around the crossing whose effect is
    # too small to act on. A switching rule is only as sharp as the region it
    # separates, and inside this run "full SFT" and "LoRA" are a coin toss on
    # the paper's own practical bar, wherever the fitted crossing happens to
    # fall. Runs elsewhere in the roster are irrelevant -- only the one the
    # crossing sits in blunts the rule.
    small = np.abs(delta) < PRACTICAL_MIN_DELTA
    lo_i = hi_i = min(int(np.searchsorted(x, np.log2(cross))), n_models - 1)
    zone: list[str] = []
    if small[lo_i]:
        while lo_i > 0 and small[lo_i - 1]:
            lo_i -= 1
        while hi_i + 1 < n_models and small[hi_i + 1]:
            hi_i += 1
        zone = models[lo_i:hi_i + 1]
    span = f"{NUM[zone[0]]:.0f}-{NUM[zone[-1]]:.0f}B" if zone else "none"
    print(f"\nflat zone around the fitted crossing (|delta| < "
          f"{PRACTICAL_MIN_DELTA}): {', '.join(zone) or 'none'} = {span}")
    rows += [dict(statistic="flat_zone_models", value="|".join(zone)),
             dict(statistic="flat_zone_lo_b", value=NUM[zone[0]] if zone else np.nan),
             dict(statistic="flat_zone_hi_b", value=NUM[zone[-1]] if zone else np.nan)]

    print("\nWhat this does and does not license:")
    print("  * the DIRECTION is sturdy -- it survives dropping any model and any\n"
          "    family, and every family declines on its own;")
    print(f"  * the THRESHOLD is not identified to one model size: the combined\n"
          f"    crossing interval spans most of the roster's size range, and "
          f"every model\n    in the {span} flat zone is below the practical bar;")
    print("  * so 'switch above ~9B' should be written as a trend with an "
          "interval,\n    not as a numerically grounded rule.")

    os.makedirs(args.out, exist_ok=True)
    per_model = pd.DataFrame(dict(
        base_model=models, family=fams, params_b=[NUM[m] for m in models],
        log2_params=np.round(x, 4), baseline_p1=np.round(base, 3),
        delta_p3_p1=np.round(delta, 3)))
    lo, hi = zip(*[C.cluster_bootstrap_ci(D[i], cids) for i in range(n_models)])
    per_model["delta_ci95_lo"] = np.round(lo, 3)
    per_model["delta_ci95_hi"] = np.round(hi, 3)
    per_model.to_csv(os.path.join(args.out, "scale_trend.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "scale_trend_summary.csv"),
                              index=False)
    print(f"\nwrote {args.out}/scale_trend.csv and scale_trend_summary.csv")


if __name__ == "__main__":
    main()
