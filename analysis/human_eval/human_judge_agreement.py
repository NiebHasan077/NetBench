#!/usr/bin/env python3
"""Human-judge alignment report for the HPN-QA benchmark.

Joins the merged human ratings (merge_filled_ratings.py) with the GPT-5.1
judge's scores and quantifies how well the LLM judge aligns with a human
domain expert on the identical rubric. Unlike the two-LLM report
(NetBench-LLM/evaluation/judge_agreement.py), which supports a robustness
claim, this one speaks to *construct validity* - whether the judge's scores
track expert judgement.

Views:
  • Per-response agreement, pooled over 125 human-scored responses (5 systems
    x 25 questions): Pearson r, Spearman rho, Kendall tau-b, MAE, signed bias
    (judge - human; positive = judge more lenient), quadratic-weighted kappa
    (integer dimensions), and % within +/-0.5 and +/-1. 95% CIs come from a
    cluster bootstrap that resamples *questions* (the 5 responses to one
    question share context and are correlated).
  • ``overall`` is compared two ways: against the judge-reported overall (the
    number every paper table uses) and against the judge's four dimensions
    re-weighted with the instructed formula (0.4/0.3/0.2/0.1) - the same
    formula that defines the human composite.
  • System-level: mean overall per system under human vs judge on the same
    25 questions (judge's full-benchmark mean shown for reference).
  • The embedded paired contrasts (the sample was designed around them):
    human-scored vs judge-scored versions of 2B P1->P4, 2B P1->P6, and
    gpt-4o -> 9B-P6, with the same statistics as analysis/significance.csv
    (paired Wilcoxon, seeded bootstrap CI, Cohen's d_z).

Outputs (alongside the second-judge report, same naming scheme):
    analysis/outputs/HPN_JUDGE_AGREEMENT_human_vs_gpt-5.1.{md,xlsx}
    outputs/human_vs_judge_long.csv        per-response joined scores

``_selftest()`` asserts the metric implementations on every invocation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from scipy.stats import kendalltau, pearsonr, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
ANALYSIS_OUT = HERE.parent / "outputs"

HUMAN_LABEL = "human"
JUDGE_LABEL = "gpt-5.1"

DIMENSIONS = ["correctness", "completeness", "clarity", "conciseness"]
WEIGHTS = {"correctness": 0.4, "completeness": 0.3, "clarity": 0.2, "conciseness": 0.1}

# match analysis/aggregate_scores.py
BOOTSTRAP_N = 10000
SEED = 1234

# (variant_run a, variant_run b, label). Positive diff => b > a, as in
# aggregate_scores.py. These are the contrasts the sample was built around.
CONTRASTS = [
    ("Qwen3.5-2B", "Qwen3.5-2B-cpt-full-sft",
     "qwen3.5-2b P1 -> P4 (full CPT+SFT effect)"),
    ("Qwen3.5-2B", "RAG-Qwen3.5-2B",
     "qwen3.5-2b P1 -> P6 (RAG effect)"),
    ("gpt-4o", "RAG-Qwen3.5-9B",
     "gpt-4o -> qwen3.5-9b P6 (frontier parity)"),
]


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def _corr(fn, a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(fn(a, b)[0])


def _quadratic_weighted_kappa(a, b) -> float:
    # same implementation as NetBench-LLM/evaluation/judge_agreement.py
    a = np.asarray(np.round(a), dtype=int)
    b = np.asarray(np.round(b), dtype=int)
    lo = int(min(a.min(), b.min()))
    hi = int(max(a.max(), b.max()))
    R = hi - lo + 1
    if R < 2:
        return float("nan")
    O = np.zeros((R, R), dtype=float)
    np.add.at(O, (a - lo, b - lo), 1)
    ij = np.arange(R)
    w = (ij[:, None] - ij[None, :]) ** 2 / ((R - 1) ** 2)
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / O.sum()
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else float("nan")


def krippendorff_alpha_interval(units) -> float:
    """Krippendorff's alpha with the interval difference metric. `units` is an
    iterable of per-unit rating tuples (any missing value simply absent from the
    tuple). Coincidence-matrix formulation; validated against Krippendorff's
    canonical interval example (alpha=0.849) in ``_selftest``."""
    from collections import defaultdict
    o: dict = defaultdict(float)
    for vals in units:
        vals = [v for v in vals
                if v is not None and not (isinstance(v, float) and np.isnan(v))]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(vals[i], vals[j])] += 1.0 / (m - 1)
    seen = sorted({v for pair in o for v in pair})
    n_c: dict = defaultdict(float)
    for (c, k), wt in o.items():
        n_c[c] += wt
    n = sum(n_c.values())
    num = sum(wt * (c - k) ** 2 for (c, k), wt in o.items())
    den = sum(n_c[c] * n_c[k] * (c - k) ** 2 for c in seen for k in seen)
    return float("nan") if den == 0 else 1.0 - (n - 1) * num / den


def agreement_metrics(human, judge, integer_scale: bool) -> dict:
    """All pooled agreement metrics for one (human, judge) score vector pair."""
    h = np.asarray(human, dtype=float)
    j = np.asarray(judge, dtype=float)
    d = j - h
    return {
        "n": len(h),
        "pearson": _corr(pearsonr, h, j),
        "spearman": _corr(spearmanr, h, j),
        "kendall": _corr(kendalltau, h, j),
        "mae": float(np.abs(d).mean()),
        "bias": float(d.mean()),
        "qwk": _quadratic_weighted_kappa(h, j) if integer_scale else float("nan"),
        "pct_within_05": float((np.abs(d) <= 0.5).mean() * 100),
        "pct_within_1": float((np.abs(d) <= 1.0).mean() * 100),
    }


def cluster_bootstrap_ci(df: pd.DataFrame, col_h: str, col_j: str,
                         integer_scale: bool, rng) -> dict:
    """95% CI for every agreement metric, resampling questions (clusters)."""
    groups = [g[[col_h, col_j]].to_numpy(dtype=float)
              for _, g in df.groupby("question_id")]
    k = len(groups)
    stats: dict[str, list] = {}
    for _ in range(BOOTSTRAP_N):
        take = rng.integers(0, k, size=k)
        m = np.concatenate([groups[i] for i in take])
        res = agreement_metrics(m[:, 0], m[:, 1], integer_scale)
        for key, v in res.items():
            stats.setdefault(key, []).append(v)
    out = {}
    for key, vals in stats.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        out[key] = ((float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
                    if len(vals) else (float("nan"), float("nan")))
    return out


def bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    # identical to analysis/aggregate_scores.py
    rng = np.random.default_rng(SEED)
    means = diff[rng.integers(0, len(diff), size=(BOOTSTRAP_N, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired-contrast statistics matching aggregate_scores.paired_test."""
    diff = b - a
    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(b, a, zero_method="wilcox").pvalue)
    lo, hi = bootstrap_ci(diff)
    sd = diff.std(ddof=1)
    return dict(n=len(diff), mean_a=float(a.mean()), mean_b=float(b.mean()),
                mean_diff=float(diff.mean()), median_diff=float(np.median(diff)),
                ci95_lo=lo, ci95_hi=hi, p_raw=p,
                cohens_dz=float(diff.mean() / sd) if sd > 0 else 0.0)


def _selftest() -> None:
    assert abs(_corr(pearsonr, [1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(_corr(spearmanr, [1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert abs(_corr(kendalltau, [1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    # Krippendorff (2011) canonical interval example -> alpha = 0.849
    _k = [[1, 1, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2],
          [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [5, 5, 5], [1, 1]]
    assert abs(krippendorff_alpha_interval(_k) - 0.849) < 5e-4
    assert abs(_quadratic_weighted_kappa([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(_quadratic_weighted_kappa([1, 1, 2, 2], [1, 2, 1, 2])) < 1e-9
    m = agreement_metrics([1, 2, 3], [1, 2, 4], integer_scale=True)
    assert abs(m["mae"] - 1 / 3) < 1e-9 and abs(m["bias"] - 1 / 3) < 1e-9
    assert abs(m["pct_within_1"] - 100.0) < 1e-9
    assert abs(m["pct_within_05"] - 200 / 3) < 1e-9
    w = sum(WEIGHTS.values())
    assert abs(w - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════

def load_joined(human_csv: Path, judge_csv: Path) -> pd.DataFrame:
    human = pd.read_csv(human_csv)
    judge_long = pd.read_csv(judge_csv)
    judge_long = judge_long[judge_long.variant_run.isin(human.variant_run.unique())]
    judge = judge_long.pivot_table(index=["variant_run", "question_id"],
                                   columns="dimension", values="score").reset_index()
    judge["overall_weighted"] = sum(WEIGHTS[d] * judge[d] for d in DIMENSIONS)
    joined = human.merge(judge, on=["variant_run", "question_id"],
                         suffixes=("_human", "_judge"), validate="one_to_one")
    if len(joined) != len(human):
        missing = set(zip(human.variant_run, human.question_id)) - \
            set(zip(joined.variant_run, joined.question_id))
        raise SystemExit(f"❌ {len(human) - len(joined)} responses lack judge scores, "
                         f"e.g. {sorted(missing)[:3]}")
    return joined.rename(columns={"overall_human": "overall_human",
                                  "overall_judge": "overall_reported"})


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute(joined: pd.DataFrame, judge_full: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)

    # --- per-response pooled agreement (with cluster-bootstrap CIs) ---
    rows = []
    specs = [(d, f"{d}_human", f"{d}_judge", True) for d in DIMENSIONS]
    specs += [("overall (judge-reported)", "overall_human", "overall_reported", False),
              ("overall (judge reweighted)", "overall_human", "overall_weighted", False)]
    for label, ch, cj, integer_scale in specs:
        m = agreement_metrics(joined[ch], joined[cj], integer_scale)
        ci = cluster_bootstrap_ci(joined, ch, cj, integer_scale, rng)
        rows.append({"dimension": label, **m,
                     **{f"{k}_ci": ci[k] for k in ("pearson", "spearman", "mae",
                                                   "bias", "qwk", "pct_within_1")}})

    # --- system-level leaderboard on the 25 sampled questions ---
    lb = joined.groupby("variant_run").agg(
        human_mean=("overall_human", "mean"),
        judge_mean=("overall_reported", "mean")).reset_index()
    full = judge_full[judge_full.dimension == "overall"] \
        .groupby("variant_run").score.mean()
    lb["judge_mean_full"] = lb.variant_run.map(full)
    lb["rank_human"] = lb.human_mean.rank(ascending=False, method="first").astype(int)
    lb["rank_judge"] = lb.judge_mean.rank(ascending=False, method="first").astype(int)
    lb = lb.sort_values("rank_human")
    system = {
        "kendall_tau": _corr(kendalltau, lb.human_mean, lb.judge_mean),
        "spearman": _corr(spearmanr, lb.human_mean, lb.judge_mean),
        "pearson": _corr(pearsonr, lb.human_mean, lb.judge_mean),
    }

    # Formula-metric variants (docs/JUDGE_OVERALL_AUDIT.md): the human
    # composite is already the instructed formula, so only the judge side
    # changes, from its reported overall to the deterministic reweighting.
    lb["judge_mean_formula"] = lb.variant_run.map(
        joined.groupby("variant_run").overall_weighted.mean())
    lb["rank_judge_formula"] = lb.judge_mean_formula.rank(
        ascending=False, method="first").astype(int)
    full_f = judge_full[judge_full.dimension == "overall_formula"] \
        .groupby("variant_run").score.mean()
    lb["judge_mean_full_formula"] = lb.variant_run.map(full_f)
    system_formula = {
        "kendall_tau": _corr(kendalltau, lb.human_mean, lb.judge_mean_formula),
        "spearman": _corr(spearmanr, lb.human_mean, lb.judge_mean_formula),
        "pearson": _corr(pearsonr, lb.human_mean, lb.judge_mean_formula),
    }

    # --- embedded paired contrasts, human-scored vs judge-scored ---
    contrast_rows = []
    full_ov = judge_full[judge_full.dimension == "overall"].pivot_table(
        index="question_id", columns="variant_run", values="score")
    piv_h = joined.pivot_table(index="question_id", columns="variant_run",
                               values="overall_human")
    piv_j = joined.pivot_table(index="question_id", columns="variant_run",
                               values="overall_reported")
    piv_jf = joined.pivot_table(index="question_id", columns="variant_run",
                                values="overall_weighted")
    full_ov_f = judge_full[judge_full.dimension == "overall_formula"].pivot_table(
        index="question_id", columns="variant_run", values="score")
    for run_a, run_b, label in CONTRASTS:
        ref = float((full_ov[run_b] - full_ov[run_a]).mean())
        ref_f = float((full_ov_f[run_b] - full_ov_f[run_a]).mean())
        for scorer, piv in ((HUMAN_LABEL, piv_h), (JUDGE_LABEL, piv_j),
                            (f"{JUDGE_LABEL} (formula)", piv_jf)):
            st = paired_stats(piv[run_a].to_numpy(), piv[run_b].to_numpy())
            contrast_rows.append({"contrast": label, "scorer": scorer, **st,
                                  "judge_full_mean_diff": ref,
                                  "judge_full_mean_diff_formula": ref_f})

    return {"agreement": rows, "leaderboard": lb, "system": system,
            "system_formula": system_formula,
            "contrasts": contrast_rows,
            "n_responses": len(joined),
            "n_questions": joined.question_id.nunique(),
            "n_systems": joined.variant_run.nunique(),
            "reported_vs_weighted_mad": float(
                (joined.overall_reported - joined.overall_weighted).abs().mean())}


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def _fmt(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _fmt_ci(ci) -> str:
    lo, hi = ci
    return "n/a" if np.isnan(lo) else f"[{lo:.2f}, {hi:.2f}]"


def write_markdown(res: dict, path: Path, n_raters: int = 1) -> None:
    s = res["system"]
    two = n_raters >= 2
    who = ("Two domain experts (authors) independently blind-scored"
           if two else "One domain expert blind-scored")
    reference = (" The `human` reference below is their per-response consensus "
                 "(mean of the two raters); inter-rater reliability is reported "
                 "separately in `HPN_INTERRATER_rater1_vs_rater2.md`." if two
                 else "")
    calibration = ("Two-rater consensus: values reflect the two experts' mean "
                   "calibration." if two else
                   "Single rater: values reflect one expert's calibration.")
    L = []
    L.append(f"# HPN-QA Human-Judge Alignment: `{HUMAN_LABEL}` vs `{JUDGE_LABEL}`\n")
    L.append(
        f"{who} **{res['n_responses']} responses** "
        f"({res['n_systems']} systems x {res['n_questions']} stratified questions; "
        "system identity hidden, order re-randomized per question) on the judge's "
        f"exact rubric.{reference} Composite `overall` for the human is the judge's "
        "instructed weighting (0.4 correctness + 0.3 completeness + 0.2 clarity "
        "+ 0.1 conciseness).\n")
    L.append("> Unlike the two-LLM agreement report, this measures *construct "
             f"validity* - whether the LLM judge tracks expert judgement. {calibration} "
             "95% CIs are cluster "
             "bootstrap over questions (responses to one question are correlated); "
             f"contrast CIs use the seeded bootstrap from `aggregate_scores.py` "
             f"(B={BOOTSTRAP_N}).\n")

    L.append("## Per-response agreement (pooled)\n")
    L.append("| Dimension | n | Pearson [95% CI] | Spearman | Kendall tau-b | "
             "MAE [95% CI] | Bias (judge-human) [95% CI] | QWK [95% CI] | "
             "% within +/-0.5 | % within +/-1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["agreement"]:
        qwk = ("n/a" if np.isnan(r["qwk"])
               else f"{_fmt(r['qwk'])} {_fmt_ci(r['qwk_ci'])}")
        L.append(
            f"| {r['dimension']} | {r['n']} "
            f"| {_fmt(r['pearson'])} {_fmt_ci(r['pearson_ci'])} "
            f"| {_fmt(r['spearman'])} | {_fmt(r['kendall'])} "
            f"| {_fmt(r['mae'])} {_fmt_ci(r['mae_ci'])} "
            f"| {_fmt(r['bias'], 3)} {_fmt_ci(r['bias_ci'])} "
            f"| {qwk} "
            f"| {_fmt(r['pct_within_05'], 1)} | {_fmt(r['pct_within_1'], 1)} |")
    L.append("\n_For the four integer dimensions, % within +/-0.5 equals exact "
             "agreement. `overall (judge-reported)` compares the human composite "
             "with the judge's own reported overall (the number used in every "
             "paper table); `overall (judge reweighted)` applies the instructed "
             "formula to the judge's four dimension scores, removing the judge's "
             "formula deviation (mean |reported - reweighted| = "
             f"{res['reported_vs_weighted_mad']:.3f} on this sample)._\n")

    L.append("## System-level (leaderboard on the 25 sampled questions)\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Kendall tau-b (system ranking) | **{_fmt(s['kendall_tau'])}** |")
    L.append(f"| Spearman rho (system ranking)  | **{_fmt(s['spearman'])}** |")
    L.append(f"| Pearson r (mean overall)       | {_fmt(s['pearson'])} |\n")
    L.append(f"| System | {HUMAN_LABEL} mean | rank | {JUDGE_LABEL} mean (25q) | "
             "rank | judge mean (full benchmark) |")
    L.append("|---|---|---|---|---|---|")
    for _, d in res["leaderboard"].iterrows():
        L.append(f"| {d.variant_run} | {d.human_mean:.3f} | {d.rank_human} "
                 f"| {d.judge_mean:.3f} | {d.rank_judge} | {d.judge_mean_full:.3f} |")

    L.append("\n## Embedded paired contrasts (human-scored vs judge-scored)\n")
    L.append("_Same statistics as `analysis/outputs/significance.csv` (paired "
             "Wilcoxon, seeded bootstrap 95% CI, Cohen's d_z), computed on the 25 "
             "sampled questions under each scorer. `judge full` is the judge's "
             "mean difference on the complete benchmark, for reference._\n")
    L.append("| Contrast | Scorer | n | mean A | mean B | mean diff | 95% CI | "
             "Wilcoxon p | d_z | judge full |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in res["contrasts"]:
        L.append(
            f"| {c['contrast']} | {c['scorer']} | {c['n']} "
            f"| {c['mean_a']:.3f} | {c['mean_b']:.3f} | **{c['mean_diff']:+.3f}** "
            f"| [{c['ci95_lo']:.3f}, {c['ci95_hi']:.3f}] | {c['p_raw']:.4f} "
            f"| {c['cohens_dz']:.3f} | {c['judge_full_mean_diff']:+.3f} |")
    sf = res["system_formula"]
    L.append("\n## Formula-metric variants\n")
    L.append("_The deterministic rubric composite (0.4/0.3/0.2/0.1 over the "
             "judge's dimension scores; see `docs/JUDGE_OVERALL_AUDIT.md`) is "
             "the EACL paper's primary metric. Per-response agreement for it is "
             "the `overall (judge reweighted)` row above; `judge (formula)` "
             "contrast rows and the system-level ranking below use it too._\n")
    L.append(f"| Kendall tau_b (ranking, formula) | {_fmt(sf['kendall_tau'])} |")
    L.append(f"| Spearman rho (ranking, formula)  | {_fmt(sf['spearman'])} |")
    L.append(f"| Pearson r (mean overall, formula) | {_fmt(sf['pearson'])} |")
    L.append("")
    path.write_text("\n".join(L))


def write_excel(res: dict, path: Path) -> None:
    wb = Workbook()
    bold = Font(bold=True)
    fill = PatternFill("solid", fgColor="DDEBF7")

    def header(ws, headers):
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = bold
            cell.fill = fill

    ws = wb.active
    ws.title = "Summary"
    header(ws, ["Metric", "Value"])
    s = res["system"]
    ov = next(r for r in res["agreement"] if r["dimension"] == "overall (judge-reported)")
    summary = [
        ("rater_a", HUMAN_LABEL), ("rater_b", JUDGE_LABEL),
        ("systems", res["n_systems"]), ("questions", res["n_questions"]),
        ("responses", res["n_responses"]),
        ("overall_pearson_r", round(ov["pearson"], 4)),
        ("overall_spearman_rho", round(ov["spearman"], 4)),
        ("overall_mae", round(ov["mae"], 4)),
        ("overall_bias_judge_minus_human", round(ov["bias"], 4)),
        ("kendall_tau_b_ranking", round(s["kendall_tau"], 4)),
        ("spearman_rho_ranking", round(s["spearman"], 4)),
    ]
    sf = res["system_formula"]
    ovf = next(r for r in res["agreement"]
               if r["dimension"] == "overall (judge reweighted)")
    summary += [
        ("overall_formula_pearson_r", round(ovf["pearson"], 4)),
        ("overall_formula_spearman_rho", round(ovf["spearman"], 4)),
        ("overall_formula_mae", round(ovf["mae"], 4)),
        ("overall_formula_bias_judge_minus_human", round(ovf["bias"], 4)),
        ("kendall_tau_b_ranking_formula", round(sf["kendall_tau"], 4)),
        ("spearman_rho_ranking_formula", round(sf["spearman"], 4)),
    ]
    for i, (k, v) in enumerate(summary, 2):
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=v)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 24

    ws2 = wb.create_sheet("PerResponseAgreement")
    cols = ["dimension", "n", "pearson", "pearson_ci_lo", "pearson_ci_hi",
            "spearman", "kendall", "mae", "mae_ci_lo", "mae_ci_hi",
            "bias", "bias_ci_lo", "bias_ci_hi", "qwk", "qwk_ci_lo", "qwk_ci_hi",
            "pct_within_05", "pct_within_1"]
    header(ws2, cols)
    for i, r in enumerate(res["agreement"], 2):
        flat = {**r}
        for k in ("pearson", "mae", "bias", "qwk"):
            flat[f"{k}_ci_lo"], flat[f"{k}_ci_hi"] = r[f"{k}_ci"]
        for c, name in enumerate(cols, 1):
            v = flat[name]
            if isinstance(v, float):
                v = None if np.isnan(v) else round(v, 4)
            ws2.cell(row=i, column=c, value=v)
    ws2.column_dimensions["A"].width = 26

    ws3 = wb.create_sheet("Leaderboard")
    header(ws3, ["system", "human_mean", "human_rank", "judge_mean_25q",
                 "judge_rank", "judge_mean_full", "judge_mean_25q_formula",
                 "judge_rank_formula", "judge_mean_full_formula"])
    for i, (_, d) in enumerate(res["leaderboard"].iterrows(), 2):
        for c, v in enumerate([d.variant_run, round(d.human_mean, 4), d.rank_human,
                               round(d.judge_mean, 4), d.rank_judge,
                               round(d.judge_mean_full, 4),
                               round(d.judge_mean_formula, 4), d.rank_judge_formula,
                               round(d.judge_mean_full_formula, 4)], 1):
            ws3.cell(row=i, column=c, value=v)
    ws3.column_dimensions["A"].width = 30

    ws4 = wb.create_sheet("Contrasts")
    ccols = ["contrast", "scorer", "n", "mean_a", "mean_b", "mean_diff",
             "median_diff", "ci95_lo", "ci95_hi", "p_raw", "cohens_dz",
             "judge_full_mean_diff", "judge_full_mean_diff_formula"]
    header(ws4, ccols)
    for i, c in enumerate(res["contrasts"], 2):
        for j, name in enumerate(ccols, 1):
            v = c[name]
            ws4.cell(row=i, column=j, value=round(v, 4) if isinstance(v, float) else v)
    ws4.column_dimensions["A"].width = 44

    wb.save(path)


# ═══════════════════════════════════════════════════════════════════════
# Two-rater consensus + inter-rater reliability
# ═══════════════════════════════════════════════════════════════════════

def build_consensus(r1: pd.DataFrame, r2: pd.DataFrame) -> pd.DataFrame:
    """Per-response consensus = mean of the two raters' dimension scores; the
    composite is recomputed with the instructed weights (identical to the mean
    of the two raters' composites, the weights being linear)."""
    a = r1.set_index("response_id")
    b = r2.set_index("response_id")
    if set(a.index) != set(b.index):
        raise SystemExit("❌ the two raters cover different response sets")
    b = b.loc[a.index]
    cons = a.copy()
    for d in DIMENSIONS:
        cons[d] = (a[d].to_numpy(float) + b[d].to_numpy(float)) / 2.0
    cons["overall"] = sum(WEIGHTS[d] * cons[d] for d in DIMENSIONS).round(4)
    cons["rater"] = "consensus"
    return cons.reset_index()


def compute_interrater(r1: pd.DataFrame, r2: pd.DataFrame,
                       joined: pd.DataFrame) -> list[dict]:
    """Rater-1 vs rater-2 agreement per dimension and on the composite, with
    Krippendorff's alpha (interval) for the two human raters and a 3-way alpha
    adding the judge (reweighted composite / dimension scores) as a third coder."""
    a = r1.set_index("response_id")
    b = r2.set_index("response_id").loc[a.index]
    j = joined.set_index("response_id").loc[a.index]
    rows = []
    for d in [*DIMENSIONS, "overall"]:
        x, y = a[d].to_numpy(float), b[d].to_numpy(float)
        jc = (j[f"{d}_judge"] if d in DIMENSIONS
              else j["overall_weighted"]).to_numpy(float)
        rows.append({
            "dimension": d, "n": len(x),
            "pearson": _corr(pearsonr, x, y),
            "spearman": _corr(spearmanr, x, y),
            "mae": float(np.abs(x - y).mean()),
            "bias_r1_minus_r2": float((x - y).mean()),
            "qwk": _quadratic_weighted_kappa(x, y) if d in DIMENSIONS else float("nan"),
            "alpha_2rater": krippendorff_alpha_interval(list(zip(x, y))),
            "alpha_3way_with_judge": krippendorff_alpha_interval(list(zip(x, y, jc))),
        })
    return rows


def write_interrater(rows: list[dict], out_dir: Path) -> None:
    """Emit the inter-rater report: CSV + xlsx (InterRater sheet) + md."""
    cols = ["dimension", "n", "pearson", "spearman", "mae", "bias_r1_minus_r2",
            "qwk", "alpha_2rater", "alpha_3way_with_judge"]
    # Round before writing. At full float64 repr these are 17 significant
    # digits of a correlation over 125 ratings -- false precision, and the last
    # digit is not stable across CPU architectures: a Pearson r came back one
    # ULP apart on a CI runner (…4425 vs …4424) and failed byte-comparison.
    # Six decimals is nine orders of magnitude clear of that noise and still
    # far finer than the two decimals the paper reports.
    pd.DataFrame(rows)[cols].round(6).to_csv(
        out_dir / "HPN_INTERRATER_rater1_vs_rater2.csv", index=False)
    wb = Workbook()
    ws = wb.active
    ws.title = "InterRater"
    for c, h in enumerate(cols, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")
    for i, r in enumerate(rows, 2):
        for c, name in enumerate(cols, 1):
            v = r[name]
            if isinstance(v, float):
                v = None if np.isnan(v) else round(v, 4)
            ws.cell(row=i, column=c, value=v)
    ws.column_dimensions["A"].width = 26
    wb.save(out_dir / "HPN_INTERRATER_rater1_vs_rater2.xlsx")
    ov = next(r for r in rows if r["dimension"] == "overall")
    L = ["# HPN-QA inter-rater reliability: rater 1 vs rater 2\n",
         f"Two domain experts (authors) independently blind-scored the same "
         f"{ov['n']} responses under the judge's rubric. Krippendorff's alpha uses "
         "the interval metric; `alpha (3-way)` adds the judge as a third coder.\n",
         "| Dimension | Pearson | QWK | Krippendorff alpha | alpha (3-way, +judge) "
         "| MAE | Bias (R1-R2) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        qwk = "--" if np.isnan(r["qwk"]) else f"{r['qwk']:.3f}"
        L.append(f"| {r['dimension']} | {r['pearson']:.3f} | {qwk} "
                 f"| {r['alpha_2rater']:.3f} | {r['alpha_3way_with_judge']:.3f} "
                 f"| {r['mae']:.3f} | {r['bias_r1_minus_r2']:+.3f} |")
    (out_dir / "HPN_INTERRATER_rater1_vs_rater2.md").write_text("\n".join(L) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    _selftest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", default=str(HERE / "outputs/human_scores_long.csv"),
                        help="Rater 1's merged ratings (merge_filled_ratings.py output).")
    parser.add_argument("--human2", default=str(HERE / "outputs/human_scores_rater2_long.csv"),
                        help="Rater 2's merged ratings; if present, the human reference "
                             "becomes the two raters' per-response consensus and an "
                             "inter-rater reliability report is written alongside.")
    parser.add_argument("--judge", default=str(ANALYSIS_OUT / "scores_long.csv"),
                        help="Judge scores in long format (aggregate_scores.py output).")
    parser.add_argument("--out_dir", default=str(ANALYSIS_OUT),
                        help="Directory for the report (md + xlsx).")
    args = parser.parse_args()

    # Two raters -> the "human" reference is their per-response consensus (mean);
    # rater 1 and rater 2 individually feed only the inter-rater reliability report.
    r1 = pd.read_csv(args.human)
    r2 = pd.read_csv(args.human2) if args.human2 and Path(args.human2).exists() else None
    if r2 is not None:
        human_csv = HERE / "outputs/human_scores_consensus_long.csv"
        build_consensus(r1, r2).to_csv(human_csv, index=False)
    else:
        human_csv = Path(args.human)

    joined = load_joined(human_csv, Path(args.judge))
    judge_full = pd.read_csv(args.judge)
    judge_full = judge_full[judge_full.variant_run.isin(joined.variant_run.unique())]
    res = compute(joined, judge_full)

    keep = ["response_id", "question_no", "question_id", "base_model", "p_id",
            "setting", "variant_run", "category", "difficulty",
            *[f"{d}_human" for d in DIMENSIONS], "overall_human",
            *[f"{d}_judge" for d in DIMENSIONS], "overall_reported",
            "overall_weighted"]
    joined_path = HERE / "outputs/human_vs_judge_long.csv"
    joined[keep].to_csv(joined_path, index=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"HPN_JUDGE_AGREEMENT_{HUMAN_LABEL}_vs_{JUDGE_LABEL}"
    write_markdown(res, out_dir / f"{stem}.md", n_raters=2 if r2 is not None else 1)
    write_excel(res, out_dir / f"{stem}.xlsx")

    interrater = None
    if r2 is not None:
        interrater = compute_interrater(r1, r2, joined)
        write_interrater(interrater, out_dir)

    ref = "consensus" if r2 is not None else "rater 1"
    ov = next(r for r in res["agreement"] if r["dimension"] == "overall (judge-reported)")
    ovf = next(r for r in res["agreement"] if r["dimension"] == "overall (judge reweighted)")
    print(f"✅ {res['n_responses']} responses ({res['n_systems']} systems x "
          f"{res['n_questions']} questions); human reference = {ref}")
    print(f"   judge-vs-{ref} Pearson r (reported/formula): "
          f"{_fmt(ov['pearson'])} / {_fmt(ovf['pearson'])}")
    print(f"   bias judge - {ref} (reported/formula)      : "
          f"{_fmt(ov['bias'])} / {_fmt(ovf['bias'])}")
    print(f"   ranking Kendall tau-b : {_fmt(res['system']['kendall_tau'])}")
    if interrater is not None:
        io = next(r for r in interrater if r["dimension"] == "overall")
        print(f"   inter-rater (R1 vs R2) overall: Pearson {io['pearson']:.3f}, "
              f"alpha {io['alpha_2rater']:.3f}, 3-way alpha {io['alpha_3way_with_judge']:.3f}, "
              f"bias(R1-R2) {io['bias_r1_minus_r2']:+.3f}")
    print(f"   -> {out_dir / (stem + '.md')}")
    print(f"   -> {out_dir / (stem + '.xlsx')}")
    print(f"   -> {joined_path}")


if __name__ == "__main__":
    main()
