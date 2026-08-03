#!/usr/bin/env python3
"""Generate the paper's figures (PDF, vector) from the analysis outputs + profiling.

Writes paper/figures/{fig1_heatmap,fig2_quality_cost,fig3_forest}.pdf.
Re-run after analysis/aggregate_scores.py (e.g. once qwen is regenerated):
    analysis/.venv/bin/python analysis/make_figures.py
Provisional (think-corrupted) cells are marked; provisional contrasts are already
absent from significance.csv.

--paper {ieee,eacl} selects the paper target: ieee (default) writes paper/figures
exactly as before; eacl writes paper_eacl/figures at ACL column geometry and
additionally emits the EACL-only figures (benchmark validation, scale effect).

--outdir DIR redirects the output only; --paper still selects geometry and metric.
This lets the released artifact regenerate every figure with no paper/ tree
present, e.g.
    analysis/.venv/bin/python analysis/make_figures.py --paper eacl \\
        --outdir analysis/outputs/figures
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "paper/figures"
PAPER = "ieee"   # target paper; "eacl" switches ACL-specific figure variants
A = "analysis/outputs"
BENCHMARK = "NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl"
EXCLUDED = "analysis/excluded_items.csv"
# figure sizes (inches) per paper target; ieee values are the historical defaults
GEOM = {"heatmap": (7.0, 2.9), "quality_cost": (3.4, 2.8), "forest": (7.0, 4.2),
        "radar": (3.7, 3.7), "strata": (7.0, 2.7)}
GEOM_EACL = {"heatmap": (6.3, 2.45), "quality_cost": (3.03, 2.7), "forest": (6.3, 4.2),
             "radar": (3.3, 3.3), "strata": (6.3, 2.6),
             "benchval": (6.3, 2.2), "scale": (5.2, 3.0)}
PS = [f"P{i}" for i in range(1, 9)]
ORDER = ["llama-3.2-1b", "gemma-3-1b", "qwen3.5-2b", "llama-3.1-8b", "qwen3.5-9b",
         "gemma-3-12b", "gemma-3-27b", "qwen3.5-27b"]
NUM = {"llama-3.2-1b": 1.2, "gemma-3-1b": 1.0, "qwen3.5-2b": 2, "llama-3.1-8b": 8,
       "qwen3.5-9b": 9, "gemma-3-12b": 12, "gemma-3-27b": 27, "qwen3.5-27b": 27}
FAM = {"llama": "#1f77b4", "qwen": "#d62728", "gemma": "#2ca02c"}
def fam(m): return "llama" if "llama" in m else ("qwen" if "qwen" in m else "gemma")
CB, OB, AP = "#1f77b4", "#d62728", "#7f7f7f"   # closed-book / open-book / API

# Heatmap colour scale. viridis is perceptually uniform, colour-blind safe, and
# monotonic in luminance, so the figure survives grayscale printing -- the ACL
# accessibility guidance. (It replaces RdYlGn, whose red/green extremes are
# indistinguishable under the most common colour-vision deficiency and collapse
# to near-identical gray when printed.)
HEAT_CMAP, HEAT_VMIN, HEAT_VMAX = "viridis", 2.0, 4.5


def _cell_ink(value, provisional=False):
    """Annotation colour for a heatmap cell: light ink on dark cells, dark ink on
    light ones, so the printed value stays legible across the whole scale."""
    if np.isnan(value):
        return "gray"
    t = min(max((value - HEAT_VMIN) / (HEAT_VMAX - HEAT_VMIN), 0.0), 1.0)
    r, g, b, _ = plt.get_cmap(HEAT_CMAP)(t)
    dark = (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5
    if provisional:
        return "#a6d8ff" if dark else "navy"
    return "white" if dark else "black"
SHORT_CAT = {
    "Transfer Parameters: Definitions and Roles": "Transfer\nParams",
    "Bottleneck Diagnosis and End-to-End Reasoning": "Bottleneck\nDiag.",
    "BDP-Based Reasoning and Window Sizing": "BDP /\nWindow",
    "Adaptive and Online Optimization": "Adaptive\nOpt.",
    "Practical HPN Scenarios and Design": "Practical\nDesign",
    "Fairness, Stability, and Shared Networks": "Fairness",
    "Pipelining and Small-File Optimization": "Pipelining",
    "Parallelism and Large-File Optimization": "Parallelism",
    "Concurrency Tuning and Scaling": "Concurrency",
    "Dataset Partitioning and Mixed Workloads": "Partition",
}


def gpu_tps(with_setting=False):
    out = {}
    for f in glob.glob("NetBench-LLM/outputs/by_model/*/profiling_results/inference_profile_summary_*.csv"):
        model = f.split("/by_model/")[1].split("/")[0]
        df = pd.read_csv(f)
        for ptype in ("direct", "rag"):
            d = df[(df.profile_type == ptype) & (df.device == "cuda")]
            if not d.empty:
                tps = float(d.iloc[0]["avg_tokens_per_second"])
                out[model] = (tps, ptype) if with_setting else tps
                break
    return out


def heatmap(summ):
    ov = summ.pivot_table(index="base_model", columns="p_id", values="overall")
    pv = summ.pivot_table(index="base_model", columns="p_id", values="provisional", aggfunc="max")
    rows = [m for m in ORDER if m in ov.index]
    M = ov.reindex(rows)[PS].to_numpy(dtype=float)
    P = pv.reindex(rows)[PS].fillna(False).to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=GEOM["heatmap"])
    im = ax.imshow(M, cmap=HEAT_CMAP, vmin=HEAT_VMIN, vmax=HEAT_VMAX, aspect="auto")
    ax.set_xticks(range(len(PS))); ax.set_xticklabels(PS)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.axvline(4.5, color="k", lw=1.2)  # closed-book (P1-P5) | open-book (P6-P8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="gray"); continue
            lab = f"{M[i,j]:.2f}" + ("$^\\dagger$" if P[i, j] else "")
            ax.text(j, i, lab, ha="center", va="center", fontsize=6.5,
                    color=_cell_ink(M[i, j], P[i, j]))
    if PAPER != "eacl":   # eacl: the LaTeX caption already carries this text
        ax.set_title("Overall HPN score: closed-book (P1--P5) | open-book/RAG (P6--P8)   "
                     "($\\dagger$ provisional)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="overall (1--5)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_heatmap.pdf"); plt.close(fig)
    print("wrote fig1_heatmap.pdf")


def heatmap_column(summ):
    """Single-column (\\columnwidth) heatmap variant for eacl: abbreviated
    model labels, no colorbar (cells are value-annotated; the caption states
    the color encoding), tighter fonts. Emitted alongside the full-width
    fig1_heatmap.pdf so the paper can switch between them."""
    ov = summ.pivot_table(index="base_model", columns="p_id", values="overall")
    pv = summ.pivot_table(index="base_model", columns="p_id", values="provisional", aggfunc="max")
    rows = [m for m in ORDER if m in ov.index]
    M = ov.reindex(rows)[PS].to_numpy(dtype=float)
    P = pv.reindex(rows)[PS].fillna(False).to_numpy(dtype=bool)
    abbr = {"llama-3.2-1b": "L-1B", "gemma-3-1b": "G-1B", "qwen3.5-2b": "Q-2B",
            "llama-3.1-8b": "L-8B", "qwen3.5-9b": "Q-9B", "gemma-3-12b": "G-12B",
            "gemma-3-27b": "G-27B", "qwen3.5-27b": "Q-27B"}
    fig, ax = plt.subplots(figsize=(3.03, 2.3))
    ax.imshow(M, cmap=HEAT_CMAP, vmin=HEAT_VMIN, vmax=HEAT_VMAX, aspect="auto")
    ax.set_xticks(range(len(PS))); ax.set_xticklabels(PS, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([abbr.get(m, m) for m in rows], fontsize=7)
    ax.tick_params(length=0)
    ax.axvline(4.5, color="k", lw=1.0)  # closed-book | open-book
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "--", ha="center", va="center", fontsize=5.5, color="gray"); continue
            lab = f"{M[i,j]:.2f}" + ("$^\\dagger$" if P[i, j] else "")
            ax.text(j, i, lab, ha="center", va="center", fontsize=5.2,
                    color=_cell_ink(M[i, j], P[i, j]))
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig1_heatmap_col.pdf"); plt.close(fig)
    print("wrote fig1_heatmap_col.pdf")


def quality_cost(summ):
    clean = summ[~summ.provisional]
    tps = gpu_tps()
    if PAPER == "eacl":
        # Row-aligned form: one row per model (no overlapping labels), quality
        # dots + frontier band on the left, log-scale throughput bars on the
        # right. Replaces the scatter, whose labels collide at ACL column size.
        tpsf = gpu_tps(with_setting=True)
        rows = [m for m in ORDER if m in set(clean.base_model) and m in tpsf]
        best, bvar = {}, {}
        for m in rows:
            sub = clean[clean.base_model == m]
            i = sub.overall.idxmax()
            best[m], bvar[m] = sub.loc[i, "overall"], sub.loc[i, "p_id"]
        api = summ[summ.size_group == "api-baseline"]["overall"]
        ypos = np.arange(len(rows))[::-1]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=GEOM["quality_cost"], sharey=True,
                                     gridspec_kw={"width_ratios": [1.25, 1]})
        a1.axvspan(api.min(), api.max(), color="gray", alpha=0.18, zorder=1)
        a1.axvline(api.max(), color="gray", lw=0.6, ls="--", zorder=1)
        a1.scatter([best[m] for m in rows], ypos, color=CB, s=18, zorder=3)
        for yy, m in zip(ypos, rows):
            a1.text(best[m] - 0.14, yy, bvar[m], fontsize=5.5, ha="right",
                    va="center", color="#4d4d4d")
        a1.set_yticks(list(ypos)); a1.set_yticklabels(rows, fontsize=6.5)
        a1.set_xlim(2.55, 4.6)
        a1.set_xlabel("best overall (1--5)", fontsize=7.5)
        a1.set_title("(a) Quality", fontsize=7.5)
        a1.tick_params(labelsize=6.5)
        a2.barh(ypos, [tpsf[m][0] for m in rows], height=0.55, color="#8fb3d1")
        a2.set_xscale("log")
        for yy, m in zip(ypos, rows):
            v, ptype = tpsf[m]
            a2.text(v * 1.18, yy, f"{v:.0f}" + ("*" if ptype == "rag" else ""),
                    fontsize=5.5, va="center", color="#4d4d4d")
        a2.set_xlim(right=max(v for v, _ in tpsf.values()) * 3.2)
        a2.set_xlabel("GPU tok/s (log)", fontsize=7.5)
        a2.set_title("(b) Throughput", fontsize=7.5)
        a2.tick_params(labelsize=6.5)
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig2_quality_cost.pdf"); plt.close(fig)
        print("wrote fig2_quality_cost.pdf (eacl row-aligned variant)")
        return
    fig, ax = plt.subplots(figsize=GEOM["quality_cost"])
    for m in ORDER:
        sub = clean[clean.base_model == m]
        if sub.empty or m not in tps:
            continue
        y = sub.overall.max()
        ax.scatter(tps[m], y, s=20 + 6 * NUM[m], color=FAM[fam(m)], alpha=0.8, edgecolor="k", lw=0.4)
        ax.annotate(m.replace("-", "‑"), (tps[m], y), fontsize=5.5,
                    xytext=(3, 2), textcoords="offset points")
    api = summ[summ.size_group == "api-baseline"]["overall"]
    if not api.empty:
        ax.axhspan(api.min(), api.max(), color="gray", alpha=0.18)
        ax.axhline(api.max(), color="gray", lw=0.6, ls="--")
        ax.text(ax.get_xlim()[1], api.max(), " frontier API\n (closed-book)", fontsize=5.5, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("GPU throughput (tok/s, log)", fontsize=8)
    ax.set_ylabel("best overall score", fontsize=8)
    ax.set_title("Quality vs. inference cost (best clean variant)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_quality_cost.pdf"); plt.close(fig)
    print("wrote fig2_quality_cost.pdf")


def forest(sig):
    keep = sig[sig.contrast.isin(["P3 vs P4", "P4 vs P7", "P5 vs P8"])].copy()
    keep["key"] = keep.base_model + "  " + keep.contrast
    keep = keep.sort_values(["contrast", "base_model"])
    y = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=GEOM["forest"])
    ax.axvspan(-0.25, 0.25, color="gray", alpha=0.15, label="practical threshold $|\\Delta|<0.25$")
    ax.axvline(0, color="k", lw=0.8)
    for i, (_, r) in enumerate(keep.iterrows()):
        c = "#2ca02c" if r.practically_sig else "#888888"
        ax.errorbar(r.mean_diff, i, xerr=[[r.mean_diff - r.ci95_lo], [r.ci95_hi - r.mean_diff]],
                    fmt="o", color=c, ecolor=c, ms=4, capsize=2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(keep.key, fontsize=6.5)
    ax.set_xlabel("paired $\\Delta$ overall (second $-$ first), 95\\% CI", fontsize=8)
    ax.set_title("CPT-over-SFT (P3$\\to$P4) and matched RAG effect (P4$\\to$P7, P5$\\to$P8). "
                 "Green = practically significant.", fontsize=7.5)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_forest.pdf"); plt.close(fig)
    print("wrote fig3_forest.pdf")


def radar(scores):
    """Per-category competence radar (mean over adapted models), closed vs open-book."""
    df = scores[scores.size_group != "api-baseline"]
    piv = df.pivot_table(index="category", columns="setting", values="overall", aggfunc="mean")
    cats = list(piv.index)
    labels = [SHORT_CAT.get(c, c) for c in cats]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    ang = angles + angles[:1]
    fig = plt.figure(figsize=GEOM["radar"])
    ax = fig.add_subplot(111, polar=True)
    for col, color, lab in [("closed_book", CB, "Closed-book"), ("open_book", OB, "Open-book (RAG)")]:
        v = piv[col].tolist() + [piv[col].tolist()[0]]
        ax.plot(ang, v, color=color, lw=1.6, label=lab)
        ax.fill(ang, v, color=color, alpha=0.12)
    ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_ylim(2.8, 4.2); ax.set_yticks([3.0, 3.5, 4.0]); ax.tick_params(axis="y", labelsize=5.5)
    ax.set_title("Per-category competence (mean over models)", fontsize=8, pad=14)
    if PAPER == "eacl":
        # In the narrower EACL column the in-axes legend collides with the
        # top category labels; park it below the plot instead.
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=2, frameon=False)
    else:
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.28, 1.13))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_radar.pdf"); plt.close(fig)
    print("wrote fig4_radar.pdf")


def strata_bars(scores):
    """Two panels: mean overall by difficulty, and by quality dimension (closed/open/API)."""
    df = scores[scores.size_group != "api-baseline"]
    api = scores[scores.size_group == "api-baseline"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=GEOM["strata"])
    diffs = ["easy", "medium", "hard"]
    cb = [df[(df.setting == "closed_book") & (df.difficulty == d)].overall.mean() for d in diffs]
    ob = [df[(df.setting == "open_book") & (df.difficulty == d)].overall.mean() for d in diffs]
    x = np.arange(3); w = 0.38
    a1.bar(x - w / 2, cb, w, label="Closed-book", color=CB)
    a1.bar(x + w / 2, ob, w, label="Open-book (RAG)", color=OB)
    a1.set_xticks(x); a1.set_xticklabels([d.capitalize() for d in diffs], fontsize=7)
    a1.set_ylabel("mean overall", fontsize=8); a1.set_ylim(2.5, 4.2)
    a1.set_title("(a) By difficulty", fontsize=8); a1.tick_params(labelsize=7)
    dims = ["correctness", "completeness", "clarity", "conciseness"]
    cbd = [df[df.setting == "closed_book"][d].mean() for d in dims]
    obd = [df[df.setting == "open_book"][d].mean() for d in dims]
    apd = [api[d].mean() for d in dims]
    x2 = np.arange(4); w2 = 0.27
    lb_c, lb_o = (("Closed-book", "Open-book (RAG)") if PAPER == "eacl"
                  else ("Closed", "Open (RAG)"))
    a2.bar(x2 - w2, cbd, w2, label=lb_c, color=CB)
    a2.bar(x2, obd, w2, label=lb_o, color=OB)
    a2.bar(x2 + w2, apd, w2, label="Frontier API", color=AP)
    a2.set_xticks(x2); a2.set_xticklabels(["Corr.", "Compl.", "Clar.", "Conc."], fontsize=7)
    a2.set_ylim(2.5, 5.0); a2.set_title("(b) By quality dimension", fontsize=8)
    a2.tick_params(labelsize=7)
    if PAPER == "eacl":
        # One shared legend above both panels: an in-axes legend overlaps the
        # tall frontier-API clarity bar at ACL geometry.
        fig.legend(*a2.get_legend_handles_labels(), ncol=3, loc="upper center",
                   fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        a1.legend(fontsize=6); a2.legend(fontsize=6)
        fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_strata.pdf"); plt.close(fig)
    print("wrote fig5_strata.pdf")


# ---------------- EACL-only figures (plan Phase 1) ----------------

def benchmark_validation(scores):
    """Three panels: (a) difficulty distribution, (b) question-type distribution
    (both from the benchmark JSONL minus audited exclusions, as in make_tables
    composition()), (c) per-model mean overall by difficulty — the empirical
    Easy>Medium>Hard validation. Identity is not the message in (c), so model
    lines are neutral gray with direct group labels (no legend)."""
    excluded = set(pd.read_csv(EXCLUDED)["id"].astype(str).str.strip())
    kept = [r for r in (json.loads(l) for l in open(BENCHMARK, encoding="utf-8"))
            if str(r["id"]) not in excluded]
    diffs = ["easy", "medium", "hard"]
    dcount = [sum(1 for r in kept if r["difficulty"] == d) for d in diffs]
    qt = {}
    for r in kept:
        qt[r["question_type"]] = qt.get(r["question_type"], 0) + 1
    qt = dict(sorted(qt.items(), key=lambda kv: kv[1]))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=GEOM["benchval"],
                                     gridspec_kw={"width_ratios": [1, 1.3, 1.5]})
    b = a1.bar(range(3), dcount, 0.62, color=CB)
    a1.bar_label(b, fontsize=6.5, padding=1)
    a1.set_xticks(range(3)); a1.set_xticklabels([d.capitalize() for d in diffs], fontsize=7)
    a1.set_ylim(0, max(dcount) * 1.18); a1.set_ylabel("questions", fontsize=7.5)
    a1.set_title("(a) Difficulty", fontsize=8); a1.tick_params(labelsize=6.5)
    b = a2.barh(range(len(qt)), list(qt.values()), 0.62, color=CB)
    a2.bar_label(b, fontsize=6.5, padding=2)
    a2.set_yticks(range(len(qt)))
    a2.set_yticklabels([k.capitalize() for k in qt], fontsize=6.5)
    a2.set_xlim(0, max(qt.values()) * 1.16)
    a2.set_title("(b) Question type", fontsize=8); a2.tick_params(labelsize=6.5)
    df = scores[(scores.size_group != "api-baseline") & (scores.setting == "closed_book")]
    api = scores[scores.size_group == "api-baseline"]
    for m in ORDER:
        v = [df[(df.base_model == m) & (df.difficulty == d)].overall.mean() for d in diffs]
        a3.plot(range(3), v, color="#b3b8bd", lw=0.9, zorder=1)
    for m in sorted(api.base_model.unique()):
        v = [api[(api.base_model == m) & (api.difficulty == d)].overall.mean() for d in diffs]
        a3.plot(range(3), v, color="#4d4d4d", lw=1.1, ls="--", zorder=2)
    ohard = df[df.difficulty == "hard"].groupby("base_model").overall.mean().mean()
    ahard = api[api.difficulty == "hard"].overall.mean()
    bbox = dict(fc="white", ec="none", alpha=0.75, pad=0.4)
    a3.text(2.08, ahard, "frontier\nAPIs", fontsize=6, color="#4d4d4d", va="center", bbox=bbox)
    a3.text(2.08, ohard, "8 open-weight\n(closed-book)", fontsize=6, color="#7d838a",
            va="center", bbox=bbox)
    a3.set_xlim(-0.15, 2.95)
    a3.set_xticks(range(3)); a3.set_xticklabels([d.capitalize() for d in diffs], fontsize=7)
    a3.set_ylabel("mean overall", fontsize=7.5); a3.tick_params(labelsize=6.5)
    a3.set_title("(c) Score by difficulty (per model)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_benchmark_validation.pdf"); plt.close(fig)
    print("wrote fig_benchmark_validation.pdf")


def scale_effect(summ):
    """Fine-tuning gain over the instruct baseline vs. model size: full SFT
    (P1->P3) and LoRA (P1->P5), with the Spearman rho of the full-SFT decline
    and the ~12B sign crossover annotated. Blue/orange + distinct markers so
    series identity is not color-alone."""
    from scipy.stats import spearmanr
    ov = summ[summ.size_group != "api-baseline"].pivot_table(
        index="base_model", columns="p_id", values="overall")
    rows = [m for m in ORDER if m in ov.index]
    x = np.array([NUM[m] for m in rows])
    d_full = np.array([ov.loc[m, "P3"] - ov.loc[m, "P1"] for m in rows])
    d_lsft = np.array([ov.loc[m, "P2"] - ov.loc[m, "P1"] for m in rows])
    d_lora = np.array([ov.loc[m, "P5"] - ov.loc[m, "P1"] for m in rows])
    rho = spearmanr(x, d_full).statistic
    fig, ax = plt.subplots(figsize=GEOM["scale"])
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(12, color="#888888", lw=0.8, ls=":")
    # connect the per-size mean (two models share x=27); markers show every model
    for d, color, marker, lab in [(d_full, "#1f77b4", "o", "full SFT (P1$\\to$P3)"),
                                  (d_lsft, "#2ca02c", "s", "LoRA SFT (P1$\\to$P2)"),
                                  (d_lora, "#ff7f0e", "^", "LoRA CPT+SFT (P1$\\to$P5)")]:
        xs = np.array(sorted(set(x)))
        ys = np.array([d[x == v].mean() for v in xs])
        ax.plot(xs, ys, color=color, lw=1.0, zorder=2)
        ax.scatter(x, d, color=color, marker=marker, s=22, zorder=3, label=lab)
    ax.margins(y=0.16)
    # per-model name labels (abbreviated L/G/Q+size, as in Fig. 2): place above
    # the top marker or below the bottom marker of each model's 3-series cluster
    # so the near/exact x-overlap pairs (~1B, 8-9B, 27B) stay separated.
    ABBR = {"llama-3.2-1b": "L-1B", "gemma-3-1b": "G-1B", "qwen3.5-2b": "Q-2B",
            "llama-3.1-8b": "L-8B", "qwen3.5-9b": "Q-9B", "gemma-3-12b": "G-12B",
            "gemma-3-27b": "G-27B", "qwen3.5-27b": "Q-27B"}
    # (vertical side, dx pt, ha): the higher model of each overlap pair goes up,
    # the lower goes down; the two 27B models hug the right spine, so both labels
    # sit to its left (ha="right").
    PLACE = {"gemma-3-1b": ("dn", -2, "center"), "llama-3.2-1b": ("up", 2, "center"),
             "qwen3.5-2b": ("up", 0, "center"), "llama-3.1-8b": ("up", -2, "center"),
             "qwen3.5-9b": ("dn", 3, "center"), "gemma-3-12b": ("dn", 0, "center"),
             "qwen3.5-27b": ("up", -2, "right"), "gemma-3-27b": ("dn", -2, "right")}
    tops = np.maximum.reduce([d_full, d_lsft, d_lora])
    bots = np.minimum.reduce([d_full, d_lsft, d_lora])
    for i, m in enumerate(rows):
        side, ddx, ha = PLACE[m]
        yv, dy, va = (tops[i], 5, "bottom") if side == "up" else (bots[i], -5, "top")
        ax.annotate(ABBR[m], (x[i], yv), fontsize=5.5, xytext=(ddx, dy),
                    textcoords="offset points", ha=ha, va=va, color="#333333")
    ax.text(12.7, ax.get_ylim()[1] * 0.97, "$\\approx$12B", fontsize=6,
            color="#666666", va="top")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 12, 27]); ax.set_xticklabels(["1", "2", "4", "8", "12", "27"])
    ax.set_xlabel("model size (B parameters, log)", fontsize=8)
    ax.set_ylabel("$\\Delta$ overall vs. instruct (P1)", fontsize=8)
    ax.set_title("Fine-tuning gain vs. scale", fontsize=8)
    ax.tick_params(labelsize=7)
    # rho lives in the legend title, off the data (in-plot text crossed the lines)
    leg = ax.legend(fontsize=6, loc="lower left", frameon=False,
                    title=f"full SFT: Spearman $\\rho$={rho:.2f}", title_fontsize=6)
    leg._legend_box.align = "left"
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_scale_effect.pdf"); plt.close(fig)
    print("wrote fig_scale_effect.pdf")


def main():
    global OUT, GEOM, PAPER
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=["ieee", "eacl"], default="ieee",
                    help="target paper: ieee -> paper/figures (default), eacl -> paper_eacl/figures")
    ap.add_argument("--outdir", default=None,
                    help="write .pdf here instead of the paper directory; "
                         "--paper still selects geometry and metric, so the "
                         "artifact release can regenerate every figure without "
                         "a paper/ tree present")
    args = ap.parse_args()
    PAPER = args.paper
    if args.paper == "eacl":
        OUT = "paper_eacl/figures"
        GEOM = GEOM_EACL
    # --outdir changes only *where* output lands, never *what* is emitted.
    if args.outdir:
        OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)
    summ = pd.read_csv(f"{A}/summary_by_variant.csv")
    summ = summ[summ.base_model.isin(ORDER) | (summ.size_group == "api-baseline")]
    sig = pd.read_csv(f"{A}/significance.csv")
    scores = pd.read_csv(f"{A}/scores_overall.csv")
    if args.paper == "eacl":
        # EACL primary metric: deterministic rubric aggregate (plan v2 §13,
        # docs/JUDGE_OVERALL_AUDIT.md). Every figure below then renders the
        # formula metric. Delete these three lines to revert to judge-reported.
        summ["overall"] = summ["overall_formula"]
        scores["overall"] = scores["overall_formula"]
        sig = pd.read_csv(f"{A}/significance_formula.csv")
    heatmap(summ); quality_cost(summ); forest(sig); radar(scores); strata_bars(scores)
    if args.paper == "eacl":
        heatmap_column(summ); benchmark_validation(scores); scale_effect(summ)
    print("done.")


if __name__ == "__main__":
    main()
