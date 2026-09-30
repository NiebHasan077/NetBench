#!/usr/bin/env python3
"""Generate LaTeX table fragments for the paper from the analysis CSVs + profiling.

Reads analysis/outputs/{summary_by_variant,significance,completeness_grid}.csv and
the per-model NetBench-LLM/outputs/by_model/*/profiling_results/inference_profile_summary_*.csv,
and writes paper/tables/*.tex (booktabs). Provisional (think-corrupted) cells are
wrapped in \\provisional{...}; the macro + table notes are defined in paper/main.tex.

Re-run after analysis/aggregate_scores.py (e.g. once qwen is regenerated) to refresh
every table automatically:  analysis/.venv/bin/python analysis/make_tables.py

--paper {ieee,eacl} selects the paper target: ieee (default) writes paper/tables
exactly as before; eacl writes paper_eacl/tables, rewrites cross-references to the
EACL section labels, and additionally emits the EACL-only tables (models/variants,
stratified significance, judge agreement).

--outdir DIR redirects the output only; --paper still selects layout, metric, and
cross-references. This lets the released artifact regenerate every table with no
paper/ tree present, e.g.
    analysis/.venv/bin/python analysis/make_tables.py --paper eacl \\
        --outdir analysis/outputs/tables
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import statistics
import openpyxl
import pandas as pd

OUT = "paper/tables"
# Two independent axes. They used to be one, which worked while there were only
# two targets; jbd wants eacl's content with neither paper's column geometry.
#   PAPER   content profile: which captions, which metric, which tables exist.
#           "ieee" or "eacl" only -- jbd sets it to "eacl" because their content
#           is identical, including the corrected source-paper count.
#   LAYOUT  column geometry: "ieee", "eacl", or "jbd". Gate anything that exists
#           because a column is narrow or wide on THIS, never on PAPER.
PAPER = "ieee"
LAYOUT = "ieee"
SUBS: dict[str, str] = {}   # caption cross-reference rewrites (set for --paper eacl/jbd)
# Score metric (docs/JUDGE_OVERALL_AUDIT.md): ieee keeps the judge-reported
# overall; eacl switches both to the deterministic rubric aggregate.
OV = "overall"              # score column in summary/scores CSVs
SIGF = "significance"       # significance file stem
A = "analysis/outputs"
BENCHMARK = "NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl"
EXCLUDED = "analysis/excluded_items.csv"
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "api-baseline": 3}
PARAMS = {"llama-3.2-1b": "1.2B", "gemma-3-1b": "1B", "qwen3.5-2b": "2B",
          "llama-3.1-8b": "8B", "qwen3.5-9b": "9B", "gemma-3-12b": "12B",
          "gemma-3-27b": "27B", "qwen3.5-27b": "27B"}
NUMP = {"gemma-3-1b": 1.0, "llama-3.2-1b": 1.2, "qwen3.5-2b": 2, "llama-3.1-8b": 8,
        "qwen3.5-9b": 9, "gemma-3-12b": 12, "gemma-3-27b": 27, "qwen3.5-27b": 27}
PS = [f"P{i}" for i in range(1, 9)]


def cell(v, prov):
    if pd.isna(v):
        return "--"
    s = f"{v:.2f}"
    return f"\\provisional{{{s}}}" if prov else s


def single_column_floats(text):
    """Rewrite two-column float syntax for the single-column Springer class.

    Two things break there, both found by the first Overleaf compile:
      * table*/figure* are twocolumn-only. sn-jnl is single column, so a
        starred float raises "Not in outer par mode" and kills the build.
      * [H] needs the float package, which the Springer preamble deliberately
        does not load (the manual discourages forced placement). sn-jnl wraps
        every table in threeparttable (sn-jnl.cls:1332), so an unrecognised
        specifier unravels that nesting into a confusing "Extra }" instead of
        a clear error.
    Applied at the single point every table passes through, so no emitter can
    forget it.

    Placement is normalised to [htbp] here, and `p` is the load-bearing letter.
    Measured on the compiled PDFs: tab:sig was 519pt tall and tab:sigrag 539pt,
    against sn-jnl's 552pt text block. (Those two numbers are pre-caption-split;
    both floats later grew past 552pt, which is what TALL_JBD_TABLES fixes.) Read the class, not the LaTeX defaults:
    sn-jnl.cls:356 sets \topfraction{.921}, so the [t] ceiling is 508pt, and
    sn-jnl.cls:364 sets \floatpagefraction{.887}, so a float page needs 490pt.
    Both tables sit in the 11pt-wide window above 508 and below 552: too tall
    for any top slot, tall enough to fill a float page. Since LaTeX emits
    floats of one class strictly in order, the eight shorter tables queued
    behind them could not be placed either, and all ten fell through to
    end-of-document float pages -- tab:sig printed on p77 having been first
    cited on p37, tab:bycat on p79 for a p23 citation, and the same cascade hit
    the appendix (tab:stratdiff and friends, 497-539pt).
    fit_narrow_column already shrank these to fit ON a page, but fitting a page
    is not fitting a top, and only `p` asks for a float page. Do not narrow
    this back to [t]: the 2026-08-12 build confirms the repair, with all 22
    generated tables landing on a page that cites them.
    """
    text = re.sub(r"\\(begin|end)\{(table|figure)\*\}", r"\\\1{\2}", text)
    text = re.sub(r"(\\begin\{(?:table|figure)\})(?:\[[^\]]*\])?", r"\1[htbp]", text)
    return fit_narrow_column(text)


# The paired-contrast tables (significance*, significance_supp) carry a
# Description column that is a deterministic function of Contrast: five distinct
# strings repeated over 37 rows. Measured against sn-jnl's 372pt block with real
# Computer Modern advances, it costs 31% of the measure in tab:sig and 39% in
# tab:sigrag -- which is what put the latter at 108% and over the right margin.
# Dropping it for jbd leaves them at 61% of the measure at the class's own 8bp
# table size, so they end up both narrower and larger-typed than the two-column
# originals rather than needing a size cut to fit. The gloss moves to
# the caption, read once instead of 37 times; the per-stratum matrices
# (stratified_*) already report bare P-codes exactly this way.
PCODE_LEGEND = (r" Variants (\S\ref{sec:setup}): P1 instruct; P2 LoRA SFT; "
                r"P3 full SFT; P4 full CPT+SFT; P5 LoRA CPT+SFT; P6--P8 are "
                r"P1/P4/P5 answering with retrieval.")
# How to read a cell of the stratified_* matrices. Shared because jbd repeats it
# verbatim as the tablenotes legend of all four (JBD_CAPTION), and a second copy
# would drift.
STRAT_CELL_LEGEND = (r"Cell: paired mean difference $\Delta$ on overall score; "
                     r"$^{*}$: BH-significant; \textbf{bold}: also practically "
                     r"significant; --: stratum skipped ($n<10$). Full "
                     r"per-stratum statistics (CIs, $p$, $d_z$) ship in the "
                     r"released CSV.")
HDR_CONTRAST = r"Model & Contrast & Description & $\Delta$ [95\% CI] & $p_{BH}$ & $d_z$ & PS"
HDR_CONTRAST_JBD = r"Model & Contrast & $\Delta$ [95\% CI] & $p_{BH}$ & $d_z$ & PS"


def pbh_fmt(p):
    r"""Format a BH-adjusted p-value.

    "<.001" is a text-mode "<". sn-jnl leaves OT1 in force (it comments out the
    T1 \fontenc line at :144) and OT1 maps "<" to an inverted exclamation, so
    the first compiled PDF printed this column as "¡.001". Math mode gives
    the real glyph under any encoding. ieee/eacl load T1 and are byte-frozen, so
    they keep the bare form.
    """
    if p >= 0.001:
        return f"{p:.3f}"
    return r"$<$.001" if LAYOUT == "jbd" else "<.001"


def contrast_head():
    """(tabular spec, header cells, column count) for the paired-contrast tables."""
    if LAYOUT == "jbd":
        return "llrrrc", HDR_CONTRAST_JBD, 6
    return "lllrrrc", HDR_CONTRAST, 7


def contrast_row(r, ci, pbh, ps):
    cells = [str(r.base_model), str(r.contrast)]
    if LAYOUT != "jbd":
        cells.append(str(r.description))
    cells += [ci, pbh, f"{r.cohens_dz:+.2f}", ps]
    return " & ".join(cells) + r" \\"


def fit_narrow_column(text):
    r"""Refit a float authored for a wide two-column span to sn-jnl's 31pc column.

    Two independent overflows, both measured against the 372pt x 552pt text
    block (sn-jnl.cls:270) and both seen in the first compile:

    Vertical. The [referee] option makes sn-jnl load setspace and \doublespacing
    the whole document (sn-jnl.cls:125-130). That stretch is applied by
    \@setfontsize, so it reaches tabular rows too, not just prose: rows come out
    ~1.67x apart and long tables run off the bottom of the page. The 73-row
    stratified tables measured 1.76x the text height that way. \baselinestretch
    is the kernel hook setspace itself sets, so assigning it directly needs no
    package and keeps working if [referee] is dropped for the camera-ready; it
    must precede the float's size command, which is what re-selects the font and
    applies the new stretch. Scoped to the float, so prose stays double-spaced
    for the reviewers.

    Horizontal. \tabcolsep is sized for a 7in measure; at 372pt the 7- to
    12-column tables spend 72-132pt of the line on gutters alone and run past
    the right margin into the lineno numbers. 4pt recovers 4pt per gutter. The
    paired-contrast tables need more than a gutter trim and drop a whole
    redundant column instead -- see PCODE_LEGEND.

    Size. sn-jnl inverts the standard ladder: \footnotesize is 7pt and
    \scriptsize is 9pt (sn-jnl.cls:187-198), so the two-column targets' size
    commands mean the opposite of what they read as -- the tables inherited from
    ieee/eacl were landing at the smallest non-tiny size the class offers. Both
    are dropped for an explicit size, which is also the only form that cannot be
    inverted by a class. 9bp/11bp is \small's pairing and one point above the
    \tablebodyfont = 8bp the class picks itself (sn-jnl.cls:1216, applied by the
    redefined table environment at :1332); the extra point is spent because the
    measurements leave room for it and the reviewers' copy is double-spaced
    around these tables, which makes small type read smaller still. Worst case
    over the 23 jbd tables is 85% of the 372pt measure and 81% of the 552pt
    height, both on the 37-row matrices. Written out rather than calling
    \tablebodyfont, which runs before the \baselinestretch reset above and would
    keep the doubled leading.
    """
    return re.sub(r"(\\begin\{table\}(?:\[[^\]]*\])?\\centering)"
                  r"(?:\\footnotesize|\\scriptsize)?",
                  r"\1\\renewcommand{\\baselinestretch}{1}"
                  r"\\setlength{\\tabcolsep}{4pt}"
                  r"\\fontsize{9bp}{11bp}\\selectfont", text)


# Springer requires a table *title* of at most 15 words above the table and,
# separately, a legend of at most 300 words underneath it. Thirty-six of the
# paper's 38 captions were single blocks of 40 words on average, written for
# venues that have no such rule, so every one had to be split. This is a split
# and not a truncation: no wording is dropped, it just moves below the tabular.
#
# The titles are written out rather than derived by cutting the first sentence.
# A mechanical cut produces titles that read as fragments and that silently
# change whenever an emitter's caption is reworded; these are keyed to the
# label, so what a reader sees above each table is deliberate.
#
# Gated on LAYOUT == "jbd" at the single point every table passes through. ieee
# and eacl keep the one-block caption their emitters build and stay
# byte-identical -- nothing below is reachable from those targets.
#
# A value of None means "already within 15 words, leave it whole". A label that
# is missing entirely raises, so a newly added table cannot quietly ship a
# non-compliant caption.
_L = PCODE_LEGEND.lstrip()
JBD_CAPTION: dict[str, tuple[str, str] | None] = {
    "tab:baselines": None,
    "tab:bycat": (
        r"Mean overall score by HPN skill category, closed- versus open-book",
        r"Mean over the eight adapted models. $\Delta$: retrieval gain; "
        r"$n$: questions in category."),
    "tab:bydim": (
        r"Mean judge score by quality dimension",
        r"Scores are 1--5. Open-weight closed-book and open-book (RAG) against "
        r"the frontier API baselines."),
    "tab:equiv": (
        r"Frontier comparison as an equivalence test rather than a ranking",
        r"Each open-weight model's best variant against each API, paired over "
        r"the same 233 questions and resampled by source paper. $\Delta$ is "
        r"open-weight minus API. \emph{equivalent}: the difference lies inside "
        r"the pre-registered $\pm0.25$ margin by two one-sided tests. "
        r"\textbf{higher}/lower: distinguishable from zero and outside that "
        r"margin. \textsuperscript{*}distinguishable from zero but inside "
        r"it: real, yet smaller than the bar this paper sets for acting on a "
        r"difference. No comparison is inconclusive, so the benchmark is "
        r"adequately powered for its own margin at $n{=}233$ "
        r"(\S\ref{sec:stats})."),
    "tab:composition": (
        r"Composition of the 233-item HPN-QA evaluation set",
        r"After the expert audit of \S\ref{sec:benchmark}. Per-category counts "
        r"and scores are in Table~\ref{tab:bycat}."),
    "tab:cptppl": (
        r"Effect of LoRA-CPT on domain and general perplexity",
        r"Lower is better. $\Delta$Gen.\ is the signed change in general "
        r"(WikiText-2) perplexity ($-$ improved, $+$ degraded)."),
    "tab:frontier": (
        r"Frontier baselines against each open-weight model's best variant, ranked",
        r"Best variant in bold. Open-weight rows are open-book (retrieval), API "
        r"rows closed-book; the difference is information access, not model "
        r"quality (\S\ref{sec:discussion}). Full grid: Table~\ref{tab:main}."),
    "tab:humanagree": (
        r"Alignment between the LLM judge and the two-expert human consensus",
        r"Two domain experts (authors) independently blind-scored all 125 "
        r"responses (5 systems $\times$ 25 stratified questions) under the "
        r"judge's exact rubric and information set; the human reference is "
        r"their per-response consensus (mean). Bias is judge $-$ consensus "
        r"(positive $=$ judge more lenient)."),
    "tab:humancontrasts": (
        r"Headline paired contrasts re-measured under the two-expert consensus",
        r"The contrasts the sample was designed around, over its 25 questions, "
        r"with the same statistics as \S\ref{sec:stats}. \emph{judge, all}: the "
        r"primary judge's mean difference on the full 233-question benchmark."),
    "tab:interrater": (
        r"Inter-rater reliability between the two expert raters",
        r"Over the same $125$ responses: Pearson $r$, quadratic-weighted "
        r"$\kappa$ (QWK), and Krippendorff's $\alpha$ (interval). "
        r"$\alpha$\,(+judge) adds the primary judge as a third coder; "
        r"R1$-$R2 is the raters' mean leniency offset."),
    "tab:judgeagree": (
        r"Agreement between the primary judge and the second judge",
        r"GPT-5.1 against Gemini-3.5-Flash over 14,679 paired per-question "
        r"judgements across 63 evaluated systems."),
    "tab:judgedev": (
        r"Deviation of the judge-reported overall from the instructed formula",
        r"Over all released judgements. Bias is judge $-$ formula; the "
        r"setting-correlated sign is the drift toward the \emph{unweighted} "
        r"dimension mean discussed in the text."),
    "tab:main": (
        r"Overall HPN-QA score by model and adaptation variant",
        r"Scores are 1--5 under the GPT-5.1 judge. Closed-book P1--P5; "
        # 12.3b removed tab:models from the jbd document but left this
        # pointer, so the compiled PDF printed "(Table ??)". The label still
        # exists in tables/model_variants.tex, which is why a label-existence
        # check passed -- that file is simply no longer \input. Point at the
        # section that gives the reason instead, which is also more use to a
        # reader than a coverage grid was.
        r"open-book (RAG) P6--P8. Blank cells are the 27B P4/P7 variants, "
        r"omitted by design (\S\ref{sec:variants}). Bottom tier: frontier API baselines "
        r"(closed-book, no adaptation; shown under P1). Paired differences "
        r"quoted in the text are computed on unrounded per-question scores and "
        r"can differ from differences of the rounded cells shown here by $0.01$."),
    "tab:models": (
        r"Evaluated open-weight models and adaptation-variant coverage",
        r"\checkmark{}: variant run; --: P4/P7 (full-parameter CPT from base) "
        r"disabled by design for the 27B tier (\S\ref{sec:setup}). The three "
        r"frontier API baselines (GPT-4o, Gemini-2.5-Pro, Claude-Sonnet-4.6) "
        r"are evaluated closed-book under the identical protocol."),
    "tab:profiling": (
        r"Per-model inference cost for closed-book generation",
        r"Measured on each model's own serving configuration (GPUs column; "
        r"fixed across that model's variants). Throughput is comparable within "
        r"a configuration and indicative across them. VRAM is the peak "
        r"\emph{per-GPU} allocation under the listed sharding, not a "
        r"single-GPU footprint. Weights are recovered by summing each model's "
        r"per-shard load allocation, which returns $77$--$97\%$ of the "
        r"$2$-bytes-per-parameter expectation under uneven sharding, so the "
        r"totals are lower bounds; gemma-3-1b's was logged as $0$\,MB, so its "
        r"total is unknown rather than zero and is left blank."),
    "tab:sig": (
        r"Primary paired contrasts on overall score: closed-book adaptation",
        r"The open-book and matched-RAG contrasts continue in "
        r"Table~\ref{tab:sigrag}. $\Delta$: mean paired difference; $p_{BH}$: "
        r"Benjamini--Hochberg-adjusted $p$-value; $d_z$: paired effect size; "
        r"\textbf{PS}: practically significant. Test, FDR control, and "
        r"thresholds in \S\ref{sec:stats}. " + _L),
    "tab:sigrag": (
        r"Primary paired contrasts on overall score: open-book and matched-model retrieval",
        r"Continuing Table~\ref{tab:sig}. $\Delta$: mean paired difference; "
        r"$p_{BH}$: Benjamini--Hochberg-adjusted $p$-value; $d_z$: paired "
        r"effect size; \textbf{PS}: practically significant. Test, FDR "
        r"control, and thresholds in \S\ref{sec:stats}. " + _L),
    "tab:sigsupp": (
        r"Supplementary paired contrasts outside the pre-specified primary family",
        r"Reported to back the LoRA-vs-full-SFT comparisons with uncertainty: "
        r"P1~vs~P2 isolates LoRA SFT alone, P2~vs~P3 compares LoRA SFT against "
        r"full SFT. Same test, CI, effect size, and practical rule as "
        r"Table~\ref{tab:sig}; BH correction within this supplementary family."),
    "tab:stratcat": (
        r"Primary contrasts re-run within each skill category: closed-book adaptation",
        r"Categories abbreviated; full names in Table~\ref{tab:bycat}. "
        r"Exploratory per-stratum analysis (\S\ref{sec:stats}). The remaining "
        r"contrasts continue in Table~\ref{tab:stratcatb}. " + STRAT_CELL_LEGEND),
    "tab:stratcatb": (
        r"Primary contrasts re-run within each skill category: open-book and retrieval",
        r"Categories abbreviated; full names in Table~\ref{tab:bycat}. "
        r"Exploratory per-stratum analysis (\S\ref{sec:stats}). "
        + STRAT_CELL_LEGEND),
    "tab:stratdiff": (
        r"Primary contrasts re-run within each difficulty level: closed-book adaptation",
        r"Exploratory per-stratum analysis (\S\ref{sec:stats}). The remaining "
        r"contrasts continue in Table~\ref{tab:stratdiffb}. " + STRAT_CELL_LEGEND),
    "tab:stratdiffb": (
        r"Primary contrasts re-run within each difficulty level: open-book and retrieval",
        r"Exploratory per-stratum analysis (\S\ref{sec:stats}). "
        + STRAT_CELL_LEGEND),
    "tab:threshsens": (
        r"Sensitivity of the practical-significance verdict to its thresholds",
        r"Of the $54$ BH-significant primary contrasts, the number also "
        r"clearing each $(|\Delta|, |d_z|)$ minimum. The paper's pre-specified "
        r"rule ($0.25$, $0.2$; \textbf{bold}) sits on a plateau: nearby "
        r"conventions move the count, not the conclusion."),
    "tab:traindata": (
        r"Adaptation-data scale across the corpus, instruction set, and benchmark",
        r"Tokens are \texttt{cl100k\_base}. The SFT set is held disjoint from "
        r"the benchmark's source material (\S\ref{sec:framework}); the corpus "
        r"is by design the common source for CPT, retrieval, and benchmark "
        r"generation (\S\ref{sec:threats}). The SFT source mix and split are "
        r"detailed in \S\ref{sec:setup}."),
}


def _caption_body(text, start):
    """Return (body, end_index) for the \\caption{...} whose brace opens at start."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i
        i += 1
    raise ValueError("unbalanced \\caption{")


def split_caption(text):
    r"""Split a one-block caption into a Springer title and a tablenotes legend.

    See JBD_CAPTION for why. The class supplies \begin{tablenotes}
    (sn-jnl.cls:1264) for exactly this, so the legend needs no package; it is
    placed after the tabular and before \end{table}, which is where Springer
    wants it. \raggedright is needed because the float's \centering would
    otherwise centre the legend's lines.

    A caption whose label is absent from JBD_CAPTION raises rather than passing
    through: a new table would otherwise ship a 40-word title and only be caught
    by a human reading the proof. The converse -- an emitter's caption reworded
    without updating the entry here -- is not detectable this way, so the entry
    is the source of truth for what jbd prints.
    """
    m = re.search(r"\\caption\{", text)
    if not m:
        return text
    body, end = _caption_body(text, m.end() - 1)
    lab = re.search(r"\\label\{([^}]+)\}", text)
    key = lab.group(1) if lab else None
    if key not in JBD_CAPTION:
        raise KeyError(f"no jbd caption title recorded for {key!r}; add one to "
                       f"JBD_CAPTION (Springer: table titles are max 15 words)")
    entry = JBD_CAPTION[key]
    if entry is None:
        return text
    title, legend = entry
    text = text[:m.end()] + title + text[end:]
    # tablenotes is a list environment: its content must sit inside an \item,
    # or LaTeX raises "Something's wrong--perhaps a missing \item" for every
    # table. Non-fatal under nonstopmode, which is why it went unseen.
    notes = ("\\begin{tablenotes}\\raggedright\n\\item[]" + legend
             + "\n\\end{tablenotes}\n")
    i = text.rindex(r"\end{table}")
    return text[:i] + notes + text[i:]


# The six 37-row matrices outgrew fit_narrow_column's 9bp/11bp sizing. That
# sizing was chosen against a measurement of tab:sigrag at 539pt (see
# normalise_float), which the Springer caption split then invalidated: moving a
# 300-word legend below the rule added lines to every one of these floats, and
# the first local compile reported tab:sigrag at 572pt and tab:stratdiffb at
# 562pt against sn-jnl's 552pt text block -- "Float too large for page", which
# no page can hold. 8bp on 10bp leading is the class's own \tablebodyfont size
# (sn-jnl.cls:1216) and brings both back under the ceiling.
#
# Applied to all six, not just the two that overflowed: they print as adjacent
# pairs (sig/sigrag, stratdiff/stratdiffb, stratcat/stratcatb) and a type-size
# change in one half of a pair is visible on the page.
TALL_JBD_TABLES = {"significance.tex", "significance_rag.tex",
                   "stratified_difficulty.tex", "stratified_difficulty_b.tex",
                   "stratified_category.tex", "stratified_category_b.tex"}


def write(name, text):
    for old, new in SUBS.items():
        text = text.replace(old, new)
    if LAYOUT == "jbd":
        text = single_column_floats(text)
        if name in TALL_JBD_TABLES:
            text = text.replace(r"\fontsize{9bp}{11bp}", r"\fontsize{8bp}{10bp}")
        text = split_caption(text)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write(text)
    print("wrote", os.path.join(OUT, name))


def main_results(summ):
    adapted = summ[summ.size_group != "api-baseline"].copy()
    adapted["o"] = adapted.sort_values("size_group", key=lambda s: s.map(SIZE_ORDER))["size_group"]
    models = sorted(adapted.base_model.unique(),
                    key=lambda m: (SIZE_ORDER[adapted[adapted.base_model == m].size_group.iloc[0]], m))
    ov = adapted.pivot_table(index="base_model", columns="p_id", values=OV, aggfunc="mean")
    pv = adapted.pivot_table(index="base_model", columns="p_id", values="provisional", aggfunc="max")
    dagger = (r"$^{\dagger}$ provisional (think-corrupted; see \S\ref{sec:limitations}). "
              if bool(pv.fillna(False).to_numpy().any()) else "")
    # eacl folds the frontier API baselines into this table (bottom tier, P1
    # column) so the main text needs no separate baselines table; the rounding
    # note preempts cell-arithmetic vs paired-mean discrepancies (e.g. 0.82 vs
    # +0.81 for Qwen3.5-2B P1->P4)
    api_note = (r" Bottom tier: frontier API baselines (closed-book, no "
                r"adaptation; shown under P1). Paired differences quoted in "
                r"the text are computed on unrounded per-question scores and "
                r"can differ from differences of the rounded cells shown here "
                r"by $0.01$." if PAPER == "eacl" else "")
    L = [r"\begin{table*}[t]\centering\footnotesize",
         r"\caption{Overall HPN-QA score (1--5; GPT-5.1 judge) per model and adaptation "
         r"variant. Closed-book P1--P5; open-book (RAG) P6--P8. " + dagger +
         r"Blank cells mark variants not run (Table~\ref{" +
         (r"tab:models" if PAPER == "eacl" else r"tab:coverage") + r"})." + api_note + r"}",
         r"\label{tab:main}",
         r"\begin{tabular}{ll" + "c" * 8 + "}", r"\toprule",
         r"Tier & Model & " + " & ".join(PS) + r" \\ \midrule"]
    last = None
    for m in models:
        tier = adapted[adapted.base_model == m].size_group.iloc[0]
        tcell = tier if tier != last else ""
        last = tier
        cells = [cell(ov.loc[m, p] if p in ov.columns else float("nan"),
                      bool(pv.loc[m, p]) if (p in pv.columns and not pd.isna(pv.loc[m, p])) else False)
                 for p in PS]
        L.append(f"{tcell} & {m} & " + " & ".join(cells) + r" \\")
    if PAPER == "eacl":
        api = summ[summ.size_group == "api-baseline"][["base_model", OV]].sort_values("base_model")
        L.append(r"\midrule")
        tcell = "API"
        for _, r in api.iterrows():
            L.append(f"{tcell} & {r.base_model} & {r[OV]:.2f} & " + " & ".join([""] * 7) + r" \\")
            tcell = ""
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("main_results.tex", "\n".join(L) + "\n")


def baselines(summ):
    api = summ[summ.size_group == "api-baseline"][["base_model", OV]].sort_values("base_model")
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Frontier API baselines (closed-book, identical HPN-QA / GPT-5.1 protocol).}",
         r"\label{tab:baselines}", r"\begin{tabular}{lc}", r"\toprule",
         r"Model & Overall \\ \midrule"]
    for _, r in api.iterrows():
        L.append(f"{r.base_model} & {r[OV]:.2f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("baselines.tex", "\n".join(L) + "\n")


def frontier(summ):
    """EACL-only: the frontier baselines and the best open-weight variants in one
    ranked list. The main text claims mid-sized models rival and 27B models
    exceed the frontier; without this the reader has to reconstruct it from
    Appendix F. Ranked rather than grouped so the interleaving is the point."""
    api = summ[summ.size_group == "api-baseline"][["base_model", OV]].copy()
    api["label"] = api.base_model
    api["kind"] = "API"
    ow = summ[summ.size_group != "api-baseline"]
    best = ow.loc[ow.groupby("base_model")[OV].idxmax()].copy()
    # Show only models that reach the frontier band, plus the first one below it,
    # so the table shows where the line falls instead of just the winners.
    floor = api[OV].min()
    keep = best[best[OV] >= floor]
    below = best[best[OV] < floor].nlargest(1, OV)
    best = pd.concat([keep, below])
    best["label"] = best.apply(
        lambda r: f"{r.base_model} ({r.p_id})", axis=1)
    best["kind"] = "open"
    rows = pd.concat([api, best])[["label", "kind", OV]].sort_values(OV, ascending=False)

    cap = (r"\caption{Frontier baselines vs.\ each open-weight model's best "
           r"variant (bold), ranked. Open-weight rows are open-book (retrieval), "
           r"API rows closed-book---information access, not model quality "
           r"(\S\ref{sec:discussion}). Full grid: Table~\ref{tab:main}.}"
           "\n" r"\label{tab:frontier}")
    head = [r"\begin{table}[t]\centering\footnotesize"]
    # Caption above the tabular, matching every other table in the paper and
    # Springer's house style; the eacl/ieee emission keeps the historical order.
    if LAYOUT == "jbd":
        head.append(cap)
    L = head + [r"\begin{tabular}{@{}llr@{}}", r"\toprule",
                r"System & & Overall \\", r"\midrule"]
    for _, r in rows.iterrows():
        name = r.label.replace("_", r"\_")
        if r.kind == "open":
            name = r"\textbf{" + name + "}"
        L.append(f"{name} & {r.kind} & {r[OV]:.2f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if LAYOUT != "jbd":
        L.append(cap)
    L.append(r"\end{table}")
    write("frontier.tex", "\n".join(L) + "\n")


def equivalence(path):
    """R1/R6 (Phase 12), jbd-only: the frontier comparison as a test rather than
    a ranking. tab:frontier orders point estimates, which cannot establish
    parity -- and "not significantly different" is not "equivalent". This
    reports both decisions TOST produces: distinguishable from zero, and inside
    the pre-registered +-0.25 margin. Gated on LAYOUT so the frozen eacl tree
    and the dormant ieee tree cannot move (decision 0.7)."""
    d = pd.read_csv(path)
    short = {"claude-sonnet-4-6": "Claude-Sonnet-4.6",
             "gemini-2.5-pro": "Gemini-2.5-Pro", "gpt-4o": "GPT-4o"}
    mark = {"equivalent": r"equivalent",
            "meaningfully higher": r"\textbf{higher}",
            "meaningfully lower": r"lower",
            "higher, but below the practical margin": r"higher\textsuperscript{*}",
            "lower, but below the practical margin": r"lower\textsuperscript{*}",
            "inconclusive": r"inconclusive"}
    rows = []
    for _, r in d.iterrows():
        rows.append(f"{r.open_system} & {short.get(r.api, r.api)} & "
                    f"{r.mean_diff:+.2f} & [{r.ci90_lo:+.2f},\\,{r.ci90_hi:+.2f}] & "
                    f"{mark.get(r.verdict, r.verdict)} \\\\")
    cap = (r"\caption{Frontier comparison as an equivalence test rather than a "
           r"ranking. Each open-weight model's best variant against each API, "
           r"paired over the same 233 questions and resampled by source paper. "
           r"$\Delta$ is open-weight minus API. \emph{equivalent}: the "
           r"difference lies inside the pre-registered $\pm0.25$ margin by two "
           r"one-sided tests. \textbf{higher}/lower: distinguishable from zero "
           r"and outside that margin. \textsuperscript{*}distinguishable from "
           r"zero but inside it---real, yet smaller than the bar this paper "
           r"sets for acting on a difference. No comparison is inconclusive, so "
           r"the benchmark is adequately powered for its own margin at "
           r"$n{=}233$ (\S\ref{sec:stats}).}" "\n" r"\label{tab:equiv}")
    text = ("\n".join([r"\begin{table}[t]\centering\footnotesize", cap,
                       r"\setlength{\tabcolsep}{4pt}",
                       r"\begin{tabular}{@{}llrrl@{}}", r"\toprule",
                       r"Open-weight system & vs.\ API & $\Delta$ & 90\% CI & Verdict \\",
                       r"\midrule"] + rows
                      + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))
    write("equivalence.tex", text)


def significance(sig):
    any_prov = bool(sig.provisional.any())
    s = sig[~sig.provisional].copy()
    gl = {"closed_book": "Closed-book adaptation", "open_book": "Open-book adaptation",
          "rag_effect": "Matched-model RAG effect"}
    family = "non-provisional family" if any_prov else "primary family"
    prov_note = r" Provisional (think-corrupted) contrasts are excluded." if any_prov else ""
    cap_head = (r"\caption{Primary paired contrasts on per-question overall score "
                r"(" + family + r")")
    # p_BH is glossed for eacl only; the ieee caption stays byte-identical
    # (its missing gloss is deferred to the main-branch sweep).
    pbh_def = (r"$p_{BH}$: Benjamini--Hochberg-adjusted $p$-value; "
               if PAPER == "eacl" else "")
    legend = PCODE_LEGEND if LAYOUT == "jbd" else ""
    cap_tail = (r" $\Delta$: mean paired difference; " + pbh_def +
                r"$d_z$: paired effect "
                r"size; \textbf{PS}: practically significant. Test, FDR control, and thresholds "
                r"in \S\ref{sec:stats}." + prov_note + legend + r"}")
    spec, hdr, ncol = contrast_head()

    def emit(fname, label, caption, groups):
        L = [r"\begin{table*}[t]\centering\footnotesize", caption, label,
             r"\begin{tabular}{" + spec + "}", r"\toprule", hdr + r" \\"]
        for g in groups:
            sub = s[s.group == g]
            if sub.empty:
                continue
            L.append(r"\midrule \multicolumn{" + str(ncol) + r"}{l}{\textit{" + gl[g] + r"}} \\ \midrule")
            for _, r in sub.iterrows():
                ps = r"\checkmark" if r.practically_sig else ""
                ci = f"{r.mean_diff:+.2f} [{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}]"
                pbh = pbh_fmt(r.p_bh)
                L.append(contrast_row(r, ci, pbh, ps))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        write(fname, "\n".join(L) + "\n")

    if PAPER == "eacl":
        # Two floats: the single 72-row table overflows the page and clips.
        emit("significance.tex", r"\label{tab:sig}",
             cap_head + r", closed-book contrasts; the open-book and matched-RAG "
             r"contrasts continue in Table~\ref{tab:sigrag}." + cap_tail,
             ["closed_book"])
        emit("significance_rag.tex", r"\label{tab:sigrag}",
             cap_head + r", open-book and matched-RAG contrasts, continuing "
             r"Table~\ref{tab:sig}." + cap_tail,
             ["open_book", "rag_effect"])
    else:
        emit("significance.tex", r"\label{tab:sig}", cap_head + r"." + cap_tail,
             ["closed_book", "open_book", "rag_effect"])


def completeness(path):
    g = pd.read_csv(path, index_col=0)
    g = g.replace({"yes": r"\checkmark", "—": "--", "-": "--"})
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Variant coverage. P4/P7 (full CPT from base) are disabled by design for 27B models.}",
         r"\label{tab:coverage}", r"\begin{tabular}{l" + "c" * len(g.columns) + "}", r"\toprule",
         "Model & " + " & ".join(g.columns) + r" \\ \midrule"]
    for m, row in g.iterrows():
        L.append(f"{m} & " + " & ".join(str(x) for x in row.values) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("completeness.tex", "\n".join(L) + "\n")


def gpu_config():
    """Per-model GPU serving configuration, from the profiling JSONs' system
    block (fixed across a model's variants). E.g. '2x A6000-48G'."""
    import math
    import re
    out = {}
    for f in sorted(glob.glob("NetBench-LLM/outputs/by_model/*/profiling_results/inference_*.json")):
        model = f.split("/by_model/")[1].split("/")[0]
        if model in out:
            continue
        try:
            gpus = json.load(open(f)).get("system", {}).get("gpus") or []
        except (OSError, json.JSONDecodeError):
            continue
        if not gpus:
            continue
        name = gpus[0]["name"]
        chip = next((t for t in name.replace("-", " ").split()
                     if re.fullmatch(r"[A-Z]+\d+\w*", t)), name)
        mem = math.ceil(gpus[0].get("total_memory_mb", 0) / 1024)
        out[model] = f"{len(gpus)}$\\times$~{chip}-{mem}G"
    return out


def profiling():
    num = {"gemma-3-1b": 1.0, "llama-3.2-1b": 1.2, "qwen3.5-2b": 2, "llama-3.1-8b": 8,
           "qwen3.5-9b": 9, "gemma-3-12b": 12, "gemma-3-27b": 27, "qwen3.5-27b": 27}
    # R7 (Phase 12), jbd only: the reviewer is right that a per-GPU peak under
    # 4-way sharding says nothing about a model's memory need. The weight total
    # does, so carry it beside the per-GPU peak. Recovered from the per-shard
    # load allocation by analysis/deployment_cost.py; eacl/ieee keep the frozen
    # six-column form (decision 0.7).
    global WEIGHTS_GB
    WEIGHTS_GB = {}
    wt_path = f"{A}/deployment_footprint.csv"
    if LAYOUT == "jbd" and os.path.exists(wt_path):
        wf = pd.read_csv(wt_path)
        WEIGHTS_GB = dict(zip(wf.base_model, wf.weights_total_gb))
    gpus = gpu_config()
    rows = []
    for f in sorted(glob.glob("NetBench-LLM/outputs/by_model/*/profiling_results/inference_profile_summary_*.csv")):
        model = f.split("/by_model/")[1].split("/")[0]
        df = pd.read_csv(f)

        def pick(dev):
            for ptype in ("direct", "rag"):     # prefer closed-book; fall back to RAG-only profiles
                d = df[(df.profile_type == ptype) & (df.device == dev)]
                if not d.empty:
                    s1 = d[d.label.str.contains("S1|Instruct", case=False, na=False)]
                    return (s1 if not s1.empty else d).head(1).iloc[0], ptype
            return None, None
        g, gset = pick("cuda")
        c, _ = pick("cpu")
        if g is None:
            continue
        # every model now ships direct (closed-book) profiles; a silent RAG
        # fallback would make the rows incomparable again, so fail loudly
        assert gset == "direct", f"{model}: no direct GPU profile in {f}"
        vram = g.get("peak_gpu_vram_mb", float("nan"))
        ram = c.get("peak_cpu_ram_mb", float("nan")) if c is not None else float("nan")
        rows.append({
            "model": model, "gpus": gpus.get(model, "?"),
            "wt_gb": WEIGHTS_GB.get(model, float("nan")),
            "gpu_tps": g.get("avg_tokens_per_second", float("nan")),
            "cpu_tps": c.get("avg_tokens_per_second", float("nan")) if c is not None else float("nan"),
            "vram_gb": (vram / 1024) if pd.notna(vram) else float("nan"),
            "ram_gb": (ram / 1024) if pd.notna(ram) else float("nan"),
        })
    if not rows:
        write("profiling.tex", "% no profiling summaries found\n")
        return
    pf = pd.DataFrame(rows).sort_values("model", key=lambda s: s.map(lambda m: num.get(m, 99)))
    # eacl: the ACL column is narrower than the IEEE one; scriptsize + trimmed
    # padding keep the six columns inside it (the ieee strings stay byte-identical).
    # jbd is single column and wider than either, so it takes the roomier form.
    size = r"\scriptsize" if LAYOUT == "eacl" else r"\footnotesize"
    jbd = LAYOUT == "jbd" and bool(WEIGHTS_GB)
    tabline = (r"\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llrrrr@{}}"
               if LAYOUT == "eacl" else
               r"\setlength{\tabcolsep}{3.2pt}\begin{tabular}{llrrrrr}" if jbd else
               r"\setlength{\tabcolsep}{3.2pt}\begin{tabular}{llrrrr}")
    cap = (r"\caption{Per-model inference cost (closed-book generation profiles), "
           r"measured on each model's own serving configuration (GPUs column; fixed "
           r"across that model's variants). Throughput is comparable within a "
           r"configuration and indicative across them. VRAM is the peak \emph{per-GPU} "
           r"allocation under the listed sharding, not a single-GPU footprint.}")
    if jbd:
        cap = (r"\caption{Per-model inference cost (closed-book generation profiles), "
               r"measured on each model's own serving configuration (GPUs column; fixed "
               r"across that model's variants). Throughput is comparable within a "
               r"configuration and indicative across them. \emph{Weights} is the model's "
               r"total parameter footprint summed back across its shards, which is the "
               r"binding memory constraint; \emph{VRAM/GPU} is the peak allocation on "
               r"the device holding the first shard, so it is neither a per-model total "
               r"nor a single-GPU footprint. No model was profiled on a single GPU "
               r"(\S\ref{sec:threats}).}")
    hdr = (r"Model & GPUs & \shortstack[r]{GPU\\tok/s} & \shortstack[r]{CPU\\tok/s} & "
           + (r"\shortstack[r]{Weights\\(GB)} & " if jbd else "")
           + r"\shortstack[r]{VRAM/GPU\\(GB)} & \shortstack[r]{RAM\\(GB)} \\ \midrule")
    L = [r"\begin{table}[t]\centering" + size, cap,
         r"\label{tab:profiling}", tabline, r"\toprule", hdr]
    for _, r in pf.iterrows():
        def f(x, d=2):
            return "--" if pd.isna(x) else f"{x:.{d}f}"
        wt = f"{f(r.wt_gb, 1)} & " if jbd else ""
        L.append(f"{r.model} & {r.gpus} & {f(r.gpu_tps)} & {f(r.cpu_tps)} & {wt}{f(r.vram_gb,1)} & {f(r.ram_gb,1)} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("profiling.tex", "\n".join(L) + "\n")


def by_category(path):
    df = pd.read_csv(path)
    adapted = df[df.size_group != "api-baseline"]
    g = adapted.groupby(["category", "setting"])[OV].mean().unstack().round(2)
    g["nq"] = df.drop_duplicates("question_id").groupby("category").size()
    g["delta"] = (g["open_book"] - g["closed_book"]).round(2)
    g = g.sort_values("nq", ascending=False)
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Mean overall score by HPN skill category (eight adapted models), "
         r"closed- vs.\ open-book (RAG). $\Delta$: retrieval gain; $n$: questions in category.}",
         r"\label{tab:bycat}",
         # The category column was p{0.46\columnwidth}. That silently depends on
         # what \columnwidth is *inside* the float, and sn-jnl wraps every table
         # in threeparttable (sn-jnl.cls:1332), where it is not the 372pt text
         # block: labels measuring 153pt were wrapping in a box nominally 171pt
         # wide. The wrap then hyphenated "Mixed Work-/loads" hard against the
         # n column, which is the collision the first compile showed. An
         # explicit width measured against the real text block removes the
         # dependency. 180pt clears the longest label (Bottleneck Diagnosis and
         # End-to-End Reasoning, 171.8pt at 9bp) and leaves the table at ~76%
         # of the measure. ieee/eacl keep the fraction: their columns are
         # narrower and the labels already wrap there by design.
         (r"\begin{tabular}{@{}p{180pt}rrrr@{}}" if LAYOUT == "jbd"
          else r"\begin{tabular}{@{}p{0.46\columnwidth}rrrr@{}}"), r"\toprule",
         r"Skill category & $n$ & Closed & Open & $\Delta$ \\ \midrule"]
    for cat, r in g.iterrows():
        L.append(f"{cat} & {int(r.nq)} & {r.closed_book:.2f} & {r.open_book:.2f} & {r.delta:+.2f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("by_category.tex", "\n".join(L) + "\n")


def by_dimension(path):
    df = pd.read_csv(path)
    dims = ["correctness", "completeness", "clarity", "conciseness"]
    ad = df[df.size_group != "api-baseline"]
    closed = ad[ad.setting == "closed_book"][dims].mean()
    openb = ad[ad.setting == "open_book"][dims].mean()
    api = df[df.size_group == "api-baseline"][dims].mean()
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Mean judge score by quality dimension (1--5): open-weight closed-book "
         r"and open-book (RAG) vs.\ the frontier API baselines.}",
         r"\label{tab:bydim}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Dimension & Closed-book & Open-book & Frontier API \\ \midrule"]
    for d in dims:
        L.append(f"{d.capitalize()} & {closed[d]:.2f} & {openb[d]:.2f} & {api[d]:.2f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("by_dimension.tex", "\n".join(L) + "\n")


def cpt_perplexity():
    """HPN-domain and general (WikiText-2) perplexity before/after LoRA-CPT,
    from the per-model adaptation reports. The domain column shows CPT learning
    the domain; the general column shows whether it forgets at the LM level."""
    def ppl(path, key):
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["Adaptation Report"]
        for row in ws.iter_rows(values_only=True):
            if row[0] and key in str(row[0]).lower():
                try:
                    return float(row[1]), float(row[2])
                except (TypeError, ValueError):
                    return None, None
        return None, None
    seen = {}
    for f in sorted(glob.glob("NetBench-LLM/outputs/by_model/*/evaluations/adaptation_reports/*lora-cpt*merged*.xlsx")):
        m = f.split("/by_model/")[1].split("/")[0]
        if m in seen:
            continue
        hb, ha = ppl(f, "hpn domain perplexity")
        gb, ga = ppl(f, "general lm perplexity")
        if hb is not None and ha is not None:
            seen[m] = (hb, ha, gb, ga)
    order = sorted(seen, key=lambda m: NUMP.get(m, 99))
    if not order:
        write("cpt_perplexity.tex", "% no adaptation reports found\n")
        return
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Effect of LoRA-CPT on perplexity (lower is better). $\Delta$Gen.\ "
         r"is the signed change in general (WikiText-2) perplexity ($-$ improved, "
         r"$+$ degraded).}",
         r"\label{tab:cptppl}", r"\begin{tabular}{@{}lccr@{}}", r"\toprule",
         r"& HPN-domain PPL & General PPL & \\",
         r"Model & (before$\to$after) & (WikiText-2) & $\Delta$Gen. \\ \midrule"]
    for m in order:
        hb, ha, gb, ga = seen[m]
        if gb is not None and ga is not None:
            gcol = f"{gb:.1f}$\\to${ga:.1f}"
            dgen = f"{100*(ga-gb)/gb:+.0f}\\%"
        else:
            gcol, dgen = "---", "---"
        L.append(f"{m} & {hb:.1f}$\\to${ha:.1f} & {gcol} & {dgen} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("cpt_perplexity.tex", "\n".join(L) + "\n")


def composition():
    """Benchmark composition from the source JSONL minus the audited exclusions.
    Independent of model answers, so it does not move with the score regeneration."""
    excluded = set(pd.read_csv(EXCLUDED)["id"].astype(str).str.strip())
    recs = [json.loads(l) for l in open(BENCHMARK, encoding="utf-8")]
    kept = [r for r in recs if str(r["id"]) not in excluded]
    n = len(kept)
    # distinct source papers via the items' source_papers metadata; the id-stem
    # proxy overcounts (each cross-paper SYN / adversarial item became a "paper")
    papers = len({p for r in kept for p in r["source_papers"]})
    cats = len({r["category"] for r in kept})
    diff = {d: sum(1 for r in kept if r["difficulty"] == d) for d in ("easy", "medium", "hard")}
    qt = {}
    for r in kept:
        qt[r["question_type"]] = qt.get(r["question_type"], 0) + 1
    qt = dict(sorted(qt.items(), key=lambda kv: -kv[1]))
    numeric = sum(1 for r in kept if r.get("requires_calculation"))
    wl = [len(r["reference_answer"].split()) for r in kept]
    label = {"diagnosis": "Diagnosis (troubleshooting)"}
    pct = lambda c: f"{round(100 * c / n)}\\%"
    # ACL columns are narrower than IEEE's: shorten the widest row there and
    # move its legend into the caption. jbd is wider still and keeps the full label.
    wl_note = (r" Reference-answer length is mean\,/\,median\,/\,range."
               if LAYOUT == "eacl" else "")
    wl_label = ("Ref.-answer length (words)" if LAYOUT == "eacl"
                else "Ref.-answer length (words: mean/median/range)")
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Composition of the $233$-item HPN-QA evaluation set (after the expert "
         r"audit of \S\ref{sec:benchmark}). Per-category counts and scores are in "
         r"Table~\ref{tab:bycat}." + wl_note + r"}",
         r"\label{tab:composition}", r"\begin{tabular}{@{}lr@{}}", r"\toprule",
         r"Property & Value \\ \midrule",
         f"Source papers & {papers} " + r"\\",
         f"Questions & {n} " + r"\\",
         f"Skill categories & {cats} " + r"\\",
         r"\midrule \multicolumn{2}{@{}l}{\textit{Difficulty}} \\",
         f"\\quad Easy / Medium / Hard & {diff['easy']} / {diff['medium']} / {diff['hard']} " + r"\\",
         r"\midrule \multicolumn{2}{@{}l}{\textit{Question type}} \\"]
    for t, c in qt.items():
        L.append(f"\\quad {label.get(t, t.capitalize())} & {c} ({pct(c)}) " + r"\\")
    L += [r"\midrule",
          f"Require numeric calculation & {numeric} ({pct(numeric)}) " + r"\\",
          f"{wl_label} & "
          f"{round(statistics.mean(wl))} / {round(statistics.median(wl))} / "
          f"{min(wl)}--{max(wl)} " + r"\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("composition.tex", "\n".join(L) + "\n")


def training_stats():
    """Adaptation-data scale: corpus (CPT) and SFT from analysis/outputs/training_data_stats.csv
    (produced by count_corpus_tokens.py + count_sft_tokens.py over the gitignored raw data);
    the eval row is computed from the in-repo benchmark JSONL so it always matches
    Table~\\ref{tab:composition}."""
    d = pd.read_csv(f"{A}/training_data_stats.csv", comment="#").set_index("key")["value"]
    excluded = set(pd.read_csv(EXCLUDED)["id"].astype(str).str.strip())
    kept = [json.loads(l) for l in open(BENCHMARK, encoding="utf-8")]
    kept = [r for r in kept if str(r["id"]) not in excluded]
    if PAPER == "eacl":
        # Distinct source papers from item metadata, matching composition(); the
        # id-stem proxy below overcounts (each cross-paper SYN/adversarial stem
        # becomes a phantom "paper": 177 vs the true 171).
        ev_papers = len({p for r in kept for p in r["source_papers"]})
    else:  # frozen ieee mirror keeps the id-stem count; fix deferred to main
        ev_papers = len({r["id"].rsplit("-q", 1)[0] for r in kept})
    ev_q = len(kept)
    n = lambda x: f"{int(x):,}"
    m = lambda x: f"{int(x) / 1e6:.1f}M"
    cap = (r"\caption{Adaptation-data scale (tokens: \texttt{cl100k\_base}). The SFT "
           r"set is held disjoint from the benchmark's source material "
           r"(\S\ref{sec:framework}); the corpus is by design the common source for "
           r"CPT, retrieval, and benchmark generation (see Limitations). The SFT "
           r"source mix and split are detailed in \S\ref{sec:methodology}.}"
           if PAPER == "eacl" else
           r"\caption{Adaptation-data scale (tokens: \texttt{cl100k\_base}). The corpus, "
           r"SFT set, and benchmark are mutually disjoint (\S\ref{sec:framework}); the SFT "
           r"source mix and split are detailed in \S\ref{sec:methodology}.}")
    L = ([r"\begin{table}[t]\centering\footnotesize", cap, r"\label{tab:traindata}"]
         + ([r"\setlength{\tabcolsep}{4pt}"] if PAPER == "eacl" else [])
         + [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Stage & Source papers & Examples/items & Tokens \\ \midrule",
         f"Corpus (CPT) & {n(d['corpus_papers'])} & -- & {m(d['corpus_tokens'])} " + r"\\",
         f"Instruction SFT & {n(d['sft_papers'])} & {n(d['sft_examples'])} & {m(d['sft_tokens'])} " + r"\\",
         f"HPN-QA (eval) & {ev_papers} & {ev_q} & -- " + r"\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("training_data.tex", "\n".join(L) + "\n")


# ---------------- EACL-only emitters (plan Phase 1) ----------------

def significance_supp(path):
    """App F: supplementary contrasts (P1 vs P2, P2 vs P3) outside the
    pre-specified primary family, from supplementary_contrasts.py."""
    s = pd.read_csv(path)
    L = [r"\begin{table*}[tp]\centering\footnotesize",
         r"\caption{Supplementary paired contrasts, reported \emph{outside} the "
         r"pre-specified primary family to back the LoRA-vs-full-SFT comparisons "
         r"with uncertainty: P1~vs~P2 isolates LoRA SFT alone, P2~vs~P3 compares "
         r"LoRA SFT against full SFT. Same test, CI, effect size, and practical "
         r"rule as Table~\ref{tab:sig}; BH correction within this supplementary "
         r"family.}",
         r"\label{tab:sigsupp}", r"\begin{tabular}{" + contrast_head()[0] + "}", r"\toprule",
         contrast_head()[1] + r" \\ \midrule"]
    for _, r in s.iterrows():
        ps = r"\checkmark" if r.practically_sig else ""
        ci = f"{r.mean_diff:+.2f} [{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}]"
        pbh = pbh_fmt(r.p_bh)
        L.append(contrast_row(r, ci, pbh, ps))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("significance_supp.tex", "\n".join(L) + "\n")


def threshold_sens(path):
    """App F: practical-significance count under alternative threshold pairs,
    from threshold_sensitivity.py. The paper's rule (0.25, 0.2) is bolded."""
    t = pd.read_csv(path)
    dzcols = [c for c in t.columns if c.startswith("dz_")]
    L = [r"\begin{table}[htbp]\centering\footnotesize",
         r"\caption{Threshold sensitivity of the RQ6 verdict: of the $54$ "
         r"BH-significant primary contrasts, the number also clearing each "
         r"$(|\Delta|, |d_z|)$ minimum. The paper's pre-specified rule "
         r"($0.25$, $0.2$; \textbf{bold}) sits on a plateau: nearby "
         r"conventions move the count, not the conclusion.}",
         r"\label{tab:threshsens}", r"\begin{tabular}{@{}r" + "r" * len(dzcols) + r"@{}}",
         r"\toprule",
         r"$|\Delta|\geq$ & " + " & ".join(f"$|d_z|{{\\geq}}{c[3:]}$" for c in dzcols) + r" \\ \midrule"]
    for _, r in t.iterrows():
        cells = []
        for c in dzcols:
            v = f"{int(r[c])}"
            if abs(r.delta_min - 0.25) < 1e-9 and c == "dz_0.2":
                v = r"\textbf{" + v + "}"
            cells.append(v)
        L.append(f"{r.delta_min:.2f} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("threshold_sensitivity.tex", "\n".join(L) + "\n")


def judge_deviation(path):
    """App B.2 evidence table: how the judge-reported overall deviates from the
    instructed weighted formula (docs/JUDGE_OVERALL_AUDIT.md), by setting."""
    d = pd.read_csv(path).set_index("slice")
    rows = [("all", "All judgements"), ("closed_book_local", "Closed-book (P1--P5)"),
            ("open_book", "Open-book (P6--P8)"), ("api_baseline", "API baselines")]
    L = [r"\begin{table}[H]\centering\footnotesize",
         r"\caption{Deviation of the judge-reported \texttt{overall} from the instructed "
         r"weighted formula, over all released judgements. Bias is judge $-$ formula; "
         r"the setting-correlated sign is the drift toward the \emph{unweighted} "
         r"dimension mean discussed in the text.}",
         r"\label{tab:judgedev}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Slice & $n$ & MAD & Bias & Exact & $|{\cdot}|{>}0.25$ \\ \midrule"]
    for key, name in rows:
        r = d.loc[key]
        L.append(f"{name} & {int(r.n):,} & {r.mad:.3f} & {r.signed_bias:+.3f} & "
                 f"{r.exact_pct:.0f}\\% & {r.gt_0_25_pct:.1f}\\% " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("judge_deviation.tex", "\n".join(L) + "\n")



def model_variants(grid_path):
    """EACL setup table: model roster (tier/family/params) x variant coverage."""
    g = pd.read_csv(grid_path, index_col=0)
    tier = {m: ("small" if NUMP.get(m, 99) <= 2 else "medium" if NUMP.get(m, 99) <= 12 else "large")
            for m in g.index}
    fam = lambda m: "Llama" if "llama" in m else ("Qwen" if "qwen" in m else "Gemma")
    order = sorted(g.index, key=lambda m: (NUMP.get(m, 99), m))
    L = [r"\begin{table*}[t]\centering\footnotesize",
         r"\caption{Evaluated open-weight models and adaptation-variant coverage. "
         r"\checkmark{}: variant run; --: P4/P7 (full-parameter CPT from base) disabled "
         r"by design for the 27B tier (\S\ref{sec:setup}). The three frontier API "
         r"baselines (GPT-4o, Gemini-2.5-Pro, Claude-Sonnet-4.6) are evaluated "
         r"closed-book under the identical protocol.}",
         r"\label{tab:models}",
         r"\begin{tabular}{llll" + "c" * 8 + "}", r"\toprule",
         r"Tier & Model & Family & Params & " + " & ".join(PS) + r" \\ \midrule"]
    last = None
    for m in order:
        t = tier[m]
        tcell = t if t != last else ""
        last = t
        cells = [r"\checkmark" if str(g.loc[m, p]).strip() == "yes" else "--" for p in PS]
        L.append(f"{tcell} & {m} & {fam(m)} & {PARAMS.get(m, '?')} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("model_variants.tex", "\n".join(L) + "\n")


def stratified_matrix(path, stratum_type, fname, label, groups=None, cont=None):
    """Appendix matrix of per-stratum contrasts: rows model x contrast, columns strata.
    Cell: paired mean difference; * BH-significant; bold also practically significant;
    -- stratum skipped (n<10). Compact form of significance_stratified.csv.

    groups restricts the contrast families emitted (None = all three), and cont
    is the label of the float this one continues into. Both are for jbd only:
    all 73 rows fit an ACL [p] float page but not sn-jnl's 194.25mm text block,
    so jbd emits the table in two parts split on the family boundary."""
    short_cat = {
        "Transfer Parameters: Definitions and Roles": "Transf.",
        "Bottleneck Diagnosis and End-to-End Reasoning": "Bottl.",
        "BDP-Based Reasoning and Window Sizing": "BDP",
        "Adaptive and Online Optimization": "Adapt.",
        "Practical HPN Scenarios and Design": "Pract.",
        "Fairness, Stability, and Shared Networks": "Fairn.",
        "Pipelining and Small-File Optimization": "Pipel.",
        "Parallelism and Large-File Optimization": "Parall.",
        "Concurrency Tuning and Scaling": "Concur.",
        "Dataset Partitioning and Mixed Workloads": "Partit.",
    }
    df = pd.read_csv(path)
    df = df[df.stratum_type == stratum_type].copy()
    if stratum_type == "difficulty":
        cols = ["easy", "medium", "hard"]
        heads = ["Easy", "Medium", "Hard"]
        which = "difficulty level"
    else:
        cols = sorted(df.stratum.unique(), key=lambda c: -df[df.stratum == c].n.max())
        heads = [short_cat.get(c, c[:7]) for c in cols]
        which = "skill category (abbreviated; full names in Table~\\ref{tab:bycat})"
    gorder = {"closed_book": 0, "open_book": 1, "rag_effect": 2}
    keys = (df[["base_model", "contrast", "group"]].drop_duplicates()
            .sort_values(["group", "base_model", "contrast"],
                         key=lambda s: s.map(gorder) if s.name == "group" else s))
    gl = {"closed_book": "Closed-book adaptation", "open_book": "Open-book adaptation",
          "rag_effect": "Matched-model RAG effect"}
    if groups is not None:
        keys = keys[keys.group.isin(groups)]
    idx = df.set_index(["base_model", "contrast", "stratum"])
    scope = "" if groups is None else \
        ", " + " and ".join(gl[g].lower() for g in groups) + " contrasts"
    tail = "" if cont is None else \
        r" The remaining contrasts continue in Table~\ref{" + cont + "}."
    L = [r"\begin{table*}[p]\centering\scriptsize",
         r"\caption{Pre-specified primary contrasts re-run within each " + which + scope +
         r" (exploratory per-stratum analysis, \S\ref{sec:stats})." + tail + " " +
         STRAT_CELL_LEGEND + "}",
         r"\label{" + label + "}",
         r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
         r"Model & Contrast & " + " & ".join(heads) + r" \\"]
    last_g = None
    for _, k in keys.iterrows():
        if k.group != last_g:
            L.append(r"\midrule \multicolumn{" + str(2 + len(cols)) +
                     r"}{l}{\textit{" + gl[k.group] + r"}} \\ \midrule")
            last_g = k.group
        cells = []
        for c in cols:
            try:
                r = idx.loc[(k.base_model, k.contrast, c)]
            except KeyError:
                cells.append("--")
                continue
            s = f"{r.mean_diff:+.2f}" + (r"$^{*}$" if bool(r.bh_sig) else "")
            cells.append(r"\textbf{" + s + "}" if bool(r.practically_sig) else s)
        L.append(f"{k.base_model} & {k.contrast} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(fname, "\n".join(L) + "\n")


def judge_agreement(path):
    """Appendix table: primary vs. second judge agreement, from the released workbook."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    summ = {str(r[0]): r[1] for r in wb["Summary"].iter_rows(values_only=True)}
    dims = [r for r in wb["PerQuestionAgreement"].iter_rows(values_only=True)][1:]
    # the *_formula workbook (OV switch) has the identical schema
    # [H] (float pkg, loaded by paper_eacl/main.tex): pin inside the short
    # appendix text; a [t] float here strands on an end-of-document float page.
    L = [r"\begin{table}[H]\centering\footnotesize",
         r"\caption{Agreement between the primary judge (GPT-5.1) and the second judge "
         r"(Gemini-3.5-Flash) over " + f"{int(summ['paired_question_judgements']):,}" +
         r" paired per-question judgements across " + str(int(summ['models_compared'])) +
         r" evaluated systems.}",
         r"\label{tab:judgeagree}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"\multicolumn{5}{@{}l}{\textit{System-ranking agreement}} \\",
         f"Kendall $\\tau_b$ & \\multicolumn{{4}}{{r}}{{{summ['kendall_tau_b_ranking']:.2f}}} " + r"\\",
         f"Spearman $\\rho$ & \\multicolumn{{4}}{{r}}{{{summ['spearman_rho_ranking']:.2f}}} " + r"\\",
         f"Pearson $r$ (system means) & \\multicolumn{{4}}{{r}}{{{summ['pearson_r_mean_overall']:.2f}}} " + r"\\",
         r"\midrule \multicolumn{5}{@{}l}{\textit{Per-question, by dimension}} \\",
         r"Dimension & Pearson & Spearman & MAE & QWK \\ \midrule"]
    for d in dims:
        qwk = "--" if d[5] is None else f"{d[5]:.2f}"
        L.append(f"{str(d[0]).capitalize()} & {d[2]:.2f} & {d[3]:.2f} & {d[4]:.2f} & {qwk} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("judge_agreement.tex", "\n".join(L) + "\n")


def human_agreement(path):
    """Appendix tables: judge vs human-expert alignment, from the released workbook
    (analysis/human_eval/, single expert rater on the stratified 125-response sample)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    summ = {str(r[0]): r[1] for r in wb["Summary"].iter_rows(values_only=True)}
    dims = [r for r in wb["PerResponseAgreement"].iter_rows(values_only=True)][1:]
    lead = [r for r in wb["Leaderboard"].iter_rows(values_only=True)][1:]
    n_sys, n_q, n_resp = (int(summ[k]) for k in ("systems", "questions", "responses"))
    # formula metric (OV switch): judge side = deterministic reweighting; the
    # workbook carries both (columns/rows appended, keys suffixed _formula)
    formula = OV == "overall_formula"
    fsuf = "_formula" if formula else ""
    jcol = 6 if formula else 3          # Leaderboard judge_mean_25q(_formula)
    r_sys = statistics.correlation([r[1] for r in lead], [r[jcol] for r in lead])

    L = [r"\begin{table}[H]\centering\footnotesize",
         r"\caption{Judge--consensus alignment: two domain experts (authors) "
         r"independently blind-scored all "
         f"{n_resp} responses ({n_sys} systems $\\times$ {n_q} stratified questions) "
         r"under the judge's exact rubric and information set; the human reference is "
         r"their per-response consensus (mean). Bias is judge $-$ consensus (positive "
         r"$=$ judge more lenient).}",
         r"\label{tab:humanagree}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"\multicolumn{5}{@{}l}{\textit{System-ranking agreement (" + str(n_sys) + r" systems)}} \\",
         f"Kendall $\\tau_b$ & \\multicolumn{{4}}{{r}}{{{summ['kendall_tau_b_ranking' + fsuf]:.2f}}} " + r"\\",
         f"Spearman $\\rho$ & \\multicolumn{{4}}{{r}}{{{summ['spearman_rho_ranking' + fsuf]:.2f}}} " + r"\\",
         f"Pearson $r$ (mean overall) & \\multicolumn{{4}}{{r}}{{{r_sys:.2f}}} " + r"\\",
         r"\midrule \multicolumn{5}{@{}l}{\textit{Per-response, by dimension}} \\",
         r"Dimension & Pearson & MAE & Bias & QWK \\ \midrule"]
    for d in dims:
        if str(d[0]).startswith("overall"):
            continue
        L.append(f"{str(d[0]).capitalize()} & {d[2]:.2f} & {d[7]:.2f} & "
                 f"{d[10]:+.2f} & {d[13]:.2f} " + r"\\")
    ov = next(d for d in dims if d[0] == ("overall (judge reweighted)" if formula
                                          else "overall (judge-reported)"))
    L.append(r"\midrule Overall (composite) & "
             f"{ov[2]:.2f} & {ov[7]:.2f} & {ov[10]:+.2f} & -- " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("human_agreement.tex", "\n".join(L) + "\n")

    labels = {
        "qwen3.5-2b P1 -> P4 (full CPT+SFT effect)":
            r"Qwen3.5-2B P1$\to$P4 (full CPT$+$SFT)",
        "qwen3.5-2b P1 -> P6 (RAG effect)":
            r"Qwen3.5-2B P1$\to$P6 ($+$RAG)",
        "gpt-4o -> qwen3.5-9b P6 (frontier parity)":
            r"GPT-4o $\to$ Qwen3.5-9B P6 (frontier parity)",
    }
    con = [r for r in wb["Contrasts"].iter_rows(values_only=True)][1:]
    assert {c[0] for c in con} == set(labels), "unexpected contrast set in workbook"
    L = [r"\begin{table}[H]\centering\footnotesize",
         r"\caption{The paired contrasts the sample was designed around, re-measured "
         r"under the two-expert consensus scoring (" + str(n_q) + r" questions; same "
         r"statistics as \S\ref{sec:stats}). \emph{judge, all}: the primary judge's "
         r"mean difference on the full 233-question benchmark.}",
         r"\label{tab:humancontrasts}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Scorer & $\Delta$ & 95\% CI & $d_z$ \\"]
    jl = "gpt-5.1 (formula)" if formula else "gpt-5.1"
    allcol = 12 if formula else 11
    for lab, tex in labels.items():
        rows = [c for c in con if c[0] == lab and c[1] in ("human", jl)]
        L.append(r"\midrule \multicolumn{4}{@{}l}{\textit{" + tex + r"}} \\")
        for c in rows:
            scorer = "consensus" if c[1] == "human" else "judge"
            L.append(f"{scorer} & {c[5]:+.2f} & [{c[7]:.2f}, {c[8]:.2f}] & {c[10]:.2f} " + r"\\")
        L.append(f"judge, all & {rows[0][allcol]:+.2f} & -- & -- " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("human_contrasts.tex", "\n".join(L) + "\n")


def interrater_agreement(path):
    """Appendix table: inter-rater reliability between the two expert raters
    (analysis/outputs/HPN_INTERRATER_rater1_vs_rater2.xlsx). Columns:
    dimension,n,pearson,spearman,mae,bias_r1_minus_r2,qwk,alpha_2rater,alpha_3way."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = [r for r in wb["InterRater"].iter_rows(values_only=True)][1:]
    L = [r"\begin{table}[H]\centering\footnotesize",
         r"\caption{Inter-rater reliability between the two expert raters on the same "
         r"$125$ responses: Pearson $r$, quadratic-weighted $\kappa$ (QWK), and "
         r"Krippendorff's $\alpha$ (interval). $\alpha$\,(+judge) adds the primary judge "
         r"as a third coder; R1$-$R2 is the raters' mean leniency offset.}",
         r"\label{tab:interrater}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Dimension & Pearson & QWK & $\alpha$ & $\alpha$\,(+judge) & R1$-$R2 \\ \midrule"]
    for d in rows:
        if str(d[0]) == "overall":
            L.append(r"\midrule Overall (composite) & "
                     f"{d[2]:.2f} & -- & {d[7]:.2f} & {d[8]:.2f} & {d[5]:+.2f} " + r"\\")
        else:
            L.append(f"{str(d[0]).capitalize()} & {d[2]:.2f} & {d[6]:.2f} & {d[7]:.2f} & "
                     f"{d[8]:.2f} & {d[5]:+.2f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("human_interrater.tex", "\n".join(L) + "\n")


def main():
    global OUT, SUBS, PAPER, LAYOUT, OV, SIGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=["ieee", "eacl", "jbd"], default="ieee",
                    help="target paper: ieee -> paper/tables (default), "
                         "eacl -> paper_eacl/tables, jbd -> paper_jbd/tables")
    ap.add_argument("--outdir", default=None,
                    help="write .tex here instead of the paper directory; "
                         "--paper still selects layout and cross-references, so "
                         "the artifact release can regenerate every table "
                         "without a paper/ tree present")
    args = ap.parse_args()
    LAYOUT = args.paper
    # jbd shares eacl's content profile entirely: same composite metric, same
    # captions, same set of tables. Only the geometry differs, and that is
    # LAYOUT's job.
    PAPER = "eacl" if args.paper in ("eacl", "jbd") else args.paper
    if PAPER == "eacl":
        OUT = f"paper_{LAYOUT}/tables"
        SUBS = {"sec:methodology": "sec:setup"}
        # EACL primary metric: deterministic rubric aggregate (plan v2 §13).
        # To revert to the judge-reported metric, delete the next line.
        OV, SIGF = "overall_formula", "significance_formula"
    # --outdir changes only *where* output lands, never *what* is emitted.
    if args.outdir:
        OUT = args.outdir
    summ = pd.read_csv(f"{A}/summary_by_variant.csv")
    sig = pd.read_csv(f"{A}/{SIGF}.csv")
    composition()
    training_stats()
    main_results(summ)
    baselines(summ)
    significance(sig)
    # EACL folds variant coverage into the setup-table model roster
    # (tab:models, model_variants); the standalone tab:coverage grid is
    # redundant there and only strands on its own float page, so emit it for
    # ieee only (where model_variants is not produced).
    if PAPER != "eacl":
        completeness(f"{A}/completeness_grid.csv")
    by_category(f"{A}/scores_overall.csv")
    by_dimension(f"{A}/scores_overall.csv")
    cpt_perplexity()
    profiling()
    if PAPER == "eacl":   # eacl and jbd both take this set
        frontier(summ)
        model_variants(f"{A}/completeness_grid.csv")
        # 73 contrast rows: one ACL [p] float page holds them, sn-jnl's 552pt
        # text block does not (~584pt single-spaced, more with the caption), so
        # jbd splits each matrix on the closed-book/open-book family boundary.
        # Part A keeps the original label, so every existing \ref still resolves.
        for st, stem, lab in (("difficulty", "stratified_difficulty", "tab:stratdiff"),
                              ("category", "stratified_category", "tab:stratcat")):
            src = f"{A}/{SIGF}_stratified.csv"
            if LAYOUT == "jbd":
                stratified_matrix(src, st, f"{stem}.tex", lab,
                                  groups=["closed_book"], cont=lab + "b")
                stratified_matrix(src, st, f"{stem}_b.tex", lab + "b",
                                  groups=["open_book", "rag_effect"])
            else:
                stratified_matrix(src, st, f"{stem}.tex", lab)
        agr_suf = "_formula" if OV == "overall_formula" else ""
        judge_agreement(f"{A}/HPN_JUDGE_AGREEMENT_gpt-5.1_vs_gemini-3.5-flash{agr_suf}.xlsx")
        human_agreement(f"{A}/HPN_JUDGE_AGREEMENT_human_vs_gpt-5.1.xlsx")
        interrater_agreement(f"{A}/HPN_INTERRATER_rater1_vs_rater2.xlsx")
        judge_deviation(f"{A}/judge_overall_deviation.csv")
        significance_supp(f"{A}/significance_formula_supplementary.csv")
        threshold_sens(f"{A}/threshold_sensitivity.csv")
        if LAYOUT == "jbd":   # Phase 12 additions; eacl/ieee stay frozen
            equivalence(f"{A}/equivalence_frontier.csv")
    print("done.")


if __name__ == "__main__":
    main()
