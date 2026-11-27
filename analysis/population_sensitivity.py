#!/usr/bin/env python3
"""Question-population sensitivity: does each finding hold on restricted subsets?

The primary analysis scores all 233 questions. This re-runs it on six
restricted populations, defined by labels and audits already committed:

  Test-labeled          benchmark split == "test"                       (192)
  Nondev                split != "dev"                                  (204)
  Without weak quotes   drops the 7 items whose evidence quote is not
                        verbatim in the corpus (evidence_quote_audit.csv) (226)
  Test, no weak quotes  both of the above                               (185)
  Source unflagged      no source paper shared with the SFT data
                        (sft_overlap_audit.csv)                        (164)
  Source flagged        the complement                                  (69)

For each population and each judge it recomputes the 72-contrast primary family
with source-paper cluster sign-flip tests and Benjamini-Hochberg *within the
population* (the same tests as clustered_uncertainty.py), the API equivalence
tests (each model's variant fixed to its full-set primary-judge best, as in
equivalence_tests.py), and the size-vs-full-SFT Spearman trend (scale_trend.py).
It also splits the matched retrieval effect by whether the four retrieved
passages include a source paper of the question (retrieval_provenance.csv), and
the human-rated sample by split.

Reads only committed outputs (scores_long.csv, the judged workbooks, the
benchmark JSONL, and the audit CSVs) -- no model, judge, or retrieval run.

    analysis/.venv/bin/python analysis/population_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clusters as C                                        # noqa: E402
import scale_trend as ST                                    # noqa: E402
from aggregate_scores import (DZ_MIN, FDR_Q, MIN_N,          # noqa: E402
                              PRACTICAL_MIN_DELTA, PRIMARY_CONTRASTS,
                              SIZE_GROUP, cohen_dz)
from threshold_anchor import load_judge                     # noqa: E402

BENCHMARK = "NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl"
METRIC = "overall_formula"
MARGIN = PRACTICAL_MIN_DELTA
APIS = [m for m, g in SIZE_GROUP.items() if g == "api-baseline"]
JUDGES = {"primary": "gpt-5.1", "second": "gemini-3.5-flash"}
TOP_K = 4                     # passages the model receives (rerank top 4)


def populations(scored: list[str], bench: dict, weak: set[str],
                flagged: set[str]) -> dict[str, set[str]]:
    test = {q for q in scored if bench[q]["split"] == "test"}
    return {
        "All scored": set(scored),
        "Test-labeled": test,
        "Nondev": {q for q in scored if bench[q]["split"] != "dev"},
        "Without weak quotes": set(scored) - weak,
        "Test, no weak quotes": test - weak,
        "Source unflagged": set(scored) - flagged,
        "Source flagged": set(scored) & flagged,
    }


def contrast_family(piv: pd.DataFrame, pop: set[str], cl: dict) -> pd.DataFrame:
    rows = []
    for model in sorted(piv.index.get_level_values(0).unique()):
        if SIZE_GROUP.get(model) == "api-baseline":
            continue
        sub = piv.loc[model]
        sub = sub[sub.index.isin(pop)]
        for a, b, _group, _desc in PRIMARY_CONTRASTS:
            if a not in sub or b not in sub:
                continue
            pair = sub[[a, b]].dropna()
            if len(pair) < MIN_N:
                continue
            diff = (pair[b] - pair[a]).to_numpy()
            cids = pair.index.map(cl).to_numpy()
            lo, hi = C.cluster_bootstrap_ci(diff, cids)
            rows.append(dict(base_model=model, contrast=f"{a} vs {b}", n=len(pair),
                             n_clusters=int(len(set(cids))),
                             mean_diff=round(float(diff.mean()), 4),
                             ci95_lo_clustered=round(lo, 4), ci95_hi_clustered=round(hi, 4),
                             cohens_dz=round(cohen_dz(diff), 4),
                             p_perm_clustered=round(C.cluster_signflip_p(diff, cids), 5)))
    out = pd.DataFrame(rows)
    out["p_bh"] = np.round(false_discovery_control(out["p_perm_clustered"].to_numpy(),
                                                   method="bh"), 5)
    out["bh_sig"] = out["p_bh"] <= FDR_Q
    out["practically_sig"] = ((out["mean_diff"].abs() >= PRACTICAL_MIN_DELTA)
                              & (out["cohens_dz"].abs() >= DZ_MIN) & out["bh_sig"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="analysis/outputs")
    ap.add_argument("--scores-long", default="analysis/outputs/scores_long.csv")
    ap.add_argument("--by-model", default="NetBench-LLM/outputs/by_model")
    ap.add_argument("--excluded", default="analysis/excluded_items.csv")
    ap.add_argument("--quote-audit", default="analysis/outputs/evidence_quote_audit.csv")
    ap.add_argument("--sft-audit", default="analysis/outputs/sft_overlap_audit.csv")
    ap.add_argument("--provenance", default="NetBench-RAG/outputs/retrieval_provenance.csv")
    ap.add_argument("--human", default="analysis/human_eval/outputs/human_vs_judge_long.csv")
    args = ap.parse_args()

    bench = {r["id"]: r for r in map(json.loads, open(BENCHMARK, encoding="utf-8"))}
    d = pd.read_csv(args.scores_long)
    d = d[d.dimension == METRIC]
    scored = sorted(d.question_id.unique())
    qa = pd.read_csv(args.quote_audit)
    weak = set(qa.loc[~qa["verbatim"].astype(bool), "item_id"])
    sft = pd.read_csv(args.sft_audit)
    flagged = set(sft.loc[sft["shared_source_paper"] == 1, "id"])
    pops = populations(scored, bench, weak, flagged)
    cl = C.question_clusters()

    excluded = set(pd.read_csv(args.excluded)["id"].astype(str).str.strip())
    second = load_judge(args.by_model, JUDGES["second"], excluded)
    piv = {"primary": d.pivot_table(index=["base_model", "question_id"],
                                    columns="p_id", values="score"),
           "second": second.pivot_table(index=["base_model", "question_id"],
                                        columns="p_id", values=METRIC)}

    # --- (1) the 72-contrast family per population and judge ---
    fam = []
    for judge, p in piv.items():
        for name, pop in pops.items():
            f = contrast_family(p, pop, cl)
            f.insert(0, "judge", judge)
            f.insert(0, "n_questions", len(pop))
            f.insert(0, "population", name)
            fam.append(f)
    fam = pd.concat(fam, ignore_index=True)

    summary = []
    key = lambda x: set(zip(x.base_model[x.practically_sig], x.contrast[x.practically_sig]))
    for (name, judge), f in fam.groupby(["population", "judge"], sort=False):
        full = fam[(fam.population == "All scored") & (fam.judge == judge)]
        prim = key(fam[(fam.population == name) & (fam.judge == "primary")])
        kept, ref = key(f) & key(full), key(full)
        rag = f[f.contrast == "P1 vs P6"]
        cpt = f[f.contrast == "P2 vs P5"]
        summary.append(dict(
            population=name, n_questions=int(f.n_questions.iloc[0]), judge=judge,
            contrasts=len(f), bh_sig=int(f.bh_sig.sum()), practical=int(f.practically_sig.sum()),
            practical_retrieval_p1_p6=int(rag.practically_sig.sum()),
            practical_lora_cpt_p2_p5=int(cpt.practically_sig.sum()),
            retrieval_p1_p6_positive=int((rag.mean_diff > 0).sum()),
            retrieval_p1_p6_over_margin=int((rag.mean_diff > MARGIN).sum()),
            practical_kept_from_all=len(kept), practical_lost=len(ref - kept),
            practical_gained=len(key(f) - ref),
            # same population, this judge against the primary judge
            practical_kept_from_primary=len(key(f) & prim),
            practical_changed_vs_primary=len(key(f) ^ prim)))
    summary = pd.DataFrame(summary)

    # --- (2) API equivalence, variants fixed to the full-set primary best ---
    best = {}
    for model in sorted(piv["primary"].index.get_level_values(0).unique()):
        if SIZE_GROUP.get(model) != "api-baseline":
            best[model] = str(piv["primary"].loc[model].mean().idxmax())
    eq = []
    for judge, p in piv.items():
        for name, pop in pops.items():
            for model, variant in best.items():
                for api in APIS:
                    pair = pd.concat([p.loc[model][variant].rename("open"),
                                      p.loc[api]["BASE-API"].rename("api")], axis=1).dropna()
                    pair = pair[pair.index.isin(pop)]
                    diff = (pair["open"] - pair["api"]).to_numpy()
                    cids = pair.index.map(cl).to_numpy()
                    lo, hi = C.cluster_bootstrap_ci(diff, cids, level=90.0)
                    p_tost = max(C.cluster_signflip_p(diff + MARGIN, cids, alternative="greater"),
                                 C.cluster_signflip_p(diff - MARGIN, cids, alternative="less"))
                    eq.append(dict(population=name, judge=judge,
                                   open_system=f"{model} ({variant})", api=api, n=len(pair),
                                   mean_diff=round(float(diff.mean()), 4),
                                   ci90_lo=round(lo, 4), ci90_hi=round(hi, 4),
                                   p_tost=round(p_tost, 5), equivalent=p_tost < 0.05))
    eq = pd.DataFrame(eq)

    # --- (3) size vs full-SFT gain, per population, judge, and roster ---
    models = [m for m in ST.ORDER if m in piv["primary"].index.get_level_values(0)]
    rosters = {"All eight": models,
               "Without Qwen3.5-9B": [m for m in models if m != "qwen3.5-9b"],
               "Without Qwen family": [m for m in models if not m.startswith("qwen")]}
    sc = []
    for judge, p in piv.items():
        for name, pop in pops.items():
            for roster, ms in rosters.items():
                x = np.log2([ST.NUM[m] for m in ms])
                y = []
                for m in ms:
                    sub = p.loc[m]
                    pair = sub[sub.index.isin(pop)][["P1", "P3"]].dropna()
                    y.append(float((pair["P3"] - pair["P1"]).mean()))
                y = np.array(y)
                p_exact, n_perm = ST.exact_permutation_p(x, y)
                sc.append(dict(population=name, judge=judge, roster=roster, n_models=len(ms),
                               spearman_rho=round(ST.spearman(x, y), 4),
                               exact_p=round(p_exact, 5), n_permutations=n_perm))
    sc = pd.DataFrame(sc)

    # --- (4) matched retrieval effect by whether a source paper was retrieved ---
    prov = pd.read_csv(args.provenance)
    top = prov[prov["rank"] <= TOP_K].groupby("question_id")["paper_id"].apply(set)
    hit = {q for q in scored if top.get(q, set()) & set(bench[q]["source_papers"])}
    rh = []
    for model in models:
        for subset, pop in (("source paper retrieved", hit), ("not retrieved", set(scored) - hit)):
            sub = piv["primary"].loc[model]
            pair = sub[sub.index.isin(pop)][["P1", "P6"]].dropna()
            diff = (pair["P6"] - pair["P1"]).to_numpy()
            lo, hi = C.cluster_bootstrap_ci(diff, pair.index.map(cl).to_numpy())
            rh.append(dict(base_model=model, subset=subset, n=len(pair), contrast="P1 vs P6",
                           mean_diff=round(float(diff.mean()), 4),
                           ci95_lo_clustered=round(lo, 4), ci95_hi_clustered=round(hi, 4)))
    rh = pd.DataFrame(rh)

    # --- (5) the human-rated sample by split ---
    h = pd.read_csv(args.human)
    h = h[h.base_model == "qwen3.5-2b"].copy()
    h["split"] = h.question_id.map(lambda q: bench[q]["split"])
    hs = []
    for split in sorted(h.split.unique()):
        s = h[h.split == split]
        for scorer, col in (("human consensus", "overall_human"),
                            ("primary judge", "overall_weighted")):
            w = s.pivot_table(index="question_id", columns="p_id", values=col)
            for b in ("P4", "P6"):
                diff = (w[b] - w["P1"]).dropna()
                hs.append(dict(split=split, scorer=scorer, base_model="qwen3.5-2b",
                               contrast=f"P1 vs {b}", n=len(diff),
                               mean_diff=round(float(diff.mean()), 4)))
    hs = pd.DataFrame(hs)

    os.makedirs(args.out, exist_ok=True)
    for frame, name in ((fam, "population_contrasts.csv"), (summary, "population_summary.csv"),
                        (eq, "population_api_equivalence.csv"), (sc, "population_scale_trend.csv"),
                        (rh, "retrieval_source_hit_effects.csv"), (hs, "human_split_effects.csv")):
        frame.to_csv(os.path.join(args.out, name), index=False)

    print(f"populations: " + ", ".join(f"{k} {len(v)}" for k, v in pops.items()))
    print(f"source paper among the top {TOP_K}: {len(hit)} of {len(scored)}\n")
    print(summary[["population", "judge", "n_questions", "practical",
                   "practical_retrieval_p1_p6", "practical_lora_cpt_p2_p5"]].to_string(index=False))


if __name__ == "__main__":
    main()
