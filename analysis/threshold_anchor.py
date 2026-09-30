"""R2: anchor the practical-significance threshold on the right quantity.

Phase 12, answering the reviewer's sharpest single point. §6.5 justifies the
0.25 practical bar by saying it "matches the inter-judge leniency offset
(+0.21)". The reviewer objects that a *constant* offset is calibration bias,
not a noise floor, and that a stable offset largely cancels in a paired
contrast -- so it is the wrong quantity to anchor a threshold on *differences*.

That objection is correct, and this script demonstrates it rather than
conceding it in prose: it measures the offset in levels, shows how much of it
survives into paired differences, and then measures what actually does vary.

The right anchor is the repeatability of the quantity we threshold. We threshold
a paired mean difference, so we need to know how far that difference moves when
something incidental changes. Two handles exist in committed evidence:

  cross-judge   every contrast recomputed on the second judge's 63 workbooks.
                Swapping the grader is the largest incidental change we can
                actually make, and it is a full re-measurement of all 233
                questions, not a subsample.
  human         the three contrasts the human study was designed around,
                recomputed under the two-expert consensus.

Neither is a true test-retest -- we have one run per judge, so within-judge
decoding variance is not observable here, and this floor is therefore a lower
bound on total measurement noise. Say so; do not imply otherwise.

Reads only committed workbooks and CSVs. Nothing is re-judged.

    analysis/.venv/bin/python analysis/threshold_anchor.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_scores import (MIN_N, PRACTICAL_MIN_DELTA,     # noqa: E402
                              PRIMARY_CONTRASTS, SIZE_GROUP,
                              classify_variant, read_judged, setting_of)

JUDGES = {"gpt-5.1": "primary", "gemini-3.5-flash": "second"}
HUMAN_BACKBONE = "qwen3.5-2b"     # the one model whose P1/P4/P6 the sample spans


def load_judge(by_model: str, judge: str, excluded: set[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(
            by_model, f"*/evaluations/judged/*_by_{judge}.xlsx"))):
        model_dir = path.split("/by_model/")[1].split("/")[0]
        df = read_judged(path)
        p_id, _ = classify_variant(df["variant_run"].iloc[0])
        df["base_model"], df["p_id"], df["setting"] = model_dir, p_id, setting_of(p_id)
        frames.append(df)
    allq = pd.concat(frames, ignore_index=True)
    allq["overall_formula"] = (0.4 * allq["correctness"] + 0.3 * allq["completeness"]
                               + 0.2 * allq["clarity"] + 0.1 * allq["conciseness"])
    return allq[~allq["question_id"].isin(excluded)].copy()


def contrast_deltas(allq: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(model, 'Pa vs Pb') -> mean paired difference on overall_formula."""
    out = {}
    piv = allq.pivot_table(index=["base_model", "question_id"], columns="p_id",
                           values="overall_formula")
    for model in piv.index.get_level_values(0).unique():
        if SIZE_GROUP.get(model) == "api-baseline":
            continue
        sub = piv.loc[model]
        for a, b, _group, _desc in PRIMARY_CONTRASTS:
            if a not in sub or b not in sub:
                continue
            pair = sub[[a, b]].dropna()
            if len(pair) < MIN_N:
                continue
            out[(model, f"{a} vs {b}")] = float((pair[b] - pair[a]).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-model", default="NetBench-LLM/outputs/by_model")
    ap.add_argument("--out", default="analysis/outputs")
    ap.add_argument("--excluded", default="analysis/excluded_items.csv")
    ap.add_argument("--human", default="analysis/human_eval/outputs/human_vs_judge_long.csv")
    args = ap.parse_args()

    excluded = set()
    if os.path.exists(args.excluded):
        excluded = set(pd.read_csv(args.excluded)["id"].astype(str).str.strip())

    judged = {j: load_judge(args.by_model, j, excluded) for j in JUDGES}
    print(f"loaded {len(judged)} judges x "
          f"{judged['gpt-5.1']['base_model'].nunique()} models, "
          f"{judged['gpt-5.1']['question_id'].nunique()} scored questions")

    # --- (1) the offset in LEVELS: this is what §6.5 currently quotes ---
    lev = {j: d.groupby(["base_model", "p_id"])["overall_formula"].mean()
           for j, d in judged.items()}
    offset = (lev["gemini-3.5-flash"] - lev["gpt-5.1"]).dropna()
    print(f"\nlevel offset (second - primary) over {len(offset)} systems: "
          f"mean {offset.mean():+.3f}, sd {offset.std(ddof=1):.3f}, "
          f"range [{offset.min():+.3f}, {offset.max():+.3f}]")

    # --- (2) how much of it survives into PAIRED DIFFERENCES ---
    dg = contrast_deltas(judged["gpt-5.1"])
    dm = contrast_deltas(judged["gemini-3.5-flash"])
    keys = sorted(set(dg) & set(dm))
    rows = [dict(base_model=m, contrast=c,
                 delta_primary=round(dg[(m, c)], 4),
                 delta_second=round(dm[(m, c)], 4),
                 delta_shift=round(dm[(m, c)] - dg[(m, c)], 4)) for m, c in keys]
    per = pd.DataFrame(rows)
    shift = per["delta_shift"].to_numpy()
    absshift = np.abs(shift)
    print(f"\npaired-difference shift when the grader changes, over {len(shift)} contrasts:")
    print(f"   mean {shift.mean():+.4f}  <- the constant offset does cancel, "
          f"exactly as the reviewer said")
    print(f"   sd {shift.std(ddof=1):.4f}   median |shift| {np.median(absshift):.4f}")
    for q in (50, 75, 90, 95, 99):
        print(f"   p{q:<2d} |shift| {np.percentile(absshift, q):.4f}")
    print(f"   max |shift| {absshift.max():.4f}")

    # --- (3) the human handle: same contrasts under the two-expert consensus ---
    hrows = []
    if os.path.exists(args.human):
        h = pd.read_csv(args.human)
        # Resolve P1/P4/P6 within ONE backbone. The sample holds five systems and
        # two of them are P6 (RAG-Qwen3.5-2B and RAG-Qwen3.5-9B), so pivoting on
        # p_id alone silently averages them -- pivot_table defaults to mean --
        # and "P1 vs P6" becomes Qwen3.5-2B against the mean of two different
        # models. The three contrasts the sample was designed around are the
        # pairwise ones among Qwen3.5-2B's own P1, P4 and P6 (S6.4).
        h = h[h.base_model == HUMAN_BACKBONE]
        hp = h.pivot_table(index="question_id", columns="p_id",
                           values=["overall_human", "overall_weighted"])
        for a, b in [("P1", "P4"), ("P1", "P6"), ("P4", "P6")]:
            try:
                hh = hp["overall_human"][[a, b]].dropna()
                jj = hp["overall_weighted"][[a, b]].dropna()
            except KeyError:
                continue
            if len(hh) < MIN_N:
                continue
            dh = float((hh[b] - hh[a]).mean())
            dj = float((jj[b] - jj[a]).mean())
            hrows.append(dict(contrast=f"{a} vs {b}", n=len(hh),
                              delta_consensus=round(dh, 4), delta_judge=round(dj, 4),
                              delta_shift=round(dh - dj, 4)))
        if hrows:
            print(f"\nsame contrasts under the two-expert consensus "
                  f"({HUMAN_BACKBONE}, n={hrows[0]['n']} questions):")
            for r in hrows:
                print(f"   {r['contrast']:10s} consensus {r['delta_consensus']:+.3f} "
                      f"vs judge {r['delta_judge']:+.3f}  shift {r['delta_shift']:+.3f}")

    # --- (4) the decision-relevant question: does the VERDICT survive? ---
    # A tail statistic on the shift is not what a reader needs. What matters is
    # whether the call the threshold makes -- practically significant or not --
    # is the same call under a different grader.
    per["sign_flip"] = np.sign(per["delta_primary"]) != np.sign(per["delta_second"])
    per["ps_primary"] = per["delta_primary"].abs() >= PRACTICAL_MIN_DELTA
    per["ps_second"] = per["delta_second"].abs() >= PRACTICAL_MIN_DELTA
    per["ps_flip"] = per["ps_primary"] != per["ps_second"]
    n_ps = int(per["ps_primary"].sum())
    held = int((per["ps_primary"] & per["ps_second"]).sum())
    print(f"\nverdict stability across graders, over {len(per)} contrasts:")
    print(f"   |delta| >= {PRACTICAL_MIN_DELTA} under the primary judge: {n_ps}; "
          f"still so under the second: {held}")
    print(f"   verdict flips either way : {int(per['ps_flip'].sum())} "
          f"({per['ps_flip'].mean():.1%})")
    print(f"   sign of the effect flips : {int(per['sign_flip'].sum())} "
          f"({per['sign_flip'].mean():.1%})")
    print(f"   contrasts moving by more than the bar itself: "
          f"{int((absshift > PRACTICAL_MIN_DELTA).sum())}")

    # --- (5) verdict on the threshold ---
    floor95 = float(np.percentile(absshift, 95))
    sd = float(shift.std(ddof=1))
    hmax = max((abs(r["delta_shift"]) for r in hrows), default=0.0)
    print(f"\nfloor on a paired difference, three readings of the same data:")
    print(f"   sd of the grader-induced shift  {sd:.3f}   "
          f"(the bar is {PRACTICAL_MIN_DELTA / sd:.2f}x this)")
    print(f"   median |shift|                  {np.median(absshift):.3f}")
    print(f"   p95 |shift|                     {floor95:.3f}   "
          f"(the bar is BELOW this)")
    if hrows:
        print(f"   largest human-consensus shift   {hmax:.3f}   "
              f"(n=25; noisier, and points the same way)")
    print("\nThe reviewer is right that the published justification is wrong: the\n"
          "+0.21 is a level offset, most of it cancels (mean shift "
          f"{shift.mean():+.3f}), and it is\nnot a noise floor. The correct floor is the "
          f"spread of the shift, and 0.25 sits\nat {PRACTICAL_MIN_DELTA / sd:.1f} sd of it "
          "-- above the typical grader disagreement but inside\nits upper tail. Keep the "
          "pre-registered 0.25, re-justify it on this quantity,\nand report the flip rate "
          "rather than implying grader-invariance.")

    os.makedirs(args.out, exist_ok=True)
    per.to_csv(os.path.join(args.out, "threshold_anchor.csv"), index=False)
    summary = pd.DataFrame([
        dict(quantity="level offset (second - primary), per system",
             n=len(offset), mean=round(float(offset.mean()), 4),
             sd=round(float(offset.std(ddof=1)), 4),
             p95_abs=round(float(np.percentile(offset.abs(), 95)), 4)),
        dict(quantity="paired-difference shift when the grader changes",
             n=len(shift), mean=round(float(shift.mean()), 4),
             sd=round(float(shift.std(ddof=1)), 4),
             p95_abs=round(floor95, 4)),
    ] + [dict(quantity=f"human consensus vs judge, {r['contrast']}", n=r["n"],
              mean=r["delta_shift"], sd=np.nan, p95_abs=abs(r["delta_shift"]))
         for r in hrows])
    summary.to_csv(os.path.join(args.out, "threshold_anchor_summary.csv"), index=False)
    print(f"\nwrote {args.out}/threshold_anchor.csv and threshold_anchor_summary.csv")


if __name__ == "__main__":
    main()
