#!/usr/bin/env python3
"""95% intervals for response-level agreement between the two LLM judges.

``HPN_JUDGE_AGREEMENT_gpt-5.1_vs_gemini-3.5-flash_formula.md`` reports Pearson r
and MAE over the 14,679 paired judgements with no interval, while the
human-vs-judge report has one. This supplies the matching interval, computed
the way the human study computes its own: a percentile bootstrap resampling
questions, since the 63 responses to one question share its difficulty and are
not independent. B = 10,000, seed 1234.

Pairs come from ``judge_agreement.py``'s own loader, exclusion list, and
``overall_formula`` composite, and the script stops unless its full-sample
estimates reproduce that report's.

    analysis/.venv/bin/python analysis/judge_agreement_ci.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "NetBench-LLM", "evaluation"))
from judge_agreement import ALL_FIELDS, _load_judge_set, compute_agreement  # noqa: E402

BY_MODEL = os.path.join(ROOT, "NetBench-LLM", "outputs", "by_model")
EXCLUDED = os.path.join(ROOT, "analysis", "excluded_items.csv")
OUT = os.path.join(ROOT, "analysis", "outputs", "judge_agreement_ci.csv")
SEED = 1234
BOOTSTRAP_N = 10000


def main() -> None:
    with open(EXCLUDED) as fh:
        excluded = {row["id"].strip() for row in csv.DictReader(fh)}
    set_a, _ = _load_judge_set(os.path.join(BY_MODEL, "**", "hpn_judged_*_by_gpt-5.1.xlsx"),
                               excluded, "overall_formula")
    set_b, _ = _load_judge_set(os.path.join(BY_MODEL, "**", "hpn_judged_*_by_gemini-3.5-flash.xlsx"),
                               excluded, "overall_formula")
    reported = compute_agreement(set_a, set_b)["per_question"]

    qids, a, b = [], {f: [] for f in ALL_FIELDS}, {f: [] for f in ALL_FIELDS}
    for m in sorted(set(set_a) & set(set_b)):
        for rid in sorted(set(set_a[m]) & set(set_b[m])):
            qids.append(rid)
            for f in ALL_FIELDS:
                a[f].append(set_a[m][rid][f])
                b[f].append(set_b[m][rid][f])

    # Per-question sufficient statistics make every bootstrap replicate a sum
    # over resampled questions, exactly equal to concatenating their responses.
    codes, uniq = pd.factorize(pd.Series(qids), sort=True)
    k = len(uniq)
    draws = np.random.default_rng(SEED).integers(0, k, size=(BOOTSTRAP_N, k))

    def per_q(v: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=v, minlength=k)

    rows = []
    for f in ALL_FIELDS:
        x, y = np.asarray(a[f]), np.asarray(b[f])
        stats = [per_q(np.ones_like(x)), per_q(x), per_q(y), per_q(x * x),
                 per_q(y * y), per_q(x * y), per_q(np.abs(x - y))]
        full = [s.sum() for s in stats]
        boot = [s[draws].sum(axis=1) for s in stats]

        def pearson(n, sx, sy, sxx, syy, sxy):
            return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))

        r_full, mae_full = pearson(*full[:6]), full[6] / full[0]
        if not (np.isclose(r_full, reported[f]["pearson"]) and np.isclose(mae_full, reported[f]["mae"])):
            raise SystemExit(f"{f}: estimates differ from judge_agreement.py "
                             f"(r {r_full} vs {reported[f]['pearson']}, "
                             f"MAE {mae_full} vs {reported[f]['mae']})")
        r_boot, mae_boot = pearson(*boot[:6]), boot[6] / boot[0]
        rows.append({"dimension": f, "n": int(full[0]), "n_questions": k,
                     "pearson": round(float(r_full), 3),
                     "pearson_lo": round(float(np.percentile(r_boot, 2.5)), 3),
                     "pearson_hi": round(float(np.percentile(r_boot, 97.5)), 3),
                     "mae": round(float(mae_full), 3),
                     "mae_lo": round(float(np.percentile(mae_boot, 2.5)), 3),
                     "mae_hi": round(float(np.percentile(mae_boot, 97.5)), 3)})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"✅ {rows[0]['n']} paired judgements over {k} questions -> {OUT}")
    for r in rows:
        print(f"   {r['dimension']:13s} r {r['pearson']:.3f} [{r['pearson_lo']:.3f}, {r['pearson_hi']:.3f}]"
              f"   MAE {r['mae']:.3f} [{r['mae_lo']:.3f}, {r['mae_hi']:.3f}]")


if __name__ == "__main__":
    main()
