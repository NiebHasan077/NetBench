#!/usr/bin/env python3
"""Model-independent validation of the behavioral difficulty grades.

The grades are assigned by a three-model capability ladder whose rungs share
model families with the evaluated roster (and one rung, Qwen3.5-9B, is itself
evaluated), so per-system monotonicity is partly expected by construction for
ladder-adjacent systems. Two checks that do not inherit that circularity:

1. API-only: per-item mean overall (formula metric) across the three frontier
   API baselines, which played no role in construction or calibration,
   correlated with the assigned grade.
2. IRT: a Rasch model fit jointly over the full 233-item x 63-system response
   matrix (correctness dichotomized at >=4), so item difficulty is a latent
   property estimated from all systems at once rather than from any ladder.
   Robustness: alternative dichotomization thresholds, and a refit excluding
   the rung model's (Qwen3.5-9B) eight variants.

R4 (Phase 12) adds the family-level version the reviewer asked for, in two
forms, because the one they named is the weaker of the two:

3. Leave-one-family-out: refit with every Gemma system removed, then every
   Llama, then every Qwen. This is what was requested, and it is worth
   reporting -- but each refit keeps 41 to 47 of the 63 systems, so a high
   correlation with the full fit is close to arithmetic. It bounds the
   influence of one family; it does not show difficulty is family-independent.

4. Per-family fits, which do: estimate item difficulty from each family alone
   and correlate the three estimates with each other. These fits share **no
   system at all**, so agreement between them cannot come from overlapping
   data. If a Gemma-only difficulty ordering and a Qwen-only one agree, the
   ordering belongs to the items. The price is that a 16-to-22-system fit has
   items every one of its systems answered alike, whose difficulty is bounded
   only by the ridge; those are counted and reported rather than dropped.

Input : analysis/outputs/scores_long.csv
Output: analysis/outputs/difficulty_validation.csv
Cited in Appendix A (benchmark pipeline) and Section 4.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

A = Path(__file__).resolve().parent / "outputs"
GRADE = {"easy": 1, "medium": 2, "hard": 3}
RUNG_MODEL = "qwen3.5-9b"  # ladder rung that is itself in the evaluated roster
L2 = 0.01  # weak ridge; keeps never-missed/never-solved items finite
# The calibration ladder took one rung from each of these three, so leaving a
# family out removes that rung's influence on the grades entirely.
FAMILIES = ("gemma", "llama", "qwen")
# Split-half ceiling: 8 systems per fit, the largest size at which the smallest
# family (Llama, 16 systems) can supply two disjoint halves. 50 draws averages
# out which systems happen to land on which side.
HALF, DRAWS, SEED = 8, 50, 1234


def family_of(system: str) -> str:
    """Vendor family of a `base_model/variant_run` system id; APIs are "api"."""
    base = system.split("/")[0]
    return next((f for f in FAMILIES if base.startswith(f)), "api")


def fit_rasch(mat: np.ndarray) -> np.ndarray:
    """JMLE Rasch fit; mat is systems x items in {0,1}. Returns item difficulty b."""
    n_sys, n_item = mat.shape

    def nll(params):
        theta = params[:n_sys]
        b = params[n_sys:]
        z = theta[:, None] - b[None, :]
        # log-likelihood of Bernoulli with logit z, numerically stable
        ll = mat * z - np.logaddexp(0.0, z)
        return -ll.sum() + L2 * (theta @ theta + b @ b)

    def grad(params):
        theta = params[:n_sys]
        b = params[n_sys:]
        z = theta[:, None] - b[None, :]
        p = 1.0 / (1.0 + np.exp(-z))
        resid = mat - p
        return np.concatenate([
            -resid.sum(axis=1) + 2 * L2 * theta,
            resid.sum(axis=0) + 2 * L2 * b,
        ])

    x0 = np.zeros(n_sys + n_item)
    # The objective is strictly convex (Bernoulli NLL + ridge), so the minimiser
    # is unique and it is worth actually reaching it. L-BFGS-B's default gtol of
    # 1e-5 stopped with b accurate to only ~3e-3, which left the fit dependent on
    # the arithmetic of the machine that ran it.
    res = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
    assert res.success, res.message
    b = res.x[n_sys:]
    b = b - b.mean()  # center: identification is up to a shift
    # Round to the precision the fit actually resolves (~1e-6 after the above).
    # Downstream this is rank-correlated against the difficulty grades, and
    # Spearman gives distinct ranks to values that differ in the 8th decimal —
    # so without this, items whose difficulty is indistinguishable get ordered by
    # floating-point noise and rho moves by ±0.003 between machines. Rounding
    # makes them tied, which is what they are, and ties get averaged ranks.
    return np.round(b, 6)


def main() -> None:
    long = pd.read_csv(A / "scores_long.csv")
    long["system"] = long["base_model"] + "/" + long["variant_run"]
    grades = (long[["question_id", "difficulty"]].drop_duplicates()
              .set_index("question_id")["difficulty"].map(GRADE))

    rows = []

    # -- 1. API-only association (fully ladder-independent) --
    api = long[(long["p_id"] == "BASE-API") &
               (long["dimension"] == "overall_formula")]
    api_mean = api.groupby("question_id")["score"].mean()
    g = grades.loc[api_mean.index]
    rho, p = spearmanr(g, api_mean)
    rows.append(("api_only", "n_items", len(api_mean)))
    rows.append(("api_only", "n_systems", api["system"].nunique()))
    rows.append(("api_only", "spearman_rho", round(rho, 3)))
    rows.append(("api_only", "spearman_p", f"{p:.2e}"))
    for name, code in GRADE.items():
        rows.append(("api_only", f"mean_overall_{name}",
                     round(api_mean[g == code].mean(), 2)))

    # -- 2. Rasch IRT over the full response matrix --
    corr = long[long["dimension"] == "correctness"]
    piv = corr.pivot_table(index="system", columns="question_id",
                           values="score", aggfunc="first")
    assert piv.notna().all().all(), "response matrix has holes"
    g_items = grades.loc[piv.columns]

    b_main = fit_rasch((piv.values >= 4).astype(float))
    rho_b, p_b = spearmanr(g_items, b_main)
    rows.append(("irt", "n_items", piv.shape[1]))
    rows.append(("irt", "n_systems", piv.shape[0]))
    rows.append(("irt", "dichotomization", "correctness>=4"))
    rows.append(("irt", "spearman_rho_b_vs_grade", round(rho_b, 3)))
    rows.append(("irt", "spearman_p", f"{p_b:.2e}"))
    for name, code in GRADE.items():
        rows.append(("irt", f"mean_b_{name}",
                     round(b_main[(g_items == code).values].mean(), 2)))

    # -- 3. Robustness: thresholds and rung-model exclusion --
    for label, thresh in [("correctness>=3", 3), ("correctness=5", 5)]:
        b = fit_rasch((piv.values >= thresh).astype(float))
        rows.append(("irt_robustness", f"spearman_{label}",
                     round(spearmanr(g_items, b)[0], 3)))

    keep = ~piv.index.str.startswith(RUNG_MODEL + "/")
    b_drop = fit_rasch((piv.values[keep] >= 4).astype(float))
    rows.append(("irt_robustness", "n_systems_excl_rung", int(keep.sum())))
    rows.append(("irt_robustness", "pearson_b_full_vs_excl_rung",
                 round(np.corrcoef(b_main, b_drop)[0, 1], 3)))
    rows.append(("irt_robustness", "spearman_excl_rung",
                 round(spearmanr(g_items, b_drop)[0], 3)))

    # -- 4. R4: leave-one-family-out, and the disjoint per-family fits --
    resp = (piv.values >= 4).astype(float)
    fams = np.array([family_of(s) for s in piv.index])
    b_by_family = {}
    for f in FAMILIES:
        out_mask = fams != f
        b_lofo = fit_rasch(resp[out_mask])
        rows.append(("irt_lofo", f"n_systems_excl_{f}", int(out_mask.sum())))
        rows.append(("irt_lofo", f"pearson_b_full_vs_excl_{f}",
                     round(np.corrcoef(b_main, b_lofo)[0, 1], 3)))
        rows.append(("irt_lofo", f"spearman_grade_excl_{f}",
                     round(spearmanr(g_items, b_lofo)[0], 3)))

        in_mask = fams == f
        b_f = fit_rasch(resp[in_mask])
        b_by_family[f] = b_f
        # Items this family answered uniformly: their difficulty carries no
        # information from this fit and is held finite only by the ridge.
        col = resp[in_mask].sum(axis=0)
        rows.append(("irt_family_only", f"n_systems_{f}", int(in_mask.sum())))
        rows.append(("irt_family_only", f"uniform_items_{f}",
                     int(((col == 0) | (col == in_mask.sum())).sum())))
        rows.append(("irt_family_only", f"spearman_grade_{f}",
                     round(spearmanr(g_items, b_f)[0], 3)))
        for name, code in GRADE.items():
            rows.append(("irt_family_only", f"mean_b_{name}_{f}",
                         round(b_f[(g_items == code).values].mean(), 2)))
    for a, b in itertools.combinations(FAMILIES, 2):
        rows.append(("irt_family_only", f"pearson_b_{a}_vs_{b}",
                     round(np.corrcoef(b_by_family[a], b_by_family[b])[0, 1], 3)))
        rows.append(("irt_family_only", f"spearman_b_{a}_vs_{b}",
                     round(spearmanr(b_by_family[a], b_by_family[b])[0], 3)))

    # -- 5. R4: is the cross-family agreement above high or low? --
    # On its own a correlation of ~0.8 between two family fits is not
    # interpretable: no fit of this size reproduces itself perfectly, so some of
    # the gap from 1.0 is estimation noise rather than family disagreement. The
    # ceiling is measured, not assumed -- split ONE family into two disjoint
    # halves and correlate their fits. Both sides then use HALF systems, so the
    # within- and between-family numbers are estimated at matched precision and
    # differ only in whether the two halves come from the same family.
    fam_idx = {f: np.flatnonzero(fams == f) for f in FAMILIES}
    rng = np.random.default_rng(SEED)
    within, cross = {}, {}
    for f in FAMILIES:
        vals = [np.corrcoef(*[fit_rasch(resp[s]) for s in
                              np.split(rng.permutation(fam_idx[f])[:2 * HALF], 2)])[0, 1]
                for _ in range(DRAWS)]
        within[f] = float(np.mean(vals))
        rows.append(("irt_splithalf", f"within_{f}", round(within[f], 3)))
    for a, b in itertools.combinations(FAMILIES, 2):
        vals = [np.corrcoef(fit_rasch(resp[rng.permutation(fam_idx[a])[:HALF]]),
                            fit_rasch(resp[rng.permutation(fam_idx[b])[:HALF]]))[0, 1]
                for _ in range(DRAWS)]
        cross[(a, b)] = float(np.mean(vals))
        rows.append(("irt_splithalf", f"cross_{a}_{b}", round(cross[(a, b)], 3)))
    rows.append(("irt_splithalf", "systems_per_fit", HALF))
    rows.append(("irt_splithalf", "draws", DRAWS))
    rows.append(("irt_splithalf", "within_mean", round(np.mean(list(within.values())), 3)))
    rows.append(("irt_splithalf", "cross_mean", round(np.mean(list(cross.values())), 3)))

    out = pd.DataFrame(rows, columns=["section", "metric", "value"])
    out.to_csv(A / "difficulty_validation.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
