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

Input : analysis/outputs/scores_long.csv
Output: analysis/outputs/difficulty_validation.csv
Cited in Appendix A (benchmark pipeline) and Section 4.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

A = Path(__file__).resolve().parent / "outputs"
GRADE = {"easy": 1, "medium": 2, "hard": 3}
RUNG_MODEL = "qwen3.5-9b"  # ladder rung that is itself in the evaluated roster
L2 = 0.01  # weak ridge; keeps never-missed/never-solved items finite


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

    out = pd.DataFrame(rows, columns=["section", "metric", "value"])
    out.to_csv(A / "difficulty_validation.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
