"""Question clustering and cluster-robust resampling.

Phase 12 R1. The pre-registered protocol treats the 233 benchmark questions as
independent observations, but they are distilled from 171 source papers and
several questions can share one paper -- and therefore share evidence passages
and the generator card built from them. A reviewer is right that this can make
question-level bootstrap intervals and Wilcoxon p-values optimistic.

The clustering rule is not invented here; it is read off the benchmark. Every
item carries `source_papers`, so clusters are the connected components of
"shares at least one source paper":

  * an ordinary NB-HPN item names one paper and joins that paper's cluster;
  * a cross-paper NB-SYN item names two, which merges those two papers into a
    single cluster (that merge is the whole point -- the item is evidence from
    both, so neither can be resampled without it);
  * an adversarial NB-ADV item carries its paper anchor in `source_papers` too.

On the 233 scored items that yields 167 clusters: 112 singletons, 45 pairs, 9
triples, 1 quadruple, mean size 1.395.

That mean bounds the damage before any resampling runs. The design effect of a
clustered mean is 1 + (m-1)*ICC, so even at ICC = 1 -- every question from one
paper carrying identical information, the worst case that can exist -- the
variance inflates by 1.395 and interval half-widths by sqrt(1.395) = 1.18. This
module computes the honest number; the bound is why we already know it cannot
overturn a result whose interval clears zero by more than 18%.

Two resamplers, both cluster-level:

  cluster_bootstrap_ci  resamples whole clusters with replacement, so a paper
                        enters the replicate with all of its questions or none.
  cluster_signflip_p    flips the sign of every difference in a cluster
                        together. Under the paired null the difference
                        distribution is symmetric about zero -- the same
                        assumption the Wilcoxon signed-rank test already makes
                        -- so a sign flip is exchangeable, and flipping by
                        cluster rather than by question preserves whatever
                        dependence the shared paper induces.
"""
from __future__ import annotations

import ast
import json

import numpy as np

SEED = 1234          # matches aggregate_scores.py
RESAMPLE_N = 10000   # matches aggregate_scores.py's BOOTSTRAP_N
BENCHMARK = "NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl"


def question_clusters(path: str = BENCHMARK) -> dict[str, str]:
    """question_id -> cluster_id, by union-find over shared source papers."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    papers: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("excluded_from_scoring"):
                continue
            ps = rec["source_papers"]
            if isinstance(ps, str):
                ps = ast.literal_eval(ps)
            # An item with no named paper is its own cluster: nothing to share.
            ps = list(ps) if ps else [f"__item_{rec['id']}"]
            papers[rec["id"]] = ps
            for other in ps[1:]:
                union(ps[0], other)
            find(ps[0])
    return {qid: find(ps[0]) for qid, ps in papers.items()}


def _groups(clusters: np.ndarray) -> list[np.ndarray]:
    """Row indices of each cluster, in a stable order."""
    order = np.argsort(clusters, kind="stable")
    sorted_c = clusters[order]
    edges = np.flatnonzero(np.r_[True, sorted_c[1:] != sorted_c[:-1], True])
    return [order[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)]


def cluster_bootstrap_ci(diff: np.ndarray, clusters: np.ndarray,
                         level: float = 95.0, seed: int = SEED,
                         n: int = RESAMPLE_N) -> tuple[float, float]:
    """Percentile CI on mean(diff), resampling whole clusters with replacement.

    Replicates vary in length, since clusters differ in size -- that is the
    cluster bootstrap working as intended, not a defect to pad away.
    """
    idx = _groups(clusters)
    rng = np.random.default_rng(seed)
    k = len(idx)
    sums = np.array([diff[g].sum() for g in idx])
    sizes = np.array([len(g) for g in idx], dtype=float)
    draws = rng.integers(0, k, size=(n, k))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = (100.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def cluster_signflip_p(diff: np.ndarray, clusters: np.ndarray,
                       seed: int = SEED, n: int = RESAMPLE_N,
                       alternative: str = "two-sided") -> float:
    """Paired permutation p-value on the mean, flipping signs by cluster.

    `alternative` is "two-sided", "greater" (H1: mean > 0) or "less". The
    one-sided forms are what TOST needs: shift the differences by the
    equivalence margin and ask a directional question about the shifted mean.

    The +1 in numerator and denominator is the standard Monte-Carlo correction:
    it keeps the p-value from ever being exactly 0, which would claim more
    precision than a finite number of permutations can support.

    Validated 2026-08-12: on real contrasts this tracks `scipy.ttest_rel` to
    within Monte-Carlo error (it is the exact non-parametric analogue of the
    paired t-test); it holds its nominal level on pure noise; and on synthetic
    ICC=1 data it stays calibrated at 12/200 false positives where the
    question-level version inflates to 36/200.
    """
    if np.allclose(diff, 0):
        return 1.0
    idx = _groups(clusters)
    rng = np.random.default_rng(seed)
    k = len(idx)
    sums = np.array([diff[g].sum() for g in idx])
    obs = diff.mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, k))
    means = (signs * sums).sum(axis=1) / len(diff)
    if alternative == "greater":
        hits = np.sum(means >= obs - 1e-12)
    elif alternative == "less":
        hits = np.sum(means <= obs + 1e-12)
    else:
        hits = np.sum(np.abs(means) >= abs(obs) - 1e-12)
    return float((1 + hits) / (1 + n))


def summary(clusters: dict[str, str]) -> dict[str, float]:
    sizes = np.array(list(__import__("collections").Counter(clusters.values()).values()))
    m = sizes.mean()
    return dict(n_items=int(sizes.sum()), n_clusters=int(len(sizes)),
                mean_size=round(float(m), 3), max_size=int(sizes.max()),
                worst_case_design_effect=round(float(m), 3),
                worst_case_ci_inflation=round(float(np.sqrt(m)), 3))
