# Judge-reported vs deterministic `overall`: audit and metric decision

**Decision (2026-07-17, professor-approved):** the EACL paper
(`paper_eacl/`, branch `eacl-2027`) reports the **deterministic rubric
aggregate** as its primary metric:

```
overall_formula = 0.4*correctness + 0.3*completeness + 0.2*clarity + 0.1*conciseness
```

computed from the judge's integer dimension scores. The judge-reported
`overall` remains released with every judged sheet and is reported in the
paper's App B.2 as an audited measurement finding. The IEEE paper (`paper/`,
`main`) is **not** switched this round; `analysis/` therefore emits both
metrics side by side (`significance.csv` = judge-reported, used by `paper/`;
`significance_formula*.csv` = deterministic, used by `paper_eacl/`).
Decision recorded here; the accompanying paper reports the deterministic metric.

**Release note (2026-07-18):** every judged workbook under
`NetBench-LLM/outputs/by_model/*/evaluations/judged/` now carries an
`overall_formula` column (appended after the last original column, with a
provenance row in each workbook's Metadata sheet) so the paper's primary
metric is directly visible next to the judge-reported value. Backfilled by
`NetBench-LLM/evaluation/backfill_overall_formula.py`; both judging pipelines
(`NetBench-LLM` and `NetBench-RAG` `judge_responses.py`) emit the column
natively for future runs. Snapshot branches are untouched.

Everything below is the audit that motivated the decision. It was produced by
re-running `aggregate_scores.py`'s own contrast/BH/practical procedure twice
(judge-reported vs deterministic) on all judged sheets; the judge-reported
pass reproduced the committed `significance.csv` exactly before any
comparison was made.

## 1. Deviation audit (overall_judge - overall_formula), full data

Scope: 63 judged sheets, 14,679 judgements with complete rubric rows
(post-audit, 233 questions).

| slice | n | MAD | signed bias | exact % | >0.25 % | max abs dev |
|---|---|---|---|---|---|---|
| **all** | 14,679 | 0.100 | +0.0048 | 26.5 | 3.9 | 0.60 |
| closed_book | 9,553 | 0.096 | +0.0322 | 25.8 | 3.0 | 0.60 |
| open_book | 5,126 | 0.108 | -0.0464 | 27.7 | 5.4 | 0.40 |
| API baselines | 699 | 0.128 | -0.0808 | 16.2 | 12.3 | 0.40 |
| P1 | 1,864 | 0.121 | +0.0296 | 17.8 | 3.9 | 0.40 |
| P2 | 1,864 | 0.085 | +0.0446 | 29.9 | 1.5 | 0.50 |
| P3 | 1,864 | 0.088 | +0.0464 | 28.3 | 2.0 | 0.50 |
| P4 | 1,398 | 0.090 | +0.0483 | 26.8 | 1.8 | 0.60 |
| P5 | 1,864 | 0.085 | +0.0385 | 30.3 | 2.1 | 0.50 |
| P6 | 1,864 | 0.113 | -0.0468 | 24.3 | 6.8 | 0.40 |
| P7 | 1,398 | 0.100 | -0.0353 | 31.8 | 4.1 | 0.40 |
| P8 | 1,864 | 0.108 | -0.0541 | 28.1 | 5.1 | 0.40 |

Per-model MAD range: 0.091-0.147. The paper's earlier 12-sheet sampled audit
(MAD ~0.10, ~28% exact, ~4% > 0.25, max 0.60) is confirmed on the full data.

## 2. Mechanism: the judge drifts toward the UNWEIGHTED mean

corr(deviation, unweighted_mean - formula) = 0.57; among nonzero deviations,
87.6% point toward the unweighted mean. The judge's `overall` is not noisy
arithmetic around the instructed 0.4/0.3/0.2/0.1 weighting - it slides
toward a plain average of the four dimensions. Answers with high correctness
but low conciseness (open-book and API answers) get their overall deflated
(open-book bias -0.046, API -0.081) while closed-book answers are slightly
inflated (+0.032). Judge-reported numbers therefore *understate* the RAG
effect and API strength; the deviation is NOT symmetric across the two sides
of cross-setting contrasts (it is, to good approximation, within-setting).

## 3. Verdict diff (primary family, deterministic vs judge-reported)

- BH-significance flips: **7**, all False->True, all below the practical bar.
- Practical-significance flips: **2**, both at the 0.25 boundary.

| model | contrast | mean_diff old->new | dz old->new | bh | practical |
|---|---|---|---|---|---|
| llama-3.2-1b | P2 vs P5 | +0.088 -> +0.091 | 0.13 -> 0.13 | F -> T | F -> F |
| gemma-3-1b | P1 vs P4 | +0.076 -> +0.115 | 0.11 -> 0.16 | F -> T | F -> F |
| gemma-3-1b | P5 vs P8 | +0.126 -> +0.166 | 0.14 -> 0.17 | F -> T | F -> F |
| qwen3.5-9b | P3 vs P4 | +0.097 -> +0.114 | 0.13 -> 0.14 | F -> T | F -> F |
| qwen3.5-9b | P6 vs P8 | -0.047 -> -0.088 | -0.08 -> -0.13 | F -> T | F -> F |
| qwen3.5-27b | P1 vs P3 | -0.063 -> -0.151 | -0.08 -> -0.18 | F -> T | F -> F |
| qwen3.5-27b | P6 vs P8 | -0.017 -> -0.063 | -0.04 -> -0.12 | F -> T | F -> F |
| llama-3.2-1b | P1 vs P3 | +0.245 -> +0.273 | 0.36 -> 0.39 | T -> T | F -> T |
| llama-3.1-8b | P1 vs P5 | +0.250 -> +0.248 | 0.28 -> 0.26 | T -> T | T -> F |

Largest |mean_diff| shift across all 72 contrasts: 0.133 (mean 0.044).

## 4. Headline numbers, judge-reported -> deterministic

| stat | judge-reported | deterministic |
|---|---|---|
| RQ1 Qwen2B P1->P4 | +0.79 dz=1.03 | +0.81 dz=1.01 |
| RQ1 Llama1B P1->P3 | +0.24 dz=0.36 (not practical) | +0.27 dz=0.39 (practical) |
| RQ1 Qwen2B P1->P6 (RAG) | +1.14 dz=1.21 | +1.25 dz=1.21 |
| RQ2 Gemma27B P1->P3 | -0.25 dz=-0.28 | -0.30 dz=-0.31 |
| RQ2 Qwen27B P1->P3 | -0.06 n.s. | -0.15 BH-sig, not practical |
| RQ3 closed-book easy/med/hard | 3.78/3.30/3.06 | 3.78/3.25/2.98 |
| RQ3 open-book easy/med/hard | 3.92/3.77/3.73 | 3.99/3.82/3.76 |
| RQ6 verdict counts (of 72) | 47 BH-sig, 30 practical, 17 det-not-prac | 54 BH-sig, 30 practical, 24 det-not-prac |

Variant-mean grid: max cell change 0.152, mean 0.057. API baselines:
gpt-4o 4.17->4.22, gemini-2.5-pro 4.28->4.36, claude-sonnet-4-6 4.29->4.40.

## 5. RQ5 claim check (27B+RAG vs frontier APIs)

| cell | judge | deterministic |
|---|---|---|
| qwen3.5-27b P6 | 4.38 | 4.53 |
| gemma-3-27b P6 | 4.35 | 4.43 |
| qwen3.5-9b P6 | 4.22 | 4.36 |
| gemma-3-12b P6 | 4.26 | 4.30 |
| best API (claude-sonnet-4-6) | 4.29 | 4.40 |

"27B+RAG exceeds all three APIs" and "9-13B tier surpasses GPT-4o" both hold
with equal or larger margins. Note the tier ordering swap: qwen3.5-9b P6
(4.36) moves above gemma-3-12b P6 (4.30).

## 6. Bottom line

No headline claim weakens under the deterministic aggregate; RQ1, RQ2, RQ3,
RQ6 strengthen slightly and RQ5 holds with better margins. The
setting-correlated bias (§2) is the decisive argument: keeping the
judge-reported overall means knowingly reporting numbers with a small,
measured, systematic anti-RAG/anti-API bias.
