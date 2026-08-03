# HPN-QA Human-Judge Alignment: `human` vs `gpt-5.1`

Two domain experts (authors) independently blind-scored **125 responses** (5 systems x 25 stratified questions; system identity hidden, order re-randomized per question) on the judge's exact rubric. The `human` reference below is their per-response consensus (mean of the two raters); inter-rater reliability is reported separately in `HPN_INTERRATER_rater1_vs_rater2.md`. Composite `overall` for the human is the judge's instructed weighting (0.4 correctness + 0.3 completeness + 0.2 clarity + 0.1 conciseness).

> Unlike the two-LLM agreement report, this measures *construct validity* - whether the LLM judge tracks expert judgement. Two-rater consensus: values reflect the two experts' mean calibration. 95% CIs are cluster bootstrap over questions (responses to one question are correlated); contrast CIs use the seeded bootstrap from `aggregate_scores.py` (B=10000).

## Per-response agreement (pooled)

| Dimension | n | Pearson [95% CI] | Spearman | Kendall tau-b | MAE [95% CI] | Bias (judge-human) [95% CI] | QWK [95% CI] | % within +/-0.5 | % within +/-1 |
|---|---|---|---|---|---|---|---|---|---|
| correctness | 125 | 0.856 [0.80, 0.91] | 0.850 | 0.757 | 0.496 [0.39, 0.61] | 0.312 [0.18, 0.45] | 0.775 [0.68, 0.85] | 74.4 | 88.8 |
| completeness | 125 | 0.865 [0.80, 0.91] | 0.846 | 0.748 | 0.468 [0.38, 0.56] | 0.204 [0.06, 0.35] | 0.822 [0.75, 0.88] | 75.2 | 94.4 |
| clarity | 125 | 0.700 [0.61, 0.78] | 0.707 | 0.643 | 0.332 [0.27, 0.39] | 0.116 [0.02, 0.21] | 0.682 [0.61, 0.74] | 78.4 | 98.4 |
| conciseness | 125 | 0.683 [0.59, 0.77] | 0.671 | 0.603 | 0.428 [0.35, 0.51] | -0.044 [-0.14, 0.05] | 0.545 [0.43, 0.65] | 76.8 | 98.4 |
| overall (judge-reported) | 125 | 0.892 [0.84, 0.93] | 0.870 | 0.707 | 0.376 [0.31, 0.45] | 0.176 [0.10, 0.26] | n/a | 74.4 | 96.8 |
| overall (judge reweighted) | 125 | 0.892 [0.85, 0.93] | 0.876 | 0.712 | 0.375 [0.31, 0.45] | 0.205 [0.12, 0.30] | n/a | 73.6 | 95.2 |

_For the four integer dimensions, % within +/-0.5 equals exact agreement. `overall (judge-reported)` compares the human composite with the judge's own reported overall (the number used in every paper table); `overall (judge reweighted)` applies the instructed formula to the judge's four dimension scores, removing the judge's formula deviation (mean |reported - reweighted| = 0.117 on this sample)._

## System-level (leaderboard on the 25 sampled questions)

| Metric | Value |
|---|---|
| Kendall tau-b (system ranking) | **1.000** |
| Spearman rho (system ranking)  | **1.000** |
| Pearson r (mean overall)       | 0.989 |

| System | human mean | rank | gpt-5.1 mean (25q) | rank | judge mean (full benchmark) |
|---|---|---|---|---|---|
| RAG-Qwen3.5-9B | 4.334 | 1 | 4.288 | 1 | 4.220 |
| gpt-4o | 4.190 | 2 | 4.236 | 2 | 4.167 |
| RAG-Qwen3.5-2B | 3.820 | 3 | 3.836 | 3 | 3.857 |
| Qwen3.5-2B-cpt-full-sft | 3.186 | 4 | 3.600 | 4 | 3.498 |
| Qwen3.5-2B | 2.394 | 5 | 2.844 | 5 | 2.712 |

## Embedded paired contrasts (human-scored vs judge-scored)

_Same statistics as `analysis/outputs/significance.csv` (paired Wilcoxon, seeded bootstrap 95% CI, Cohen's d_z), computed on the 25 sampled questions under each scorer. `judge full` is the judge's mean difference on the complete benchmark, for reference._

| Contrast | Scorer | n | mean A | mean B | mean diff | 95% CI | Wilcoxon p | d_z | judge full |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b P1 -> P4 (full CPT+SFT effect) | human | 25 | 2.394 | 3.186 | **+0.792** | [0.562, 1.030] | 0.0000 | 1.317 | +0.786 |
| qwen3.5-2b P1 -> P4 (full CPT+SFT effect) | gpt-5.1 | 25 | 2.844 | 3.600 | **+0.756** | [0.556, 0.964] | 0.0000 | 1.424 | +0.786 |
| qwen3.5-2b P1 -> P4 (full CPT+SFT effect) | gpt-5.1 (formula) | 25 | 2.808 | 3.544 | **+0.736** | [0.516, 0.972] | 0.0001 | 1.259 | +0.786 |
| qwen3.5-2b P1 -> P6 (RAG effect) | human | 25 | 2.394 | 3.820 | **+1.426** | [0.942, 1.834] | 0.0002 | 1.199 | +1.144 |
| qwen3.5-2b P1 -> P6 (RAG effect) | gpt-5.1 | 25 | 2.844 | 3.836 | **+0.992** | [0.552, 1.368] | 0.0002 | 0.929 | +1.144 |
| qwen3.5-2b P1 -> P6 (RAG effect) | gpt-5.1 (formula) | 25 | 2.808 | 3.848 | **+1.040** | [0.556, 1.456] | 0.0004 | 0.884 | +1.144 |
| gpt-4o -> qwen3.5-9b P6 (frontier parity) | human | 25 | 4.190 | 4.334 | **+0.144** | [-0.188, 0.500] | 0.6765 | 0.161 | +0.053 |
| gpt-4o -> qwen3.5-9b P6 (frontier parity) | gpt-5.1 | 25 | 4.236 | 4.288 | **+0.052** | [-0.184, 0.324] | 0.8960 | 0.079 | +0.053 |
| gpt-4o -> qwen3.5-9b P6 (frontier parity) | gpt-5.1 (formula) | 25 | 4.308 | 4.440 | **+0.132** | [-0.156, 0.456] | 0.7650 | 0.169 | +0.053 |

## Formula-metric variants

_The deterministic rubric composite (0.4/0.3/0.2/0.1 over the judge's dimension scores; see `docs/JUDGE_OVERALL_AUDIT.md`) is the EACL paper's primary metric. Per-response agreement for it is the `overall (judge reweighted)` row above; `judge (formula)` contrast rows and the system-level ranking below use it too._

| Kendall tau_b (ranking, formula) | 1.000 |
| Spearman rho (ranking, formula)  | 1.000 |
| Pearson r (mean overall, formula) | 0.992 |
