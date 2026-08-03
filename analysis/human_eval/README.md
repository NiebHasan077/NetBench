# Human-judge alignment study

Measures agreement between the GPT-5.1 LLM judge and human expert scoring on
a stratified subsample of the HPN-QA benchmark, using the identical rubric.
This addresses the validation gap named in the paper's limitations section.

## Files

- `make_judging_sheets.py` — generates the blinded rating workbook
  (deterministic, seed 42). Rerunning reproduces byte-identical sampling.
- `outputs/HPN_human_judging_v1.xlsx` — the blank rating workbook: an
  `Instructions` sheet (rubric verbatim from
  `NetBench-LLM/evaluation/judge_responses.py`) and a `Scoring` sheet with
  125 blinded responses and blank 1–5 dropdown score cells.
- `outputs/sampling_key.csv` — response-code → system mapping.
  **Raters must not open this until scoring is complete.**
- `outputs/filled/` — the rater's filled workbooks (five 5-question parts).
- `merge_filled_ratings.py` — validates and merges the filled workbook(s)
  into `outputs/human_scores_long.csv` and the single consolidated
  `outputs/HPN_human_judging_v1_filled.xlsx`.
- `human_judge_agreement.py` — joins the merged human ratings with the
  GPT-5.1 judge scores and writes the alignment report to
  `analysis/outputs/HPN_JUDGE_AGREEMENT_human_vs_gpt-5.1.{md,xlsx}` plus the
  per-response join `outputs/human_vs_judge_long.csv`.

## Design

**Variants (5).** Chosen to span the observed quality range (≈2.7–4.2 mean
overall) and to embed the paper's headline paired contrasts, so human scores
can validate the contrasts themselves, not just per-answer agreement:

| Code space | Variant | Judge mean | Why |
|---|---|---|---|
| — | qwen3.5-2b **P1** (instruct, closed-book) | 2.71 | low anchor; shared baseline for both contrasts below |
| — | qwen3.5-2b **P4** (full CPT→SFT) | 3.50 | carries the largest fine-tuning effect (+0.79, d_z≈1.0) |
| — | qwen3.5-2b **P6** (instruct + RAG) | 3.86 | carries the largest RAG effect (+1.14, d_z≈1.2) |
| — | qwen3.5-9b **P6** (instruct + RAG) | 4.22 | best-tier open-weight system; frontier-parity claim |
| — | **gpt-4o** (closed-book API) | 4.17 | frontier anchor |

**Questions (25; 125 responses).** 19 of the first 25 benchmark items
(raters know these deeply), minus the audit-excluded NB-HPN-2121-q2 and five
drops from over-represented cells (3 of 7 Bottleneck Diagnosis, 1 BDP medium,
1 Transfer easy); plus NB-HPN-1676-q1 and NB-HPN-2032-q3 to cover the tenth
category (Dataset Partitioning) and the calculation quota; plus 4
seeded-random fills. Final composition: 8 easy / 10 medium / 7 hard
(proportional to the benchmark's 74/96/63), all 10 categories, 3
calculation items.

**Blinding.** Raters see exactly the judge's information set — question,
reference answer, model answer — nothing else (no category, difficulty, or
system identity). The five responses are re-shuffled independently per
question; codes are `Qnn-a … Qnn-e`.

**Scoring.** Integers 1–5 on the four rubric dimensions only. No overall
score is entered — it is computed at merge time with the judge's instructed
weights (0.4·correctness + 0.3·completeness + 0.2·clarity +
0.1·conciseness), so the human composite is defined identically to the
judge's.

## Workflow

1. Rater scores the `Scoring` sheet of `HPN_human_judging_v1.xlsx` (whole,
   or split into partial copies whose union covers all 125 responses) and
   places the filled workbook(s) in `outputs/filled/`.
2. Only then may `sampling_key.csv` be opened.
3. `merge_filled_ratings.py` validates every row (integer 1–5 scores; answer
   text unchanged from the blinded original, modulo newline-encoding
   differences from LibreOffice re-saves; exact 125-response coverage) and
   writes the merged CSV and consolidated workbook.
4. `human_judge_agreement.py` computes per-response agreement (Pearson,
   Spearman, Kendall, MAE, signed bias, QWK, ±0.5/±1 bands, cluster-bootstrap
   CIs over questions), the system-level ranking comparison, and human- vs
   judge-scored versions of the embedded paired contrasts.

## Result (rater 1, 2026-07-12)

Overall: Pearson r = 0.904 [0.86, 0.94], MAE = 0.32, bias (judge − human)
= −0.03 [−0.10, 0.05]; system ranking Kendall τ_b = 1.000. Content
dimensions agree strongly (QWK 0.86 correctness / 0.87 completeness),
stylistic ones less (0.69 clarity / 0.55 conciseness; the judge is stricter
on conciseness by −0.26). All three embedded contrasts reproduce under
human scoring with the same sign, significance, and comparable magnitude.
Full report: `analysis/outputs/HPN_JUDGE_AGREEMENT_human_vs_gpt-5.1.md`.
