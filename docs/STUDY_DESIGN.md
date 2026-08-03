# Study design

The pre-specified design of the empirical study: research questions, the variant
taxonomy, the evaluation settings, and the statistical protocol. It records what
was decided *before* the results were analysed; `analysis/aggregate_scores.py`
implements the protocol in the section of that name below.

## Context

The study is an **empirical investigation** of how domain adaptation (continued
pre-training, supervised fine-tuning, LoRA, RAG) affects LLM competence on
High-Performance Networking (HPN) tasks. The **framework** (collect -> corpus ->
benchmark + instruction-data -> train -> eval -> RAG -> profile) is a secondary,
engineering contribution. The benchmark and its generation pipeline are a
supporting artifact.

The design is 8 adapted models x up to 8 training/eval variants (P1-P8), each
scored by a **GPT-5.1 judge** on the **242-question HPN v5.0 benchmark**
(10 categories x 3 difficulties; Correctness / Completeness / Clarity /
Conciseness), plus **3 frontier API baselines** (gpt-4o, gemini-2.5-pro,
claude-sonnet-4-6) judged identically, plus per-model profiling.

## Research questions (RQ1–RQ5)

Revised to match the evidence actually in hand; the original `Bench-LLM.docx`
RQs are superseded. RQ1–RQ4 are the empirical core, RQ5 is a cross-cutting
methodological contribution. Each maps to concrete artifacts in
`analysis/outputs/`.

- **RQ1 — Mechanism.** Which adaptation recipe (full SFT, CPT+SFT, LoRA SFT,
  LoRA CPT+SFT) moves HPN competence, and does continued pre-training add
  measurable value *beyond* instruction tuning? *Evidence:* closed-book
  P1↔P3, P1↔P4, P3↔P4, P2↔P5, P1↔P5; LM-adaptation perplexity/forgetting.
  *Signal so far:* most gain arrives at SFT; CPT-over-SFT (P3↔P4) is rarely
  practically significant.
- **RQ2 — Heterogeneity (for whom / where).** How consistent are effects across
  model family (Llama/Qwen/Gemma), scale (1B–27B), skill category, and
  difficulty? *Evidence:* model×contrast matrix + `significance_stratified.csv`.
  *Signal:* highly family/scale-dependent — large Qwen gains, gemma-3-12b ~null,
  gemma-3-27b SFT *hurts*, 124 stratum sign-flips.
- **RQ3 — Retrieval vs parametric adaptation.** Holding the model fixed, what
  does retrieval contribute, and does RAG complement or substitute for
  fine-tuning? *Evidence:* matched-model RAG effect P1↔P6, P4↔P7, P5↔P8 +
  `significance_cross_setting.csv`. *Signal:* RAG helps adapted models but not
  uniformly (Qwen closed-book FT ≫ Instruct+RAG; gemma/llama the reverse).
  Strictly an information-access comparison, never model quality.
- **RQ4 — Frontier gap & inference cost (usability).** Can domain-adapted
  open-weight models reach frontier-API HPN quality, and at what inference cost?
  *Evidence:* spine vs API baselines + `profiling_results/`. *Signal:*
  closed-book, best open-weight peaks at 4.09 (just below gpt-4o 4.15 / gemini
  4.28 / claude 4.27); with retrieval, ≥9B open-weight match or exceed all three
  — but APIs were run **closed-book**, so this is an information-access advantage
  (API+RAG untested → limitation/future work). Profiling quantifies
  latency/throughput/memory (CPU vs GPU).
- **RQ5 — Measurement (cross-cutting).** In LLM-judged domain benchmarking, how
  often does statistical significance overstate *practically meaningful*
  improvement, and how stable are conclusions across skill strata and under a
  single strong judge? *Evidence:* BH-significant (52) vs practically-significant
  (36) primary contrasts; 124 stratum sign-flips; single-judge (gpt-5.1) caveat.

Bounding limitations (threats-to-validity, not RQs): single judge, single seed,
`recovered_from_artifact` provenance, an LLM-generated benchmark, and no
API-baseline + RAG condition.

## Locked decisions
1. Framing: empirical adaptation study (RQ1–RQ5; see above) primary; framework
   engineering secondary.
2. Reproducibility bar: fresh-venv install + imports + CLI `--help` + tests/smoke
   + tiny-sample quickstart. No retraining and no exact-score reproduction is
   claimed; the reproducible unit is the analysis from the judged outputs.
3. Statistics: follow the pre-registered **Statistical Analysis Protocol**
   (below) — paired Wilcoxon + Benjamini–Hochberg FDR, effect-size bands, a
   practical-significance rule, and category/difficulty stratification, all on
   **within-setting** contrasts. Fixed before any results prose was written.

## Variant taxonomy (P↔S)
| P | Setting | Definition | run-name pattern | profiling S-tag |
|---|---|---|---|---|
| P1 | closed-book | Original Instruct (distribution) | `<Model>` | S1 |
| P2 | closed-book | Original Instruct + LoRA HPN-SFT | `*-instruct-lora-sft-merged` | S2 |
| P3 | closed-book | Original Instruct + full HPN-SFT | `*-instruct-full-hpn-sft` | S3 |
| P4 | closed-book | Base + full HPN-CPT + full HPN-SFT | `*-cpt-full-sft` | S4 |
| P5 | closed-book | Original Instruct + LoRA CPT + LoRA SFT | `*-instruct-lora-cpt-then-sft-merged` | S5 |
| P6 | open-book (RAG) | Original Instruct + RAG | `RAG-<Model>` | S6 |
| P7 | open-book (RAG) | Base + full CPT + SFT + RAG | `RAG-*-cpt-full-sft` | S7 |
| P8 | open-book (RAG) | Original Instruct + LoRA CPT + LoRA SFT + RAG | `RAG-*-instruct-lora-cpt-then-sft-merged` | S8 |

P4/P7 are **absent** for the large (27B) models: base continued-pretraining is
disabled by design, so they are adapted from the official instruct checkpoint
via LoRA only.

## Evaluation settings: closed-book vs open-book

RAG variants have **privileged access to source documents at inference time**;
closed-book variants rely only on parametric knowledge. A "RAG beats
fine-tuning" gap is therefore an **information-access** effect, not a pure
model-quality claim, and must never be reported as the latter.

- **Closed-book (direct generation): P1–P5** — parametric knowledge only.
- **Open-book (retrieval-augmented): P6–P8** — retrieves from the HPN corpus.

Reporting rules:
- **Primary comparisons are within-setting.** Closed-book contrasts (P1 vs
  P3/P4/P5, etc.) feed RQ1–RQ2; matched-model RAG contrasts feed RQ3; open-book
  adaptation contrasts (P6 vs P7/P8) test whether adaptation still helps under
  retrieval.
- **Cross-setting contrasts** (best closed-book vs best open-book, or P4 vs P6)
  go in a separate table, explicitly labelled an *information-access*
  comparison, with the caveat that RAG sees retrieved passages the closed-book
  models do not.
- `analysis/aggregate_scores.py` gains a `setting` column
  (`closed_book` / `open_book`) and groups contrasts accordingly.
- API frontier baselines are **closed-book** unless explicitly run with RAG.

## Statistical analysis protocol (pre-registered)

Fixed **before** any Results prose is written, to avoid post-hoc threshold
selection. Computed over per-question judged scores (242 Q × variant; the
`overall` 1–5 score is primary, the 4 sub-dimensions secondary).

1. **Test.** Paired Wilcoxon signed-rank on per-question scores, paired by
   question `id`, two-sided. Report n, median paired difference, and the mean
   difference with a 10k-resample bootstrap 95% CI.
2. **Contrast family (fixed).** closed-book adaptation {P1↔P3, P1↔P4, P3↔P4,
   P2↔P5, P1↔P5}; open-book adaptation {P6↔P7, P6↔P8}; matched-model RAG effect
   {P1↔P6, P4↔P7, P5↔P8} (same model ± retrieval, which cleanly isolates RAG).
   Unmatched cross-setting contrasts (e.g. P4↔P6 = best closed-adapted vs
   instruct+RAG) are **exploratory**, reported in a separate information-access
   table, and **excluded** from the primary family.
3. **Multiple-comparison correction.** Benjamini–Hochberg FDR at q = 0.05 pooled
   across **all** primary (model × contrast) tests; report raw p and BH-adjusted p.
4. **Effect size.** Paired Cohen's d_z (and rank-biserial). Bands: negligible
   < 0.2, small 0.2–0.5, medium 0.5–0.8, large > 0.8.
5. **Practical-significance rule.** A contrast is *practically significant* only
   if it clears **all three**: |Δ overall| ≥ 0.25 (on the 1–5 scale),
   d_z ≥ 0.2, and BH-significant. Statistical significance alone (easy at
   n = 242) is necessary but not sufficient — this rule is itself a stated
   contribution on domain-LLM benchmarking rigor.
6. **Stratified analyses.** Re-run each primary contrast within each of the 10
   categories and 3 difficulty levels; report where the sign flips or
   significance/practical-significance is lost. Per-stratum results are
   exploratory (smaller n; BH within each stratum family).
7. **Reproducibility.** Fixed bootstrap seed; protocol + contrast family
   recorded here and emitted into `analysis/outputs/`.

## API-baseline provenance requirements

API model identifiers drift — providers silently update weights behind a stable
name — so each API-baseline `PROVENANCE.md` must record, at minimum:
- **exact model string as invoked** (e.g. `gpt-4o`) and any provider-side dated
  snapshot/version if exposed;
- **provider** (OpenAI / Google / Anthropic);
- **answer-generation run date** (answer workbook `Metadata.timestamp`);
- **judge model + judge run date** (judged workbook `Metadata`);
- **benchmark version** (HPN v5.0) and decoding params if known;
- an explicit **drift caveat** — results pin a point-in-time API, not a fixed
  artifact.

Values already on hand: answer runs and judging both **2026-06-15**, judge
**gpt-5.1**. Enriching the three API `PROVENANCE.md` files with these fields is
the immediate Phase-A follow-up.

## Model roster (8 adapted + 3 API baselines)
Small: llama-3.2-1b, gemma-3-1b, qwen3.5-2b · Medium: llama-3.1-8b, qwen3.5-9b,
gemma-3-12b · Large: gemma-3-27b, qwen3.5-27b · API baselines: gpt-4o,
gemini-2.5-pro, claude-sonnet-4-6.
