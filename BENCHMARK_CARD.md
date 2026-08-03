# HPN-QA v5.0 — benchmark card

A question-answering benchmark on **high-performance networking** (HPN): bulk
data transfer, congestion control, bottleneck diagnosis, parallelism and
concurrency tuning, and the parameters that govern them. Items are generated
from a corpus of open-access research papers and graded against a reference
answer.

| | |
|---|---|
| Version | v5.0 |
| Items | 242 released, **233 scored** (9 excluded, see below) |
| Language | English |
| Task | open-ended generation, graded on four dimensions |
| Generated | 2026-04-27, seed 42 |
| File | `NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl` (JSON Lines) |
| Manifest | `NetBench-LLM/data/prompts/hpn_benchmark_v5.0_provenance.json` |
| Licence | see `LICENSE-DATA` — **evidence quotes are not CC-BY** |

## Composition

**Splits.** `test` 200 · `dev` 30 · `adversarial` 8 · `synthesis` 4.
`dev` is for prompt and protocol development; `test` carries the reported
results. `synthesis` items require combining two papers; `adversarial` items
are built to defeat surface pattern-matching. Both are small and reported
separately, not folded into the headline numbers.

**Difficulty** — `easy` 77 · `medium` 99 · `hard` 66. This label is *measured,
not authored*: a three-model calibration ladder (weak / mid / strong) answers
every item, and the label is the outcome — solved by all three → `easy` (77);
solved by mid and strong but not weak → `medium` (99); anything else → `hard`
(63 solved only by the strong model, 3 by strong and weak but not mid). The
ladder overrode the generator's own difficulty
guess on 144 of 242 items, which is why the field is trustworthy in a way an
authored label would not be. Each item keeps both under
`calibration.difficulty_original` and `calibration.ladder`.

**Question type** — reasoning 90 · concept 70 · scenario 42 · diagnosis 19 ·
calculation 17 · comparison 4. 26 items require arithmetic
(`requires_calculation`); 65 pose a hypothetical deployment
(`synthetic_scenario`).

**Category** (10, deliberately uneven — it tracks the literature's own
emphasis, not a target quota):

| n | category |
|---:|---|
| 57 | Transfer Parameters: Definitions and Roles |
| 53 | Bottleneck Diagnosis and End-to-End Reasoning |
| 31 | BDP-Based Reasoning and Window Sizing |
| 26 | Adaptive and Online Optimization |
| 22 | Practical HPN Scenarios and Design |
| 16 | Fairness, Stability, and Shared Networks |
| 13 | Pipelining and Small-File Optimization |
| 10 | Parallelism and Large-File Optimization |
| 9 | Concurrency Tuning and Scaling |
| 5 | Dataset Partitioning and Mixed Workloads |

**Size.** Questions run 15–93 words (median 26); reference answers 58–134
(median 90). Answers are deliberately bounded — long enough to require
explanation, short enough that conciseness is scoreable.

**Sources.** 175 distinct publications, 301 evidence quotes (median 1 per item,
maximum 3).

## Fields

| Field | Meaning |
|---|---|
| `id` | stable item id (`NB-HPN-*` per-paper, `NB-SYN-*` synthesis, `NB-ADV-*` adversarial) |
| `question`, `reference_answer` | the graded pair |
| `category`, `difficulty`, `question_type`, `split` | labels described above |
| `requires_calculation`, `synthetic_scenario` | booleans |
| `keywords` | topic terms |
| `evidence[]` | `paper_id`, `passage_id`, `quote` (verbatim source passage), `source_doi`, `source_license` |
| `source_papers`, `inspired_by_papers` | corpus paper ids |
| `calibration` | `ladder{weak,mid,strong}` and `difficulty_original` |
| `validation` | `checks_passed` and any recorded `issues` |
| `excluded_from_scoring`, `exclusion_reason` | the 9 items removed from every reported score |
| `generator_model`, `critic_model`, `calibration_model`, `prompt_version`, `seed` | generation provenance |

`paper_id` is an index into a corpus that is not redistributed. It is made
resolvable by `analysis/outputs/source_papers.csv` (title, DOI, year, licence
for all 175) and by the `source_doi` / `source_license` fields on each quote.

## How it was built

Eight pipeline stages, from corpus chunking to the final split, grouped here
into six steps. Every item is written, validated, and calibrated by **local
open-weight models**: no paid API generates, filters, validates, or calibrates
any item. Paid API models appear in construction at one point only — a separate
calibration of the generation-internal grading against a human grader and two
API judges (GPT-4o, Gemini-2.5-Pro) on a held-out sample.

1. **Corpus** — open-access PDFs collected and deduplicated, then converted to
   cleaned text (headers, footers, reference lists, captions, and extraction
   artifacts stripped).
2. **Relevance filter and paper cards** — an embedding filter selects HPN
   papers; `qwen3.5:9b` summarises each into a structured card with evidence
   anchors.
3. **Question generation** — `gemma4:26b` writes per-paper items
   (`v5.0-questions`), cross-paper synthesis items (`v5.0-synthesis`), and
   adversarial items (`v5.0-adversarial`).
4. **Validation, deterministic first** — schema; evidence-quote substring
   match; 8-gram memorisation-leak check against the training corpus plus a
   cosine ceiling; embedding near-duplicate removal; then, last and smallest, an
   LLM critic for grounding, answerability, and genericness. The cheap
   auditable gates carry the load; a single rewrite attempt is allowed, and
   only on re-passing the deterministic gates.
5. **Calibration** — the three-model ladder that assigns `difficulty`.
6. **Split and enrichment** — splits assigned, then per-item calibration and
   validation records joined back on, plus the provenance manifest recording
   generation models, prompt versions, seed, and the corpus SHA-256.

**Expert review.** Automated gates cannot judge whether a question is
technically well-posed or its reference answer substantively correct, so all
three authors vetted every question. That review is what produced the 9
exclusions.

The generator families (Qwen, Gemma) overlap with families later evaluated.
This confers no advantage: items are grounded in and answerable from the
literature, not from generator-specific knowledge, and every evaluated system
answers the identical items under the identical protocol.

## Intended use

Measuring whether a language model can reason about high-performance networking
— and, in the accompanying work, whether domain adaptation (continued
pre-training, instruction tuning, LoRA, retrieval) moves that ability.
Items are answered **closed-book** (question only) or **open-book** (question
plus retrieved passages), and scored by an LLM judge on correctness (0.4),
completeness (0.3), clarity (0.2), and conciseness (0.1).

**Not suitable for:** training data (it is an evaluation set — see below);
certifying operational network configurations; or claims about languages other
than English or domains outside HPN.

**Do not train on it.** The benchmark and the instruction-tuning data in
`Instruct-FTD/` are derived from the same corpus but kept strictly separate by
design. Fine-tuning on HPN-QA invalidates any score reported against it.

## Known limitations

- **The 9 excluded items.** Expert review found 9 items with a defective
  reference answer or a dangling reference to source text the reader is not
  given. They ship, flagged `excluded_from_scoring` with a reason, and are
  removed from every reported score (242 generated → 233 scored). They are
  released rather than deleted so the count chain is auditable; see
  `DATA_RELEASE.md`.
- **7 evidence quotes are not verbatim.** An end-of-pipeline audit against the
  corpus of record (`tools/audit_evidence_quotes.py`, hash-verified against the
  manifest) finds **294 of 301 quotes (97.7%) are exact substrings** of the
  paper they cite. Of the 7 that are not, one is a truncation that preserves
  the sentence (similarity 0.85) and six are paraphrases of the source rather
  than quotations from it. Per-quote results, with the closest corpus window
  for each miss, are in `analysis/outputs/evidence_quote_audit.csv`.
  **This does not affect any reported score:** `evidence` is provenance
  metadata and is never placed in the answer-generation prompt or shown to the
  judge, which see only the question and the reference answer. It does mean the
  grounding trail for those 7 items is weaker than for the other 226 scored
  items. All 7 are in the `test` split and none was independently caught by the
  expert review.
- **15 items carry a recorded validation issue.** Eight passed only after a
  rewrite following an LLM-critic objection, and expert review later removed
  three of those eight — a useful signal that the critic's objections were
  often real. Seven synthesis/adversarial items retain a logged quote-match
  issue from before quote repair; all seven are verbatim in the shipped data.
  Every one of the 15 is inspectable in `validation.issues`.
- **Reference answers are model-written.** Generated by `gemma4:26b` from the
  source passage, then filtered by the deterministic gates and reviewed by the
  authors. They are a defensible target, not ground truth authored from
  scratch by a domain expert.
- **Category coverage is uneven** (57 items in the largest category, 5 in the
  smallest). Per-category scores on the small categories are noisy; the
  accompanying analysis reports stratified results with that in mind.
- **Small non-standard splits.** With 8 adversarial and 4 synthesis items,
  those splits indicate direction only. Do not report them as headline numbers.
- **Corpus exposure is possible.** Items derive from open-access papers, so any
  model — open-weight or frontier — may have seen the source material in
  pre-training. The 8-gram leak check and cosine ceiling limit verbatim recall
  of reference answers, but cannot rule exposure out. It is symmetric across
  systems and so does not bias within-model paired comparisons.
- **One domain, one language.** English, high-performance networking. Nothing
  here supports generalisation beyond that.

## Licence

Questions, reference answers, labels, and all derived outputs: **CC-BY-4.0**.
Verbatim evidence quotes: **not covered** — they belong to their sources and
are included as short attributed excerpts. Read `LICENSE-DATA` before
redistributing. Code is Apache-2.0 (`LICENSE`).
