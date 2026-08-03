# NetBench-LLM Benchmark Generator — Pipeline Plan

Automated pipeline that turns 2371 HPN research-paper texts into a high-quality
HPN-skill benchmark (**~300 QA pairs in v1**, across dev / test / synthesis /
adversarial splits; hidden split deferred to v5.1), using **only local Ollama
models** for generation and validation. Paid APIs (GPT-4o, Gemini 2.5 Pro) are
used **only for final judging and inter-judge agreement** — never for content
creation.

The corpus that backs this benchmark already trained the target model
([`NetBench-LLM Direct Pipeline/data/raw/research_corpus_v3.json`](../NetBench-LLM/data/raw/research_corpus_v3.json)),
so the pipeline must defend against memorization leakage.

---

## 1. Goals and constraints

### Goals
- Produce a benchmark that measures **transferable HPN skill**, not paper recall.
- Ship a reproducible, mostly-deterministic pipeline that can re-run end-to-end.
- Match the existing evaluator schema in
  [`NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py`](../NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py)
  so the new JSONL drops in with minimal code change.

### Hard constraints
- **No paid APIs for generation, validation, or filtering.** Ollama only.
- **Hardware: 2 × 48 GB GPUs (96 GB total).**
- **Manual work ≤ ~1 hour total** (only judge-prompt sanity check on ~30 questions).
- Pipeline must be re-runnable from any stage on crash.

### Non-goals (deliberately cut for v1 to keep scope small)
- GROBID / structured PDF re-ingestion. We use the existing flat-text corpus
  with sliding-window passage anchors.
- SFT-data generation. The benchmark generator must not double-purpose as an
  SFT-data generator — that creates contamination.
- Multiple-choice variants. Open-ended only in v1.

---

## 2. Models

### Local (Ollama) — generation, cards, validation
| Role | Model (currently installed) | Size | Notes |
|---|---|---|---|
| Relevance filter | none (embeddings) | — | Sentence-transformer; no LLM call needed |
| Card generation | `qwen3.5:9b` | 6.6 GB | Via `/api/chat` with `think=False` and schema enforcement; reliable structured output, ~13s/card |
| Question generation | `gemma4:26b` | 17 GB | Optional upgrade: `qwen2.5:72b-instruct-q4_K_M` (~43 GB, fits one 48 GB GPU) for stronger JSON + reasoning |
| Critic / grounding | `gemma4:26b` | 17 GB | Cheap-enough second opinion |
| Difficulty calibration (3 models) | `llama3.2:3b` (weak), `qwen3.5:9b` (mid), `gemma4:26b` (strong) | mixed | Empirical difficulty signal. Swap 72B in as `strong` if installed. |

Pin model + tag in `manifest.json` so the run is reproducible.

### API (judging only)
- **GPT-4o** — Judge A. Uses [`NetBench-LLM Direct Pipeline/OPENAI_API_KEY`](OPENAI_API_KEY).
- **Gemini 2.5 Pro** — Judge B. Uses [`NetBench-LLM Direct Pipeline/GEMINI_API_KEY`](GEMINI_API_KEY).

Different vendors → contamination-resistant judging. Compute Cohen's kappa
between A and B as a question-quality signal.

### Embeddings (local, free)
- `BAAI/bge-large-en-v1.5` (sentence-transformers) — for relevance filter,
  topic clustering, dedup.

---

## 3. Pipeline overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  research_corpus_v3.json (2371 papers, flat text)                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
   [1] Passage chunking (deterministic, no LLM)
       → passages.jsonl  (paper_id, passage_id, text)
                                 │
                                 ▼
   [2] Auto-relevance filter (embeddings only, no LLM)
       seed = v4 benchmark questions ∪ v4 reference answers
       keep top-K papers by max-passage cosine to seed
       → filtered_papers.jsonl  (~600-800 papers)
                                 │
                                 ▼
   [3] Paper cards (Ollama: qwen3.5:9b via /api/chat, schema-enforced)
       1 call per paper → structured JSON card with passage_anchor_ids
       → paper_cards.jsonl
                                 │
                                 ▼
   [4] Topic clustering (embeddings on cards, no LLM)
       HDBSCAN over card embeddings → cluster_id per card
       → paper_cards_clustered.jsonl
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
   [5a] Per-paper question gen                       [5b] Cross-paper synthesis
        (gemma4:26b, 4 Q/card)                           (gemma4:26b, 1 Q per cluster-pair)
        → candidate_questions.jsonl                      → synthesis_candidates.jsonl
        ▲                                                 │
        │                                                 ▼
   [5c] Adversarial / misconception gen
        (gemma4:26b, seeded by misconception list)
        → adversarial_candidates.jsonl
                                 │
                                 ▼
   [6] Validation gauntlet (mostly deterministic)
       6.1 schema (pydantic)
       6.2 evidence-quote substring match  ← deterministic, not LLM
       6.3 8-gram memorization-leak check  ← deterministic
       6.4 embedding dedup
       6.5 critic LLM (gemma4:26b)         ← grounding + answerability + genericness
       → validated_questions.jsonl
                                 │
                                 ▼
   [7] Difficulty calibration (3 Ollama models)
       Tag empirical difficulty; reject "all-wrong" (usually ambiguous)
       → calibrated_questions.jsonl
                                 │
                                 ▼
   [8] Splits + manifest
       dev / test / synthesis / adversarial / hidden
       → benchmark/hpn_benchmark_v5_*.jsonl
                                 │
                                 ▼
   [9] Judge calibration (paid APIs, ~30 Q manual + 100 Q dual-judge)
       human kappa, A-vs-B kappa
       → reports/judge_calibration.md
```

Total Ollama calls across the whole pipeline (300-question v1): **~3,000**.
At ~30 sec/call on 2×48 GB with `OLLAMA_NUM_PARALLEL=4`, expect **~6-10 hours
of GPU time** end-to-end. Code effort: **~1 person-week**.

---

## 4. Stage details

### Stage 1 — Passage chunking
- Split each paper's `text` into ~1000-token windows with 100-token overlap.
- Stable IDs: `paper_<idx>__p<chunk>`.
- Store in `data/passages.jsonl`. This is the **only ground truth** for evidence
  quotes downstream.

### Stage 2 — Auto-relevance filter (no manual picking)
**Replaces "manually pick 50 seed papers."** Fully automatic.

1. Build seed text = concatenation of every question + reference_answer in
   [`hpn_qa_benchmark_v4_general_skills.json`](../NetBench-LLM/data/prompts/hpn_qa_benchmark_v4_general_skills.json).
   This is your existing curated HPN signal — no manual work needed.
2. Embed seed text (chunked) and every paper passage with `bge-large-en-v1.5`.
3. For each paper, score = max cosine(seed_chunks, paper_passages).
4. Keep top-K papers (default K=600). Tunable via `config.yaml`.
5. Drop papers below an absolute threshold (default 0.45) even if they're in
   the top-K — handles the case where the corpus is mostly off-topic.

**No LLM calls. Runs in ~30 minutes on one GPU.**

### Stage 3 — Paper cards
One Ollama call per surviving paper. Card schema:

```json
{
  "paper_id": "paper_0182",
  "main_problem": "string",
  "system_context": "string",
  "key_concepts": ["string"],
  "important_formulas": ["string"],
  "important_numbers": ["string with units"],
  "failure_modes": ["string"],
  "possible_question_topics": ["string"],
  "evidence_anchors": [
    {"passage_id": "paper_0182__p3", "claim": "string"},
    {"passage_id": "paper_0182__p7", "claim": "string"}
  ]
}
```

`evidence_anchors[].passage_id` MUST be drawn from the actual passage IDs of
this paper. Validate post-hoc with a deterministic substring-match step (the
LLM will hallucinate IDs otherwise).

Prompt template lives in [`prompts/card_generation.txt`](../prompts/card_generation.txt).

### Stage 4 — Topic clustering
- Embed each card's `main_problem + key_concepts + possible_question_topics` (BGE-large, normalized).
- HDBSCAN with `min_cluster_size=3, cluster_selection_method=leaf` (defaults in `config.yaml`).
  - The original guess of `mcs=5, eom` collapsed the corpus to 3 clusters with 78% noise — embeddings are very tightly packed in HPN-domain text. `leaf+mcs=3` exposes the right amount of granularity (6 clusters in v1).
- Save cluster keywords (top-10 TF-IDF terms per cluster) to `reports/clusters.md`.
- **Decision gate (5-min eyeball):** does the cluster keyword list cover the
  taxonomy in §6? If a category has zero coverage even in noise, drop it from the plan.
- v1 actual outcome: 6 clusters (Falcon-style transfer tuning, BBR/RTT, DCTCP/incast,
  wireless, RL-CC, HPC storage I/O) + 230 noise cards. Phase 5b synthesis pulls
  from clusters first, then supplements from noise via embedding-similarity pairs.

### Stage 5 — Question generation (3 sub-stages)

#### 5a. Per-paper questions
For each card, prompt the question generator (`gemma4:26b`, or `qwen2.5:72b-instruct-q4_K_M` if installed) to emit **exactly 4** candidate questions:
1 concept, 1 reasoning, 1 calculation-or-diagnosis, 1 scenario.

Hard schema requirement: every question must reference ≥1 `passage_id` from
its source card's `evidence_anchors`. Otherwise rejected at Stage 6.2.

#### 5b. Cross-paper synthesis
For each cluster, sample pairs of cards within the cluster. For each pair,
prompt the generator with **both cards** and ask for 1 synthesis question
("compare", "combine", "when does X fail but Y succeed").

Because the HPN corpus clusters tightly (v1: 6 clusters, 68 clustered cards,
230 noise), cluster pairs alone (~41) are insufficient to reach the synthesis
target. Stage 05b supplements with **noise-pool similarity pairs**: take the
top-25% cosine-similar pairs among noise-cluster cards (using Stage 04's cached
BGE-large embeddings) and random-sample from them to fill the remaining slots.
This is reflected in `build_pairs()` in `stage_05b_synthesis.py`.

#### 5c. Adversarial / misconception
Hardcoded misconception list (10-15 items) covering common LLM HPN errors.
Examples:
- "more parallel streams always increases throughput"
- "TCP fairness is automatic on shared links"
- "concurrency and parallelism are interchangeable"

For each misconception, sample 3-5 cards relevant to the topic; prompt the
generator to write a question that elicits the misconception.

This list lives in [`prompts/misconceptions.yaml`](../prompts/misconceptions.yaml).

### Stage 6 — Validation gauntlet
Order matters: cheapest-and-most-decisive first.

| # | Check | Type | Reject rate (expected) |
|---|---|---|---|
| 6.1 | Pydantic schema | deterministic | ~5% |
| 6.2 | Each `evidence.quote` is exact substring of its `passage_id` | deterministic | ~30% |
| 6.3 | No 8-gram of `reference_answer` appears in training corpus; cosine to nearest training passage < 0.92 | deterministic | ~10-15% |
| 6.4 | No embedding-near-duplicate with prior accepted Q (cosine > 0.88) | deterministic | ~20% |
| 6.5 | Critic LLM (`gemma4:26b`) — single combined prompt: "is this grounded? answerable without paper? not generic CS?" | LLM | ~10-15% |

**Rule:** Stage 6.2 and 6.3 are non-negotiable hard rejects (no rewriting).
Stage 6.5 outputs a `rewrite_suggestion` field; rewrites are accepted only on
re-pass through 6.1–6.4.

### Stage 7 — Difficulty calibration
Run 3 Ollama models on each surviving question, judge with one judge call
each (still local, gemma 27b judge):

| Behavior | Tag |
|---|---|
| All 3 correct | `easy` |
| Top 2 correct, weakest wrong | `medium` |
| Only strongest correct | `hard` |
| All 3 wrong | `flag-for-review` (usually ambiguous, not hard) |

`flag-for-review` items are dropped from v1 unless trivially fixable. Don't
bother manually reviewing them all — just drop and move on.

### Stage 8 — Splits

| Split | Source | Size target (v1, 300 total) |
|---|---|---|
| dev | random sample of validated Qs | 30 |
| test | random sample, no overlap with dev | 200 |
| synthesis | all cluster-synthesis Qs | 50 |
| adversarial | all misconception Qs | 20 |
| hidden | v5.1: 30-50 fresh PDFs, re-run pipeline | (deferred) |

Each split is its own JSONL. `manifest.json` records counts, generator model
versions, prompt versions, seed hashes.

### Stage 9 — Judge calibration
- **Manual** (~1 hour, the only manual step): grade ~30 Qs from the test split.
- Run **GPT-4o** and **Gemini 2.5 Pro** as judges over 100 Qs (the same 30 +
  70 more). Compute:
  - Cohen's kappa(human, GPT-4o)
  - Cohen's kappa(human, Gemini)
  - Cohen's kappa(GPT-4o, Gemini)
- If any kappa < 0.6, the questions in that subset are likely ambiguous —
  flag them for fix or drop, regenerate, repeat.

---

## 5. Final question schema (JSONL line)

```json
{
  "id": "NB-HPN-000421",
  "benchmark_version": "v5.0",
  "category": "Bottleneck Diagnosis and End-to-End Reasoning",
  "difficulty": "hard",
  "question_type": "diagnosis",
  "question": "A transfer over a 100 Gbps WAN reaches only 12 Gbps...",
  "reference_answer": "The likely bottleneck is destination storage write...",
  "source_papers": ["paper_0182", "paper_0441"],
  "evidence": [
    {
      "paper_id": "paper_0182",
      "passage_id": "paper_0182__p7",
      "quote": "destination disk write bandwidth ceiling..."
    }
  ],
  "keywords": ["bottleneck", "iperf", "storage"],
  "requires_calculation": false,
  "synthetic_scenario": true,
  "inspired_by_papers": ["paper_0182", "paper_0441"],
  "generator_model": "gemma4:26b",
  "critic_model": "gemma4:26b",
  "prompt_version": "v5.0-questions",
  "seed": 42
}
```

Shape is back-compatible with v4: existing fields are kept, new fields are
additive. The existing
[`hpn_qa_benchmark.py`](../NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py)
loader needs only a small change to read JSONL and ignore the extra fields.

---

## 6. Categories (kept aligned with v4)

Use v4's 10 categories so the existing evaluator and dashboards don't break.
Stage 4 clustering tells you which categories the corpus actually supports;
drop empty ones, don't invent new ones for v1.

```
- Transfer Parameters: Definitions and Roles
- Concurrency Tuning and Scaling
- Pipelining and Small-File Optimization
- Parallelism and Large-File Optimization
- Dataset Partitioning and Mixed Workloads
- BDP-Based Reasoning and Window Sizing
- Bottleneck Diagnosis and End-to-End Reasoning
- Adaptive and Online Optimization
- Fairness, Stability, and Shared Networks
- Practical HPN Scenarios and Design
```

Coverage target per category: 50-100 questions, balanced across difficulty
(25/45/30) and type (concept/reasoning/calculation/comparison/diagnosis/
scenario/task per v4).

---

## 7. Anti-contamination rules (non-negotiable)

1. **No 8-gram from reference_answer may appear verbatim in
   `research_corpus_v3.json`.** Deterministic check, Stage 6.3.
2. **Cosine similarity** of reference_answer to nearest training passage
   < 0.92. Stage 6.3.
3. **Generator and judges live in different vendor families.** Generator =
   Ollama (Qwen/Gemma); judges = OpenAI + Google.
4. **The benchmark must NEVER feed back into SFT or CPT data.**
5. **Hidden split** (Stage 8) eventually built from PDFs not in
   `research_corpus_v3.json` — gives a true generalization signal. v1 ships
   without hidden split; v5.1 adds it after sourcing 30-50 fresh PDFs.

---

## 8. Repo layout

```
Benchmark-Generator/
├── PLAN.md                     ← this file
├── README.md
├── config.yaml                 ← K, thresholds, model names, seeds
├── requirements.txt
├── manifest.json               ← written at run end (versions, counts)
│
├── prompts/
│   ├── card_generation.txt
│   ├── question_generation.txt
│   ├── synthesis_generation.txt
│   ├── adversarial_generation.txt
│   ├── critic.txt
│   ├── misconceptions.yaml
│   └── judge.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py               ← typed loader for config.yaml
│   ├── io_utils.py             ← JSONL streaming I/O w/ resume
│   ├── ollama_client.py        ← sync client w/ retry, tolerant JSON parsing
│   ├── schema.py               ← pydantic models
│   ├── embeddings.py           ← sentence-transformer wrapper, used from Phase 2 on
│   ├── validators.py           (Phase 6, Stage 6 deterministic checks)
│   ├── stage_01_chunk.py
│   ├── stage_02_filter.py
│   ├── stage_03_cards.py
│   ├── stage_04_cluster.py
│   ├── stage_05a_questions.py
│   ├── stage_05b_synthesis.py
│   ├── stage_05c_adversarial.py
│   ├── stage_06_validate.py
│   ├── stage_07_calibrate.py
│   ├── stage_08_split.py
│   └── stage_09_judge.py
│
├── scripts/
│   └── smoke_test.py           ← Phase 0 verifier (config/schema/io/ollama)
│
├── data/
│   ├── passages.jsonl
│   ├── filtered_papers.jsonl
│   ├── paper_cards.jsonl
│   ├── paper_cards_clustered.jsonl
│   ├── candidate_questions.jsonl
│   ├── synthesis_candidates.jsonl
│   ├── adversarial_candidates.jsonl
│   ├── validated_questions.jsonl
│   └── calibrated_questions.jsonl
│
├── benchmark/
│   ├── hpn_benchmark_v5_dev.jsonl
│   ├── hpn_benchmark_v5_test.jsonl
│   ├── hpn_benchmark_v5_synthesis.jsonl
│   ├── hpn_benchmark_v5_adversarial.jsonl
│   └── hpn_benchmark_v5_hidden.jsonl   (v5.1)
│
└── reports/
    ├── stage_stats.md          ← reject rates per stage
    ├── difficulty_distribution.md
    └── judge_calibration.md
```

Each `stage_NN_*.py` has a single CLI: `python -m src.stage_03_cards
--input data/filtered_papers.jsonl --output data/paper_cards.jsonl --resume`.
The `--resume` flag reads existing output and skips already-processed inputs
— this is what makes the pipeline crash-recoverable.

---

## 9. Operational notes

- **Ollama parallelism:** `OLLAMA_NUM_PARALLEL=4` for Gemma 27B (≈ 4 × 17 GB
  on one GPU works at Q4). For the 72B model, drop to `OLLAMA_NUM_PARALLEL=1`
  per GPU and run two GPUs concurrently with two ollama hosts on different
  ports.
- **Streaming I/O:** every stage appends to its output JSONL line-by-line so
  a SIGKILL doesn't lose work. `--resume` reads completed IDs.
- **JSON parsing from local LLMs:** Gemma/Qwen will sometimes wrap JSON in
  markdown fences or trail commentary. Use a tolerant extractor (regex for
  the outermost `{...}` or `[...]`, then `json.loads`). Reject + retry once
  with temperature=0.2 before giving up on that input.
- **Log everything:** `data/_logs/<stage>.jsonl` with `{input_id,
  output_status, model, latency_ms, retries}`. Lets you compute reject rates
  and find slow papers.
- **Cost ceiling:** $0 for generation. Budget for judges:
  ~100 Qs × 2 vendors × ~2K tokens ≈ $5 total. Set hard caps in code anyway.

---

## 10. Concrete week-1 milestones

| Day | Output |
|---|---|
| 1 | `config.yaml`, `requirements.txt`, `src/ollama_client.py`, `src/schema.py`. Stage 1 (`stage_01_chunk.py`) end-to-end on the corpus. |
| 2 | Stage 2 filter implemented; embeddings cached; verify ~600 surviving papers and eyeball top-50 / bottom-50 by score. |
| 3 | Stage 3 card generation on a 30-paper pilot. **Manually review 10 cards.** Iterate prompt until cards are clean. |
| 4 | Stage 3 full run; Stage 4 clustering; eyeball cluster keywords. Lock category list. |
| 5 | Stage 5a/b/c on a 50-card pilot. **Run Stage 6 on the pilot output and look at reject rates.** Tune prompts if Stage 6.5 critic rejects > 30%. |
| 6 | Full Stage 5 + Stage 6 run. |
| 7 | Stage 7 calibration, Stage 8 splits, Stage 9 manual + judge calibration. Ship `hpn_benchmark_v5_dev.jsonl` + `hpn_benchmark_v5_test.jsonl`. |

After day 7, the only remaining work is collecting fresh PDFs for the
hidden split (v5.1).

---

## 11. What this plan deliberately omits (and why)

- **GROBID re-ingestion** — adds days of setup; sliding-window passages are
  good enough to support evidence quotes via substring match.
- **Per-validator LLM calls (one per check)** — combined critic prompt does
  grounding + answerability + genericness in one Ollama call. Roughly 4×
  cheaper, slightly noisier; the deterministic checks (6.1–6.4) carry the load.
- **Multi-agent rewriter loop** — single rewrite attempt only. If a Q fails
  twice, drop it. The corpus has 600 papers worth of fresh material; we don't
  need to salvage every candidate.
- **MCQ variant** — open-ended only in v1.
- **Phase 1 "expand from v4"** — left out entirely. Paper-driven generation
  populates the categories without amplifying v4's blind spots.

These are explicitly v5.1+ work, not v1.

---

## 12. Open decisions (default below if unspecified)

| Decision | Default |
|---|---|
| Use 72B generator? | Optional install of `qwen2.5:72b-instruct-q4_K_M`. v1 default uses already-installed `gemma4:26b`. |
| K (papers kept by relevance filter) | 300 |
| Min cluster size (HDBSCAN) | 3 (with `cluster_selection_method=leaf`) |
| Per-card question count | 4 |
| Synthesis Qs per cluster pair | 1 |
| Total target benchmark size | **300** (dev 30, test 200, synth 50, adversarial 20). Hidden split deferred to v5.1. |
| Memorization-leak n-gram size | 8 |
| Memorization-leak cosine threshold | 0.92 |
| Dedup cosine threshold | 0.88 |

Override any default in `config.yaml` before running.
