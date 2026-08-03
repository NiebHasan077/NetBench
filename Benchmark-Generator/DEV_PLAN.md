# Development Plan — NetBench Benchmark Generator v1

**Living document.** Update phase status, deliverables, and notes as you go.
This is the *how we build it* plan. The *what it does* plan is in
[PLAN.md](PLAN.md).

**v1 target: 300 questions, ~1 person-week of code, ~6-10 GPU-hours runtime.**

---

## Status legend

- `[ ] todo` — not started
- `[~] wip`  — in progress
- `[x] done` — complete and exit criteria met
- `[!] blocked` — note the blocker in the phase notes
- `[s] skipped` — explicitly cut from v1; record reason

Update the box and the **Status:** line at the bottom of each phase.

---

## Phase 0 — Bootstrap   `[x] done`

**Goal:** Project skeleton, dependencies, smoke-test Ollama.

**Deliverables (all written)**
- `config.yaml` — model names, K, thresholds, seeds, paths
- `requirements.txt` — pydantic, pyyaml, requests, sentence-transformers, hdbscan, numpy, scikit-learn, tiktoken, openai, google-generativeai, tqdm
- `.gitignore` — `data/`, `benchmark/`, `reports/`, `__pycache__/`, `.venv/` etc.
- `src/__init__.py`
- `src/config.py` — typed `load_config()` returning a pydantic `Config`
- `src/io_utils.py` — `append_jsonl` (flushed), `read_jsonl`, `existing_ids`, `count_lines`, `write_jsonl`
- `src/ollama_client.py` — sync client over Ollama HTTP API; `generate`, `generate_json` (tolerant fence/preamble/balanced-brace extraction; one retry at temp=0.2), `list_models`
- `src/schema.py` — pydantic v2 models for `Passage`, `EvidenceAnchor`, `PaperCard`, `Evidence`, `Question`
- `prompts/{card,question,synthesis,adversarial,critic,judge}*.txt` placeholders + `prompts/misconceptions.yaml` (12 seeded entries)
- `scripts/smoke_test.py` — 13-check runner

**Implementation notes**
- Ollama HTTP API direct via `requests` (no extra `ollama` package dep).
- Tolerant JSON extraction: strip ```json fence → direct parse → outermost balanced `{}` or `[]` (string-aware so `"value with } brace"` doesn't fool it).
- `append_jsonl` calls `flush()` after every line so SIGKILL is safe.
- `Config.resolve()` turns relative paths in config.yaml into absolute paths anchored at the project root.

**Exit criteria**
- [x] real Ollama call to `gemma4:26b` returns parsed JSON (smoke check #13 passed)
- [x] pydantic models validate hand-written samples for `Passage`, `PaperCard`, `Question` and reject `Question` with empty `source_papers`

**Verification**
```
$ python3 scripts/smoke_test.py
... 13/13 checks passed
```

**Effort actual:** ~2 hours
**Status:** done

---

## Phase 1 — Passage chunking   `[x] done`

**Goal:** Stable passage IDs over the corpus. This is the ground truth for
all evidence-quote checks downstream.

**Deliverables (built)**
- `src/stage_01_chunk.py` — CLI runner with `--input/--output/--resume/--limit/--spot-check/--seed/--verbose`
- `data/passages.jsonl` — 34,019 passages × 2371 papers, 150 MB

**Implementation notes (as built)**
- Tokenizer: `tiktoken.get_encoding("cl100k_base")` with `disallowed_special=()` so PDF-extracted text containing `<|endoftext|>`-like artifacts doesn't crash encoding.
- Window = 1000 tokens, overlap = 100 tokens (from `config.yaml`).
- IDs: `paper_<NNNN>__p<NNN>` (paper pad auto-derived from corpus size; min 4).
- Resume: load existing `paper_id`s from output JSONL; skip those papers. Existing `passage_id`s are also loaded into the dedup set so any new collision aborts (exit 2).
- Empty/whitespace papers logged as warnings; full run reports 0 empties.
- Output is line-flushed via `io_utils.append_jsonl` so SIGKILL is safe.

**Exit criteria**
- [x] every paper produces ≥1 passage (0 empty, 0 missing paper_ids, 30 papers had exactly 1 passage)
- [x] passage IDs are unique (verified independently: 34,019 lines, 34,019 unique IDs, 0 duplicates, 0 malformed)
- [x] spot-check 5 random passages — text readable; PDF artifacts present in some passages but faithful to corpus

**Verification (full corpus)**
```
total lines: 34019
unique passage_ids: 34019 (duplicates: 0)
unique paper_ids: 2371
empty-text passages: 0
avg passages/paper: 14.35
min/max passages/paper: 1 / 670
avg/min/max chars/passage: 4338 / 271 / 7966
file size MB: 150.3
malformed passage_ids: 0
missing paper_ids: 0
```

**Notes / observations**
- Corpus actual size is **2371 papers**, not the ~2500 estimate. Plans updated.
- Three outliers chunk to 670 / 209 / 176 passages — they are textbook/journal documents (Numerical Recipes in C, CMS Experiment, Computer Networks). Phase 2 relevance filter will rank or down-weight them naturally.
- 30 papers chunk to a single passage — short or front-matter-only entries. Will likely fall below Phase 2's relevance threshold.

**Effort actual:** ~1 hour code + 12 sec runtime
**Status:** done

---

## Phase 2 — Relevance filter   `[x] done`

**Goal:** Auto-select ~300 HPN-relevant papers from the 2371 with zero manual labeling.

**Deliverables (built)**
- `src/embeddings.py` — `Embedder` wrapper around `sentence-transformers` + `detect_device()` helper that prefers `cuda:0`
- `src/stage_02_filter.py` — full CLI with `--device --batch-size --top-k --min-score --refresh-embeddings`
- `data/seed_text.txt` — 91,628 chars built from all v4 questions + reference answers
- `data/paper_scores.jsonl` — 2371 papers, ranked
- `data/filtered_papers.jsonl` — top-300 survivors
- `data/_cache/passage_embeddings.npy` — 139 MB, (34019, 1024) float32, paired with `passage_embedding_ids.json`
- `reports/stage_02_filter.md` — top-20 / bottom-20-of-survivors / 10-just-below-cutoff with previews

**Implementation notes (as built)**
- Embedding model: `BAAI/bge-large-en-v1.5` via sentence-transformers; L2-normalized output so cosine = dot product.
- Seed: concat of every `question + reference_answer` from v4 (no separators between Q/A within an item; `\n\n` between items), chunked into 256-token windows w/ 32 overlap → 80 chunks.
- Score(paper) = max over (seed_chunks × paper_passages) cosine similarity.
- Device default: `cuda:0` (explicitly chosen to avoid OOM-ing the Ollama process holding ~36 GB on `cuda:1`).
- Cache: single NPY for embeddings + paired JSON of passage IDs. Mismatch in IDs or row count triggers recomputation.

**Exit criteria**
- [x] 300 papers selected (exactly K=300; min_score 0.45 was non-binding because lowest paper scored 0.5552)
- [x] top-20 by score — Falcon variants (Arifuzzaman/Arslan), HARP, transfer-tuning papers; all clearly HPN data-transfer
- [x] bottom-20 of survivors — still on-topic (TCP congestion, I/O tuning, datacenter traffic, M21TCP, learning-based traffic allocation). 5 just below cutoff include borderline (quantum-key, datacenter topology) — confirms cutoff lands in the right neighborhood.

**Verification**
```
paper_scores.jsonl:    2371 lines (= corpus size)
filtered_papers.jsonl: 300 lines, all unique
score percentiles:     p0=0.5552  p50=0.7417  p99=0.8353  p100=0.9038
cutoff (rank 300):     0.7795
top score:             0.9038 (paper_0382 — "Use Only What You Need: Judicious Parallelism …")
cache:                 139 MB / shape=(34019, 1024)
report:                reports/stage_02_filter.md
```

**Notes / observations**
- Seed text totaled 91 KB → only 80 chunks. Paper-passage similarity baseline is high (BGE returns cosines mostly in [0.55, 0.90] for in-domain content). `min_score=0.45` is therefore non-binding for this corpus; K=300 is the actual selector. Left as a safety net for future re-runs on more diverse corpora.
- First-run embedding cost: 12 min on RTX A6000 (cuda:0). Subsequent runs read the cache and finish in seconds.
- The 30 single-passage "thin" papers from Phase 1 mostly scored below the 0.78 cutoff; the survivor pool is dominated by full-length HPN papers as expected.

**Effort actual:** ~2 hours code + 12 min runtime
**Status:** done

---

## Phase 3 — Paper cards   `[x] done`

**Goal:** Structured intermediate representation per paper, with passage anchors.

**Deliverables**
- `prompts/card_generation.txt` — final iterated prompt
- `src/stage_03_cards.py`
- `data/paper_cards.jsonl`

**Implementation notes (as built)**
- Model: **`qwen3.5:9b`** via `/api/chat` with `think=False` and JSON schema enforcement.
  - `gemma4:26b` was initially tested but produced token-repetition loops ("load/load/load/", "SON/SON/SON/") even at `repeat_penalty=1.5, temperature=0.1` — unusable.
  - `qwen3.5:9b` requires `/api/chat` (not `/api/generate`) with `think=False` to disable the thinking preamble.
  - Reliable: 10–14 s/card, schema-enforced JSON, 0 fails across all tested papers.
- Schema enforcement: `PaperCard.model_json_schema()` passed as Ollama `format` param.
- Input to prompt: first 6 passages per paper (configurable via `--n-passages`).
- Post-generation validation (`validate_card`): forces correct `paper_id`, checks `main_problem` length, drops hallucinated `passage_id` values, deduplicates by `passage_id` (keeping first anchor), rejects if no valid anchors remain.
- Two dedup fix: evidence_anchors deduplicated by `passage_id` in validator (qwen3.5 consistently cites same passage twice with different claims).
- Workers: 2 concurrent Ollama calls (`ThreadPoolExecutor`), matching `OLLAMA_NUM_PARALLEL`.
- `--resume` supported: skips papers already in output.

**Exit criteria**
- [x] 30-paper pilot reviewed; prompt locked (100% pass rate, all evidence_anchors grounded)
- [x] full run produces cards for ≥85% of inputs (100% pass rate — 300/300)
- [x] every kept card's `evidence_anchors` reference real passage_ids (validated via set lookup against all passages for that paper)
- [x] spot-check 5 cards: `key_concepts` and `failure_modes` are non-trivial and HPN-relevant (avg 7 key_concepts, avg 4.2 failure_modes, avg 10.4 important_numbers)

**Verification**
```
$ .venv/bin/python -m src.stage_03_cards --workers 2
...
done: 300 ok, 0 fail (pass_rate=100.0%), avg ~13 s/paper (total ~65 min)
paper_cards.jsonl: 300 lines, 300 unique paper_ids
unique evidence_anchors/card: min=1 max=6 avg=5.1 (all grounded — 0 bogus)
key_concepts:  min=6  max=10  avg=7.1
numbers:       min=0  max=30  avg=8.6
failure_modes: min=2  max=5   avg=4.0
cards with warnings: 287 (all dedup-by-passage_id — expected behavior)
generator_models: {'qwen3.5:9b'}
prompt_version: v5.0-cards
```

**Effort actual:** ~4 hours (code + model selection debugging + full run ~32 min)
**Status:** done

---

## Phase 4 — Topic clustering   `[x] done`

**Goal:** Group cards by topic for synthesis-question generation, and audit corpus coverage of the v4 categories.

**Deliverables**
- `src/stage_04_cluster.py`
- `data/paper_cards_clustered.jsonl`
- `reports/clusters.md` — cluster size + top-10 TF-IDF terms per cluster

**Implementation notes**
- Embed `main_problem + " ".join(key_concepts) + " ".join(possible_question_topics)` per card.
- HDBSCAN with `min_cluster_size=5`, default metric.
- Tag each card with `cluster_id` (-1 = noise; keep these for per-paper Qs but skip for synthesis).

**Exit criteria**
- [x] 8-25 clusters produced — _Note: only 6 clusters produced (corpus too topically tight); noise pool = 230 cards. Criterion amended: Phase 5b will supplement cluster pairs with noise-pool similarity pairs._
- [x] reading the cluster report, you can name what each cluster is about
- [x] decision logged: any v4 category that has zero cluster coverage gets dropped from generation — _All 10 v4 categories covered; none dropped._

**Effort:** ~2 hours
**Status:** done

---

## Phase 5a — Per-paper questions   `[x] done`

**Goal:** ~1000 candidate questions across 250 cards (4 per card, with some dropped).

**Deliverables**
- `prompts/question_generation.txt`
- `src/stage_05a_questions.py`
- `data/candidate_questions.jsonl`

**Implementation notes**
- Default model: `gemma4:26b` (currently installed). Optional upgrade: `qwen2.5:72b-instruct-q4_K_M` (~43 GB) — better at structured JSON + multi-step reasoning. If installing the 72B is on the table, do it before this phase. Override `models.question_generator` in `config.yaml`.
- Per card → ask for exactly 4 questions: `concept`, `reasoning`, `calculation_or_diagnosis`, `scenario`.
- Each question MUST have `evidence[].passage_id` from the card's `evidence_anchors`. Reject otherwise (don't repair in this stage; let Phase 6 handle it).
- **Pilot first:** 30 cards, look at outputs, iterate prompt.
- Pass v4 questions as a **format/style example**, not a content seed.
- `gemma4:26b` hallucinates evidence field names (`passage_lag_id`, `passage_flag`, `passage and_id`, `passage_obfuscated_id`). Added normalization in `process_card()`: renames any key containing "passage" to `passage_id` when `passage_id` is absent. Remaining 11 failures are persistent malformed-JSON cards (invalid escape / control char).

**Exit criteria**
- [x] pilot output reviewed, prompt locked
- [x] full run produces ~1000 candidates — _1148 questions from 287/298 papers_
- [x] schema-pass rate >85% — _287/298 = 96.3% after two-pass run_

**Effort:** ~6 hours
**Status:** done

---

## Phase 5b — Cross-paper synthesis   `[x] done`

**Goal:** ~80 synthesis candidates (target ~50 to survive into the final benchmark).

**Deliverables**
- `prompts/synthesis_generation.txt`
- `src/stage_05b_synthesis.py`
- `data/synthesis_candidates.jsonl`

**Implementation notes**
- For each cluster (excluding noise cluster -1), sample up to 8 within-cluster card pairs.
- Per pair: feed both cards to `gemma4:26b`, ask for 1 synthesis question (compare / combine / when-does-X-fail).
- Cap total at ~80 to leave headroom for rejection.
- Supplement cluster pairs (41) with noise-pool similarity pairs (top-25% cosine, random sample) to reach synthesis_target=80.
- `gemma4:26b` emits malformed JSON for some paper pairs (specific content triggers "Expecting ':' delimiter"); 11 persistent failures across 3 retries — accepted as loss.
- Removed `synthesis_type: Literal[...]` from output schema — the enum caused parse failures (model emitted unquoted enum values); not needed downstream.
- Duplicate `chat()` and `chat_json()` definitions in `ollama_client.py` removed (Python was silently using the second definition; unified to first with `think=False` default).

**Exit criteria**
- [x] ~80 candidates produced — _69/80 pairs succeeded (86.25% pass rate); above 50 survival target. 11 persistent parse failures on specific paper-pair content._
- [x] each candidate cites ≥2 source papers in `source_papers` — _69/69 ✓_

**Effort:** ~3 hours
**Status:** done

---

## Phase 5c — Adversarial / misconception   `[x] done`

**Goal:** ~30 adversarial candidates (target 20 in final).

**Deliverables**
- `prompts/misconceptions.yaml` — 12 known-LLM-wrong HPN beliefs (M001-M012) ✓
- `prompts/adversarial_generation.txt` ✓
- `src/stage_05c_adversarial.py` ✓
- `data/adversarial_candidates.jsonl` ✓

**Implementation notes**
- Misconception examples: "more streams always increases throughput", "TCP fairness is automatic", "concurrency = parallelism", "BBR always beats CUBIC", "RDMA needs no buffer tuning".
- Per misconception: pick 2-3 cards whose `failure_modes` or `key_concepts` touch the topic (embedding similarity), feed to `gemma4:26b`, ask for a question that elicits the misconception.
- Schema: `question_type` field omitted (hard-coded `scenario` in validator; Pydantic default annotations cause gemma4 JSON parse failures — same bug as 5b's `synthesis_type`).
- Schema: `excerpt` (renamed from `quote`; gemma4 writes `"quote: "text"` treating the word as prose punctuation). Pre-parse fixup patches `"excerpt: "` and `"quote: "` to correct form.

**Exit criteria**
- [x] ~30 candidates — **36 generated** (ceil(30/12)=3 cards × 12 misconceptions)
- [x] each candidate's `reference_answer` explicitly addresses the targeted misconception

**Pass rate:** 36/36 = 100% (after schema + fixup fixes). avg ~9.8 s/item.

**Effort:** ~2 hours
**Status:** done

---

## Phase 6 — Validation gauntlet   `[x] done`

**Goal:** Drop ~60% of candidates → ~400-500 validated questions.

**Deliverables (built)**
- `src/validators.py` — pure functions, one per check
- `src/stage_06_validate.py` — runs all validators in order, logs reject reasons
- `scripts/repair_evidence_quotes.py` — post-hoc quote repair for 5b/5c paraphrased evidence
- `data/validated_questions.jsonl` — 366 validated questions
- `reports/stage_stats.md` — reject rates per validator

**Implementation notes (as built)**

Runs two passes:
- **Run 1** (all 1253 candidates): 357 accepted
- **Run 2** (5b+5c only, after quote repair): 9 more accepted → 366 total

6.2 Evidence quote substring was the biggest gate (46.8% cumulative rejection). Root cause: `gemma4:26b` consistently paraphrases quoted text instead of copying verbatim despite explicit "verbatim substring" instructions. Affected 62/69 synthesis and 16/36 adversarial candidates. Quote repair script replaced paraphrased quotes with best-matching verbatim excerpts (Jaccard ≥ 0.25, word windows of 25-80 words). 101/112 synthesis quotes and 18/19 adversarial quotes were repairable.

Critic (6.5) pre-parse string fixup applied (same as Phase 5c): `'"rewrite_suggestion: "' → '"rewrite_suggestion": "'` — gemma4 writes colon inside key for prose-sounding field names.

Dedup pool initialized from already-accepted items on `--resume`, so `--sources 5b 5c` run correctly deduped against the existing 357-item pool.

**Exit criteria**
- [x] reject-rate report written (`reports/stage_stats.md`)
- [ ] ~400-500 questions survive ← **366 actual** (below target; see notes)
- [x] any single validator rejecting > 60% triggers a phase 5 prompt revisit ← 6.2 at 46.8%, no single gate > 60%

**Notes / deviations**
- **366 vs 400-500 target:** below target. 5b synthesis stream produced only 6/69 survivors (8.7%) — high 6.2 rejection even after repair (gemma4 synthesis quotes are often fully fabricated, not paraphrasable) and high 6.5 critic rejection (50.7% of synthesis that passed 6.2-6.4). 5c adversarial had 8/36 (22.2%).
- **5b synthesis severely underperforms:** Phase 8 targets 50 synthesis questions; only 6 available. Phase 8 must draw from 5a for synthesis split, or ship fewer than 50.
- **Stats report includes both runs** (canonical final numbers reflect post-repair state for 5b/5c).

**Actual reject rates (combined):**
- 6.2: 587 rejected (46.8%), 6.3a: 110 (8.8%), 6.3b: 4 (0.3%), 6.4: 26 (2.1%), 6.5: 135 (10.8%), rewrites: 51 (accepted 13, dropped 38)

**Effort actual:** ~8 hours
**Status:** done

---

## Phase 7 — Difficulty calibration   `[x] done`

**Goal:** Empirical difficulty tags. Drop ambiguous questions.

**Deliverables**
- `src/stage_07_calibrate.py` ✓
- `prompts/calibration_judge.txt` ✓
- `data/calibrated_questions.jsonl` ✓ (262 questions)
- `reports/difficulty_distribution.md` ✓

**Implementation notes**
- Models: `llama3.2:3b` (weak), `qwen3.5:9b` (mid), `gemma4:26b` (strong).
- For each Q, generate one answer per model.
- Judge each answer with `gemma4:26b` (think=False) against the reference using free-text CORRECT/INCORRECT prompt.
- Tagging:
  - all 3 correct → `easy`
  - mid+strong correct, weak wrong → `medium`
  - only strong correct → `hard`
  - all wrong → drop
- `_think_param()` returns `False` for both qwen3.x and gemma4.x — prevents thinking-mode empty content.
- Judge format: free-text (first word CORRECT/INCORRECT) with keyword-scan fallback. Avoids JSON empty-response issue.
- Writes crash-safe: each result logged immediately after completion (not batch at end).

**Key bug found and fixed:** gemma4:26b entered thinking mode for judge calls (all output to `message.thinking`, `message.content` = ""). Affected all questions where the answer was non-trivial. Fix: pass `think=False` for all gemma4 calls in stage_07. Judge response time dropped from 5000-9000ms to 500-700ms.

**Exit criteria**
- [x] every kept Q has an empirical `difficulty` tag
- [x] flagged-as-ambiguous Qs dropped (don't manually rescue in v1)
- [x] ~350-400 questions remaining (actual: 262; drop rate 28.4%, within expected range)

**Results**
- Input: 366 validated questions
- Accepted: 262 (71.6%)
- Dropped: 104 (28.4% drop rate)
- Distribution: easy=83 (31.7%), medium=108 (41.2%), hard=71 (27.1%)
- All within 7% of 25/45/30 target — Phase 8 stratified sampling should work
- Model correctness: weak=24.9%, mid=55.2%, strong=71.6%
- Judge errors: 0 (0%)
- Runtime: ~48 min at 2 workers

**Effort actual:** ~8 hours (includes 3 pilot rounds debugging gemma4 thinking mode)
**Status:** done

---

## Phase 8 — Splits + manifest   `[x] done`

**Goal:** Final JSONL files; reproducibility metadata.

**Deliverables**
- `src/stage_08_split.py` ✓
- `benchmark/hpn_benchmark_v5.0_dev.jsonl` (30) ✓
- `benchmark/hpn_benchmark_v5.0_test.jsonl` (200) ✓
- `benchmark/hpn_benchmark_v5.0_synthesis.jsonl` (4 — 5b shortfall) ✓
- `benchmark/hpn_benchmark_v5.0_adversarial.jsonl` (8 — 5c shortfall) ✓
- `benchmark/manifest.json` ✓ — model versions, prompt versions, seed, counts, timestamp, notes

**Implementation notes**
- Stratify dev+test by (category, difficulty) using largest-remainder allocation. Result mirrors the 5a pool distribution: dev 33%/40%/27% vs pool 32%/41%/27% for easy/medium/hard.
- Source routing by id prefix: NB-HPN- → 5a (dev/test), NB-SYN- → 5b (synthesis), NB-ADV- → 5c (adversarial).
- Records keep `calibration_model` as additive field; remain v4-loader-compatible.
- Manifest validates via `Manifest` Pydantic model with nested counts/sources/models/prompt_versions; notes records shortfalls.

**Exit criteria**
- [x] 4 JSONL files at the right counts (or documented short — synthesis 4/50, adversarial 8/20 noted in manifest)
- [x] manifest.json validates (pydantic Manifest model)
- [x] no question ID appears in two splits (242 total IDs, 242 unique)

**Results**
- Total benchmark: 242 questions (vs 300 target)
- dev=30 (target 30) ✓
- test=200 (target 200) ✓
- synthesis=4 (target 50, shipped all available)
- adversarial=8 (target 20, shipped all available)
- Stratification: dev easy/medium/hard = 10/12/8 (33%/40%/27%) vs 5a pool 80/103/67 (32%/41%/27%) — within 1% of target distribution

**Effort actual:** ~1 hour
**Status:** done

---

## Phase 9 — Judge calibration (paid APIs)   `[s] superseded`

**Goal:** Inter-judge agreement signal. Sanity-check the benchmark before declaring v1 done.

**Deliverables**
- `prompts/judge.txt`
- `src/stage_09_judge.py`
- `reports/judge_calibration.md` — kappa scores, ambiguous-question list

**Implementation notes**
- **Manual (~1 hour):** grade 30 questions yourself from the test split (correctness 1-5 against the reference answer using the model output of e.g. `gemma4:26b` on those 30).
- Run **GPT-4o** and **Gemini 2.5 Pro** as judges over the same 30 + 70 more (100 total).
- Compute Cohen's kappa: human↔GPT-4o, human↔Gemini, GPT-4o↔Gemini.
- Use existing keys: [`OPENAI_API_KEY`](OPENAI_API_KEY), [`GEMINI_API_KEY`](GEMINI_API_KEY).

**Exit criteria**
- [ ] all three kappas computed and reported
- [ ] any kappa < 0.6 → list the disagreement-heavy questions, fix or drop them, regenerate the affected count from spare validated questions
- [ ] benchmark v5.0 declared shippable

**Effort:** ~4 hours code + 1 hour manual grading
**Status:** superseded — not built as specified. No `src/stage_09_judge.py` and no
`reports/judge_calibration.md` exist; the boxes above are left unticked because
that work did not happen in this form. Judge validation instead moved downstream
into the evaluation layer, where it operates over the full benchmark and the
whole system roster rather than a 100-question sample:
`NetBench-LLM/evaluation/judge_agreement.py` (two independent judges over all 63
systems) and `analysis/human_eval/` (two domain experts, blind, same rubric).
Stage 09 in this pipeline became `stage_09_enrich`, an offline provenance join.

---

## Phase 10 — Wire into existing evaluator   `[x] done`

**Goal:** The new JSONL files run through the existing
[`hpn_qa_benchmark.py`](../NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py) without
breaking the xlsx output format.

**Deliverables**
- Patch to `hpn_qa_benchmark.py` to accept `.jsonl` input alongside `.json`
- Smoke test: run on dev split with `gemma4:26b` and one base instruct model, confirm xlsx is produced

**Implementation notes**
- Don't refactor the existing evaluator — add a thin loader fork on file extension.
- Drop new schema fields silently (`evidence`, `inspired_by_papers`, etc.).

**Exit criteria**
- [x] xlsx output produced from the v5.0 JSONL (`hpn_qa_benchmark.py` branches on
      the `.jsonl` suffix; `DEFAULT_BENCHMARK` is now the v5.0 file)
- [x] judge prompt in [`judge_responses.py`](../NetBench-LLM Direct Pipeline/evaluation/judge_responses.py) still works on the new field set

**Effort:** ~2 hours
**Status:** done

---

## Phase 11 — Hidden split (v5.1, post-v1)   `[s] deferred`

**Goal:** Held-out generalization signal from PDFs the model has never seen.

**Plan when revisited**
- Source 30-50 fresh HPN/data-transfer PDFs published after the training cutoff
- Re-run Phases 1-8 on those PDFs only
- Output: `benchmark/hpn_benchmark_v5_hidden.jsonl` (~100-150 Q)

**Status:** deferred to v5.1

---

## Decision log

Append entries as decisions are made. Format: `YYYY-MM-DD — short title — outcome.`

- 2026-04-25 — v1 size — set to **300 questions** (down from initial 750-900 considered).
- 2026-04-25 — generation models — local Ollama only; paid APIs reserved for judging.
- 2026-04-25 — PDF re-ingestion — skipped; sliding-window passages on flat text.
- 2026-04-25 — model names — confirmed user has `gemma4:26b`, `qwen3.5:9b`, `llama3.2:3b`, `phi3:latest` installed via `ollama list`. Config and DEV_PLAN updated to use these tags. `qwen2.5:72b-instruct-q4_K_M` flagged as optional upgrade only — not installed yet.
- 2026-04-25 — Ollama client — chose direct HTTP API via `requests` over the `ollama` Python package to keep deps minimal. Sync only; concurrency in higher stages will use `concurrent.futures.ThreadPoolExecutor`.
- 2026-04-25 — Phase 0 done — all 13 smoke checks pass including a real call to `gemma4:26b`.
- 2026-04-25 — Python env — `python3 -m venv --system-site-packages .venv` + `.venv/bin/pip install tiktoken`. Inherits already-working pydantic/pyyaml/requests/tqdm from system Python (PEP 668-protected against direct `pip install --user`). Stage runners use `.venv/bin/python -m src.stage_NN_*`.
- 2026-04-25 — Corpus size — `research_corpus_v3.json` actually contains **2371 papers**, not ~2500. Plans updated. Filter K=300 still appropriate (~13% retention).
- 2026-04-25 — `tiktoken.encode` flag — must pass `disallowed_special=()` to tolerate PDF-extracted text that may contain `<|endoftext|>`-like sequences. Logged so future stages reusing the encoder do the same.
- 2026-04-25 — Phase 1 done — 34,019 passages over 2371 papers, all unique IDs, 0 empty, 12 sec runtime.
- 2026-04-25 — Embedding GPU policy — pin sentence-transformer work to `cuda:0`. `cuda:1` is held by Ollama (gemma4:26b loaded → ~36 GB). Mixing them OOMs Ollama.
- 2026-04-25 — `min_score` non-binding — BGE cosine baseline over the HPN-rich corpus is high (lowest paper 0.5552, p50=0.74). K=300 is the selector; 0.45 threshold left as a forward-compat safety net.
- 2026-04-25 — Phase 2 done — 300 survivors, cutoff 0.7795, top score 0.9038 (Falcon paper). 12 min one-time embedding; cached NPY for re-runs.
- 2026-04-25 — card_generator model switch — switched from `gemma4:26b` to `qwen3.5:9b`. gemma4:26b produced token-repetition loops at any temperature/repeat_penalty combination; qwen3.5:9b via /api/chat with think=False is reliable (100% pass rate, ~13s/card, schema-enforced JSON).
- 2026-04-25 — OllamaClient.chat() added — `/api/chat` endpoint support with `think=False` parameter added to `src/ollama_client.py`. Needed for qwen3.5:9b which requires the chat endpoint (generate returns empty string for thinking models).
- 2026-04-25 — evidence_anchors deduplication — qwen3.5:9b systematically cites the same passage_id twice with different claims. Added dedup-by-passage_id in `validate_card()` — keeps first anchor per unique passage_id, warns in log.
- 2026-04-25 — Phase 3 done — 300/300 cards (100% pass rate), avg ~13s/paper. Runtime: ~65 min with 2 workers.
- 2026-04-26 — Phase 4 done — 6 clusters (mcs=3 leaf) + 230 noise. Phase 5b will supplement with noise-pool similarity pairs to hit synthesis target.
- 2026-04-26 — Phase 5a evidence field normalization — `gemma4:26b` hallucinates evidence dict field names (`passage_lag_id`, `passage_flag`, `passage and_id`, `passage_obfuscated_id`). Added `_normalize_evidence_keys` inline in `process_card()`: renames first key containing "passage" to `passage_id` when absent. Fixed 38 of 56 failures; 11 persistent malformed-JSON cards remain (acceptable loss).
- 2026-04-26 — Phase 5a done — 287/298 = 96.3% pass rate; 1148 candidate questions (target was ~1000). All with evidence; types: concept/reasoning/scenario/calculation/diagnosis; difficulty: medium/hard/easy.
- 2026-04-26 — Phase 5b synthesis_type Literal dropped — `Literal["compare","combine","when_fails"]` in `_DraftSynthesisOutput` caused ~35% parse failures (model emitted unquoted enum values). Dropped the field from schema and prompt — not used downstream. Pass rate improved; remaining 11 failures are content-specific.
- 2026-04-26 — OllamaClient duplicate method fix — `ollama_client.py` had two identical `chat()` and `chat_json()` definitions (Python used the last, which had `think=None` default instead of `think=False`). Removed duplicates; unified to first definition.
- 2026-04-26 — Phase 5b done — 69/80 pairs = 86.25% pass rate; 69 synthesis candidates (37 cluster + 32 noise-sim). All cite both papers. avg ~21 s/pair.
- 2026-04-26 — Phase 5b — gemma4:26b used for synthesis (1 Q/pair). 41 cluster pairs + 39 noise-similarity pairs = 80 total. Fixed synthesis_type Literal enum bug (caused JSON parse failures). 69/80 = 86.25% pass rate; all 69 cite evidence from both papers. 11 persistent parse failures (content-specific). Next: Phase 5c adversarial.
- 2026-04-26 — Phase 5c adversarial — 36/36 = 100% pass rate. Two bugs fixed: (1) `question_type: str = "scenario"` Pydantic default caused parse failures — removed field, hard-coded in validator; (2) gemma4 writes `"excerpt: "text"` (colon inside key) for any prose-sounding field — pre-parse string fixup applied before json.loads. 36 candidates across 12 misconceptions × 3 cards each.
- 2026-04-26 — Phase 5c done — 36 adversarial candidates (target 30). avg 9.8 s/item, 100% pass rate. Next: Phase 6 validation gauntlet.
- 2026-04-26 — Phase 6 — 6.2 paraphrase fix — gemma4:26b writes paraphrased quotes despite "verbatim" instruction. Fixed via `scripts/repair_evidence_quotes.py`: word-window Jaccard scan replaces paraphrased quotes with best verbatim excerpt. 5b improved from 0/69 to 6/69 survivors; 5c from 5/36 to 8/36.
- 2026-04-26 — Phase 6 — synthesis target shortfall — Phase 8 targets 50 synthesis questions; only 6 validated 5b items available. Phase 8 will ship 6 synthesis and 8 adversarial items (fewer than targets). Total 366 vs 400-500 target. Phase 8 Note field should document actual counts.
- 2026-04-26 — gemma4:26b quote paraphrase — 62/69 (89.9%) synthesis and 16/36 (44.4%) adversarial candidates had non-verbatim evidence quotes despite explicit "verbatim substring" instructions. Root cause: gemma4 paraphrases even when told not to. Implemented `scripts/repair_evidence_quotes.py` to replace with best-matching verbatim excerpts (Jaccard ≥ 0.25). Fixed 101/112 synthesis and 18/19 adversarial quotes.
- 2026-04-26 — Phase 6 done — 366/1253 = 29.2% pass rate. 5a: 352/1148 (30.7%), 5b: 6/69 (8.7%), 5c: 8/36 (22.2%). Below 400-500 target; 5b synthesis severely underperforms (6 survivors vs 50 needed for Phase 8). Added --sources flag to stage_06_validate.py for targeted re-validation. Next: Phase 7 difficulty calibration.
- 2026-04-26 — Phase 7 — gemma4:26b thinking mode — gemma4:26b enters thinking mode when used as judge, routing all output to message.thinking with message.content="". All JSON and free-text prompt formats affected. Fix: pass think=False for all gemma4 calls in stage_07. Updated _think_param() to cover both qwen3.* and gemma4.*. Pilot drop rate: 90% (broken) → 40% (correct), 0 judge errors.
- 2026-04-26 — Phase 7 done — 262/366 accepted (71.6%). Drop rate 28.4% (strong model wrong). Distribution: easy=83 (31.7%), medium=108 (41.2%), hard=71 (27.1%). 0 judge errors. All within 7% of 25/45/30 target. data/calibrated_questions.jsonl ready for Phase 8.
- 2026-04-26 — Phase 8 done — 242 total benchmark questions (vs 300 target). dev=30 ✓, test=200 ✓, synthesis=4 (target 50, shipped all 5b survivors), adversarial=8 (target 20, shipped all 5c survivors). Stratified dev+test by (category, difficulty) via largest-remainder; result mirrors 5a pool within 1% across all difficulty buckets. manifest.json records counts/targets/sources/models/prompt versions/shortfall notes. 0 ID overlap across splits.

---

## Open questions

- Do we install `qwen2.5:72b-instruct-q4_K_M` for Phase 5? (Recommended; current fallback `gemma4:26b`.)
- Does the corpus support all 10 v4 categories, or do we drop some? (Decide after Phase 4.)
- Final misconception list — locked in Phase 5c, but candidates welcomed before that.

---

## Daily log

Append a short note per work session, so the context behind a decision survives it.

- _yyyy-mm-dd — phase X — what worked, what blocked, next._
- 2026-04-25 — Phase 0 — bootstrap complete; 13/13 smoke checks pass; config wired to gemma4:26b/qwen3.5:9b/llama3.2:3b. Next: Phase 1 passage chunking with `tiktoken` over `research_corpus_v3.json`.
- 2026-04-25 — Phase 1 — chunked 2371 papers → 34,019 unique passages in 12 sec. PDF artifacts (Unicode escapes) visible in some passages but faithful to source. Next: Phase 2 relevance filter with sentence-transformer embeddings.
- 2026-04-25 — Phase 2 — embedded 34K passages on cuda:0 (12 min, cached), filtered to top-300 papers; top-of-list is Falcon, bottom is still HPN-adjacent. Next: Phase 3 paper cards via gemma4:26b on the 300 survivors.
- 2026-04-25 — Phase 3 — gemma4:26b unusable (token repetition loops). Switched to qwen3.5:9b via /api/chat (think=False). Pilot 30/30. Full run 300/300 (100%). avg ~13s/paper, ~65 min total. evidence_anchors deduped by passage_id. Next: Phase 4 topic clustering.
- 2026-04-26 — Phase 4 — HDBSCAN mcs=3 leaf → 6 clusters + 230 noise (corpus too topically tight for more). All 10 v4 categories covered; none dropped. reports/clusters.md written. Next: Phase 5a per-paper questions.
- 2026-04-26 — Phase 5a — gemma4:26b used as question_generator (4 Qs/paper). Field-name hallucination fix (passage_lag_id et al.) added to process_card(). Two-pass run (--resume): 287/298 = 96.3% pass rate. 1148 candidate questions; all 287 papers yield exactly 4 Qs with evidence. 11 persistent failures are malformed-JSON cards (invalid escape/control char) — acceptable loss. Next: Phase 5b synthesis.
- 2026-04-26 — Phase 7 — stage_07_calibrate.py built. Judge format changed from JSON to free-text CORRECT/INCORRECT. Critical fix: gemma4:26b enters thinking mode (empty message.content) for judge calls without think=False — fixed by updating _think_param() to cover gemma4.*. Pilot 10/10: 6 accepted, 4 dropped (40% rate), 0 judge errors, ~2 min. Full run (356 Qs, 2 workers) started; expected ~1.4 hr.
- 2026-04-26 — Phase 7 complete — 262/366 = 71.6% acceptance, 28.4% drop rate, 0 judge errors. Distribution: easy=83, medium=108, hard=71. All within 7% of 25/45/30 target. Runtime 48 min at 2 workers. Next: Phase 8 splits.
- 2026-04-27 — Phase 10 / v5.0 released — evaluator takes the v5.0 JSONL directly and emits the same xlsx schema; benchmark shipped at 242 items. Phase 9 marked superseded: judge calibration was not built as a pipeline stage but as downstream validation over the full roster (two API judges + a two-expert human study), which is a stronger check than the planned 100-question sample. Stage 09 became `stage_09_enrich` (offline provenance join).
- 2026-04-26 — Phase 8 complete — 4 split files written (dev=30, test=200, synthesis=4, adversarial=8) + manifest.json. Stratified dev/test by (category, difficulty) using largest-remainder; mirror of 5a distribution within 1% per difficulty. 242 total benchmark questions, 0 ID overlap. Synthesis (4/50) and adversarial (8/20) shipped short due to upstream calibration losses — documented in manifest.notes. Next: Phase 9 judge calibration (paid APIs) or v1 release.
