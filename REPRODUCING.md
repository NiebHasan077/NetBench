# Reproducing NetBench

This file answers one question: **what, exactly, can you get back — and what
can you not?**

The repository makes reproducibility claims of three different strengths, and
they are not interchangeable. Conflating them wastes your time, so they are
separated here, strongest first. Each tier says what it needs, what it gives
you, and where it stops.

| Tier | What you re-derive | Strength | What it costs |
|---|---|---|---|
| **1. Analysis** | every number, table, and figure in the paper | **exact** — byte-identical, gated in CI | one command, ~3 min, no GPU |
| **2. Evaluation** | model answers and judge scores | re-runnable, **not** identical | model weights + API keys, GPU |
| **3. Full pipeline** | corpus, benchmark, training data, models | **cannot** return the same inputs | days of compute; see §4 |

Tier 1 is what the paper's claims rest on and what a reviewer can check in
minutes. Tiers 2 and 3 are how the evidence was produced in the first place.

---

## 1. Tier 1 — every reported number, exactly

```bash
make setup       # build analysis/.venv from requirements.lock (Python 3.11)
make reproduce   # regenerate everything, then verify it matches
```

`make reproduce` exits non-zero if anything comes back different. It runs the
seventeen analysis steps in order, then compares:

- **CSV, Markdown, LaTeX** — byte for byte, no normalisation.
- **XLSX and PDF** — by content, ignoring the creation timestamp each format
  embeds and, for PDF, the byte offsets that timestamp's *length* determines
  (`tools/compare_outputs.py`). A UTC-offset suffix is six bytes longer than
  `Z`, which shifts `startxref` on a machine in another timezone.

This runs on every push (`.github/workflows/reproduce.yml`), so the claim is
checked by something other than the authors' laptop.

**Two committed records are outside the gate** because their inputs are not
redistributed (`ARTIFACTS.md`): the retrieval-audit CSVs
(`analysis/retrieval_recall.py`, which needs the RAG chunk cache) and
`analysis/outputs/sft_overlap_audit.csv` (`make audit-sft`, which needs the
final instruction-tuning records). Both scripts ship and re-run wherever those
inputs exist.

**Where the chain starts.** Tier 1 consumes the *committed judged workbooks*
under `NetBench-LLM/outputs/` and `NetBench-RAG/outputs/`, plus the merged
human rating CSVs. Everything upstream of those is model inference or human
data entry, not analysis. `PROVENANCE.md` traces a single number all the way
back.

---

## 2. Tier 2 — re-running the evaluation

You supply the models; the protocol is here. Answers will be close but not
identical to ours: decoding is greedy (`temperature=0.0`, `do_sample=False`)
and dataset splits are seeded, but GPU non-determinism, library versions, and
above all **hosted judge models that change under a fixed name** all move the
numbers.

**Prerequisites**

| Need | Where it goes | Notes |
|---|---|---|
| Hugging Face token | `NetBench-LLM/hf_token.txt` | to fetch base and instruct weights |
| OpenAI key | `openai_api_key.txt` in the module dir | judge, and the GPT baseline |
| Gemini key | `gemini_api_key.txt` in the module dir | second judge |
| Ollama | `http://localhost:11434` | only for Tier 3 generation, not judging |
| GPU | 2× 48 GB used for the reported runs | 27B variants will not fit on less |

Generation and judging deliberately use **different vendor families**. Keep it
that way if you re-run: a model judging its own family is the failure mode the
split exists to avoid.

**The three phases** (identical in both modules, by design — same prompts,
rubric, and output schema, so results stay comparable):

```bash
# NetBench-LLM (direct)
python evaluation/hpn_qa_benchmark.py   # 1. generate answers   -> evaluations/answers/*.xlsx
python evaluation/judge_responses.py    # 2. LLM-as-judge       -> evaluations/judged/*.xlsx
python evaluation/benchmark_report.py   # 3. comparative report

# NetBench-RAG (retrieval)
python index_corpus.py                  # once, ~30 min on GPU
python evaluation/evaluate_rag.py --model local
python evaluation/judge_responses.py --answer_files outputs/answers/<file>.xlsx
python evaluation/benchmark_report.py
```

`NetBench-LLM/scripts/run_pipeline.sh` orchestrates all eight adaptation
variants (S1–S8) for one model, with hardware auto-detection. Read
`NetBench-LLM/DEVELOPMENT.md` first — it is the run-book for the full matrix.

Drop the resulting judged workbooks into the same paths and Tier 1 will
re-derive the analysis over *your* numbers.

---

## 3. Tier 3 — the full pipeline, module by module

Six modules, one data flow. Each is self-contained: its own `requirements.txt`,
its own virtualenv, its own README. **Always `cd` into the module and use that
module's venv.** There is no top-level build.

The seam between modules is a file. That file is the contract:

| # | Module | Consumes | Produces (the handoff) |
|---|---|---|---|
| 1 | `Collect-papers/` | a domain profile | a directory of open-access PDFs |
| 2 | `Paper2Corpus/` | those PDFs | `research_corpus_v3.json` — 2,371 text records, 133.7 MB |
| 3 | `Benchmark-Generator/` | the corpus | `hpn_benchmark_v5.0.jsonl` — 242 items |
| 4 | `Instruct-FTD/` | the corpus | instruction SFT train/validation JSONL |
| 5 | `NetBench-LLM/` | corpus + SFT data + benchmark | adapted models, answers, judged workbooks |
| 6 | `NetBench-RAG/` | corpus + benchmark | retrieval index, answers, judged workbooks |

Modules 3 and 4 both read the corpus and are independent of each other.
Modules 5 and 6 both read the benchmark. **The benchmark must never feed back
into training data** — the eval set and the train set stay strictly separate.

### 3.1 Acquire papers

```bash
cd Collect-papers && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python collect_papers.py --dry-run   # preview, writes nothing
.venv/bin/python collect_papers.py
```

Queries OpenAlex (and optionally Semantic Scholar, `export S2_API_KEY=...`),
screens for relevance with an LLM filter, and deduplicates against a persistent
paperbase. **This is where exact reproduction becomes impossible** — see §4.

### 3.2 Build the corpus

```bash
cd Paper2Corpus && python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp /path/to/pdfs/*.pdf pdfs/
python3 pipeline.py --keep-intermediate
```

Strips headers, footers, reference lists, captions, and extraction artifacts.
Output is `[{"text": ...}]`. Everything downstream inherits its quality.

### 3.3 Generate the benchmark

```bash
cd Benchmark-Generator && python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py        # 13/13 must pass before you start
```

Then eleven numbered stages, every one supporting `--resume`. Measured on the
reference hardware:

```bash
.venv/bin/python -m src.stage_01_chunk                      # ~12 sec
.venv/bin/python -m src.stage_02_filter --device cuda:0     # ~12 min (cached after)
.venv/bin/python -m src.stage_03_cards --workers 2          # ~62 min, 300 papers
.venv/bin/python -m src.stage_04_cluster
.venv/bin/python -m src.stage_05a_questions --workers 2     # ~35 min
.venv/bin/python -m src.stage_05b_synthesis --workers 2     # ~15 min
.venv/bin/python -m src.stage_05c_adversarial --workers 2   # ~6 min
.venv/bin/python -m src.stage_06_validate                   # ~8 min, 1253 candidates
.venv/bin/python -m src.stage_07_calibrate                  # ~48 min, 366 questions
.venv/bin/python -m src.stage_08_split
.venv/bin/python -m src.stage_09_enrich
```

Roughly **3 hours** end to end. Ollama pins a large model (~36 GB resident) on
`cuda:1`, so any other GPU work must use `cuda:0`. Read
`Benchmark-Generator/DEVELOPMENT.md` — it documents the model failure modes
(thinking-mode blanking, schema parse failures, quote paraphrasing) you will
otherwise rediscover.

### 3.4 Build instruction data

```bash
cd Instruct-FTD && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 scripts/normalize_corpus.py
python3 scripts/build_chunks.py
python3 scripts/run_candidate_generation.py --model <ollama-model> --output_dir <dir>
python3 scripts/filter_candidates.py  --output_dir <dir>
python3 scripts/import_generic_anchors.py
python3 scripts/build_final_dataset.py --output_dir data/final/<run> --strict_mix
```

Long Ollama phases checkpoint; resume with `--resume` and the *same*
`--output_dir`. See `Instruct-FTD/quickstart.md` for the full flag set.

### 3.5 Adapt, generate, judge

Training scripts are under `NetBench-LLM/training/` (continued pre-training,
LoRA, full SFT, adapter merge); evaluation and profiling as in Tier 2.
`docs/STUDY_DESIGN.md` defines the variant taxonomy (P1–P8), and
`docs/ARCHITECTURE.md` maps P-codes to S-codes to run names.

Total training cost across the reported matrix is **not recorded**, and it
dominates everything else here. Budget in GPU-days, not hours.

---

## 4. What you will not get back, and why

These are properties of the problem, not gaps in the packaging. Stating them is
the point of this file.

**The corpus cannot be re-obtained.** `Collect-papers` queries live scholarly
indexes. OpenAlex and Semantic Scholar change continuously — new records,
revised metadata, changed open-access status. A run today returns a different
paper set than the one behind `research_corpus_v3.json`, whose SHA-256
(`68c1148d…`) is recorded in `DATA_PROVENANCE.md` as an **identity record, not
a download link**. Everything downstream is conditioned on that corpus.

**Source PDFs are not redistributable.** They are copyrighted by their
publishers. Only the evidence quotes travel, and even those are carved out of
the CC-BY grant — see `LICENSE-DATA`.

**Generation models are named by tag, not digest.** The provenance manifest
records `gemma4:26b`, `qwen3.5:9b`, `llama3.2:3b`. Ollama tags are mutable;
pulling one now may not give you the weights we used.

**Judge models are hosted APIs.** `gpt-5.1` and `gemini-3.5-flash` will be
revised and eventually retired. A fixed model name is not a fixed model.

**Weights, checkpoints, adapters, and retrieval indexes are not shipped.**
`ARTIFACTS.md` records their checksums as provenance. Regenerate them.

**Snapshot tags are recovered, not captured.** Every registered run carries
`evidence_status = recovered_from_artifact`. The experiments ran across three
compute sites with independently staged working copies and no stage stamped its
commit at launch, so a tag is a reconciled post-run state. `PROVENANCE.md` §4
sets out exactly how far that holds; read a run's `limitations` column in
`experiment_registry/run_index.csv` before citing its tag.

**What this does *not* undermine.** The analysis layer consumes only committed
evaluation outputs, so every cell in every table traces to the raw judged
scores, configuration, and prompts that produced it. A re-run of Tiers 2–3 is
**auditable against** ours — same seed, same prompts, same rubric, same
recorded model names — even where it is not identical to it. That is the
claim, and Tier 1 is where it is enforced.

---

## 5. Getting help from the right file

| | |
|---|---|
| a number → the run that produced it | `PROVENANCE.md` |
| what the benchmark contains | `BENCHMARK_CARD.md` |
| 242 generated → 233 scored | `DATA_RELEASE.md` |
| research questions, variants, statistics | `docs/STUDY_DESIGN.md` |
| module contracts and data flow | `docs/ARCHITECTURE.md` |
| per-run snapshots and their limits | `experiment_registry/`, `REPRODUCIBILITY.md` |
| what you may redistribute | `LICENSE`, `LICENSE-DATA` |
