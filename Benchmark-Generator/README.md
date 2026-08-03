# Benchmark-Generator

Automated pipeline that turns the 2371-paper HPN research corpus into a
**242-question HPN-skill benchmark** for evaluating locally-trained LLMs.

Generation runs entirely on local Ollama models. Paid APIs (GPT-4o,
Gemini 2.5 Pro) are reserved for Phase 9 inter-judge calibration only.

## Status

**Phases 0-8 complete.** Phase 9 (paid-API judge calibration) is the only
remaining stage and is optional for v1 release.

| Phase | Stage | Output | Status |
|---|---|---|---|
| 0 | Bootstrap | smoke test, schemas | done |
| 1 | Passage chunking | 34,019 passages | done |
| 2 | Relevance filter | 300 papers (cosine ≥ 0.78) | done |
| 3 | Paper cards | 300 cards (qwen3.5:9b) | done |
| 4 | Topic clustering | 6 clusters + 230 noise | done |
| 5a | Per-paper questions | 1148 candidates (gemma4:26b) | done |
| 5b | Synthesis questions | 69 candidates | done |
| 5c | Adversarial questions | 36 candidates | done |
| 6 | Validation gauntlet | 366 accepted (29.2%) | done |
| 7 | Difficulty calibration | 262 accepted (71.6%) | done |
| 8 | Splits + manifest | **242 question benchmark** | done |
| 9 | Judge calibration | dual-API kappa | not started |

Detailed phase notes, decision log, and daily log: [DEV_PLAN.md](DEV_PLAN.md).

## What you get

`benchmark/` contains the v5.0 benchmark (gitignored — produced by running
the pipeline):

| File | Count | Source |
|---|---|---|
| `hpn_benchmark_v5.0_dev.jsonl` | 30 | Stratified sample from per-paper Qs |
| `hpn_benchmark_v5.0_test.jsonl` | 200 | Stratified sample from per-paper Qs |
| `hpn_benchmark_v5.0_synthesis.jsonl` | 4 | All cluster-synthesis Qs that survived |
| `hpn_benchmark_v5.0_adversarial.jsonl` | 8 | All misconception-targeting Qs that survived |
| `manifest.json` | — | Models, prompt versions, seed, counts, shortfall notes |

Synthesis (4/50) and adversarial (8/20) shipped short of target — the
upstream evidence-quote validator was strict and the candidate streams
were small to begin with. Documented in `manifest.notes`.

Each split mirrors the source distribution within ~1% per difficulty:
**~32% easy / 41% medium / 27% hard**, across 10 HPN categories.

## Quick read

- **Get running in 5 minutes:** [QUICKSTART.md](QUICKSTART.md)
- **Question schema reference:** [docs/SCHEMA.md](docs/SCHEMA.md)
- **What the pipeline does (architecture):** [PLAN.md](PLAN.md)
- **How it was built (status, decisions, daily log):** [DEV_PLAN.md](DEV_PLAN.md)
- **Hard rules and conventions:** [DEVELOPMENT.md](DEVELOPMENT.md)
- **Distribution stats:** [reports/difficulty_distribution.md](reports/difficulty_distribution.md), [reports/clusters.md](reports/clusters.md), [reports/stage_stats.md](reports/stage_stats.md)

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py    # 13/13 checks should pass
```

You also need Ollama running locally with `gemma4:26b`, `qwen3.5:9b`, and
`llama3.2:3b` pulled. See [DEVELOPMENT.md](DEVELOPMENT.md) for the GPU policy
(stage runners pin to `cuda:0`; Ollama keeps gemma4:26b on `cuda:1`).

## Layout

```
config.yaml                  runtime config (models, K, thresholds, seed, paths)
requirements.txt             Python deps
PLAN.md                      pipeline architecture (stages, schemas, prompts)
DEV_PLAN.md                  phase-by-phase build status + decision log
DEVELOPMENT.md               developer rules and gotchas
QUICKSTART.md                runnable steps for users
docs/
  SCHEMA.md                  question JSONL schema reference
src/
  config.py                  typed config loader
  io_utils.py                JSONL streaming with --resume
  schema.py                  Pydantic v2 models
  ollama_client.py           sync HTTP client + tolerant JSON extraction
  embeddings.py              sentence-transformer wrapper (cuda:0)
  validators.py              deterministic Phase 6 validators
  stage_01_chunk.py          passage chunking
  stage_02_filter.py         relevance filter
  stage_03_cards.py          paper cards (qwen3.5:9b chat + think=False)
  stage_04_cluster.py        HDBSCAN topic clustering
  stage_05a_questions.py     per-paper questions
  stage_05b_synthesis.py     cross-paper synthesis questions
  stage_05c_adversarial.py   misconception-targeting questions
  stage_06_validate.py       schema → quote → leak → dedup → critic
  stage_07_calibrate.py      empirical difficulty + drop strong-wrong
  stage_08_split.py          dev/test/synthesis/adversarial + manifest
  stage_09_enrich.py         provenance join (calibration ladder + validation) + sidecar
prompts/                     prompt templates per stage
scripts/
  smoke_test.py              13-check sanity runner
  repair_evidence_quotes.py  fixes paraphrased quotes before Phase 6
data/                        intermediate JSONL (gitignored)
benchmark/                   final v5.0 splits + manifest (gitignored)
reports/                     auto-generated stats and reports
```

## Key model choices

| Role | Model | Notes |
|---|---|---|
| Card generator | `qwen3.5:9b` | via `/api/chat` with `think=False`. gemma4:26b loops on long output. |
| Question / synthesis / adversarial | `gemma4:26b` | shorter structured outputs work fine |
| Critic (Phase 6) | `gemma4:26b` | last filter; deterministic gates carry the load |
| Calibration weak / mid / strong | `llama3.2:3b` / `qwen3.5:9b` / `gemma4:26b` | three-tier capability ladder |
| Calibration judge | `gemma4:26b` | with `think=False` to avoid empty-content thinking-mode bug |
| Embeddings | `BAAI/bge-large-en-v1.5` | on `cuda:0` |
| Inter-judge (Phase 9) | `gpt-4o`, `gemini-2.5-pro` | different vendor families = contamination resistance |

See [DEVELOPMENT.md](DEVELOPMENT.md) for known gemma4:26b behaviors (token-loop
threshold, paraphrased-quote tendency, thinking-mode empty content) and
the workarounds.

## Reproducing the benchmark

The full pipeline is driven by `seed: 42` in `config.yaml`. Running every
stage end-to-end on the same corpus reproduces the same 242-question
benchmark byte-for-byte, modulo upstream model nondeterminism (see notes
on temperature in [PLAN.md](PLAN.md)).

End-to-end runtime on 2× RTX A6000 with `OLLAMA_NUM_PARALLEL=4`:
**~6-10 GPU-hours** (chunking 12 sec, filter 12 min one-time embed,
cards ~62 min, 5a ~35 min, 5b ~15 min, 5c ~6 min, validate ~8 min,
calibrate ~48 min, split <1 sec).
