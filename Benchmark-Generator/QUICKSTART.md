# Quickstart

Two paths depending on what you want.

## Path A — Use the existing v5.0 benchmark

If `benchmark/hpn_benchmark_v5.0_*.jsonl` already exists locally, you can
load and evaluate without re-running the pipeline.

```python
import json

def load(split):
    with open(f"benchmark/hpn_benchmark_v5.0_{split}.jsonl") as f:
        return [json.loads(line) for line in f]

dev = load("dev")          # 30  — for prompt tuning, no test contamination
test = load("test")        # 200 — primary scoring set
synthesis = load("synthesis")    # 4  — cross-paper reasoning
adversarial = load("adversarial")# 8  — misconception probes

manifest = json.load(open("benchmark/manifest.json"))
print(manifest["counts"], manifest["models"])
```

For the question schema, see [docs/SCHEMA.md](docs/SCHEMA.md).

The existing v4 evaluator at
`../NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py` reads this JSONL with no
schema changes — additive fields are ignored.

## Path B — Regenerate the benchmark from scratch

### Prerequisites

- 2× GPUs with ≥40 GB each (e.g. RTX A6000). One is held by Ollama for
  `gemma4:26b` (~36 GB resident); embedding/clustering pins to `cuda:0`.
- Ollama running on `localhost:11434` with these models pulled:
  ```
  ollama pull gemma4:26b
  ollama pull qwen3.5:9b
  ollama pull llama3.2:3b
  ```
- The HPN corpus at `../NetBench-LLM/data/raw/research_corpus_v3.json`
  (2371 papers; the relative path is set in `config.yaml`).

### Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py    # 13/13 checks must pass
```

If smoke check #13 fails, Ollama is not reachable or the configured
`card_generator` model is not pulled.

### Run the pipeline

Each stage is its own runnable module. All support `--resume`. Wall-clock
times are for 2× RTX A6000 with `OLLAMA_NUM_PARALLEL=4`.

```bash
# Phase 1 — chunk the corpus into 1000-token passages (~12 sec)
.venv/bin/python -m src.stage_01_chunk

# Phase 2 — embed + filter to top-K papers (~12 min first time, cached after)
.venv/bin/python -m src.stage_02_filter --device cuda:0

# Phase 3 — generate paper cards (~62 min, 300 papers, 2 workers)
.venv/bin/python -m src.stage_03_cards --workers 2

# Phase 4 — HDBSCAN topic clustering (seconds)
.venv/bin/python -m src.stage_04_cluster --device cuda:0

# Phase 5 — three candidate streams
.venv/bin/python -m src.stage_05a_questions --workers 2     # ~35 min
.venv/bin/python -m src.stage_05b_synthesis --workers 2     # ~15 min
.venv/bin/python -m src.stage_05c_adversarial --workers 2   # ~6 min

# Repair gemma4-paraphrased evidence quotes before validation
.venv/bin/python scripts/repair_evidence_quotes.py

# Phase 6 — validation gauntlet (~8 min, 1253 candidates)
.venv/bin/python -m src.stage_06_validate --workers 2

# Phase 7 — empirical difficulty calibration (~48 min, 366 questions)
.venv/bin/python -m src.stage_07_calibrate --workers 2

# Phase 8 — write the four splits + manifest.json (<1 sec)
.venv/bin/python -m src.stage_08_split
```

Output files appear under `benchmark/` after Phase 8.

### Resuming after a crash

Every stage skips IDs already present in its output JSONL. Re-run with
`--resume` to pick up where it stopped.

```bash
.venv/bin/python -m src.stage_05a_questions --resume --workers 2
```

Phase 7 is special: `--resume` reads the `data/_logs/stage_07.jsonl` log
(not the output file), so previously-dropped questions aren't
re-processed.

### Pilot mode

Most stages accept `--limit N` for a fast sanity check. Recommended order
when first wiring up the pipeline:

```bash
.venv/bin/python -m src.stage_03_cards --limit 5 --workers 1
.venv/bin/python -m src.stage_05a_questions --limit 10 --workers 1
.venv/bin/python -m src.stage_07_calibrate --limit 10 --workers 1 --verbose
```

Pilot output goes to the same files as the full run; clear them between
pilot rounds if you want a clean slate.

## Path C — Phase 9 (paid-API judge calibration)

Not yet implemented. The plan is in [PLAN.md §3 stage 9](PLAN.md#stage-9--judge-calibration)
and [DEV_PLAN.md Phase 9](DEV_PLAN.md). Requires `OPENAI_API_KEY`
and `GEMINI_API_KEY` at the paths configured in `config.yaml`.

## Troubleshooting

- **`OllamaError: chat failed`** — Ollama not running, or the model isn't
  pulled. `curl localhost:11434/api/tags` to check.
- **Empty responses from `gemma4:26b`** — thinking-mode bug. Pass
  `think=False`. Already handled in stage_07; if you see it elsewhere,
  search `_think_param` in `stage_07_calibrate.py`.
- **OOM on `cuda:1`** — Ollama already holds gemma4:26b there (~36 GB).
  Pin embedding/clustering to `cuda:0` via `--device cuda:0`.
- **Phase 5b/5c shows 0 survivors after Phase 6** — gemma4:26b
  paraphrased the evidence quotes. Run
  `scripts/repair_evidence_quotes.py` and re-run validation with
  `--sources 5b 5c --resume`.
- **Stage 6 stats report only shows the latest sources run** — combine
  manually from `data/_logs/stage_06.jsonl`. The report is overwritten,
  not merged.

For deeper context (decision history, model gotchas, daily log), see
[DEV_PLAN.md](DEV_PLAN.md) and [DEVELOPMENT.md](DEVELOPMENT.md).
