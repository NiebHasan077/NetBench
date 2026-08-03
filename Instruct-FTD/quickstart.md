# Quickstart

This project builds a local instruction fine-tuning dataset for HPN-focused
models. The final output is:

- `data/final/train.jsonl`
- `data/final/validation.jsonl`

Each row uses the canonical contract:

- `system`
- `question`
- `response`
- `category`

## Prerequisites

- Python 3.10+
- Local Ollama if you want synthetic HPN/RAG generation
- `gemma4:26b` or another local teacher model already pulled in Ollama
- Optional: Hugging Face `datasets` if you want to import OpenOrca/Dolly from HF

## Create A Virtual Environment

Create and activate the virtual environment before running any pipeline
commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Once the environment is active, install dependencies:

```bash
pip install -r requirements.txt
```

Recommended checks:

```bash
which python3
python3 --version
pip --version
```

The active interpreter should now resolve from `.venv/`.

## Dependency Download Suggestions

The dependency setup is intentionally minimal:

- `pip install -r requirements.txt`
  Use this as the default setup. It installs `datasets`, which is only needed
  if you plan to import OpenOrca/Dolly directly from Hugging Face in Phase 8.
- If you will use only local JSON/JSONL files for generic anchors, the same
  environment is still fine. The extra dependency simply will not be used.
- If you want the leanest possible environment, you can skip `pip install -r requirements.txt`
  and run only the standard-library phases plus Ollama-backed phases. In that
  case, Phase 8 must use `--orca_local` and `--dolly_local`.

Suggested approach by workflow:

- Full workflow with HF anchor import:
  Create the venv, activate it, and run `pip install -r requirements.txt`.
- Local-only workflow:
  Create the venv and activate it. Installing `requirements.txt` is still fine,
  but optional if you will not use HF anchor import.
- Offline or restricted environment:
  Use local anchor files and avoid the HF path entirely.

Ollama itself is not installed through `requirements.txt`. It is expected to be
installed separately on the machine, with the teacher model already available.

## Run The Test Suite

```bash
python3 -m unittest discover -s tests -v
```

## Minimal End-To-End Run

Run these commands from the project root.

Suggested convention:

- use an explicit run name such as `demo_run`, `pilot_v1`, or `full_run_20260411`
- keep the same `--output_dir` if you want resume support on long-running phases

### 1. Normalize the corpus

```bash
python3 scripts/normalize_corpus.py
```

Outputs:

- `data/intermediate/normalized/normalized_papers.jsonl`
- `reports/phase2_normalization_report.json`

The script exits immediately with a clear error if the corpus file is missing.
This phase is deterministic and not resumable. Re-run it directly if needed.

### 2. Build short and RAG chunks

```bash
python3 scripts/build_chunks.py
```

Outputs:

- `data/intermediate/chunks/short_chunks.jsonl`
- `data/intermediate/chunks/rag_chunks.jsonl`
- `data/intermediate/chunks/rag_prompt_bundles.jsonl`
- `reports/phase3_chunking_report.json`

The script exits immediately with a clear error if the normalized papers file is
missing. This phase is deterministic and not resumable. Re-run it directly if
needed.

### 3. Run a small pilot with Ollama

To see all available prompt families before choosing:

```bash
python3 scripts/run_pilot_generation.py --list_families
```

```bash
python3 scripts/run_pilot_generation.py \
  --model gemma4:26b \
  --paper_limit 3 \
  --output_dir data/intermediate/pilot/demo_run
```

Progress is printed to stderr as the run proceeds:

```
[13:04:21] Phase 5 pilot  |  model=gemma4:26b  |  papers=3  |  requests=18 (0 already done, 18 pending)  |  run=demo_run
[13:04:22]   [1/18] pilot_req_000000 | hpn_fact_qa | paper_0003 | easy  ->  ok
...
[13:06:14] Phase 5 pilot done  |  elapsed=1m 53s  |  success=16  failures=2
```

If the run is interrupted, resume it with:

```bash
python3 scripts/run_pilot_generation.py \
  --model gemma4:26b \
  --paper_limit 3 \
  --output_dir data/intermediate/pilot/demo_run \
  --resume
```

The startup banner shows how many requests are already done and how many are
pending, so you can confirm the checkpoint is being honoured.

Review:

- `data/intermediate/pilot/demo_run/pilot_examples.jsonl`
- `data/intermediate/pilot/demo_run/pilot_failures.jsonl`
- `data/intermediate/pilot/demo_run/pilot_manifest.json`
- `data/intermediate/pilot/demo_run/pilot_checkpoint.jsonl`

`pilot_checkpoint.jsonl` is the resume log for completed pilot requests.

### 4. Generate full candidate examples

To see all available families before configuring the run:

```bash
python3 scripts/run_candidate_generation.py --list_families
```

```bash
python3 scripts/run_candidate_generation.py \
  --model gemma4:26b \
  --paper_limit 50 \
  --output_dir data/intermediate/candidates/demo_run
```

Progress is printed to stderr as the run proceeds (one line per request):

```
[13:10:05] Phase 6 candidate generation  |  model=gemma4:26b  |  papers=50  |  requests=231 (0 already done, 231 pending)  |  run=demo_run
[13:10:06]   [1/231] req_000000 | hpn_fact_qa | paper_0001 | easy  ->  ok
...
[13:52:18] Phase 6 candidate generation done  |  elapsed=42m 13s  |  success=218 (hpn=176, rag=42)  failures=13
```

Resume a stopped run:

```bash
python3 scripts/run_candidate_generation.py \
  --model gemma4:26b \
  --paper_limit 50 \
  --output_dir data/intermediate/candidates/demo_run \
  --resume
```

Outputs:

- `candidate_hpn.jsonl`
- `candidate_rag.jsonl`
- `candidate_failures.jsonl`
- `candidate_manifest.json`
- `candidate_checkpoint.jsonl`

`candidate_checkpoint.jsonl` is the resume log for completed candidate requests.

### 5. Filter and deduplicate candidates

`--input_hpn` and `--input_rag` must be passed explicitly — the script will
exit with a clear error if either file is missing:

```bash
python3 scripts/filter_candidates.py \
  --input_hpn data/intermediate/candidates/demo_run/candidate_hpn.jsonl \
  --input_rag data/intermediate/candidates/demo_run/candidate_rag.jsonl \
  --output_dir data/intermediate/filtered/demo_run
```

Optional local judge:

```bash
python3 scripts/filter_candidates.py \
  --input_hpn data/intermediate/candidates/demo_run/candidate_hpn.jsonl \
  --input_rag data/intermediate/candidates/demo_run/candidate_rag.jsonl \
  --output_dir data/intermediate/filtered/demo_run_judged \
  --judge_model qwen3.5:9b
```

Resume a stopped judged run:

```bash
python3 scripts/filter_candidates.py \
  --input_hpn data/intermediate/candidates/demo_run/candidate_hpn.jsonl \
  --input_rag data/intermediate/candidates/demo_run/candidate_rag.jsonl \
  --output_dir data/intermediate/filtered/demo_run_judged \
  --judge_model qwen3.5:9b \
  --resume
```

`judge_results.jsonl` is used as the resume checkpoint for judged filtering.

### 6. Import generic anchors

From local files:

```bash
python3 scripts/import_generic_anchors.py \
  --orca_local /path/to/openorca_like.jsonl \
  --dolly_local /path/to/dolly_like.jsonl
```

Or from HF datasets if `datasets` is installed:

```bash
python3 scripts/import_generic_anchors.py
```

Output:

- `data/intermediate/anchors/generic_anchor.jsonl`

This phase is not resumable. Re-run it directly if needed.

### 7. Build the final mixed dataset

`--filtered_hpn` and `--filtered_rag` must be passed explicitly — the script
will exit with a clear error if either file is missing:

```bash
python3 scripts/build_final_dataset.py \
  --filtered_hpn data/intermediate/filtered/demo_run/filtered_hpn.jsonl \
  --filtered_rag data/intermediate/filtered/demo_run/filtered_rag.jsonl \
  --generic_anchors data/intermediate/anchors/generic_anchor.jsonl \
  --output_dir data/final/demo_run \
  --strict_mix
```

Outputs:

- `data/final/demo_run/train.jsonl`
- `data/final/demo_run/validation.jsonl`
- `data/final/demo_run/mix_report.json`

This phase is not resumable. Re-run it directly if needed.

## Expected Dataset Mix

In strict mode, the final dataset uses:

- `60%` HPN
- `20%` RAG
- `20%` generic anchors

## Notes

- Generated artifacts are ignored by git.
- All long-running phases (pilots, candidate generation, and judged filtering)
  print timestamped per-request progress lines to stderr, so you always know
  how far along a run is.
- Pilot and candidate generation continue past individual Ollama failures and
  log them to failure JSONL files.
- Checkpoint/resume is implemented for Ollama-backed phases: pilot generation,
  candidate generation, and judged filtering. Pass `--resume` together with the
  same `--output_dir` to continue an interrupted run. The startup banner shows
  how many requests were already completed.
- The `--list_families` flag on the generation scripts prints all available
  prompt family names and exits without starting a run.
- `filter_candidates.py` and `build_final_dataset.py` require explicit input
  paths — no stale defaults. Missing files produce a clear error.
- `NetBench-LLM` integration is intentionally separate from this repo.
