# Instruct-FTD

`Instruct-FTD` prepares a synthetic instruction fine-tuning dataset for
high-performance networking and data-transfer research. It is designed to turn
your local paper corpus into a mixed instruction dataset that can later be
consumed by `NetBench-LLM` with minimal downstream changes.

The final handoff format is model-agnostic and intentionally simple:

- `system`
- `question`
- `response`
- `category`

Additional metadata is preserved for filtering, traceability, and analysis:

- `source_split`
- `task_type`
- `paper_id`
- `paper_title`
- `difficulty`
- `question_type`
- `prompt_length_bucket`
- `evidence`

The target final mix is:

- `60%` HPN instruction data
- `20%` RAG-style grounded data
- `20%` generic assistant anchors

## What The Repository Does

The pipeline covers Phases 1 through 9:

1. define the dataset contract
2. normalize the OCR/text corpus
3. build short and RAG chunk inventories
4. define teacher prompt families
5. run pilot generation against local Ollama
6. generate full HPN and RAG candidates
7. filter, judge, and deduplicate
8. import generic OpenOrca/Dolly anchors
9. build final train/validation JSONL datasets

Phase 10, integration into `NetBench-LLM`, is intentionally outside this repo.

## Repository Layout

```text
Instruct-FTD/
├── config/
├── docs/
├── scripts/
├── src/instruct_ftd/
├── tests/
├── pdfs/
├── research_corpus_new.json
├── README.md
└── quickstart.md
```

Key directories:

- `src/instruct_ftd/`: core pipeline logic
- `scripts/`: command-line entrypoints for each phase
- `docs/`: phase-by-phase implementation notes
- `tests/`: standard-library unit suite for the pure-Python core
- `pdfs/`: local paper PDFs

Generated outputs are written under:

- `data/intermediate/`
- `data/final/`
- `reports/`

These are ignored by git.

## Requirements

Core requirements:

- Python 3.10+
- local file access to:
  - `research_corpus_new.json`
  - `pdfs/`

Optional requirements:

- Ollama for Phases 5 to 7
- a local teacher model such as `gemma4:26b`
- Hugging Face `datasets` if you want to import OpenOrca, Dolly, or Hermes JSON anchors directly in Phase 8

The test suite uses only the Python standard library.

## Virtual Environment

This repo is intended to be used from a local Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal because the core pipeline uses the
standard library. The only current install-time dependency is for the optional
Hugging Face anchor-import path.

## Documentation Map

Use the documents in this order:

- [quickstart.md](quickstart.md)
  Fastest path to a working local run.
- [docs/phase-01-dataset-contract.md](docs/phase-01-dataset-contract.md)
  Final schema and mixture policy.
- [docs/phase-02-normalization.md](docs/phase-02-normalization.md)
- [docs/phase-03-chunking.md](docs/phase-03-chunking.md)
- [docs/phase-04-teacher-prompts.md](docs/phase-04-teacher-prompts.md)
- [docs/phase-05-pilot-generation.md](docs/phase-05-pilot-generation.md)
- [docs/phase-06-candidate-generation.md](docs/phase-06-candidate-generation.md)
- [docs/phase-07-filtering.md](docs/phase-07-filtering.md)
- [docs/phase-08-anchor-import.md](docs/phase-08-anchor-import.md)
- [docs/phase-09-mixing.md](docs/phase-09-mixing.md)

## Pipeline Summary

### Phase 1: Dataset Contract

Defines the canonical record schema and target mixture policy.

References:

- [docs/phase-01-dataset-contract.md](docs/phase-01-dataset-contract.md)
- [config/dataset_contract.json](config/dataset_contract.json)

### Phase 2: Normalization

Normalizes the corpus into structured paper objects with:

- stable `paper_id`
- extracted title, authors, abstract
- heuristic section segmentation
- token estimates
- best-effort PDF mapping

Primary outputs:

- `data/intermediate/normalized/normalized_papers.jsonl`
- `reports/phase2_normalization_report.json`

Reference:

- [docs/phase-02-normalization.md](docs/phase-02-normalization.md)

### Phase 3: Chunking

Builds two evidence views:

- `short_chunks.jsonl` for local HPN QA generation
- `rag_chunks.jsonl` and `rag_prompt_bundles.jsonl` for grounded multi-excerpt prompts

Reference:

- [docs/phase-03-chunking.md](docs/phase-03-chunking.md)

### Phase 4: Teacher Prompt Families

Defines prompt families for:

- HPN fact QA
- concept explanation
- comparison
- limitation analysis
- calculation
- diagnosis
- scenario
- task-oriented recommendation
- grounded RAG QA
- unanswerable RAG QA

Reference:

- [docs/phase-04-teacher-prompts.md](docs/phase-04-teacher-prompts.md)

### Phase 5: Pilot Generation

Runs a small sample against local Ollama before full generation.

Reference:

- [docs/phase-05-pilot-generation.md](docs/phase-05-pilot-generation.md)

### Phase 6: Candidate Generation

Builds the full synthetic candidate pool over selected HPN-relevant papers.

Reference:

- [docs/phase-06-candidate-generation.md](docs/phase-06-candidate-generation.md)

### Phase 7: Filtering

Applies:

- structural validation
- heuristic grounding checks
- exact and near-duplicate removal
- optional Ollama judging

Reference:

- [docs/phase-07-filtering.md](docs/phase-07-filtering.md)

### Phase 8: Generic Anchor Import

Normalizes OpenOrca, Dolly, and optional Hermes JSON-mode examples into the
same contract as the HPN/RAG examples.

Reference:

- [docs/phase-08-anchor-import.md](docs/phase-08-anchor-import.md)

### Phase 9: Final Mixing

Builds:

- `train.jsonl`
- `validation.jsonl`
- `mix_report.json`

Strict mode enforces the target `60/20/20` mix. Split assignment is leak-aware:
HPN and RAG examples from the same `paper_id` stay in the same split.
An append-only helper can add validated `generic_json` anchors after strict
mixing without removing any existing final examples.

Reference:

- [docs/phase-09-mixing.md](docs/phase-09-mixing.md)

## Main Commands

The fastest way to get started is in [quickstart.md](quickstart.md).

Typical commands:

```bash
python3 scripts/normalize_corpus.py
python3 scripts/build_chunks.py
python3 scripts/run_pilot_generation.py --model gemma4:26b --paper_limit 3 --output_dir data/intermediate/pilot/demo_run
python3 scripts/run_candidate_generation.py --model gemma4:26b --paper_limit 50 --output_dir data/intermediate/candidates/demo_run
python3 scripts/filter_candidates.py --input_hpn ... --input_rag ... --output_dir data/intermediate/filtered/demo_run
python3 scripts/import_generic_anchors.py
python3 scripts/build_final_dataset.py --filtered_hpn ... --filtered_rag ... --generic_anchors ... --output_dir data/final/demo_run --strict_mix
python3 scripts/import_hermes_json_anchors.py --target_count 166 --output_jsonl data/intermediate/anchors/hermes_json_anchor.jsonl
python3 scripts/append_json_anchors_to_final.py --input_train data/final/demo_run/train.jsonl --input_validation data/final/demo_run/validation.jsonl --json_anchors data/intermediate/anchors/hermes_json_anchor.jsonl --count 166 --output_dir data/final/demo_run_plus_json
```

## Test Suite

Run the local unit suite with:

```bash
python3 -m unittest discover -s tests -v
```

Current test coverage focuses on the non-networked core:

- normalization and chunking
- prompt/generation helper behavior
- filtering and judge parsing
- anchor import
- final mixing and leakage control

## Checkpointing And Resume

Checkpointing is implemented for the long-running Ollama-backed phases:

- Phase 5 pilot generation
- Phase 6 full candidate generation
- Phase 7 filtering when `--judge_model` is enabled

Resume behavior is explicit:

- use the same `--output_dir`
- rerun the same command with `--resume`

Checkpoint artifacts:

- Phase 5: `pilot_checkpoint.jsonl`
- Phase 6: `candidate_checkpoint.jsonl`
- Phase 7 judge path: `judge_results.jsonl`

If an output directory already contains run artifacts, the scripts now refuse to
reuse it unless `--resume` is supplied. That prevents accidental duplication or
mixing of unrelated runs.

Phases without checkpointing:

- Phase 2 normalization
- Phase 3 chunking
- Phase 8 anchor import
- Phase 9 final mixing

Those phases are deterministic file-building steps and are expected to be
rerun directly rather than resumed.

## Final Deliverables

Production outputs from this repo are:

- `data/final/<run>/train.jsonl`
- `data/final/<run>/validation.jsonl`
- `data/final/<run>/mix_report.json`

These files are the intended handoff into `NetBench-LLM`.

## Current Limitations

- normalization still depends on OCR-heavy heuristics
- PDF mapping is positional and best-effort
- RAG bundle generation is useful but still conservative
- there is no integrated trainer in this repository by design

## Recommended Workflow

1. run the tests
2. normalize the corpus
3. build chunks
4. run a pilot with Ollama
5. inspect pilot outputs
6. generate and filter the full candidate set
7. import generic anchors
8. build the final strict-mix dataset

That keeps failures visible early instead of discovering them after a long full-corpus generation run.

For long-running stages, always provide an explicit `--output_dir` so the run
can be resumed cleanly later.
