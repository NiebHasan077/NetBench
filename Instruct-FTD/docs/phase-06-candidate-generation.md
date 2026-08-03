# Phase 6: Full Candidate Generation

Phase 6 scales synthetic generation from the pilot stage to the corpus-wide
candidate dataset stage.

## Purpose

This phase consumes:

- normalized papers from Phase 2
- short chunks and RAG bundles from Phase 3
- prompt families from Phase 4
- a local Ollama teacher model validated in Phase 5

and produces raw candidate examples for later filtering.

## Implementation

Full candidate generation is implemented in:

- [src/instruct_ftd/candidate_generation.py](src/instruct_ftd/candidate_generation.py)
- [scripts/run_candidate_generation.py](scripts/run_candidate_generation.py)

## Default Per-Paper Plan

When no `--families` override is supplied, each selected paper receives a
default 4-5 example plan:

- `hpn_fact_qa`
- `hpn_concept_explanation`
- one rotating reasoning family:
  - `hpn_diagnosis` or `hpn_scenario`
- one rotating advanced family:
  - `hpn_comparison`
  - `hpn_limitation_analysis`
  - `hpn_task`
  - `hpn_calculation`
- `rag_grounded_qa`

Optionally:

- `rag_unanswerable` every N papers via `--include_unanswerable_every`

## Selection Rules

The runner currently filters to papers that are:

- structured parses
- above a minimum parse-confidence threshold
- not obviously low-quality front matter
- at least weakly HPN-relevant by keyword score

This avoids wasting generation budget on the worst OCR or off-domain records.

## Output Artifacts

Each run writes a directory under:

- `data/intermediate/candidates/run_<timestamp>/`

Artifacts:

- `candidate_raw_generations.jsonl`
- `candidate_failures.jsonl`
- `candidate_hpn.jsonl`
- `candidate_rag.jsonl`
- `candidate_checkpoint.jsonl`
- `candidate_manifest.json`

## Typical Usage

Full default plan over all eligible papers:

```bash
python3 scripts/run_candidate_generation.py --model gemma4:26b
```

Smaller smoke test with specific families:

```bash
python3 scripts/run_candidate_generation.py \
  --model gemma4:26b \
  --paper_limit 1 \
  --families hpn_fact_qa rag_grounded_qa
```

List all available family names and exit:

```bash
python3 scripts/run_candidate_generation.py --list_families
```

Dry run without contacting Ollama:

```bash
python3 scripts/run_candidate_generation.py \
  --paper_limit 2 \
  --dry_run
```

Resume a stopped run in the same directory:

```bash
python3 scripts/run_candidate_generation.py \
  --model gemma4:26b \
  --output_dir data/intermediate/candidates/<run> \
  --resume
```

The checkpoint file records completed request IDs so resume can skip finished
requests instead of regenerating everything.
If the output directory already contains candidate artifacts, the runner
requires `--resume` before it will reuse that directory.

## Progress Output

The runner emits timestamped status lines to stderr as it works:

```
[13:10:05] Phase 6 candidate generation  |  model=gemma4:26b  |  papers=212  |  requests=923 (0 already done, 923 pending)  |  run=run_20260412T131005Z
[13:10:06]   [1/923] req_000000 | hpn_fact_qa | paper_0001 | easy  ->  ok
[13:10:09]   [2/923] req_000001 | hpn_concept_explanation | paper_0001 | medium  ->  ok
...
[14:52:31] Phase 6 candidate generation done  |  elapsed=1h 42m 26s  |  success=891 (hpn=712, rag=179)  failures=32
```

Each line shows `[done/total]`, the request ID, family, paper, difficulty, and outcome.
Failed requests are written to `candidate_failures.jsonl` and do not stop the run.

## Notes

Phase 6 intentionally over-generates raw candidates. It does not try to solve
quality and deduplication completely. Those are the main responsibilities of
Phase 7.
