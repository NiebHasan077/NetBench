# Phase 5: Pilot Generation

This phase validates the prompt families and artifact pipeline against a real
local teacher model in Ollama before full candidate generation.

## Inputs

- `data/intermediate/normalized/normalized_papers.jsonl`
- `data/intermediate/chunks/short_chunks.jsonl`
- `data/intermediate/chunks/rag_chunks.jsonl`
- `data/intermediate/chunks/rag_prompt_bundles.jsonl`

## Runner

Pilot generation is implemented by:

- [scripts/run_pilot_generation.py](scripts/run_pilot_generation.py)

The runner:

- selects high-confidence structured papers
- chooses evidence according to each prompt family
- calls the local Ollama `/api/chat` endpoint
- stores raw model responses
- parses successful examples into canonical dataset rows
- stores failures separately for inspection

## Default Pilot Scope

By default the pilot runs a representative subset of families:

- `hpn_fact_qa`
- `hpn_concept_explanation`
- `hpn_comparison`
- `hpn_diagnosis`
- `rag_grounded_qa`
- `rag_unanswerable`

Default teacher model:

- `gemma4:26b`

## Output Artifacts

Each run writes a timestamped directory under:

- `data/intermediate/pilot/run_<timestamp>/`

Artifacts in that directory:

- `pilot_raw_generations.jsonl`
- `pilot_examples.jsonl`
- `pilot_failures.jsonl`
- `pilot_checkpoint.jsonl`
- `pilot_manifest.json`

## Typical Usage

```bash
python3 scripts/run_pilot_generation.py \
  --model gemma4:26b \
  --paper_limit 4
```

For a lighter smoke test with specific families:

```bash
python3 scripts/run_pilot_generation.py \
  --model gemma4:26b \
  --paper_limit 1 \
  --families hpn_fact_qa rag_grounded_qa
```

List all available family names and exit:

```bash
python3 scripts/run_pilot_generation.py --list_families
```

Resume a stopped run in the same directory:

```bash
python3 scripts/run_pilot_generation.py \
  --model gemma4:26b \
  --paper_limit 4 \
  --output_dir data/intermediate/pilot/<run> \
  --resume
```

The checkpoint file is append-only. Completed request IDs are skipped on resume.
If the output directory already contains pilot artifacts, the runner requires
`--resume` before it will reuse that directory.

## Progress Output

The runner emits timestamped status lines to stderr as it works:

```
[13:04:21] Phase 5 pilot  |  model=gemma4:26b  |  papers=4  |  requests=24 (0 already done, 24 pending)  |  run=run_20260412T130421Z
[13:04:21] DRY RUN — Ollama will not be contacted.
[13:04:22]   [1/24] pilot_req_000000 | hpn_fact_qa | paper_0003 | easy  ->  ok
[13:04:25]   [2/24] pilot_req_000001 | rag_grounded_qa | paper_0003 | medium  ->  ok
...
[13:07:44] Phase 5 pilot done  |  elapsed=3m 23s  |  success=21  failures=3
```

Each line shows `[done/total]`, the request ID, family, paper, difficulty, and outcome.
Failed requests are written to `pilot_failures.jsonl` and do not stop the run.

## Review Guidance

After a pilot run, inspect:

- whether the response JSON parses reliably
- whether the generated question is non-trivial
- whether the response is grounded in the evidence
- whether answer length is on target
- whether RAG examples abstain correctly when unanswerable
