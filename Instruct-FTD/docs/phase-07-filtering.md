# Phase 7: Filtering, Judging, And Deduplication

Phase 7 turns raw Phase 6 candidates into a cleaner intermediate dataset that
is ready for later mixing and train/validation splitting.

## Purpose

Candidate generation intentionally over-produces examples. This phase removes:

- malformed examples
- low-value front-matter-derived examples
- weakly grounded examples
- duplicate and near-duplicate examples
- optional judge-rejected examples

## Implementation

Filtering is implemented in:

- [src/instruct_ftd/filtering.py](src/instruct_ftd/filtering.py)
- [scripts/filter_candidates.py](scripts/filter_candidates.py)

## Pipeline Stages

### 1. Heuristic filter

Reject examples that have:

- missing required fields
- bad question formatting
- missing evidence
- obvious front matter / table-of-contents noise
- assistant meta-language
- responses that are too short or too long
- very low lexical overlap with the supplied evidence

### 2. Deduplication

The dedup stage removes:

- exact duplicate question+response pairs
- exact duplicate questions
- near-duplicate questions using a lightweight SimHash comparison

### 3. Optional judge

If `--judge_model` is supplied, the pipeline asks a local Ollama model to rate:

- groundedness
- specificity
- usefulness

and to return a binary keep/reject decision in JSON.

## Outputs

Each run writes:

- `filtered_hpn.jsonl`
- `filtered_rag.jsonl`
- `reject_log.jsonl`
- `filtering_report.json`

If judge mode is enabled:

- `judge_results.jsonl`
- `judge_results.jsonl` also acts as the resume checkpoint log for judged runs

## Typical Usage

`--input_hpn` and `--input_rag` have no default values and must always be
passed explicitly:

Heuristic-only filtering:

```bash
python3 scripts/filter_candidates.py \
  --input_hpn data/intermediate/candidates/<run>/candidate_hpn.jsonl \
  --input_rag data/intermediate/candidates/<run>/candidate_rag.jsonl
```

Heuristic filter plus local judge:

```bash
python3 scripts/filter_candidates.py \
  --input_hpn data/intermediate/candidates/<run>/candidate_hpn.jsonl \
  --input_rag data/intermediate/candidates/<run>/candidate_rag.jsonl \
  --judge_model qwen3.5:9b
```

Resume a stopped judged run:

```bash
python3 scripts/filter_candidates.py \
  --input_hpn data/intermediate/candidates/<run>/candidate_hpn.jsonl \
  --input_rag data/intermediate/candidates/<run>/candidate_rag.jsonl \
  --judge_model qwen3.5:9b \
  --output_dir data/intermediate/filtered/<run> \
  --resume
```

When judge mode is enabled, `judge_results.jsonl` acts as the checkpoint log for
completed judge decisions.

Heuristic-only filtering is not checkpointed because it is fast and
deterministic. Resume support is only relevant when the local judge model is in
the loop.

## Progress Output

All phases emit timestamped status lines to stderr. Example for a judged run:

```
[14:55:00] Phase 7 filtering  |  hpn_in=712  rag_in=179  judge=qwen3.5:9b
[14:55:00] Heuristic filter done  |  hpn: 712 -> 689 kept (23 rejected)  |  rag: 179 -> 171 kept (8 rejected)
[14:55:00] Deduplication done  |  860 -> 831 unique (29 removed)
[14:55:01] Phase 7 judge  |  model=qwen3.5:9b  |  records=831 (0 already done, 831 pending)
[14:55:01]   [1/831] req_000042 | hpn_fact_qa | paper_0001  ->  keep
[14:55:03]   [2/831] req_000043 | rag_grounded_qa | paper_0001  ->  rejected (low groundedness)
...
[15:08:44] Phase 7 judge done  |  elapsed=13m 43s  |  kept=794  rejected=37
[15:08:44] Phase 7 filtering done  |  elapsed=13m 44s  |  hpn=641  rag=153  total_rejected=97
```

## Current Notes

The first implementation is intentionally conservative and lightweight. It
prefers strong obvious rejections and simple duplicate removal over complex
semantic ranking. If later runs show too many false positives or false
negatives, the thresholds can be tightened or relaxed.
