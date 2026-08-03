# Phase 3: Evidence Chunking

Phase 3 turns normalized papers into reusable evidence units for synthetic
instruction-data generation.

## Purpose

The generation pipeline needs two distinct evidence views:

- short evidence for direct HPN QA generation
- larger evidence bundles that resemble downstream RAG prompts

This phase builds both.

## Implementation

Chunking is implemented in:

- [src/instruct_ftd/chunking.py](src/instruct_ftd/chunking.py)
- [scripts/build_chunks.py](scripts/build_chunks.py)

Two chunk profiles are currently defined:

- `short`
  - target about 220 tokens
  - used for HPN fact, explanation, diagnosis, comparison, and task families

- `rag`
  - target about 900 tokens
  - used for grounded multi-excerpt generation

The chunker:

- preserves `paper_id`, section, and title metadata
- emits a context prefix for later evidence formatting
- assigns stable `chunk_id`s
- builds deterministic RAG bundle candidates from per-paper chunk windows

## Output Artifacts

- `data/intermediate/chunks/short_chunks.jsonl`
- `data/intermediate/chunks/rag_chunks.jsonl`
- `data/intermediate/chunks/rag_prompt_bundles.jsonl`
- `reports/phase3_chunking_report.json`

## RAG Bundles

Bundle candidates currently include:

- adjacent same-paper chunk pairs
- abstract-plus-body pairs
- some cross-paper heading-aligned pairs

These bundles are the raw material for later grounded and unanswerable RAG
question generation.

## Typical Usage

```bash
python3 scripts/build_chunks.py
```

Override any path explicitly:

```bash
python3 scripts/build_chunks.py \
  --normalized_jsonl data/intermediate/normalized/normalized_papers.jsonl \
  --short_chunks_out data/intermediate/chunks/short_chunks.jsonl \
  --rag_chunks_out data/intermediate/chunks/rag_chunks.jsonl \
  --rag_bundles_out data/intermediate/chunks/rag_prompt_bundles.jsonl \
  --report_path reports/phase3_chunking_report.json
```

## Output

The script prints a summary on completion:

```
Building chunks from 212 papers...
Done in 0.8s  |  short_chunks=1847  rag_chunks=624  rag_bundles=891
```

If the normalized papers file is missing, the script exits immediately with a
clear error.

## Current Notes

Chunking operates on normalized OCR text and uses offline token estimation.
Some chunks still contain noisy front matter or page residue. Phase 5 pilot
generation and Phase 7 filtering are expected to reject the worst of those
examples.

This phase is deterministic and does not implement checkpointing. Re-run it
directly after updating normalized papers or chunking settings.
