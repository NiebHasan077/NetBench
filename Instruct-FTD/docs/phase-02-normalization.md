# Phase 2: Corpus Normalization

Phase 2 converts a local raw OCR-flattened corpus, typically
`NetBench-LLM/data/raw/research_corpus_new.json`, into structured paper records
that later phases can reason over. The corpus itself is not distributed in Git.

## Input

- `research_corpus_new.json`
- optional positional PDF mapping from `pdfs/`

The current corpus is a JSON list of records where each record contains one
flattened `text` field. The text is noisy:

- whitespace is collapsed
- OCR has injected split characters into words
- section headings are inconsistent
- some records contain front matter or table-of-contents-like text

## Implementation

Normalization is implemented in:

- [src/instruct_ftd/normalize.py](src/instruct_ftd/normalize.py)
- [scripts/normalize_corpus.py](scripts/normalize_corpus.py)

The normalizer performs:

- title and author extraction from the leading text
- abstract detection
- section splitting using Roman and numeric heading heuristics
- fallback body extraction when structure cannot be recovered
- offline token estimation
- stable `paper_id` assignment
- best-effort PDF mapping using sorted positional correspondence

## Output Artifacts

- `data/intermediate/normalized/normalized_papers.jsonl`
- `reports/phase2_normalization_report.json`

Each normalized paper keeps:

- `paper_id`
- `title`
- `authors`
- `abstract`
- `sections`
- `parse_confidence`
- `parse_method`
- `source_pdf`

## Typical Usage

```bash
python3 scripts/normalize_corpus.py
```

Override any path explicitly:

```bash
python3 scripts/normalize_corpus.py \
  --corpus_path ../NetBench-LLM/data/raw/research_corpus_new.json \
  --output_jsonl data/intermediate/normalized/normalized_papers.jsonl \
  --report_path reports/phase2_normalization_report.json
```

## Output

The script prints a summary line on completion:

```
Normalizing 212 papers from research_corpus_new.json...
Done in 1.4s  |  papers=212
```

If the corpus file is missing, the script exits immediately with a clear error
rather than crashing later with an unhelpful traceback.

## Current Notes

The normalizer is intentionally heuristic and offline-safe. It does not depend
on remote services. Title extraction and section recovery are good enough for
chunking and generation, but some OCR-heavy papers still contain noisy front
matter. Later filtering phases are expected to reject or downweight those cases.

This phase is deterministic and does not implement checkpointing. Re-run it
directly if the input corpus changes or the output needs to be regenerated.
