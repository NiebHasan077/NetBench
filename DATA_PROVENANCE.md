# Data Provenance

NetBench uses a six-module publication layout. The full paper corpus, PDFs,
generated paper indexes, intermediate chunks, final instruction datasets, model
artifacts, and retrieval stores are not distributed in Git.

## Provenance Chain

```text
Collect-papers/
  -> local PDF collection + metadata
  -> Paper2Corpus/
     -> research_corpus_v3.json
     -> Benchmark-Generator/
        -> hpn_benchmark_v5.0*.jsonl
     -> Instruct-FTD/
        -> train.jsonl / validation.jsonl
     -> NetBench-LLM/ and NetBench-RAG/
        -> model adaptation, direct evaluation, retrieval, and RAG evaluation
```

## Published Modules

- `Collect-papers/` contains source, profiles, paperbase duplicate-detection
  tools, and tests for acquiring open-access papers from OpenAlex and Semantic
  Scholar.
- `Paper2Corpus/` contains the local PDF extraction, cleaning, and JSON corpus
  construction pipeline.
- `Benchmark-Generator/` contains the benchmark-generation source, prompts,
  configuration, and docs.
- `Instruct-FTD/` contains the instruction fine-tuning dataset construction
  source, schemas, configuration, docs, and tests.
- `NetBench-LLM/` consumes benchmark JSONL files and optional instruction JSONL
  files for training/adaptation/evaluation.
- `NetBench-RAG/` consumes a local corpus and regenerates retrieval artifacts.

Legacy `Check-Exists/` was preserved privately as the predecessor to
`Collect-papers/paperbase`. It is not imported as a public root module because
the maintained duplicate-detection interface now lives inside `Collect-papers`.

## Private Inputs And Generated Outputs

The following classes are intentionally excluded from Git:

- downloaded paper PDFs and merged PDF collections;
- generated paperbase indexes, duplicate candidates, and deletion logs;
- full-text paper corpora such as `research_corpus_new.json` and
  `research_corpus_v3.json`;
- Paper2Corpus intermediate text files;
- Benchmark-Generator work directories and generated reports;
- Instruct-FTD generated `data/`, `reports/`, and final instruction JSONL
  outputs;
- model weights, checkpoints, retrieval indexes, local virtual environments,
  credentials, and model artifacts.

Users must provide or regenerate these materials locally, subject to the access
terms of the original sources.

## Checksums Of Undistributed Inputs

The inputs below are not redistributed (see the exclusion policy above), so
their checksums are recorded here as identity evidence. They let anyone who
regenerates or otherwise obtains these files confirm they hold the same bytes
the reported results were produced from. They are not download links.

| Artifact | Observed Local Size/Count | SHA-256 |
| --- | ---: | --- |
| `Paper2Corpus/research_corpus_v3.json` | 2371 paper-text records, 133.7 MB | `68c1148da5ec5688efd40c8f8c269cee51ee0617c6df28d33c43fcc1c5bb1048` |
| `Paper2Corpus/research_corpus.json` | 753 paper-text records, 42.7 MB | `0c7573819e006e9d9f7e2a74bde3b4115f748f9f564404c5f9a5965f06c76326` |
| `Collect-papers/paperbase/title_index.json` | generated paperbase index | `cc6783d1bb8b056ed14c2af8f10174562d48c53ee69ad31e00ccf746865dfb1c` |
| `Instruct-FTD/data/final/v3_run_plus_json/train.jsonl` | 3320 rows, 19.0 MB | `a7d01645b3f35c78a74da8f944ecbbf371bf7d805bb8e0e57d384295feafa2e6` |
| `Instruct-FTD/data/final/v3_run_plus_json/validation.jsonl` | 174 rows, 1.0 MB | `5abd7b48d08a59e1cdf98cb075ed01d9c27f7c6006bec72d39d9e74e535b08bf` |
| `NetBench-LLM/data/prompts/hpn_benchmark_v5.0_all.jsonl` (as released, annotated) | 242 rows | `6cd13a184f198e3a0d4679eca69dbe9e8f163fadc89c29695f92bd472c8470a6` |


