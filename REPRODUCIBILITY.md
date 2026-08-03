# Reproducibility Record

## Publication Baseline

This repository is a stable publication baseline: a maintained source state that
supports the released benchmark, the analysis, and the documentation. It is
deliberately **not** an assertion that one code state generated every reported
result. Per-run source states are recorded separately as snapshot tags; see
`PROVENANCE.md` for what those tags do and do not guarantee.

## Exclusions Applied To This Baseline

The baseline omits material that is excluded by policy rather than by accident:
credentials of any kind, raw corpora, model weights and checkpoints, and
retrieval indexes. `.gitignore` is the authoritative exclusion policy, and
`ARTIFACTS.md` records what is excluded and why.

Inherited source formatting is preserved rather than normalised, so the baseline
does not carry unrelated whitespace churn.

## Dataset-Provenance Modules

The publication repository now includes two source-only provenance modules in
addition to the experimental modules:

- `Benchmark-Generator/` creates the HPN v5.0 benchmark from a local paper
  corpus. Generated work directories and private corpora are excluded from Git.
- `Instruct-FTD/` creates the instruction fine-tuning dataset from a local paper
  corpus plus generic anchor sources. Generated `data/`, `reports/`, source
  PDFs, and full-text corpora are excluded from Git.

The stable handoff contract is source-module based: regenerate benchmark JSONL
and instruction JSONL locally, then provide those files to `NetBench-LLM/` and
regenerate retrieval stores for `NetBench-RAG/`. This repository does not
distribute the full paper corpus, source PDFs, or final instruction datasets.

## Historical Run Evidence

Each currently available paper-reported run has:

- one monorepo snapshot commit and immutable tag;
- a per-run metadata summary and human-readable run summary;
- references and checksums for external artifacts;
- an evidence classification of `exact`, `recovered_from_artifact`, or
  `partially_recovered`.

Run-specific snapshots remain the authoritative record of the **source state**
for each reported run.

Currently registered recovered runs are classified as
`recovered_from_artifact`. See `PROVENANCE.md` for what that status means and
how far the evidence chain holds, and `experiment_registry/run_index.csv` for
the authoritative snapshot tags and per-run limitations.

## Consolidated Results-of-Record

The per-model evaluation outputs are consolidated onto `main` under
`NetBench-LLM/outputs/by_model/<model>/` (benchmark reports, judged workbooks,
adaptation reports, run metadata, and profiling results), so the full
cross-model comparison can be built and reviewed from a single branch. Each
model directory carries a `PROVENANCE.md` linking the consolidated outputs back
to their authoritative snapshot commit and tag. Raw per-question answer shards
and logs are intentionally excluded; large weights, checkpoints, corpora, and
retrieval indexes remain undistributed as described below.

`main` is therefore the aggregated **results-of-record** for the paper, while
`snapshot/*` branches/tags remain the immutable **source-state provenance**.

## Large Artifact Availability

Large artifacts are not distributed with this publication repository and will
not be deposited in Zenodo, OSF, Hugging Face, or institutional artifact
storage for this release. This includes model weights, fine-tuned
checkpoints, raw corpora, and RAG index stores.

`ARTIFACTS.md` records hashes for large evidence that survived recovery where
available; those hashes document provenance rather than public download
locations. Users who execute the code must use the supplied scripts to
download accessible dependencies and regenerate derived artifacts, subject to
required model/data access. This repository does not claim turnkey exact
score reproduction from hosted historical weights or indexes.

## Paper Acquisition And Corpus Construction

The publication repository includes source-only provenance modules for the
upstream data path:

- `Collect-papers/` acquires open-access papers and deduplicates local PDF
  collections. Its downloaded PDFs, metadata logs, generated indexes, and API
  credentials are excluded from Git.
- `Paper2Corpus/` converts a local PDF collection into JSON records shaped as
  `[{"text": "..."}]`. Its generated corpora and intermediate text files are
  excluded from Git.

The maintained duplicate-detection interface is `Collect-papers/paperbase`.
