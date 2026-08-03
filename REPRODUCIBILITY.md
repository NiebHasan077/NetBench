# Reproducibility Record

## Publication Baseline

The single-repository baseline is constructed from the component commits that
were recorded by the original outer repository before flattening:

| Module | Pinned pre-conversion commit |
| --- | --- |
| `NetBench-LLM/` (Direct Pipeline) | `cdd85199ad2cb50c908238be84301921619647a9` |
| `NetBench-RAG/` | `65b0616192f3c5f38d9fa1e785c3f6e97132c0a2` |

The baseline is intentionally a stable publication starting point, not an
assertion that one code state generated every reported result.

## Sanitized Conversion Deviations

The source component history was preserved privately before conversion. The
publication baseline differs from its pinned inputs only where required for
security or artifact policy:

- `NetBench-LLM/gemini_api_key.txt` is omitted because credentials must never
  be published.
- `NetBench-LLM/docs/QA_BENCHMARK_GUIDE.md` is sanitized to use an environment
  variable example rather than an embedded credential literal.
- `NetBench-RAG/data/raw/research_corpus_new.json` is omitted under the
  raw-corpus exclusion policy and because it triggered credential-pattern
  review during conversion.

The repository owner confirmed on 2026-05-24 that affected exposed
credentials were revoked or rotated.

The conversion intentionally preserves inherited source formatting rather than
creating unrelated code churn; existing whitespace warnings in materialized
module files are therefore not normalized in the baseline commit.

## Historical Run Evidence

Each currently available paper-reported run has:

- one monorepo snapshot commit and immutable tag;
- a per-run metadata summary and human-readable run summary;
- references and checksums for external artifacts;
- an evidence classification of `exact`, `recovered_from_artifact`, or
  `partially_recovered`.

Run-specific snapshots, rather than this baseline branch, are the
authoritative records for reported experimental results.

All five currently registered runs are classified as
`recovered_from_artifact`. See `PAPER_REPRODUCIBILITY_STATEMENT.md` for
paper/supplement wording and `experiment_registry/run_index.csv` for the
authoritative snapshot tags and per-run limitations.

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

## Private Recovery Provenance

Pre-flattening component Git bundles and raw evidence captures are retained
outside the publication repository. The Direct Pipeline history bundle
contains revoked historical credential material and must remain private. No
private bundle is to be committed, published, pushed, or attached to an
artifact release.
