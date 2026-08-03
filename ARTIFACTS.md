# External Artifacts

Large or sensitive materials are excluded from Git and must be referenced by
identifier and checksum when available for documenting a reported run.

Excluded artifact classes include:

- base, fine-tuned, merged, and adapter model weight files;
- checkpoints and optimizer/training-state archives not suitable for Git;
- RAG vector stores and serialized retrieval indexes;
- raw corpora whose size or licensing does not permit publication in Git;
- virtual environments, credentials, and private recovery bundles.

## Distribution Policy

Large artifacts are **not distributed** with this repository and will not be
deposited in Zenodo, OSF, Hugging Face, institutional storage, or another
public artifact host for this release.

Users who run this code must execute the provided pipeline scripts to download
accessible base-model or data prerequisites and to generate derived
fine-tuning, retrieval, and evaluation artifacts, subject to the access terms
of the original sources. Historical fine-tuned weights, checkpoints, and
retrieval indexes are not offered as downloadable release assets. Therefore,
the repository provides recovered code, metadata, selected reported outputs,
and provenance hashes rather than a turnkey exact rerun package.

The hashes below identify large evidence that survived recovery. They are
retained as provenance references only and do not identify public downloads.

## Recovered Large-Artifact Register

One row is recorded for each surviving large artifact for which a checksum was
captured during recovery.

| Run ID | Artifact Type | Recovery Reference | SHA-256 | Availability Notes |
| --- | --- | --- | --- | --- |
| `site-c/llama-3.1-8b` | RAG BM25 index | Private transferred evidence | `f5642a0b40481c520cd19ffec5592b4682cf83f7a54a19562f44adc84184c043` | Not distributed; regenerate by running the RAG indexing workflow. |
| `site-c/llama-3.1-8b` | RAG chunks cache | Private transferred evidence | `d855b3523422c889b7b5417a725e9af3d8735fbda9c29ec5e73989a315a2e53a` | Not distributed; regenerate by running the RAG indexing workflow. |
| `site-c/llama-3.1-8b` | RAG source corpus v3 | Private transferred evidence | `68c1148da5ec5688efd40c8f8c269cee51ee0617c6df28d33c43fcc1c5bb1048` | Not distributed; retained privately and excluded from publication. |
| `site-c/llama-3.1-8b` | RAG source corpus legacy copy | Private transferred evidence | `0c7573819e006e9d9f7e2a74bde3b4115f748f9f564404c5f9a5965f06c76326` | Not distributed; retained privately and excluded from publication. |
| `site-b/gemma-3-1b` | RAG BM25 index | Private transferred evidence | `f5642a0b40481c520cd19ffec5592b4682cf83f7a54a19562f44adc84184c043` | Not distributed; regenerate by running the RAG indexing workflow. |
| `site-b/gemma-3-1b` | RAG chunks cache | Private transferred evidence | `d855b3523422c889b7b5417a725e9af3d8735fbda9c29ec5e73989a315a2e53a` | Not distributed; regenerate by running the RAG indexing workflow. |
| `site-b/gemma-3-1b` | RAG Qdrant storage SQLite | Private transferred evidence | `00b284e4235c7809fb5baa03c706f7d0114797cd207c8659aa103892453abbb0` | Not distributed; regenerate by running the RAG indexing workflow. |
| `site-b/gemma-3-1b` | RAG source corpus v3 | Private transferred evidence | `68c1148da5ec5688efd40c8f8c269cee51ee0617c6df28d33c43fcc1c5bb1048` | Not distributed; retained privately and excluded from publication after security screening. |

Model weights and checkpoint binaries for `site-c/llama-3.1-8b` were not
included in the transfer and are not distributed.

Model weights and checkpoint binaries for `site-b/gemma-3-1b` were not
included in the transfer and are not distributed. Its RAG BM25, chunks-cache,
and source-corpus hashes match privately retained evidence for
`site-c/llama-3.1-8b`; this records artifact identity but does not by itself
establish a shared execution-time index deployment.

Model weights, checkpoint binaries, raw corpora, and RAG index artifacts for
`site-b/gemma-3-12b` were not included in the transfer and are not
distributed. A private non-secret incoming-file SHA-256 manifest is retained as
recovery provenance only.

Private pre-flattening recovery packages are preservation evidence, not
publication artifacts, and must not be listed as downloadable release assets.
