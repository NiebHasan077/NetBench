# External Artifacts

Large or sensitive materials are excluded from Git and must be referenced by
identifier and checksum when required for reproducing a reported run.

Excluded artifact classes include:

- base, fine-tuned, merged, and adapter model weight files;
- checkpoints and optimizer/training-state archives not suitable for Git;
- RAG vector stores and serialized retrieval indexes;
- raw corpora whose size or licensing does not permit publication in Git;
- virtual environments, credentials, and private recovery bundles.

## Artifact Register

Populate one row per external artifact required by a reported run.

| Run ID | Artifact Type | External Identifier/Location | SHA-256 | Access Notes |
| --- | --- | --- | --- | --- |
| `site-c/llama-3.1-8b` | RAG BM25 index | Transferred intake `site-c/NetBench-RAG/data/bm25_index.pkl` | `f5642a0b40481c520cd19ffec5592b4682cf83f7a54a19562f44adc84184c043` | Excluded from Git; checksum also recorded in the private recovery package; public artifact location pending. |
| `site-c/llama-3.1-8b` | RAG chunks cache | Transferred intake `site-c/NetBench-RAG/data/chunks_cache.pkl` | `d855b3523422c889b7b5417a725e9af3d8735fbda9c29ec5e73989a315a2e53a` | Excluded from Git; checksum also recorded in the private recovery package; public artifact location pending. |
| `site-c/llama-3.1-8b` | RAG source corpus v3 | Transferred intake `site-c/NetBench-RAG/data/raw/research_corpus_v3.json` | `68c1148da5ec5688efd40c8f8c269cee51ee0617c6df28d33c43fcc1c5bb1048` | Excluded from Git; checksum recorded privately; screen for sensitive content and licensing before release. |
| `site-c/llama-3.1-8b` | RAG source corpus legacy copy | Transferred intake `site-c/NetBench-RAG/data/raw/research_corpus_new.json` | `0c7573819e006e9d9f7e2a74bde3b4115f748f9f564404c5f9a5965f06c76326` | Excluded from Git; checksum recorded privately; screen for sensitive content and licensing before release. |

Model weights and checkpoint binaries for `site-c/llama-3.1-8b` were not
included in the transfer. Their external registration remains unavailable
unless they can be located on the source machine or in separate storage.

Private pre-flattening recovery packages are preservation evidence, not
publication artifacts, and must not be listed as downloadable release assets.
