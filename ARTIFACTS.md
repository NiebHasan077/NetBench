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
| TBD | TBD | TBD | TBD | TBD |

Private pre-flattening recovery packages are preservation evidence, not
publication artifacts, and must not be listed as downloadable release assets.
