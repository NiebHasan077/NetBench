# Experiment Registry

`run_index.csv` records one row per paper-reported machine/model run. A run
references one monorepo snapshot commit and tag containing the reviewed source
state and committed evidence for both modules.

Required evidence statuses are:

- `exact`
- `recovered_from_artifact`
- `partially_recovered`

Do not use the registry to imply that the stable publication baseline generated
every result. Populate each row only after its snapshot and metadata summary
have been audited.

Paths in a registry row are resolved in that row's `snapshot_commit` or
`snapshot_tag`, which remain the authoritative source-state for each run. The
per-model evaluation outputs are collected under
`NetBench-LLM/outputs/by_model/<model>/`, each with a `PROVENANCE.md` linking
back to the snapshot tag recorded here. Those outputs are the aggregated
results of record; the tags are the source-state provenance.

Large weights, checkpoints, corpora, and retrieval indexes referenced by
registered runs are not distributed for this release. `ARTIFACTS.md` records
available recovery hashes as provenance only; users who execute the pipeline
must obtain accessible prerequisites and regenerate derived artifacts through
the supplied scripts.
