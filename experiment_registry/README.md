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
`snapshot_tag`. Model-specific outputs are intentionally held on snapshot
branches and do not need to be duplicated on the stable publication branch.

`legacy_recovery_bundles` may record private bundle identifiers as recovery
provenance. Private bundles are not publication artifacts and must not be
released with the repository.

Large weights, checkpoints, corpora, and retrieval indexes referenced by
registered runs are not distributed for this release. `ARTIFACTS.md` records
available recovery hashes as provenance only; users who execute the pipeline
must obtain accessible prerequisites and regenerate derived artifacts through
the supplied scripts.
