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
