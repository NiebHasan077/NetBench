# NetBench

NetBench is the publication and reproducibility repository for the HPN
benchmark experiments. It contains two ordinary source modules:

- `NetBench-LLM/`: the Direct Pipeline for model adaptation, direct
  generation, evaluation, and profiling.
- `NetBench-RAG/`: the retrieval-augmented generation and evaluation module.

The default publication branch is intended to hold maintained source,
documentation, metadata definitions, and the run registry. Historical
experimental evidence is recorded on run-specific snapshot branches and tags;
the default branch does not claim to be the exact source state for every
reported score.

Start with:

- `REPRODUCIBILITY_ORGANIZATION_PLAN.md` for the snapshot recovery workflow.
- `REPRODUCIBILITY.md` for baseline provenance and evidence policy.
- `PAPER_REPRODUCIBILITY_STATEMENT.md` for paper/supplement disclosure text.
- `ARTIFACTS.md` for large-artifact handling.
- `experiment_registry/` for the per-run snapshot index.

Large model artifacts, retrieval indexes, credentials, local environments, and
private recovery bundles are intentionally excluded from this repository.
Large artifacts are not hosted separately for this release; users must run the
scripts to download accessible prerequisites and regenerate derived artifacts.
