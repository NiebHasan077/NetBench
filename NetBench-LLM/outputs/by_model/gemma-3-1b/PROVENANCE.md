# Provenance — `gemma-3-1b`

These outputs are the aggregated **results of record** for this model. The tag
below is the recovered post-run **source state**; see `PROVENANCE.md` at the
repository root for how far that chain holds.

| Field | Value |
| --- | --- |
| Run ID | `site-b/gemma-3-1b` |
| Machine | `site-b` |
| Snapshot commit | `687b9015f8d09a76fc115c0ceea634b25e96c3fd` |
| Snapshot tag | `snapshot-site-b-gemma-3-1b` |
| Evidence status | `recovered_from_artifact` |
| Run summary | `NetBench-LLM/outputs/by_model/gemma-3-1b/RUN_SUMMARY.md` |

**Consolidated:** 2026-06-18 · curated subset = benchmark reports, answer and
judged workbooks, adaptation reports, run metadata, and profiling results. Raw
per-GPU/worker answer shards (`.shards/`, `.parallel_*/`) and logs were
intentionally excluded.

**Limitations (from registry):** Transferred Direct tree omitted tracked documentation/auxiliary files and recovered run_pipeline.sh currently targets gemma-3-12b despite gemma-3-1b artifacts; model/checkpoint binaries were not transferred.

**Naming exception:** the comparative report is named
`HPN_BENCHMARK_REPORT_by_gpt-5.1.md` / `.xlsx` (without the `_v5.0` infix used by
the other models). The content is the identical HPN v5.0 protocol — 242 scored
questions, the full P1–P8 variant set, judged by `gpt-5.1`. The file is kept
under its original snapshot/registry name so the consolidated copy matches its
source byte-for-byte; treat it as the v5.0 report.
