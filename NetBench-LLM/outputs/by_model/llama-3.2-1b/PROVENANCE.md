# Provenance — `llama-3.2-1b`

These outputs are the aggregated **results of record** for this model. The tag
below is the recovered post-run **source state**; see `PROVENANCE.md` at the
repository root for how far that chain holds.

| Field | Value |
| --- | --- |
| Run ID | `site-a/llama-3.2-1b` |
| Machine | `site-a` |
| Snapshot commit | `457127b80b339667a75e5bacaa1c175cc64a043d` |
| Snapshot tag | `snapshot-site-a-llama-3.2-1b` |
| Evidence status | `recovered_from_artifact` |
| Run summary | `NetBench-LLM/outputs/by_model/llama-3.2-1b/RUN_SUMMARY.md` |

**Consolidated:** 2026-06-18 · curated subset = benchmark reports, answer and
judged workbooks, adaptation reports, run metadata, and profiling results. Raw
per-GPU/worker answer shards (`.shards/`, `.parallel_*/`) and logs were
intentionally excluded.

**Limitations (from registry):** Recovered common post-run code state; exact per-run source checkout was not recorded at execution time.

**Answer-file gap:** this snapshot committed all 8 *judged* workbooks but only the 5 non-RAG *answer* workbooks; the 3 RAG answer workbooks (P6/P7/P8) were never committed upstream, so only judged scores exist for the RAG variants here. Scores are unaffected (computed from judged files).
