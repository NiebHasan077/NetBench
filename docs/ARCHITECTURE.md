# NetBench — Architecture

Deep structure of the NetBench monorepo: module contracts, the data flow, the
variant taxonomy, the evaluation protocol, and the branch/provenance model. For
the reader-level overview see [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).

## Module contracts (inputs → outputs)

| Module | Consumes | Produces | Distributed in git? |
|---|---|---|---|
| `Collect-papers/` | YAML domain profile; OpenAlex/S2 APIs | downloaded PDFs + metadata; paperbase title index | source only (PDFs/indexes local) |
| `Paper2Corpus/` | local PDFs | `research_corpus_v3.json` = `[{"text": …}]` (2,371 records) | source only (corpus local) |
| `Benchmark-Generator/` | the corpus + a v4 seed benchmark | `hpn_benchmark_v5.0*.jsonl` (242 Q) | source only (benchmark/work dirs local) |
| `Instruct-FTD/` | the corpus + generic anchors | `train.jsonl` / `validation.jsonl` (instruction SFT mix) | source only (data local) |
| `NetBench-LLM/` | corpus (CPT) + instruction JSONL (SFT) + benchmark (eval) | adapted models; answers; judged workbooks; reports; profiling | code + **consolidated eval outputs** |
| `NetBench-RAG/` | corpus (indexing) + benchmark (eval) | retrieval indexes; RAG answers; judged; reports | code + outputs (indexes local) |

**Hard data-hygiene rule:** the `Benchmark-Generator` output (evaluation set) and
the `Instruct-FTD` output (training set) must never mix; the benchmark must never
feed back into CPT/SFT.

## Data flow

```
Collect-papers ─PDFs─▶ Paper2Corpus ─corpus─┬─▶ Benchmark-Generator ─242 Q─▶ eval set
                                            └─▶ Instruct-FTD ─train/val JSONL─▶ SFT set
   corpus ─CPT─▶ NetBench-LLM ◀─SFT─ SFT set
   eval set ─▶ NetBench-LLM (direct) & NetBench-RAG (retrieval) ─answers─▶ GPT-5.1 judge ─▶ reports
```

## Adaptation pipeline (NetBench-LLM)

Models flow through immutable stages into separate directories (base models are
never modified): `base/` → `pretrained/` (CPT) → `instruction/` (SFT) and
`lora/`→`lora-merged/` (LoRA). Conventions: all training config is via CLI args;
HF auth via `hf_token.txt`; datasets via `save_to_disk`/`load_from_disk`; CPT
packs 2048-token blocks with EOS between docs; SFT masks prompt tokens with
`-100`; multi-GPU via `device_map="auto"`. Chat templates are auto-selected per
family (Open-Orca for Llama, ChatML for Qwen, Gemma turn format for Gemma).

**Size-tier asymmetry:** small/medium models run full CPT from a *base* checkpoint
(P4/P7 exist). Large (27B) models disable base-CPT (`MODEL_HF_BASE=""`) and adapt
the *official instruct* checkpoint with LoRA only — so P4/P7 are absent for them.

## RAG pipeline (NetBench-RAG)

Three-stage retrieval over the same corpus: dual search (Qdrant vector top-30 +
BM25 top-30) → Reciprocal Rank Fusion (k=60, top-20) → cross-encoder rerank
(bge-reranker-v2-m3) → top-5 chunks to the generator. **Dual-index invariant:**
`Qdrant point id == chunks_cache index == BM25 position`; breaking it causes
silent retrieval failures. RAG answer/judge artifacts use the same schema and
naming as the direct pipeline so reports are directly comparable.

## Variant taxonomy (P ↔ S ↔ run-name)

The aggregation script (`analysis/aggregate_scores.py`) maps each judged run
name to a P-id via these rules (lowercased name): `rag-` prefix ⇒ RAG branch;
`lora`+`cpt`+`sft` ⇒ P5; `lora`+`sft` ⇒ P2; `cpt`+`sft` ⇒ P4; `sft` only ⇒ P3;
none ⇒ P1. RAG remaps P1→P6, P4→P7, P5→P8.

| P | run-name pattern | profiling tag |
|---|---|---|
| P1 | `<Model>` / `<Model>-Instruct` / `<Model>-it` | S1 |
| P2 | `*-instruct-lora-sft-merged` | S2 |
| P3 | `*-instruct-full-hpn-sft` | S3 |
| P4 | `*-cpt-full-sft` | S4 |
| P5 | `*-instruct-lora-cpt-then-sft-merged` | S5 |
| P6 | `RAG-<Model>` | S6 |
| P7 | `RAG-*-cpt-full-sft` | S7 |
| P8 | `RAG-*-instruct-lora-cpt-then-sft-merged` | S8 |

## Evaluation & judged-workbook schema

Each `hpn_judged_<variant>_by_gpt-5.1.xlsx` has three sheets:
- **Metadata** (Key/Value): `model_name`, `judge_model`, `benchmark_file`,
  `total_questions`, `successfully_judged`, …
- **Judged** (242 rows): `id, category, difficulty, question, reference_answer,
  model_answer, correctness, completeness, clarity, conciseness, overall,
  justification, source_paper, keywords`. The `id` (e.g. `NB-HPN-0026-q1`) is the
  cross-variant join key enabling paired statistics.
- **Summary**: per-dimension and per-category aggregates.

`analysis/aggregate_scores.py` ingests all judged workbooks → `scores_long.csv`,
`scores_overall.csv`, `summary_by_variant.csv`, `completeness_grid.csv`,
`significance.csv` (paired Wilcoxon + bootstrap 95% CI + Cohen's d_z), and
`MASTER_RESULTS.md`. The bulk CSVs are regenerable and gitignored; the script and
small summaries are tracked.

## Repository topology

- **`main`** — the 6-module publication monorepo; the aggregated
  **results-of-record** (consolidated `NetBench-LLM/outputs/by_model/`).
- **`master`** — the original 2-module working repo (NetBench-LLM + NetBench-RAG);
  its working tree held the only copy of the API-baseline outputs (now captured).
- **`snapshot/<machine>/<model>` (+ tags)** — one immutable per-run snapshot each,
  the authoritative **source-state provenance**; indexed in
  `experiment_registry/run_index.csv`.
- Two git **worktrees** share the object store: the `main` publication checkout
  and the `master` checkout. Cross-branch consolidation is done with
  `git show` / path-checkout from the `main` worktree — never by switching
  branches in place.

## Provenance & artifact policy

Per-model consolidated outputs on `main` carry a `PROVENANCE.md` linking back to
their snapshot commit/tag, machine, and `evidence_status` (`exact`,
`recovered_from_artifact`, `partially_recovered`). Excluded from git by policy
(see `.gitignore`, `ARTIFACTS.md`, `DATA_PROVENANCE.md`): model weights,
checkpoints, LoRA adapters, RAG vector/BM25 indexes, all `research_corpus*.json`,
raw per-question answer shards, logs, credentials, and `*.bundle` recovery
material. Recorded hashes are provenance only, not download links; users
regenerate large artifacts by running the pipeline.

## Hardware assumptions

Training/eval default to a 2× RTX A6000 (~49 GB each) box via
`device_map="auto"`. In `Benchmark-Generator`, Ollama pins a large model on
`cuda:1`, so other GPU work must pin `cuda:0`.
