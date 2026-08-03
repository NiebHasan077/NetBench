# Run Summary: Llama-3.1-8B

## Identity

| Field | Value |
| --- | --- |
| Run ID | `site-c/llama-3.1-8b` |
| Machine ID | `site-c` |
| Evidence hostname | `site-c-node-1` |
| Model slug | `llama-3.1-8b` |
| Evidence status | `recovered_from_artifact` |
| Snapshot branch | `snapshot/site-c/llama-3.1-8b` |
| Snapshot tag | `snapshot-site-c-llama-3.1-8b-v1` |
| Monorepo snapshot commit | See `experiment_registry/run_index.csv` after snapshot registration. |

## Provenance And Limitation

This snapshot is reconstructed from a transferred site-c project tree and
result artifacts. Both component Git histories were transferred and preserved
privately before monorepo import. The Direct Pipeline working tree was already
modified and omitted several tracked documentation and auxiliary files relative
to its original Git commit. Consequently, this snapshot captures the recovered
code and evidence associated with the run, but is not claimed as the exact
historical source checkout used at every execution stage.

| Preserved input | Identifier |
| --- | --- |
| Direct Pipeline transferred-state commit | `e99b18e54a40277e2698685476cc5d4d8c970cb7` |
| RAG transferred-state commit | `513e6e5db18e64a6632148b75394636296920fd5` |
| Legacy recovery bundles | Private pre-flattening bundles retained outside the publication repository. |

## Result Integrity Note

The transferred run initially included a stale metadata CSV describing four
direct judged workbooks as 61-question shard outputs. Before import, corrected
judged workbooks and the corrected final report were supplied. The retained
v5.0 report records eight compared variants with `242` scored questions each.
`run_metadata.json` and `run_metadata_summary.csv` in this snapshot were
regenerated from those corrected transferred artifacts during consolidation on
2026-05-24. `run_training_metadata_summary.csv` is retained from the source
transfer because the model/checkpoint directories from which it was extracted
were deliberately not transferred.

## Source Hardware Evidence

Profiling artifacts record:

| Item | Recovered value |
| --- | --- |
| Hostname | `site-c-node-1` |
| CPU | Intel Xeon Gold 6248 CPU at 2.50 GHz |
| RAM | 187.0 GB |
| GPUs | 4 x Tesla V100-SXM2-32GB (`32494` MB each) |
| CUDA version | `11.8` |

## Recovered Evaluation Metadata

| Stage | Recovered setting or evidence |
| --- | --- |
| Direct answer generation | 5 answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512`, `top_p=0.9` recorded in workbook metadata. |
| RAG answer generation | 3 answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512` recorded in workbook metadata. |
| Judge | 8 corrected judged workbooks identify `gpt-5.1`, 242 questions each; recovered judge implementation uses `temperature=0.0` for OpenAI judging. |
| Final report | `evaluations/reports/HPN_BENCHMARK_REPORT_v5.0_by_gpt-5.1.md` and companion workbook. |

The corrected final report records overall scores of `4.02` for
`RAG-Llama-3.1-8B-cpt-full-sft` and `4.02` for
`RAG-Llama-3.1-8B-instruct-lora-cpt-then-sft-merged`, compared with `3.57`
for the highest-scoring direct trained variants.

## Recovered Training Hyperparameters

The training CSV preserves all extracted `training_args.bin` rows and SHA-256
values supplied in the source transfer. It also records unsuccessful or
superseded training artifact directories; those rows must not be mistaken for
the final reported variant.

| Recovered artifact category | Train batch | Eval batch | Grad accum | Learning rate | Epochs | Optimizer | Precision / checkpointing |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Final CPT/full-SFT artifact using 8-bit AdamW | 1 | 1 | 16 | `2e-05` | 3 | `adamw_bnb_8bit` | `bf16=True`, gradient checkpointing |
| Official-instruct full HPN SFT artifact | 1 | 1 | 16 | `2e-05` | 3 | `adamw_torch` | `bf16=True`, gradient checkpointing |
| Full CPT training artifact | 4 | 2 | 4 | `2e-05` | 3 | `paged_adamw_8bit` | `bf16=True`, gradient checkpointing |
| LoRA CPT and LoRA SFT artifacts | 8 | 2 | 2 | `0.0002` | 3 | `adamw_torch` | `fp16=True`, gradient checkpointing |

Additional transferred rows document blocked/incompatible or bad-NaN attempts
with alternative batch/optimizer settings. All recorded rows use seed `42`,
cosine scheduling and three epochs; consult the CSV when describing failed
attempts versus final model variants.

## Recovered RAG Configuration

The recovered RAG configuration file records `BAAI/bge-large-en-v1.5`
embeddings, embedding batch size `64`, `1200` target tokens per chunk with
`150` overlap tokens, vector and BM25 top-k values of `30`, RRF candidate
top-n `20`, reranker top-n `4`, and generation temperature `0.0` with a
`512` token output limit. This describes the recovered configuration tree; it
does not independently prove every effective runtime override.

## Included Evidence

| Path | Purpose |
| --- | --- |
| `run_metadata.json` | Consolidated metadata regenerated from corrected imported evidence. |
| `run_metadata_summary.csv` | Corrected answer/judged metadata and hashes; all judged variants contain 242 questions. |
| `run_training_metadata_summary.csv` | Source-transferred training hyperparameter extraction and hashes. |
| `evaluations/answers/` | Direct answer workbooks and retained shard artifacts. |
| `evaluations/rag_answers/` | RAG answer workbooks and retained shard artifacts. |
| `evaluations/judged/` | Corrected scored workbooks used in the final report. |
| `evaluations/reports/` | Corrected reported score tables and report document. |
| `profiling_results/` | Performance and source hardware evidence. |

## Excluded Or External Evidence

Model weights, checkpoint binaries, source-machine environments, raw corpora,
RAG index stores, caches, and logs are not committed to the publication
repository. Private recovery material records available checksums for the
transferred RAG corpus and serialized index files. Private Git bundles retain
historical source provenance and must not be published because historical
Direct Pipeline history contained credential material.
