# Run Summary: Gemma-3-1B

## Identity

| Field | Value |
| --- | --- |
| Run ID | `site-b/gemma-3-1b` |
| Machine ID | `site-b` |
| Evidence hostnames | `site-b-node-1`, `site-b-node-4` |
| Model slug | `gemma-3-1b` |
| Evidence status | `recovered_from_artifact` |
| Snapshot branch | `snapshot/site-b/gemma-3-1b` |
| Snapshot tag | `snapshot-site-b-gemma-3-1b-v1` |
| Monorepo snapshot commit | See `experiment_registry/run_index.csv` after snapshot registration. |

## Provenance And Limitation

This snapshot is reconstructed from a transferred Stampede3 project tree and
result artifacts. Both component Git histories were transferred and preserved
privately before monorepo import. The Direct Pipeline working tree omitted
tracked documentation and auxiliary files relative to its copied Git HEAD.
In addition, its recovered `scripts/run_pipeline.sh` currently sets
`MODEL_NAME="gemma-3-12b"`, although the retained output artifacts and logs
identify the completed run as `gemma-3-1b`. This may represent a post-run
edit or an incomplete source-tree transfer. Consequently, the snapshot
captures recovered associated code and reported evidence; it is not claimed
as the exact execution-time source checkout.

| Preserved input | Identifier |
| --- | --- |
| Direct Pipeline transferred-state commit | `4f221e21524b8ba70cda62e993dca1b9b161885a` |
| RAG transferred-state commit | `eafaad53110f680ce9aa0f30e8687659d2d8817d` |
| Legacy recovery bundles | Private pre-flattening bundles retained outside the publication repository. |

## Result Integrity Note

The transferred benchmark report records eight compared variants with `242`
scored questions each. The supplied `run_metadata_summary.csv` identifies
eight answer workbooks and eight judged workbooks, each with `242` questions;
all 16 worksheet SHA-256 values were checked against the copied files during
consolidation on `2026-05-24`. `run_training_metadata_summary.csv` is retained
from the source transfer because model and checkpoint directories were
deliberately not transferred.

## Source Hardware Evidence

Profiling artifacts record:

| Item | Recovered value |
| --- | --- |
| Hostnames | `site-b-node-1`, `site-b-node-4` |
| CPU | Intel Xeon Platinum 8468, 96 physical/logical cores |
| RAM | 1006.9 GB |
| GPUs | 4 x NVIDIA H100 (`95321` MB each) |
| CUDA version | `12.8` |
| PyTorch evidence | `2.11.0+cu128` |

## Recovered Evaluation Metadata

| Stage | Recovered setting or evidence |
| --- | --- |
| Direct answer generation | 5 answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512`, `top_p=0.9` recorded in workbook metadata. |
| RAG answer generation | 3 answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512` recorded in workbook metadata. |
| Judge | 8 judged workbooks identify `gpt-5.1`, 242 questions each; recovered judge implementation sets OpenAI judge `temperature=0.0`. |
| Final report | `evaluations/reports/HPN_BENCHMARK_REPORT_by_gpt-5.1.md` and companion workbook. |

The final report records the highest overall score as `3.07` for
`RAG-gemma-3-1b-Instruct`; the highest-scoring trained direct variant is
`gemma-3-1b-instruct-lora-cpt-then-sft-merged` at `2.81`.

## Recovered Training Hyperparameters

The source-transferred training CSV records 18 extracted training-argument
artifacts and their hashes. Multiple checkpoints for one model variant carry
the same extracted configuration and are summarized below.

| Recovered artifact category | Train batch | Eval batch | Grad accum | Learning rate | Epochs | Optimizer | Precision / checkpointing |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Full CPT artifact | 4 | 2 | 4 | `2e-05` | 3 | `adamw_torch` | `bf16=True`, no gradient checkpointing |
| Full CPT then SFT artifact | 4 | 4 | 4 | `2e-05` | 3 | `adamw_torch` | `bf16=True`, no gradient checkpointing |
| Official-instruct full HPN SFT artifact | 4 | 4 | 4 | `2e-05` | 3 | `adamw_torch` | `bf16=True`, no gradient checkpointing |
| LoRA CPT, LoRA SFT, and LoRA CPT-then-SFT artifacts | 4 | 2 | 4 | `0.0002` | 3 | `adamw_torch_fused` | `bf16=True`, no gradient checkpointing |

All recorded training rows use seed `42`, cosine scheduling and three epochs.
The CSV remains the authoritative retained extraction when mapping a reported
variant to its individual checkpoint rows.

## Recovered RAG Configuration

The recovered RAG configuration file records `BAAI/bge-large-en-v1.5`
embeddings, embedding batch size `64`, `1200` target tokens per chunk with
`150` overlap tokens, vector and BM25 top-k values of `30`, RRF candidate
top-n `20`, reranker top-n `4`, and local generation temperature `0.0` with a
`512` token output limit. This describes the recovered configuration tree; it
does not independently prove each effective runtime override.

## Included Evidence

| Path | Purpose |
| --- | --- |
| `run_metadata.json` | Source-transferred consolidated run metadata. |
| `run_metadata_summary.csv` | Answer/judged metadata and hashes; all judged variants contain 242 questions. |
| `run_training_metadata_summary.csv` | Source-transferred training hyperparameter extraction and hashes. |
| `evaluations/answers/` | Direct and RAG answer workbooks. |
| `evaluations/judged/` | Scored workbooks used in the report. |
| `evaluations/reports/` | Reported score tables and report document. |
| `evaluations/adaptation_reports/` | Retained adaptation evidence. |
| `profiling_results/` | Performance and source hardware evidence. |

## Excluded Or External Evidence

Model weights, checkpoint binaries, source-machine environments, raw corpora,
RAG index stores, caches, and logs are not committed to the publication
repository. Leftover generated data for other model slugs in the transferred
tree is outside this snapshot's scope and is excluded. The transferred
credential files were excluded after the user confirmed rotation/revocation.
Private recovery material records available hashes for RAG corpus/index
artifacts; private Git bundles retain source provenance and must not be
published because historical Direct Pipeline history contained credential
material.
