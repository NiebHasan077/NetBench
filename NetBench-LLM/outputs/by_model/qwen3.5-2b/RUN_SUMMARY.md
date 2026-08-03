# Run Summary: Qwen3.5-2B

## Identity

| Field | Value |
| --- | --- |
| Run ID | `site-a/qwen3.5-2b` |
| Machine ID | `site-a` |
| Model slug | `qwen3.5-2b` |
| Evidence status | `recovered_from_artifact` |
| Snapshot branch | `snapshot/site-a/qwen3.5-2b` |
| Snapshot tag | `snapshot-site-a-qwen3.5-2b-v1` |
| Monorepo snapshot commit | See `experiment_registry/run_index.csv` after snapshot registration. |

## Provenance And Limitation

This snapshot was reconstructed from preserved files and outputs remaining on
`site-a` after execution. It contains the recovered Direct Pipeline and RAG
code state together with the model-specific result evidence. The source code
was not committed at the moment this model originally ran, and the recovered
code tree includes later local edits made while processing other models.
Therefore this snapshot preserves recoverable evidence and associated code, but
is not claimed as the exact historical source checkout that generated every
result file.

| Preserved input | Identifier |
| --- | --- |
| Direct Pipeline recovery commit | `54a284a4d04f97b0a59dc05535f824ca9683f0e0` |
| RAG recovery commit | `4df598b97e4ea55cd6e01235308fdf5665648ee2` |
| Legacy recovery bundles | Private pre-flattening bundles retained outside the publication repository. |

## Recovered Evaluation Metadata

| Stage | Recovered setting or evidence |
| --- | --- |
| Direct answer generation | 5 answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512`, `top_p=0.9` recorded in workbooks. |
| RAG answer generation | 3 answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512` recorded in workbooks. |
| Judge | 8 judged workbooks identify `gpt-5.1`; recovered judging implementation sends OpenAI judge requests with `temperature=0.0`. |
| Final report | `evaluations/reports/HPN_BENCHMARK_REPORT_v5.0_by_gpt-5.1.md` and companion workbook. |

## Recovered Training Hyperparameters

The rows below summarize distinct `training_args.bin` settings recovered from
model artifacts. The accompanying CSV contains every recovered artifact row
and its SHA-256 hash.

| Training stage | Train batch | Eval batch | Grad accum | Learning rate | Epochs | Optimizer | Precision / checkpointing |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Full CPT | 2 | 2 | 8 | `2e-05` | 3 | `adamw_torch` | `bf16=True`, gradient checkpointing |
| Full SFT variants | 2 | 2 | 8 | `2e-05` | 3 | `adamw_torch` | `bf16=True`, gradient checkpointing |
| LoRA CPT | 4 | 2 | 4 | `0.0002` | 3 | `adamw_torch_fused` | `bf16=True`, gradient checkpointing |
| LoRA SFT variants | 4 | 2 | 4 | `0.0002` | 3 | `adamw_torch_fused` | `bf16=True`, gradient checkpointing |

All recovered training rows use seed `42`, cosine scheduling, weight decay
`0.01`, and three epochs.

## Recovered RAG Configuration

The recovered RAG code/configuration records `BAAI/bge-large-en-v1.5`
embeddings, embedding batch size `64`, vector and BM25 retrieval top-k values
of `30`, RRF candidate top-n `20`, reranker top-n `4`, and generation
temperature `0.0` with a `512` token output limit. Large index stores and
source corpora are excluded from Git and must be registered as external
artifacts before publication.

## Included Evidence

| Path | Purpose |
| --- | --- |
| `run_metadata.json` | Machine-extracted consolidated run metadata and hashes. |
| `run_metadata_summary.csv` | Generated-answer and judged-result metadata with hashes. |
| `run_training_metadata_summary.csv` | Recovered training hyperparameters and artifact hashes. |
| `evaluations/answers/` | Direct answer workbooks. |
| `evaluations/rag_answers/` | RAG answer workbooks recovered with Direct Pipeline output evidence. |
| `evaluations/judged/` | Scored workbooks used by the benchmark report. |
| `evaluations/reports/` | Reported score tables and report document. |
| `profiling_results/` | Recovered performance profiling evidence. |

## Excluded Or External Evidence

Model weights, checkpoints, environments, raw retrieval corpora, and RAG
indexes are not committed to the publication repository. Their available
external identifiers and checksums must be recorded in `ARTIFACTS.md` before
artifact release.
