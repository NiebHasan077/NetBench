# Run Summary: Gemma-3-12B

## Identity

| Field | Value |
| --- | --- |
| Run ID | `site-b/gemma-3-12b` |
| Machine ID | `site-b` |
| Evidence hostnames | `site-b-node-1`, `site-b-node-2`, `site-b-node-3` |
| Model slug | `gemma-3-12b` |
| Evidence status | `recovered_from_artifact` |
| Snapshot branch | `snapshot/site-b/gemma-3-12b` |
| Snapshot tag | `snapshot-site-b-gemma-3-12b-v1` |
| Monorepo snapshot commit | See `experiment_registry/run_index.csv` after snapshot registration. |

## Provenance And Limitation

This snapshot is reconstructed from a transferred Stampede3 project tree and
result artifacts for `gemma-3-12b`. No source `.git` directories were supplied
with this transfer, so no component Git bundle or execution-time commit could
be preserved for this specific import. The snapshot therefore captures the
recovered associated source tree, configuration files, generated outputs, and
reported evidence; it is not claimed as the exact execution-time source
checkout.

Credential files were present in the incoming copy and were excluded from the
snapshot after the user confirmed that all credentials present in or
historically associated with the copy had been revoked or rotated. Model
weights, checkpoints, raw corpora, virtual environments, RAG indexes, caches,
and source-machine environments were not included in the publication snapshot.

A private recovery record under
`NetBench-LLM-recovery/site-b-gemma-3-12b/2026-06-11_145500-0500/`
contains non-secret file inventories and SHA-256 manifests for consolidation
provenance. It is private evidence and is not part of the publication
repository.

## Result Integrity Note

The transferred benchmark report records eight compared variants with `242`
scored questions each. The generated `run_metadata_summary.csv` identifies
`8` answer workbooks and `8` judged workbooks. Each answer and judged
workbook contains `242` question rows plus a header row, and each retained
workbook is recorded with a SHA-256 digest.

## Source Hardware Evidence

Profiling artifacts and logs record:

| Item | Recovered value |
| --- | --- |
| Hostnames | `site-b-node-1`, `site-b-node-2`, `site-b-node-3` |
| CPU | Intel(R) Xeon(R) Platinum 8468 |
| RAM | 1006.9 GB |
| GPUs | 4 x NVIDIA H100 (`95321` MB each) |
| CUDA version | `12.8` |
| PyTorch evidence | `2.11.0+cu128` |

## Recovered Evaluation Metadata

| Stage | Recovered setting or evidence |
| --- | --- |
| Direct answer generation | 5 direct answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512`, `top_p=0.9` recorded or recovered from workbook metadata and pipeline settings. |
| RAG answer generation | 3 RAG answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512` recorded in workbook metadata and RAG configuration. |
| Judge | 8 judged workbooks identify `gpt-5.1`, 242 questions each; recovered judge implementation uses temperature `0.0`. |
| Final report | `evaluations/reports/HPN_BENCHMARK_REPORT_v5.0_by_gpt-5.1.md` and companion workbook. |

The final report records the highest overall score as `4.25` for
`RAG-gemma-3-12b-it`. The highest-scoring trained direct variants are
`gemma-3-12b-instruct-lora-cpt-then-sft-merged`,
`gemma-3-12b-instruct-full-hpn-sft`, and `gemma-3-12b-cpt-full-sft`, each
reported at `3.57` overall. The report's overall row is:

```text
| **Overall** | 4.25 | 4.05 | 3.57 | 3.57 | 3.62 | 3.57 | 4.02 | 3.53 |
```

## Recovered Training Hyperparameters

No model/checkpoint directory or saved `training_args.bin` files were supplied
with this incoming copy. The retained training metadata is therefore recovered
from `scripts/run_pipeline.sh` and run logs.

| Recovered stage | Train batch | Grad accum | Effective batch | Learning rate | Epochs | Other settings |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Full CPT | 1 | 16 | 16 | `2e-5` | 3 | Model-sharded single-process Gemma 3 12B profile; bf16 evidence in scripts/logs. |
| Full SFT | 1 | 16 | 16 | `2e-5` | 3 | Same Gemma 3 12B full-weight profile as CPT. |
| LoRA CPT/SFT variants | 4 | 4 | 16 | `2e-4` | 3 | rank `64`, alpha `128`, `USE_QLORA=false`; forced 4x4 to avoid fp32 logits OOM. |
| Benchmark generation | - | - | - | - | - | batch size `4`, temperature `0.0`, output limit `512`. |

`run_training_metadata_summary.csv` is the authoritative retained table for
these recovered settings.

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
| `run_metadata.json` | Consolidated recovered run metadata generated during import. |
| `run_metadata_summary.csv` | Answer/judged metadata and hashes; all judged variants contain 242 questions. |
| `run_training_metadata_summary.csv` | Recovered hyperparameters from script and log evidence. |
| `evaluations/answers/` | Direct and RAG answer workbooks. |
| `evaluations/judged/` | Scored workbooks used in the report. |
| `evaluations/reports/` | Reported score tables and report document. |
| `evaluations/adaptation_reports/` | Retained adaptation evidence. |
| `profiling_results/` | Performance and source hardware evidence for `gemma-3-12b`. |

## Excluded Or External Evidence

Model weights, checkpoint binaries, source-machine environments, raw corpora,
RAG index stores, caches, logs, credential files, and leftover `gemma-3-1b`
profiling files are not committed to this snapshot. Large artifacts are not
distributed; users must regenerate them through the published pipeline scripts
subject to the access terms of the original sources.
