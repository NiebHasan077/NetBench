# Run Summary: Gemma-3-27B

## Identity

| Field | Value |
| --- | --- |
| Run ID | `site-c/gemma-3-27b` |
| Machine ID | `site-c` |
| Evidence hostnames | `site-c-node-2` |
| Model slug | `gemma-3-27b` |
| Evidence status | `recovered_from_artifact` |
| Snapshot branch | `snapshot/site-c/gemma-3-27b` |
| Snapshot tag | `snapshot-site-c-gemma-3-27b-v1` |
| Monorepo snapshot commit | See `experiment_registry/run_index.csv` after snapshot registration. |

## Provenance And Limitation

This snapshot is reconstructed from a transferred The Mill project tree and
result artifacts for `gemma-3-27b`. No source `.git` directories were supplied
with this transfer, so no component Git bundle or execution-time commit could
be preserved for this specific import. The snapshot captures the recovered
associated source tree, configuration files, generated outputs, and reported
evidence; it is not claimed as the exact execution-time source checkout.

The pipeline for this recovered run was executed with
`scripts/run_pipeline_specific.sh`. The script records
`MODEL_NAME="gemma-3-27b"`, `MODEL_HF_INSTRUCT="google/gemma-3-27b-it"`,
`RUN_S4=false`, and `RUN_S7=false`. The final log records full-training GPU
workers `1`, benchmark GPU workers `4`, and LoRA settings `batch=1`,
`grad_accum=16`, `rank=64`, `alpha=128`, `device_map=balanced_low_0`.

The incoming output tree also contains leftover `gemma-3-12b` output evidence.
That leftover evidence is excluded from this run-specific snapshot. Direct
profiling JSON for `gemma-3-27b` was supplied under root
`outputs/profiling_results/`; in the snapshot those files are retained under
`outputs/by_model/gemma-3-27b/profiling_results/` with the other run evidence.

Credential files, model weights, checkpoints, raw corpora, virtual
environments, RAG indexes, caches, logs, and source-machine environments are
excluded from the publication snapshot. The credential confirmation supplied
for this import referred to `qwen-3.5-27b`; publication push requires explicit
corrected confirmation for The Mill `gemma-3-27b`.

A private recovery record under
`NetBench-LLM-recovery/site-c-gemma-3-27b/` contains non-secret file
inventories and SHA-256 manifests for consolidation provenance. It is private
evidence and is not part of the publication repository.

## Result Integrity Note

The transferred benchmark report records `6` compared variants with
`242` scored questions each. The generated `run_metadata_summary.csv`
identifies `4` direct answer workbooks, `2` RAG answer workbooks, and
`6` judged workbooks. Each retained answer and judged workbook contains
`242` question rows plus a header row, and each retained workbook is recorded
with a SHA-256 digest.

Models compared:

- `RAG-gemma-3-27b-instruct-lora-cpt-then-sft-merged`: 242 scored questions
- `gemma-3-27b-instruct-full-hpn-sft`: 242 scored questions
- `gemma-3-27b-instruct-lora-sft-merged`: 242 scored questions
- `gemma-3-27b-it`: 242 scored questions
- `gemma-3-27b-instruct-lora-cpt-then-sft-merged`: 242 scored questions
- `RAG-gemma-3-27b-it`: 242 scored questions

## Source Hardware Evidence

Profiling artifacts and logs record:

| Item | Recovered value |
| --- | --- |
| Hostnames | `site-c-node-2` |
| CPU | Intel(R) Xeon(R) Platinum 8480+ |
| RAM | 1006.9 GB |
| GPUs | 4 x NVIDIA H100 80GB HBM3 (`81071` MB each in profiling JSON) |
| CUDA version | `13.0` |
| PyTorch evidence | `2.12.0+cu130` |

## Recovered Evaluation Metadata

| Stage | Recovered setting or evidence |
| --- | --- |
| Direct answer generation | 4 direct answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512`, `top_p=0.9` recorded or recovered from workbook metadata and pipeline settings. |
| RAG answer generation | 2 RAG answer workbooks, 242 questions each; `temperature=0.0`, `max_tokens=512` recovered from workbook metadata and RAG configuration. |
| Judge | 6 judged workbooks identify `gpt-5.1`, 242 questions each; recovered judge implementation uses temperature `0.0`. |
| Final report | `evaluations/reports/HPN_BENCHMARK_REPORT_v5.0_by_gpt-5.1.md` and companion workbook. |

The final report records the highest overall score as `4.33` for
`RAG-gemma-3-27b-it`. The highest-scoring trained direct variant is
`gemma-3-27b-instruct-lora-cpt-then-sft-merged`, reported at `3.74` overall.
The report's overall row is:

```text
| **Overall** | 4.05 | 3.53 | 3.66 | 3.79 | 3.74 | 4.33 |
```

## Recovered Training Hyperparameters

No model/checkpoint directory or saved `training_args.bin` files were supplied
with this incoming copy. The retained training metadata is therefore recovered
from `scripts/run_pipeline_specific.sh` and run logs.

| Recovered stage | Train batch | Grad accum | Effective batch | Learning rate | Epochs | Other settings |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Full CPT setting | 1 | 16 | 16 | `2e-5` | 3 | Script setting retained; S4/S7 base-CPT variants disabled for this no-base run. |
| Full SFT | 1 | 16 | 16 | `2e-5` | 3 | Full-training GPU workers `1`; torchrun/deepspeed disabled. |
| LoRA CPT/SFT variants | 1 | 16 | 16 | `2e-4` | 3 | eval batch `1`, rank `64`, alpha `128`, `USE_QLORA=false`, `LORA_DEVICE_MAP=balanced_low_0`. |
| Benchmark generation | - | - | - | - | - | benchmark workers `4`, CUDA devices `0,1,2,3`, temperature `0.0`, output limit `512`. |

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
| `evaluations/answers/` | Direct answer workbooks and small parallel worker JSON evidence. |
| `evaluations/rag_answers/` | RAG answer workbooks. |
| `evaluations/judged/` | Scored workbooks used in the report. |
| `evaluations/reports/` | Reported score tables and report document. |
| `evaluations/adaptation_reports/` | Retained adaptation evidence. |
| `profiling_results/` | Direct and RAG performance/source hardware evidence for `gemma-3-27b`. |

## Excluded Or External Evidence

Model weights, checkpoint binaries, source-machine environments, raw corpora,
RAG index stores, caches, logs, credential files, and leftover `gemma-3-12b`
outputs are not committed to this snapshot. Large artifacts are not
distributed; users must regenerate them through the published pipeline scripts
subject to the access terms of the original sources.
