# Phase 9: Final Mixing And Splitting

Phase 9 assembles the final local instruction dataset from:

- filtered HPN examples
- filtered RAG examples
- imported generic anchors

and writes the final train/validation JSONL files used by downstream training.

## Implementation

Mixing and splitting are implemented in:

- [src/instruct_ftd/mixing.py](src/instruct_ftd/mixing.py)
- [scripts/build_final_dataset.py](scripts/build_final_dataset.py)
- [src/instruct_ftd/json_anchors.py](src/instruct_ftd/json_anchors.py)
- [scripts/append_json_anchors_to_final.py](scripts/append_json_anchors_to_final.py)

## Default Behavior

The mixer:

- keeps HPN, RAG, and generic rows in the same canonical contract
- applies leakage-aware grouping for split assignment
- groups HPN and RAG by `paper_id`
- groups generic anchors by question signature
- writes:
  - `train.jsonl`
  - `validation.jsonl`
  - `mix_report.json`

## Mixing Modes

### Strict mix

When `--strict_mix` is supplied, the mixer requires all three source splits and
builds the largest feasible dataset that satisfies the target ratio:

- `60%` HPN
- `20%` RAG
- `20%` generic

### Non-strict mix

Without `--strict_mix`, the mixer uses whatever inputs are available and reports
the achieved counts. This is useful for small local smoke tests.

### Append-only JSON augmentation

When the goal is to add structured-output behavior without removing any rows
from a completed strict-mix dataset, use the append-only JSON augmentation
script after Phase 9. It copies the existing `train.jsonl` and
`validation.jsonl`, selects an exact number of validated `generic_json` anchors,
splits those anchors by the requested validation ratio, and writes a new final
dataset directory.

For the current `v3_run` baseline:

| Split | Existing | Added JSON | New Count |
| --- | ---: | ---: | ---: |
| train | 3162 | 158 | 3320 |
| validation | 166 | 8 | 174 |
| total | 3328 | 166 | 3494 |

The resulting source distribution is:

| Source group | Count | Share |
| --- | ---: | ---: |
| HPN | 1997 | 57.15% |
| RAG | 666 | 19.06% |
| generic existing | 665 | 19.03% |
| generic JSON | 166 | 4.75% |

Within the expanded generic bucket, JSON examples are `166 / 831 = 19.98%`.
If JSON should be 25% of the expanded generic bucket, add 222 examples instead
of 166.

## Split Policy

The train/validation split is leak-aware:

- HPN and RAG examples from the same `paper_id` stay in the same split
- generic anchors are grouped by normalized question signature

This reduces train/validation leakage while remaining deterministic.

## Typical Usage

`--filtered_hpn` and `--filtered_rag` have no default values and must always be
passed explicitly:

```bash
python3 scripts/build_final_dataset.py \
  --filtered_hpn data/intermediate/filtered/<run>/filtered_hpn.jsonl \
  --filtered_rag data/intermediate/filtered/<run>/filtered_rag.jsonl \
  --generic_anchors data/intermediate/anchors/generic_anchor.jsonl \
  --output_dir data/final/<run> \
  --strict_mix
```

For a non-strict partial smoke test:

```bash
python3 scripts/build_final_dataset.py \
  --filtered_hpn data/intermediate/filtered/<run>/filtered_hpn.jsonl \
  --filtered_rag data/intermediate/filtered/<run>/filtered_rag.jsonl \
  --generic_anchors data/intermediate/anchors/generic_anchor.jsonl \
  --output_dir data/final/<run>
```

For append-only Hermes JSON augmentation:

```bash
python3 scripts/import_hermes_json_anchors.py \
  --target_count 166 \
  --output_jsonl data/intermediate/anchors/hermes_json_anchor.jsonl

python3 scripts/append_json_anchors_to_final.py \
  --input_train data/final/v3_run/train.jsonl \
  --input_validation data/final/v3_run/validation.jsonl \
  --json_anchors data/intermediate/anchors/hermes_json_anchor.jsonl \
  --count 166 \
  --output_dir data/final/v3_run_plus_json
```

## Outputs

- `data/final/<run>/train.jsonl`
- `data/final/<run>/validation.jsonl`
- `data/final/<run>/mix_report.json`
- `data/final/<run>_plus_json/json_append_report.json` for append-only JSON augmentation

## Output

The script prints a summary on completion:

```
Final dataset directory: data/final/demo_run
Done in 0.3s
```

If either of the required input files is missing, the script exits immediately
with a clear error before attempting to produce output.

## Notes

The strict path is the intended production mode. The non-strict path exists so
the pipeline can be exercised incrementally before all source splits have been
generated at full scale.

This phase does not implement checkpointing. Re-run it directly when the
upstream filtered or anchor datasets change.
