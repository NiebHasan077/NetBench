# Phase 8: Generic Anchor Import

Phase 8 imports the `20%` generic assistant anchor portion of the final
instruction dataset.

## Purpose

The HPN and RAG data in earlier phases are domain-heavy. Phase 8 adds a stable
assistant-behavior anchor set so the final mixture can preserve general
instruction-following behavior without regenerating that data locally.

## Sources

The implementation supports:

- OpenOrca
- Databricks Dolly
- NousResearch Hermes JSON-mode examples

It can load them either:

- directly from Hugging Face, or
- from local JSON / JSONL files with the same field shapes

This keeps the phase usable even when network access is restricted.

## Implementation

Anchor import is implemented in:

- [src/instruct_ftd/anchors.py](src/instruct_ftd/anchors.py)
- [scripts/import_generic_anchors.py](scripts/import_generic_anchors.py)
- [src/instruct_ftd/json_anchors.py](src/instruct_ftd/json_anchors.py)
- [scripts/import_hermes_json_anchors.py](scripts/import_hermes_json_anchors.py)

## Output Contract

Every imported anchor is normalized to the same canonical shape used elsewhere:

- `system`
- `question`
- `response`
- `category`

And augmented with:

- `source_split = "generic"`
- `task_type = "generic_anchor"`
- `anchor_source`
- `anchor_source_category`

Hermes JSON-mode anchors use:

- `category = "generic_json"`
- `task_type = "json_response"`
- `question_type = "structured_output"`
- `anchor_source = "NousResearch/hermes-function-calling-v1"`

Their `response` value is validated with `json.loads` and written as raw JSON
text, not a Markdown fenced code block.

## Outputs

- `data/intermediate/anchors/generic_anchor.jsonl`
- `data/intermediate/anchors/hermes_json_anchor.jsonl`
- `reports/phase8_anchor_stats.json`
- `reports/phase8_hermes_json_anchor_stats.json`

## Typical Usage

Using local sample files:

```bash
python3 scripts/import_generic_anchors.py \
  --orca_local /path/to/openorca_sample.jsonl \
  --dolly_local /path/to/dolly_sample.jsonl
```

Using Hugging Face directly:

```bash
python3 scripts/import_generic_anchors.py
```

Importing 166 Hermes JSON-mode anchors for append-only augmentation:

```bash
python3 scripts/import_hermes_json_anchors.py \
  --target_count 166 \
  --output_jsonl data/intermediate/anchors/hermes_json_anchor.jsonl
```

This Hugging Face path requires `pip install -r requirements.txt`, because the
project uses the optional `datasets` dependency only for this phase.

## Output

The script prints a summary on completion:

```
Done in 12.3s  |  records=18234
```

## Notes

The current implementation keeps the import path simple and conservative:

- empty questions or responses are dropped
- very short responses are dropped
- OpenOrca rows become `generic_orca`
- Dolly rows become `generic_dolly`
- valid Hermes JSON-mode rows become `generic_json`

This phase does not implement checkpointing. Re-run it directly if the source
anchor files or import settings change.

Later phases will mix these anchors with filtered HPN and RAG data to reach the
final `60 / 20 / 20` target.

Hermes JSON anchors can also be appended after Phase 9 when the goal is to add
structured-output behavior without removing any examples from an already built
final dataset.
