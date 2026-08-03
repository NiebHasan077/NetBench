# Phase 4: Teacher Prompt Families

This phase defines the prompt-family registry used to generate synthetic
instruction data from the normalized HPN corpus and chunk artifacts.

## Purpose

The teacher model is not asked to emit model-specific chat templates.
Instead it generates canonical dataset content that later flows into the
existing `NetBench-LLM` templating path.

For every generated example, the final dataset row will still use:

- `system`
- `question`
- `response`
- `category`

The prompt-family layer controls:

- what kind of question is generated
- what evidence shape is used
- what answer style is expected
- whether the example belongs to `hpn` or `rag`

## Implemented Families

HPN families:

- `hpn_fact_qa`
- `hpn_concept_explanation`
- `hpn_comparison`
- `hpn_limitation_analysis`
- `hpn_calculation`
- `hpn_diagnosis`
- `hpn_scenario`
- `hpn_task`

RAG families:

- `rag_grounded_qa`
- `rag_unanswerable`

## Evidence Modes

Each family declares one evidence mode:

- `single_short`
  Use one short chunk from Phase 3.

- `pair_short`
  Use two short chunks from the same paper.

- `rag_bundle`
  Use a numbered multi-excerpt bundle built from the RAG chunks.

## Target Assistant Behavior

HPN families target the same assistant style used by the regular HPN
benchmark flow in `NetBench-LLM`:

- expert in HPN / HPC data transfer / networking
- technically precise
- reasoned rather than purely declarative
- usually 100 to 350 words

RAG families use the NetBench-style grounded user turn:

```text
Excerpts:

[1] Paper: '...'
...

Question: ...
```

For RAG generation, the teacher emits only the bare query and the answer.
The pipeline constructs the final user turn itself so the format stays
consistent with downstream RAG usage.

## Output Shape Expected From The Teacher

The teacher must return exactly one JSON object:

```json
{
  "question": "standalone user question or bare RAG query",
  "response": "assistant answer"
}
```

No markdown fences, notes, or extra keys.

## Source Of Truth

The executable registry lives in:

- [src/instruct_ftd/prompt_families.py](src/instruct_ftd/prompt_families.py)

The CLI also writes a machine-readable registry snapshot during pilot runs to:

- `reports/phase4_prompt_families.json`

## Listing Available Families

To print all registered family names and exit without running anything:

```bash
python3 scripts/run_pilot_generation.py --list_families
python3 scripts/run_candidate_generation.py --list_families
```

This is useful when constructing a custom `--families` override.
