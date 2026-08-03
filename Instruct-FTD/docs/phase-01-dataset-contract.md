# Phase 1: Dataset Contract

This document locks the Phase 1 interface for `Instruct-FTD`.

The goal is to produce a final dataset that can be consumed by
`NetBench-LLM` with minimal modifications. The current `NetBench-LLM`
instruction data path already normalizes OpenOrca and Dolly into:

- `system`
- `question`
- `response`
- `category`

`Instruct-FTD` will use the same core schema.

## 1. Design Constraints

The contract is shaped by four constraints:

1. `NetBench-LLM` should not need a major refactor.
2. The final dataset must remain model-agnostic until `NetBench-LLM`
   applies the target model chat template.
3. The HPN portion should align with the current benchmark-style assistant
   behavior already used in `NetBench-LLM`.
4. The pipeline must preserve enough metadata for quality filtering,
   deduplication, mixture accounting, and future assistant-only masking.

## 2. Handoff Contract To NetBench-LLM Direct Pipeline

The final training dataset will be emitted as JSONL files:

- `data/final/train.jsonl`
- `data/final/validation.jsonl`

Every row must include these required fields:

- `system`
- `question`
- `response`
- `category`

This matches the existing OpenOrca/Dolly-normalized shape used by
`training/prepare_instruction_data.py` in `NetBench-LLM`.

No model-specific chat template will be applied in `Instruct-FTD`.
Template rendering remains the responsibility of `NetBench-LLM`, which
already handles Llama, Qwen, and Gemma families.

## 3. Required And Optional Fields

### Required fields

- `system`: system instruction string
- `question`: user turn string
- `response`: assistant answer string
- `category`: coarse training label

### Optional metadata fields

- `source_split`: `hpn` | `rag` | `generic`
- `task_type`: fine-grained task label
- `difficulty`: `easy` | `medium` | `hard`
- `paper_id`: stable paper identifier
- `paper_title`: source paper title
- `source_pdf`: source PDF path or filename when available
- `question_type`: benchmark-aligned question type
- `prompt_length_bucket`: `short` | `medium` | `long` | `xlong`
- `generation_family`: generation recipe used to create the example
- `evidence`: supporting evidence objects
- `metadata`: free-form stage-specific metadata

Optional fields must never replace the required four fields.

## 4. Mixture Targets

The final mixed dataset must follow these top-level targets:

- `60%` HPN domain instruction data
- `20%` RAG-style grounded data
- `20%` generic assistant anchor data

### Split definitions

- `hpn`: domain instruction data generated from the HPN paper corpus
- `rag`: retrieval-style grounded prompts and answers generated from
  excerpt bundles that mirror NetBench-RAG input structure
- `generic`: imported assistant anchor data, primarily OpenOrca and
  Databricks Dolly samples

## 5. HPN Assistant Behavior Target

The HPN and RAG portions should align with the regular HPN evaluation prompt
used in `NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py`.

Reference system behavior:

- expert in high-performance networking
- expert in HPC data transfer systems
- expert in computer networking
- answers should be focused and technically precise
- specific numbers, thresholds, or formulas should be included when relevant
- reasoning should be explained, not only the conclusion

### Response-length expectation

For synthetic HPN and RAG examples:

- default target answer length: `100-350 words`
- shorter answers are acceptable only when the question is inherently narrow
- avoid generic one-sentence answers

This follows the live generation prompt used in the HPN benchmark.

### HPN question-type alignment

The `60%` HPN portion should primarily draw from the benchmark question-type
families already used in `NetBench-LLM Direct Pipeline/data/prompts/hpn_qa_benchmark_v4_general_skills.json`:

- `concept`
- `reasoning`
- `calculation`
- `comparison`
- `diagnosis`
- `scenario`
- `task`
- `structured_output`

These labels will be preserved in `question_type` metadata where available.

## 6. Category Taxonomy

`category` should remain coarse enough to be useful in training reports,
while `task_type` can be more specific.

### Allowed coarse categories

- `hpn_fact_qa`
- `hpn_concept_explanation`
- `hpn_comparison`
- `hpn_limitation_analysis`
- `hpn_calculation`
- `hpn_diagnosis`
- `hpn_scenario`
- `hpn_task`
- `rag_grounded_qa`
- `rag_unanswerable`
- `generic_orca`
- `generic_dolly`
- `generic_json`

### Allowed fine-grained task types

- `fact_qa`
- `concept_explanation`
- `method_summary`
- `comparison`
- `limitation_analysis`
- `numerical_reasoning`
- `diagnosis`
- `scenario`
- `task`
- `rag_grounded_qa`
- `rag_unanswerable`
- `generic_anchor`
- `json_response`

## 7. Evidence Metadata Contract

The final examples will keep evidence metadata even though `NetBench-LLM`
does not need it immediately.

Each `evidence` entry may include:

- `chunk_id`
- `paper_id`
- `paper_title`
- `section`
- `page`
- `text`
- `role`

Example `role` values:

- `primary`
- `supporting`
- `distractor`

This is required for later phases that generate grounded RAG supervision and
filter unsupported examples.

## 8. Canonical Example Shapes

### HPN example

```json
{
  "system": "You are an expert in high-performance networking (HPN), HPC data transfer systems, and computer networking.",
  "question": "Why is concurrency usually the most influential application-layer parameter for high-performance file transfer throughput?",
  "response": "Concurrency is usually the strongest parameter because it increases both I/O parallelism and network parallelism at the same time. Each concurrent transfer adds a separate transfer process with its own file I/O activity and network connection, so the system can better utilize storage and keep more data in flight on high-bandwidth paths. Parallelism only adds more streams for a single file, and pipelining mainly reduces command gaps for many small files. In production HPN environments, concurrency therefore tends to have the largest overall throughput impact when it is increased carefully without causing CPU, memory, or storage contention.",
  "category": "hpn_concept_explanation",
  "source_split": "hpn",
  "task_type": "concept_explanation",
  "question_type": "reasoning",
  "difficulty": "medium",
  "paper_id": "paper_001",
  "paper_title": "Dynamic Protocol Tuning",
  "prompt_length_bucket": "short",
  "evidence": [
    {
      "chunk_id": "paper_001_sec_3_chunk_0",
      "section": "Concurrency Tuning",
      "role": "primary",
      "text": "Concurrency improves both I/O and network utilization..."
    }
  ]
}
```

### RAG example

```json
{
  "system": "You are an expert in high-performance networking (HPN), HPC data transfer systems, and computer networking.\n\nAnswer the question below in a focused response (100-350 words) using the provided excerpts as your primary source. Be technically precise: include specific numbers, thresholds, or formulas where relevant. Explain the underlying reasoning, not just the conclusion.",
  "question": "Excerpts:\n\n[1] Paper: 'Paper A' | Section: Congestion Signals\n...\n\n[2] Paper: 'Paper B' | Section: Evaluation\n...\n\nQuestion: Compare the congestion signals used by the two systems and explain the main deployment tradeoff.",
  "response": "The first system relies on ...",
  "category": "rag_grounded_qa",
  "source_split": "rag",
  "task_type": "rag_grounded_qa",
  "difficulty": "hard",
  "prompt_length_bucket": "long",
  "evidence": [
    {
      "chunk_id": "paper_a_sec_4_chunk_1",
      "paper_title": "Paper A",
      "section": "Congestion Signals",
      "role": "primary"
    },
    {
      "chunk_id": "paper_b_sec_5_chunk_0",
      "paper_title": "Paper B",
      "section": "Evaluation",
      "role": "supporting"
    }
  ]
}
```

### Generic anchor example

```json
{
  "system": "You are a helpful assistant.",
  "question": "Explain the difference between a list and a tuple in Python.",
  "response": "A list is mutable, meaning you can change its contents after creation. A tuple is immutable, meaning its contents cannot be modified in place after it is created. Lists are typically used when the collection needs to change, while tuples are useful for fixed records or when immutability is desirable.",
  "category": "generic_orca",
  "source_split": "generic",
  "task_type": "generic_anchor"
}
```

## 9. Final Artifact Names

Phase 1 reserves these artifact names for later phases:

- `data/intermediate/normalized/normalized_papers.jsonl`
- `data/intermediate/chunks/short_chunks.jsonl`
- `data/intermediate/chunks/rag_chunks.jsonl`
- `data/intermediate/chunks/rag_prompt_bundles.jsonl`
- `data/intermediate/candidates/candidate_hpn.jsonl`
- `data/intermediate/candidates/candidate_rag.jsonl`
- `data/intermediate/filtered/filtered_hpn.jsonl`
- `data/intermediate/filtered/filtered_rag.jsonl`
- `data/intermediate/anchors/generic_anchor.jsonl`
- `data/intermediate/anchors/hermes_json_anchor.jsonl`
- `data/final/train.jsonl`
- `data/final/validation.jsonl`

## 10. Minimal Change Expected In NetBench-LLM Direct Pipeline

The intended downstream change is small:

- current state: `NetBench-LLM` downloads OpenOrca + Dolly directly
- target state: `NetBench-LLM` loads `train.jsonl` and `validation.jsonl`
  from `Instruct-FTD`

Everything after dataset loading can stay conceptually the same:

- model-family detection
- template rendering
- tokenizer loading
- tokenization
- training

## 11. Out Of Scope For Phase 1

Phase 1 does not implement:

- corpus normalization
- chunking
- Ollama generation
- filtering
- deduplication
- mixing
- export scripts

It only locks the contract that later phases must satisfy.
