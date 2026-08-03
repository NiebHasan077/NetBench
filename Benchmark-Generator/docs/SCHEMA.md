# Question schema reference

Every benchmark line is one JSON object on one line (JSONL).
The shape is back-compatible with v4 — new fields are additive.
Pydantic source of truth: `src/schema.py`.

## Example

```json
{
  "id": "NB-HPN-1147-q0",
  "benchmark_version": "v5.0",
  "category": "Bottleneck Diagnosis and End-to-End Reasoning",
  "difficulty": "medium",
  "question_type": "diagnosis",
  "question": "A transfer over a 100 Gbps WAN reaches only 12 Gbps. iperf3 between the same hosts hits 92 Gbps. What is the most likely bottleneck?",
  "reference_answer": "Storage write bandwidth at the destination. iperf3 measures memory-to-memory throughput and bypasses disk; the production transfer hits a disk-write ceiling on the receiving side.",
  "source_papers": ["paper_1147"],
  "evidence": [
    {
      "paper_id": "paper_1147",
      "passage_id": "paper_1147__p007",
      "quote": "When iperf throughput exceeds production transfer throughput, destination disk write bandwidth is the dominant ceiling."
    }
  ],
  "keywords": ["bottleneck", "iperf", "destination storage"],
  "requires_calculation": false,
  "synthetic_scenario": true,
  "inspired_by_papers": ["paper_1147"],
  "generator_model": "gemma4:26b",
  "critic_model": "gemma4:26b",
  "prompt_version": "v5.0-questions",
  "seed": 42,
  "calibration_model": "gemma4:26b"
}
```

## Field reference

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable unique identifier. Prefix encodes source stream: `NB-HPN-` per-paper (5a), `NB-SYN-` synthesis (5b), `NB-ADV-` adversarial (5c). |
| `benchmark_version` | string | `v5.0` for the current release. |
| `category` | string | One of the 10 v4 HPN categories (see below). |
| `difficulty` | enum | `easy`, `medium`, `hard`. **Empirically calibrated** in Phase 7 — generator-assigned tags are overwritten. |
| `question_type` | string | One of `concept`, `reasoning`, `scenario`, `diagnosis`, `calculation`, `comparison`, `task`. |
| `question` | string | The prompt presented to the model under test. |
| `reference_answer` | string | Gold answer the judge compares against. |
| `source_papers` | list[string] | Paper IDs the question was derived from. Length 1 for 5a/5c, ≥2 for 5b synthesis. |
| `evidence` | list[Evidence] | Verbatim quotes from corpus passages backing the answer. Schema below. |
| `keywords` | list[string] | Topical hints; not used by the evaluator. |
| `requires_calculation` | bool | True when the answer demands numeric computation. |
| `synthetic_scenario` | bool | True when the question describes a hypothetical scenario rather than directly extracting a fact. |
| `inspired_by_papers` | list[string] | Looser provenance — papers that informed the question even if not directly quoted. |
| `generator_model` | string | Ollama tag of the model that drafted the question. |
| `critic_model` | string | Ollama tag of the Phase 6 critic. |
| `prompt_version` | string | Version of the prompt template used to draft the question. |
| `seed` | int | RNG seed used during generation (`42`). |
| `calibration_model` | string | Phase 7 judge tag. **Additive in v5.0** — present in JSONL but not declared on the `Question` Pydantic model (passes through via `extra="ignore"`). |

### `evidence` element

| Field | Type | Description |
|---|---|---|
| `paper_id` | string | Paper the quote comes from. |
| `passage_id` | string | Stable passage ID `<paper_id>__p<NNN>` from Phase 1 chunking. |
| `quote` | string | **Verbatim substring** of the passage. Phase 6 enforces this; gemma4 paraphrases were repaired pre-validation by `scripts/repair_evidence_quotes.py`. |

## Categories

Aligned with v4 so the existing evaluator and dashboards keep working.
Phase 4 clustering surfaces ~6 of these as tight sub-topics; the rest
are broader buckets.

```
Transfer Parameters: Definitions and Roles
Concurrency Tuning and Scaling
Pipelining and Small-File Optimization
Parallelism and Large-File Optimization
Dataset Partitioning and Mixed Workloads
BDP-Based Reasoning and Window Sizing
Bottleneck Diagnosis and End-to-End Reasoning
Adaptive and Online Optimization
Fairness, Stability, and Shared Networks
Practical HPN Scenarios and Design
```

## Difficulty calibration semantics

Tags are assigned by Phase 7 based on whether each capability tier got
the question right. The judge is gemma4:26b with `think=False`.

| Tag | Behavior |
|---|---|
| `easy` | All three tiers correct (llama3.2:3b, qwen3.5:9b, gemma4:26b). |
| `medium` | mid + strong correct, weak wrong. |
| `hard` | strong correct, mid wrong (regardless of weak). |
| _dropped_ | strong wrong → question removed from `calibrated_questions.jsonl`. |

The v5.0 distribution after Phase 7: easy 31.7%, medium 41.2%, hard 27.1%.

## Loading

```python
import json

def load_split(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

dev = load_split("benchmark/hpn_benchmark_v5.0_dev.jsonl")
```

Or with Pydantic validation:

```python
from src.schema import Question
import json

questions = [
    Question.model_validate(json.loads(line))
    for line in open("benchmark/hpn_benchmark_v5.0_test.jsonl")
]
```

## Manifest

`benchmark/manifest.json` records the run that produced the splits.
Pydantic schema: `Manifest` in `src/stage_08_split.py`.

```json
{
  "benchmark_version": "v5.0",
  "generated_at": "2026-04-27T02:02:05Z",
  "seed": 42,
  "counts":  {"dev": 30, "test": 200, "synthesis": 4, "adversarial": 8, "total": 242},
  "targets": {"dev": 30, "test": 200, "synthesis": 50, "adversarial": 20, "total": 300},
  "sources": {"per_paper_5a": 250, "synthesis_5b": 4, "adversarial_5c": 8},
  "models": {
    "card_generator": "qwen3.5:9b",
    "question_generator": "gemma4:26b",
    "critic": "gemma4:26b",
    "calibration_weak": "llama3.2:3b",
    "calibration_mid": "qwen3.5:9b",
    "calibration_strong": "gemma4:26b",
    "calibration_judge": "gemma4:26b",
    "embedding": "BAAI/bge-large-en-v1.5"
  },
  "prompt_versions": {
    "per_paper_5a": "v5.0-questions",
    "synthesis_5b": "v5.0-synthesis",
    "adversarial_5c": "v5.0-adversarial",
    "calibration":   "v5.0-calibration"
  },
  "notes": [
    "Synthesis split short: shipped 4 vs target 50 ...",
    "Adversarial split short: shipped 8 vs target 20 ..."
  ]
}
```
