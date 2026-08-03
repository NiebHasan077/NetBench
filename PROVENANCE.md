# Provenance

This file answers one question: **given a number in the paper, how do I get
back to the run that produced it — and how far does that chain actually hold?**

The short version: everything from generated answers forward is exact and
re-derivable from files in this repository. The link from a run back to the
precise source state that executed it is *reconstructed*, not recorded, and we
say so rather than let it be assumed. Section 4 explains why.

## 1. The chain

```
paper table / figure
  └─ analysis/outputs/*.csv                aggregated results of record
       └─ scores_long.csv                  one row per (variant, question, dimension)
            └─ */evaluations/judged/*_by_gpt-5.1.xlsx    per-question judge scores
                 └─ */evaluations/answers/*.xlsx         model answers as generated
                      └─ experiment_registry/run_index.csv    the run, its site, its snapshot tag
                           └─ snapshot tag                    source state for that run
```

Left to right is deterministic re-derivation. Right to left is attribution.
The two have different strengths, and the boundary is at the answer workbooks.

## 2. Worked example

Take the RAG effect on the 8B model — *adding retrieval to the instruct model
gains 0.658 composite points*.

**Step 1 — the aggregate.** In `analysis/outputs/significance_formula.csv`:

```
base_model=llama-3.1-8b, contrast=P1 vs P6, n=233,
mean_a=3.304, mean_b=3.962, mean_diff=0.658, ci95=[0.522, 0.787], p_bh=0.0
```

**Step 2 — the per-question scores it was computed from.** In
`analysis/outputs/scores_long.csv`, filter `base_model=llama-3.1-8b`,
`p_id ∈ {P1, P6}`, `dimension=overall_formula`: 233 rows each, one per scored
question. `overall_formula` is the deterministic rubric composite —
0.4 correctness + 0.3 completeness + 0.2 clarity + 0.1 conciseness — recomputed
from the four dimension scores, not the judge's own overall. (Both are carried;
`docs/JUDGE_OVERALL_AUDIT.md` records why.)

**Step 3 — the judge output.** Those rows come from

```
NetBench-LLM/outputs/by_model/llama-3.1-8b/evaluations/judged/
    hpn_judged_Llama-3.1-8B-Instruct_by_gpt-5.1.xlsx          (P1)
    hpn_judged_RAG-Llama-3.1-8B-Instruct_by_gpt-5.1.xlsx      (P6)
```

one row per question, with the four dimension scores and the judge's written
rationale. `gpt-5.1` is the primary judge; a second judge
(`gemini-3.5-flash`) scored everything independently, and the agreement
workbook is in `analysis/outputs/`.

**Step 4 — the answers that were judged.** The sibling `answers/` and
`rag_answers/` workbooks hold the generated text verbatim, before judging.

**Step 5 — the run.** `experiment_registry/run_index.csv` maps the model to its
run: the site it executed on, its snapshot tag, its run summary, its
configuration metadata, its reported-result files, its evidence status, and its
known limitations.

Steps 1–4 are exact. Step 5 is where the qualification in §4 applies.

## 3. What re-derives exactly

From the committed judged workbooks forward, the entire analysis is
deterministic and reproduces byte-for-byte:

```bash
analysis/.venv/bin/python analysis/aggregate_scores.py
analysis/.venv/bin/python analysis/make_tables.py  --paper eacl --outdir analysis/outputs/tables
analysis/.venv/bin/python analysis/make_figures.py --paper eacl --outdir analysis/outputs/figures
```

The CSVs and `.tex` tables come back byte-identical; figures are identical once
the PDF `/CreationDate` stamp is stripped. Nothing in the paper is typed by
hand — every number, table, and figure is emitted by these scripts from the
committed evidence. If a CSV changes, the paper changes with it.

The benchmark side is equally checkable: the provenance manifest records the
generation models, prompt versions, seed, and the SHA-256 of the corpus the
items were generated from, and `analysis/outputs/evidence_quote_audit.csv`
records, per quote, whether it is verbatim in the paper it cites.

## 4. Where the chain stops, and why

**All eight registered runs carry the evidence status
`recovered_from_artifact`.** A snapshot tag is a *recovered common post-run
source state*, audited against that run's committed metadata. It is not a
checkout captured at the moment of execution, and we do not claim it reproduces
the run bit-for-bit.

The reason is architectural, not clerical. Adaptation, generation, and
profiling were executed on different physical nodes across three compute sites
— one lab workstation and two supercomputing clusters — each with its own
scheduler, its own filesystem, and its own independently staged working copy,
rather than one shared version-controlled checkout. No stage stamped the
executing commit into its outputs at launch, so per-run source states were
reconciled after the fact from the transferred trees and each run's metadata.

Consequences worth knowing before you rely on a tag:

- Some runs arrived without source `.git` history at all; their limitations
  column says so explicitly.
- Some transferred trees omitted documentation or auxiliary files present in
  the maintained tree.
- Where run metadata had to be regenerated from the supplied workbooks and
  logs, the registry records that too.

Read the `evidence_status` and `limitations` columns of `run_index.csv` before
citing a snapshot as the code that produced a result. They are populated per
run and they are not uniform.

The default branch is a **stable publication baseline**, not the exact source
state behind any one score. Do not read it as the code that generated every
result.

## 5. The snapshot tags

Eight tags, one per registered run:

| Tag | Model | Site | Evidence status |
|---|---|---|---|
| `snapshot-site-a-llama-3.2-1b` | llama-3.2-1b | site-a | `recovered_from_artifact` |
| `snapshot-site-a-qwen3.5-2b` | qwen3.5-2b | site-a | `recovered_from_artifact` |
| `snapshot-site-a-qwen3.5-9b` | qwen3.5-9b | site-a | `recovered_from_artifact` |
| `snapshot-site-b-gemma-3-12b` | gemma-3-12b | site-b | `recovered_from_artifact` |
| `snapshot-site-b-gemma-3-1b` | gemma-3-1b | site-b | `recovered_from_artifact` |
| `snapshot-site-c-gemma-3-27b` | gemma-3-27b | site-c | `recovered_from_artifact` |
| `snapshot-site-c-llama-3.1-8b` | llama-3.1-8b | site-c | `recovered_from_artifact` |
| `snapshot-site-c-qwen3.5-27b` | qwen3.5-27b | site-c | `recovered_from_artifact` |

Each is a **root commit** — it has no parent and is not part of any branch's
history. That is deliberate: presenting a recovered state as though it sat in a
line of development would be provenance theatre. The commit message says what
the tag is, and nothing more is claimed for it.

What a tag contains is the module source as recovered for that run, with two
declared modifications: identifying strings replaced by site identifiers
(§4), and five files removed that are not part of the executed pipeline
(the superseded organisation and disclosure plans, two module context files, and
a hardware-adaptation prompt). Nothing was added. Every other file is
byte-identical to the recovered state.

Site identifiers are stable across the whole repository: `site-a`, `site-b`,
`site-c` are the three compute sites, and `site-b-node-2` style identifiers
distinguish the physical nodes within a site, which is what §4's account of
the provenance limit rests on.

## 6. What is not in this repository

Source PDFs, the extracted text corpus, base and adapted model weights,
checkpoints, LoRA adapters, and the retrieval indexes are not redistributed.
`ARTIFACTS.md` and `DATA_PROVENANCE.md` record their checksums as provenance
only — those are identity records, not download links. Regenerate them by
running the pipeline scripts.

This is also why two of the audits here ship as committed CSVs rather than as
re-runnable steps: `analysis/outputs/source_papers.csv` and
`analysis/outputs/evidence_quote_audit.csv` are both derived from the corpus,
so they are produced once, against a hash-verified copy, and travel with the
benchmark as the record.

## 7. Checking it yourself

```bash
# every number in the paper, re-derived from the committed evidence
make reproduce

# the benchmark's own integrity: 242 items, 9 flagged out of scoring
python -c "import json; d=[json.loads(l) for l in open('NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl')]; \
print(len(d), sum(x['excluded_from_scoring'] for x in d))"

# the scored set matches: 233 distinct questions in the aggregate
tail -n +2 analysis/outputs/scores_long.csv | cut -d, -f7 | sort -u | wc -l
```

Related reading: `REPRODUCIBILITY.md` (baseline and evidence policy),
`DATA_RELEASE.md` (the 242 → 233 count chain), `BENCHMARK_CARD.md` (the
benchmark itself), `LICENSE-DATA` (what you may redistribute).
