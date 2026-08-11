# Compute sites

The experiments ran across three compute sites. Throughout this repository —
`experiment_registry/run_index.csv`, `ARTIFACTS.md`, `PROVENANCE.md`, the
snapshot tag names, and the recorded run metadata — each site appears under a
stable anonymised identifier rather than its hostname.

The identifiers exist because the repository was first prepared for anonymous
review. They are kept now for a different reason: they are embedded in tag
names and in the recorded metadata of runs that have already happened, and
rewriting them would change commit hashes that `run_index.csv` and the paper's
provenance chain both reference. The mapping below is what makes them readable.

| Identifier | Resource | Models trained and evaluated there | Hardware |
|---|---|---|---|
| `site-a` | Local lab workstation, Missouri University of Science and Technology | Llama-3.2-1B, Qwen3.5-2B, Qwen3.5-9B | 2× NVIDIA RTX A6000, 48 GB each |
| `site-b` | Stampede3, Texas Advanced Computing Center | Gemma-3-1B, Gemma-3-12B | 4× NVIDIA H100, 96 GB nominal / 94 GB usable |
| `site-c` | The Mill, Research Support Solutions, Missouri University of Science and Technology | Llama-3.1-8B; Gemma-3-27B, Qwen3.5-27B | Llama-3.1-8B on 4× NVIDIA V100 32 GB; the 27B models on 4× NVIDIA H100 80 GB |

## Why this matters when reading the results

Inference profiling was run on each model's own site rather than on one common
host. **Cross-model cost comparisons are therefore indicative, not controlled**;
within-model comparisons are hardware-matched. The profiling table in the
analysis outputs records the configuration used for each measurement.

Site assignment was driven by queue availability and memory headroom, not by
model or method. No variant was assigned to a site in a way that could confound
a reported contrast: every variant of a given model ran on the same site, so
the within-model paired contrasts that carry the primary claims are internal to
one site.

## Acknowledgement requirements

Both shared facilities require specific wording when cited. `site-b` was
accessed through an NSF ACCESS allocation, which carries its own mandated
acknowledgement text; `site-c` is cited per the wording in its DOI record
(`10.71674/PH64-N397`). The accompanying paper carries both statements in full.

## Checking

`tools/check_anonymity.sh` continues to require that every hostname recorded in
run metadata is an anonymised site identifier. That check is about the recorded
data staying internally consistent, and this file does not change it — the
mapping lives here, in prose, and nowhere in the machine-read metadata.
