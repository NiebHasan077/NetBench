# Profile Reference

All domain configuration lives in a YAML **profile** file.  
Start from the template:

```bash
cp profiles/template.yaml profiles/my-domain.yaml
```

Then edit and run:

```bash
.venv/bin/python collect_papers.py --profile profiles/my-domain.yaml --dry-run
```

---

## Field reference

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | ✓ | — | Human-readable domain name |
| `mailto` | ✓ | — | Your email — routes requests to OpenAlex polite pool (10 req/s) |
| `journals` | one of | — | List of `{search, match}` dicts for journal-based fetch |
| `keywords` | one of | — | List of keyword query strings for full-text search |
| `year_min` | | `2018` | Oldest publication year to include |
| `year_max` | | `2026` | Newest publication year to include |
| `max_per_query` | | `150` | Max results per venue / keyword bucket |
| `domain_description` | | — | Plain-English description fed to the LLM validator |
| `subfield_filter` | | — | OpenAlex subfield ID filter to narrow keyword results |
| `paperbase_dir` | | — | Path to `paperbase/` for cross-run dedup (auto-updated after downloads) |
| `downloads_dir` | | `../downloads/<slug>/` | Where to write PDFs |
| `llm_validation` | | — | LLM filtering block — see [llm-validation.md](llm-validation.md) |
| `semantic_scholar` | | — | Semantic Scholar API config — see section below |

---

## Minimal example (keywords only)

```yaml
name: "ML Systems"
mailto: "you@example.com"

year_min: 2020
year_max: 2026

domain_description: >
  Papers about machine learning systems: training infrastructure,
  GPU scheduling, distributed training, model serving, and ML compilers.

keywords:
  - '"distributed training" GPU OR "model parallelism"'
  - '"ML compiler" OR "XLA" OR "TVM" neural'
  - '"model serving" latency OR "inference" GPU cluster'
```

---

## Journal entries

Each entry needs a `search` term (OpenAlex query) and a `match` substring (verified against the resolved display name):

```yaml
journals:
  - search: "IEEE/ACM Transactions on Networking"
    match: "Transactions on Networking"
  - search: "SIGCOMM Computer Communication Review"
    match: "Communication Review"
```

> **Why no SIGCOMM / NSDI / CoNEXT / INFOCOM?**  
> Conference proceedings don't have stable OpenAlex source IDs — each year is a separate entry.  
> Use keyword search (with `subfield_filter` if needed) to cover conferences.

---

## Subfield filter

Restrict keyword results to specific OpenAlex subfield IDs to reduce noise:

```yaml
# Networking + hardware subfields
subfield_filter: "primary_topic.subfield.id:subfields/1705|subfields/1708"
```

Find your subfield IDs at [openalex.org/subfields](https://openalex.org/subfields).

---

## Paperbase dedup

Point `paperbase_dir` at the `paperbase/` directory.  
The index is loaded at startup and updated automatically after each successful download — no manual rebuild step.

```yaml
paperbase_dir: "../paperbase"   # relative to the profile file
```

---

## Semantic Scholar config

Add a `semantic_scholar` block to search the [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/graph) alongside (or instead of) OpenAlex.  Run with `--sources s2` or `--sources all` to activate.

Only **open-access** papers with a downloadable PDF are kept.

```yaml
semantic_scholar:
  enabled: true
  api_key: ""                         # free key from https://www.semanticscholar.org/product/api
  fields_of_study:
    - "Computer Science"
  max_per_query: 200
  keywords:
    - '"RDMA" OR "InfiniBand" OR "RoCE"'
    - '"programmable switch" OR "SmartNIC" OR "NIC offload"'
    - '"congestion control" datacenter'
```

### Semantic Scholar fields

| Field | Required | Default | Description |
|---|---|---|---|
| `enabled` | ✓ | — | `true` to activate S2 search |
| `api_key` | | `""` | Free API key (raises rate limit to 1 req/s) |
| `fields_of_study` | | `["Computer Science"]` | S2 field-of-study filter |
| `max_per_query` | | `100` | Max papers per keyword bucket |
| `keywords` | ✓ | — | List of keyword query strings (S2 search syntax) |

> **Keyword syntax differences:** S2 uses plain quoted phrases and boolean
> operators (`OR`, `AND`).  OpenAlex-style subfield filters do not apply to
> S2 queries — use `fields_of_study` instead.

---

## Per-profile output paths

```
downloads/
  <profile-slug>/
    papers/                       — downloaded PDFs
    metadata.jsonl                — one record per downloaded paper
    failed_downloads.jsonl        — download failures and invalid PDFs
    validation_cache.json         — LLM validation cache (if enabled)
profiles/
  <slug>_venue_cache.json         — cached OpenAlex source IDs (journal mode)
```
