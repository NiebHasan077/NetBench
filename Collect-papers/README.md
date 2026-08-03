# Collect-papers

Generic research paper collector powered by [OpenAlex](https://openalex.org) and [Semantic Scholar](https://www.semanticscholar.org/).  
Define a YAML **profile** to describe any research domain; the tool fetches open-access papers from one or both APIs, deduplicates against an existing paperbase, and optionally validates abstracts with an LLM before downloading.

## Setup

```bash
cd Collect-papers
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quickstart

```bash
# Preview what would be downloaded (no files written)
.venv/bin/python collect_papers.py --dry-run

# Download with the default HPN profile (OpenAlex only)
.venv/bin/python collect_papers.py

# Use Semantic Scholar as an additional source
.venv/bin/python collect_papers.py --sources all

# Semantic Scholar only
.venv/bin/python collect_papers.py --sources s2

# Use a custom profile
.venv/bin/python collect_papers.py --profile profiles/my-domain.yaml

# Merge multiple PDF collections into one (deduplicating)
.venv/bin/python merge_collections.py \
    --source /path/to/manual-papers \
    --source downloads/hpn/papers \
    --target /path/to/merged --dry-run
```

## Documentation

| Document | Contents |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | Setup, common commands, CLI flags |
| [docs/workflow.md](docs/workflow.md) | End-to-end data flow, dedup layers, output file schema |
| [docs/profiles.md](docs/profiles.md) | All profile fields, journal entries, S2 config, subfield filter |
| [docs/llm-validation.md](docs/llm-validation.md) | Ollama / OpenAI setup, caching, edge cases |
| [docs/paperbase.md](docs/paperbase.md) | `merge_collections`, `build_index`, `check_paper`, `verify_duplicates`, `delete_duplicates` |

## Project layout

```
collect_papers.py       — main collection pipeline
merge_collections.py    — merge multiple PDF directories with dedup
s2_client.py            — Semantic Scholar API client
profile.py              — profile loader (OpenAlex + S2 config)
llm_validator.py        — LLM abstract validation
paperbase/              — title index + duplicate-detection tools
profiles/               — YAML domain profiles (hpn.yaml, template.yaml)
downloads/              — output PDFs and metadata (per profile)
tests/                  — offline test suite (190 tests)
docs/                   — extended documentation
```

## Run tests

```bash
.venv/bin/python -m pytest tests/ -v
```


Semantic Scholar keys are optional. Prefer `export S2_API_KEY=...` instead of placing keys in tracked profile files.
