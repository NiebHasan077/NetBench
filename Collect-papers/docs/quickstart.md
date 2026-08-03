# Quick Start

## 1. Install

```bash
cd Collect-papers
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Preview (dry run)

See what would be downloaded without writing any files:

```bash
.venv/bin/python collect_papers.py --dry-run
```

This uses the default `profiles/hpn.yaml` (High-Performance Networking).

## 3. Download

```bash
# Default profile (OpenAlex only)
.venv/bin/python collect_papers.py

# OpenAlex + Semantic Scholar
.venv/bin/python collect_papers.py --sources all

# Semantic Scholar only
.venv/bin/python collect_papers.py --sources s2

# Explicit profile
.venv/bin/python collect_papers.py --profile profiles/my-domain.yaml

# Journal sources only (OpenAlex)
.venv/bin/python collect_papers.py --mode journal

# Keyword search only
.venv/bin/python collect_papers.py --mode keyword
```

## 4. Create a new domain profile

```bash
cp profiles/template.yaml profiles/my-domain.yaml
# Edit my-domain.yaml — at minimum set: name, mailto, keywords
.venv/bin/python collect_papers.py --profile profiles/my-domain.yaml --dry-run
```

See [profiles.md](profiles.md) for all available fields.

## 5. Enable LLM filtering (optional)

Add an `llm_validation` block to your profile to filter papers by abstract
relevance before downloading. See [llm-validation.md](llm-validation.md).

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--profile PATH` | `profiles/hpn.yaml` | Profile YAML to use |
| `--mode {journal,keyword,both}` | `both` | Search strategy (OpenAlex) |
| `--sources {openalex,s2,all}` | `openalex` | API sources to query |
| `--mailto EMAIL` | profile value | Override OpenAlex polite-pool email |
| `--index PATH` | profile value | Override paperbase index path |
| `--dry-run` | off | Preview only — no files written |
| `--no-llm` | off | Skip LLM validation even if enabled in profile |

## Output location

```
downloads/<profile-slug>/
  papers/                       — downloaded PDFs
  metadata.jsonl                — one record per paper
  failed_downloads.jsonl        — download/validation failures
```

## 6. Merge multiple PDF collections

If you have papers from several sources (manual downloads, automated runs, etc.),
combine them into a single directory with duplicate detection:

```bash
# Preview (dry-run)
.venv/bin/python merge_collections.py \
    --source /path/to/manual-papers \
    --source downloads/hpn/papers \
    --target /path/to/merged \
    --dry-run

# Perform the merge
.venv/bin/python merge_collections.py \
    --source /path/to/manual-papers \
    --source downloads/hpn/papers \
    --target /path/to/merged
```

After merging, review and clean up duplicates:

```bash
.venv/bin/python paperbase/verify_duplicates.py \
    --candidates /path/to/merged/duplicate_candidates.json
.venv/bin/python paperbase/delete_duplicates.py \
    --confirmed /path/to/merged/confirmed_duplicates.json --confirm
```

See [paperbase.md](paperbase.md) for full details on the merge and dedup tools.

## Run tests

```bash
.venv/bin/python -m pytest tests/ -v
```

All network calls are mocked — tests run fully offline.
