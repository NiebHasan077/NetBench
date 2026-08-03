# High-Level Workflow

## End-to-end data flow

```
OpenAlex API  ──┐
                 ├──▶  collect_papers.py
Semantic Scholar ─┘
     │
     ├─ 1. Dedup checks (in order)
     │       a. Source ID already in metadata.jsonl?    → skip (resume support)
     │       b. DOI already seen this run?              → skip (cross-source dedup)
     │       c. Normalised title in paperbase index?    → skip
     │
     ├─ 2. LLM abstract validation (optional)
     │       Send title + abstract to LLM
     │       NOT_RELEVANT response?                     → skip
     │
     ├─ 3. Download PDF
     │       Retry with exponential backoff (3 attempts)
     │       Failure?                                   → log to failed_downloads.jsonl
     │
     ├─ 4. Validate PDF
     │       Check %PDF magic bytes
     │       Fails?                                     → delete file, log to failed_downloads.jsonl
     │
     └─ 5. Record success
             Append record to metadata.jsonl
             Update paperbase/title_index.json  ← incremental, automatic
     ▼
downloads/<slug>/
     papers/  +  metadata.jsonl  +  failed_downloads.jsonl
```

## Deduplication layers

Papers are skipped if **any** of these match:

| Layer | Data source | Purpose |
|---|---|---|
| Source ID | `metadata.jsonl` | Resume interrupted runs without re-downloading |
| DOI (exact) | In-memory set | Catch same paper across sources or search queries |
| Normalised title | In-memory set + `paperbase/title_index.json` | Catch cross-run and cross-domain duplicates |

Title normalisation: lowercase → replace punctuation with spaces → collapse whitespace.  
The same function is used by `collect_papers.py` and all `paperbase/` tools, so dedup is consistent across the whole system.

## Search strategies

| Mode | How it works |
|---|---|
| `journal` | Fetches from journals listed in the profile. OpenAlex source IDs are resolved on first run and cached. |
| `keyword` | Full-text search using `keywords` buckets. Use `subfield_filter` to narrow scope. |
| `both` (default) | Runs journal first, then keyword. Dedup prevents overlapping papers from being downloaded twice. |

## PDF validation

Every downloaded file is checked for `%PDF` magic bytes.  
Files that fail (HTML error pages, empty responses, paywalled redirects) are deleted immediately and logged to `failed_downloads.jsonl`.

## Incremental index update

After each **successful** download, `paperbase/title_index.json` is updated in-place.  
No manual `build_index.py` run is needed between collect runs.  
A full rescan is only needed when PDFs are moved or added by other means:

```bash
.venv/bin/python paperbase/build_index.py
```

## Paperbase tools (run manually as needed)

```
merge_collections.py             ← merge multiple PDF dirs into one with dedup
paperbase/build_index.py         ← full rescan (after moving PDFs)
paperbase/check_paper.py         ← spot-check one or more PDFs
paperbase/verify_duplicates.py   ← re-verify weak duplicate pairs
paperbase/delete_duplicates.py   ← remove confirmed duplicates (dry-run default)
```

See [paperbase.md](paperbase.md) for full usage.

## Merging collections

Use `merge_collections.py` to combine independently-collected PDF directories
(e.g., manual downloads + automated HPN papers) into a single, deduplicated
collection.

```
Source A (manual papers)  ──┐
Source B (HPN papers)     ──┤
  ...                     ──┘
         │
   merge_collections.py
         │
         ├─ For each PDF:
         │     IndexManager.add_pdf() ── 3-signal dedup
         │         ├─ NEW          → copy to target
         │         ├─ STRONG DUP   → skip
         │         ├─ WEAK DUP     → skip, record candidate
         │         └─ EXTRACT FAIL → copy anyway
         │
         └─ Output:
              <target>/              ← merged PDFs
              <target>/title_index.json
              <target>/duplicate_candidates.json
```

After merging, the standard review pipeline applies:

```bash
# 1. Re-verify WEAK candidate pairs
.venv/bin/python paperbase/verify_duplicates.py \
    --candidates <target>/duplicate_candidates.json

# 2. Remove confirmed duplicates
.venv/bin/python paperbase/delete_duplicates.py \
    --confirmed <target>/confirmed_duplicates.json --confirm

# 3. Rebuild the index (optional, keeps things clean)
.venv/bin/python paperbase/build_index.py --root <target> \
    --output <target>/title_index.json
```

## Output files reference

| File | Description |
|---|---|
| `downloads/<slug>/papers/*.pdf` | Downloaded papers |
| `downloads/<slug>/metadata.jsonl` | One JSON record per successfully downloaded paper |
| `downloads/<slug>/failed_downloads.jsonl` | Download failures and invalid PDFs |
| `downloads/<slug>/validation_cache.json` | LLM validation cache (if enabled) |
| `profiles/<slug>_venue_cache.json` | Cached OpenAlex source IDs (journal mode) |
| `paperbase/title_index.json` | Normalised-title → path index; auto-updated each run |
| `paperbase/duplicate_candidates.json` | Pairs flagged during indexing for review |

### Metadata record schema

Every record has a `source_id` that identifies the paper's origin.
OpenAlex-sourced records also include `openalex_id` for backward compatibility.

**OpenAlex paper:**

```json
{
  "source_id": "https://openalex.org/W...",
  "openalex_id": "https://openalex.org/W...",
  "short_id": "W...",
  "title": "Paper Title",
  "normalized_title": "paper title",
  "doi": "https://doi.org/10...",
  "year": 2023,
  "authors": ["Alice", "Bob"],
  "venue": "ACM SIGCOMM",
  "abstract": "...",
  "license": "cc-by",
  "pdf_url": "https://...",
  "local_filename": "Paper Title.pdf"
}
```

**Semantic Scholar paper:**

```json
{
  "source_id": "s2:abc123def456",
  "short_id": "abc123def456",
  "title": "Paper Title",
  "normalized_title": "paper title",
  "doi": "10.1234/example",
  "year": 2023,
  "authors": ["Alice", "Bob"],
  "venue": "NSDI",
  "abstract": "...",
  "license": null,
  "pdf_url": "https://...",
  "local_filename": "Paper Title.pdf"
}
```
