# Paperbase Tools

The `paperbase/` directory contains the title index and four standalone scripts for managing it.

---

## What is the paperbase index?

`paperbase/title_index.json` maps each paper's normalised title to its file path and extraction source.  
`collect_papers.py` loads this index at startup and updates it automatically after each successful download — no manual step needed for normal collection runs.

The index is only rebuilt manually when PDFs are added, moved, or deleted outside of `collect_papers.py`.

---

## Scripts

| Script | Purpose |
|---|---|
| `merge_collections.py` | Merge multiple PDF directories into one with dedup (project root) |
| `build_index.py` | Full rescan — walk a root directory and re-index every PDF |
| `check_paper.py` | Check whether one or more PDFs are already indexed |
| `verify_duplicates.py` | Re-verify weak duplicate pairs with 3-signal checks |
| `delete_duplicates.py` | Delete confirmed duplicate PDFs (dry-run by default) |

---

## merge_collections.py

Merge multiple PDF directories into a single target folder with title-based deduplication.
Lives in the **project root** (not `paperbase/`) because it orchestrates the full copy + dedup workflow.

```bash
# Preview the merge (dry-run)
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

# Custom index location
.venv/bin/python merge_collections.py \
    --source /path/to/collection-A \
    --source /path/to/collection-B \
    --target /path/to/merged \
    --index /path/to/merged/title_index.json
```

| Flag | Default | Description |
|---|---|---|
| `--source PATH` | *(required, repeatable)* | Directory of PDFs to merge (scanned recursively) |
| `--target PATH` | *(required)* | Destination directory for unique PDFs |
| `--index PATH` | `<target>/title_index.json` | Where to save the unified title index |
| `--dry-run` | off | Preview only — no files written |

**How it works:**

1. Scans all `--source` directories recursively for `*.pdf` files.
2. Feeds each PDF through `IndexManager.add_pdf()` — same 3-signal dedup used by `build_index.py`.
3. **Unique** papers are copied to `--target`. **STRONG** duplicates are skipped. **WEAK** candidates are skipped but recorded in `duplicate_candidates.json`.
4. Papers where title extraction failed are still copied (let the user decide).
5. Filename collisions in the target are resolved by appending `_1`, `_2`, etc.

**After merging**, use the standard review pipeline:

```bash
# Re-verify WEAK candidate pairs
.venv/bin/python paperbase/verify_duplicates.py \
    --candidates /path/to/merged/duplicate_candidates.json

# Remove confirmed duplicates (preview first, then --confirm)
.venv/bin/python paperbase/delete_duplicates.py \
    --confirmed /path/to/merged/confirmed_duplicates.json \
    --confirm
```

---

## build_index.py

Full rescan of a directory tree. Use this after moving PDFs in bulk or adding papers manually.

```bash
# Default: scan New-Networking-Papers/, write paperbase/title_index.json
.venv/bin/python paperbase/build_index.py

# Custom root and output
.venv/bin/python paperbase/build_index.py --root /path/to/papers --output paperbase/title_index.json
```

Collision handling during a full scan:

- **STRONG** duplicate (all 3 checks pass): excluded from index, recorded in `duplicate_candidates.json`  
- **WEAK** candidate (fewer than 3 checks): first occurrence kept, pair recorded for review

---

## check_paper.py

Spot-check one or more PDFs before manually adding them.

```bash
# Check a single file
.venv/bin/python paperbase/check_paper.py path/to/paper.pdf

# Check all PDFs in a directory
.venv/bin/python paperbase/check_paper.py --dir path/to/new-papers/

# Use a custom index file
.venv/bin/python paperbase/check_paper.py paper.pdf --index paperbase/title_index.json
```

Output statuses:

| Status | Meaning |
|---|---|
| `EXACT` | Normalised title matches an index entry exactly |
| `FUZZY` | Token overlap ≥ 0.85 — probable match, worth checking |
| `NOT FOUND` | No match — paper is not yet indexed |
| `EXTRACT FAILED` | Title could not be extracted from the PDF |

---

## verify_duplicates.py

Re-runs the 3-signal check on every WEAK candidate pair in `duplicate_candidates.json`  
and writes confirmed pairs to `confirmed_duplicates.json`.

The three signals:

| Check | Pass condition |
|---|---|
| `page_count` | Both PDFs have the same page count |
| `file_size` | File sizes are within 5% of each other |
| `text_sim` | Jaccard similarity of first-page word tokens ≥ 0.80 |

Confidence levels written to `confirmed_duplicates.json`:

| Level | Signals passed | Recommendation |
|---|---|---|
| `STRONG` | 3 of 3 | Safe to delete |
| `LIKELY` | 2 of 3 | Review recommended |
| `WEAK` | 1 of 3 | Excluded — probable false positive |

```bash
.venv/bin/python paperbase/verify_duplicates.py
```

---

## delete_duplicates.py

Reads `confirmed_duplicates.json` and removes the duplicate copy of each pair.  
**Dry-run by default** — prints what would be deleted without touching anything.

```bash
# Preview (dry-run)
.venv/bin/python paperbase/delete_duplicates.py

# Delete STRONG duplicates only
.venv/bin/python paperbase/delete_duplicates.py --confirm

# Delete STRONG + LIKELY duplicates
.venv/bin/python paperbase/delete_duplicates.py --confidence LIKELY --confirm
```

The `existing` path (first occurrence, kept in the title index) is always preserved.  
The `duplicate` path (second occurrence) is deleted.  
Every action is appended to `paperbase/deletion_log.json`.

After deleting, run a full rebuild to refresh the index:

```bash
.venv/bin/python paperbase/build_index.py
```

---

## Data files (all in `paperbase/`)

| File | Description |
|---|---|
| `title_index.json` | Main index: normalised title → `{raw_title, path, source}` |
| `duplicate_candidates.json` | Pairs flagged during indexing (STRONG + WEAK) |
| `confirmed_duplicates.json` | Output of `verify_duplicates.py` (STRONG + LIKELY only) |
| `deletion_log.json` | Append-only log of all delete actions |
