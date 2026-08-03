# Paper2Corpus

A Python pipeline for turning research paper PDFs into cleaned JSON text for LLM training.

## What this project does

The project processes papers in 3 stages:

1. Extract text from PDFs
2. Clean boilerplate/noise from extracted text
3. Convert cleaned text into JSON records

You can run this as:
- **One command** via `pipeline.py` (recommended)
- **Three separate scripts** (`pdf_to_text.py` → `clean_dataset.py` → `convert_to_json.py`)

## Project structure

- `pipeline.py` — all-in-one pipeline
- `pdf_to_text.py` — extracts text from all PDFs in `pdfs/`
- `clean_dataset.py` — cleans extracted text
- `convert_to_json.py` — converts cleaned text into JSON
- `remove_blank_lines.py` — optional utility
- `pdfs/` — input PDF directory
- `all_papers.txt` — intermediate raw text output
- `all_papers_cleaned.txt` — intermediate cleaned text output
- `research_corpus.json` — final JSON output

## Requirements

- Python 3.9+
- dependencies from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option A: Recommended (single script)

```bash
python3 pipeline.py
```

Useful flags:

```bash
# keep intermediate txt files
python3 pipeline.py --keep-intermediate

# custom input/output
python3 pipeline.py --pdf-dir ./pdfs --output research_corpus.json
```

### Option B: Run each stage manually

```bash
python3 pdf_to_text.py
python3 clean_dataset.py all_papers.txt all_papers_cleaned.txt
python3 convert_to_json.py all_papers_cleaned.txt research_corpus.json
```

## Input and output format

### Input

- Put all paper PDFs in `pdfs/`

### Output JSON

`research_corpus.json` is a list of objects:

```json
[
  {"text": "...cleaned paper text..."},
  {"text": "...cleaned paper text..."}
]
```

## Cleaning behavior (current)

The cleaner currently:

- removes references sections
- removes common boilerplate (acknowledgments, keywords, DOI/ISBN blocks)
- removes URLs/emails
- normalizes spaces and punctuation
- keeps paper title and body content

Note: author-name block stripping between title and abstract is intentionally disabled right now to avoid accidental title/content removal.

## Error handling

The scripts are resilient by design:

- if one PDF fails, processing continues with remaining PDFs
- if one page fails, remaining pages in that PDF still process
- unicode write errors fallback to `errors='replace'`

## Typical workflow

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp /path/to/your/papers/*.pdf pdfs/
python3 pipeline.py --keep-intermediate
ls -lh all_papers.txt all_papers_cleaned.txt research_corpus.json
```

## Troubleshooting

- `ModuleNotFoundError: pypdf`
  - install with `pip install -r requirements.txt`
- empty or tiny output
  - verify PDFs are text-based (not scanned images)
  - check extraction logs for per-file failures
- wrong Python environment
  - run with your venv Python directly, e.g. `./venv/bin/python pipeline.py`
