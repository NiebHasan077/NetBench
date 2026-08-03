# Quickstart

This guide assumes the project lives at `Paper2Corpus`.

## 1) Enter the project directory

```bash
cd Paper2Corpus
```

## 2) Create and activate a virtual environment

If you do not already have a usable `venv/` for this renamed project:

```bash
python3 -m venv venv
source venv/bin/activate
```

If `venv/` already exists and is valid, just activate it:

```bash
source venv/bin/activate
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

Current dependency set:

- `pypdf`

## 4) Add input PDFs

Put all source papers in the project `pdfs/` directory:

```bash
cp /path/to/papers/*.pdf pdfs/
```

The default input directory is:

```bash
Paper2Corpus/pdfs
```

## 5) Run the full pipeline

Recommended command:

```bash
python3 pipeline.py --keep-intermediate
```

What it does:

1. Extracts text from every PDF in `pdfs/`
2. Cleans boilerplate and formatting noise
3. Writes an LLM-ready JSON corpus

Useful variants:

```bash
# default run without keeping intermediate text files
python3 pipeline.py

# custom input directory and output file
python3 pipeline.py --pdf-dir ./pdfs --output ./research_corpus.json
```

## 6) Inspect the outputs

```bash
ls -lh all_papers.txt all_papers_cleaned.txt research_corpus.json
```

Expected outputs:

- `all_papers.txt`: raw extracted text from all PDFs
- `all_papers_cleaned.txt`: cleaned text with PDF boundaries preserved
- `research_corpus.json`: final JSON array ready for downstream training or ingestion

The JSON output has this shape:

```json
[
  {
    "text": "Cleaned text for one paper..."
  }
]
```

## 7) Run each stage manually

Use this mode if you want to inspect or debug each stage separately.

### Extract PDFs to text

```bash
python3 pdf_to_text.py
```

This reads PDFs from `pdfs/` and writes:

```bash
all_papers.txt
```

### Clean the extracted text

```bash
python3 clean_dataset.py all_papers.txt all_papers_cleaned.txt
```

### Convert cleaned text to JSON

```bash
python3 convert_to_json.py all_papers_cleaned.txt research_corpus.json
```

## 8) Common troubleshooting

### `ModuleNotFoundError: pypdf`

Install dependencies in the active environment:

```bash
pip install -r requirements.txt
```

### `python3` is using the wrong environment

Run the venv interpreter explicitly:

```bash
./venv/bin/python pipeline.py --keep-intermediate
```

### Output is empty or much smaller than expected

Check the following:

- the PDFs are text-based and not just scanned images
- the files were copied into `pdfs/`
- the script output shows successful per-file extraction

### Existing virtual environment still points to the old `PyScripts` path

