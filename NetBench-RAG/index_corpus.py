#!/usr/bin/env python3
"""
index_corpus.py — Offline indexer for the NetBench-RAG pipeline.

Parses the research corpus, chunks all papers, embeds the chunks with
BAAI/bge-large-en-v1.5, and stores them in Qdrant + BM25.

Usage
-----
  python index_corpus.py                   # JSON corpus (default)
  python index_corpus.py --source pdf      # PDFs only
  python index_corpus.py --source both     # JSON + PDFs
  python index_corpus.py --force           # wipe and rebuild from scratch

Resumability
------------
The script checks which source_files are already in the chunks_cache and
skips them.  This means interrupted runs can be continued without re-embedding
papers that were already processed.

Invariant
---------
  Qdrant point ID  ==  chunks_cache index  ==  BM25 corpus position

All three artefacts are always updated together atomically (save order:
Qdrant upsert → chunks_cache → BM25).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunker_from_config
from src.embedder import embedder_from_config
from src.parser import JSONParser, PDFParser, ParsedPaper
from src.store import (
    BM25Index,
    load_chunks_cache,
    save_chunks_cache,
    vector_store_from_config,
)

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index the HPN research corpus into Qdrant + BM25.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["json", "pdf", "both"],
        default="json",
        help=(
            "json  — flat-text JSON corpus (default)\n"
            "pdf   — source PDF files\n"
            "both  — JSON corpus + PDFs (deduplicates by source_file)\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index and rebuild from scratch.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def load_json_papers(corpus_path: str) -> list[ParsedPaper]:
    """Parse all documents from a JSON or JSONL flat-text corpus."""
    console.print(f"[cyan]Parsing JSON corpus:[/] {corpus_path}")
    corpus = Path(corpus_path)
    with open(corpus, encoding="utf-8") as f:
        if corpus.suffix.lower() == ".jsonl":
            docs = [json.loads(line) for line in f if line.strip()]
        else:
            try:
                docs = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                docs = [json.loads(line) for line in f if line.strip()]
            if isinstance(docs, dict):
                for key in ("documents", "records", "data", "questions"):
                    if isinstance(docs.get(key), list):
                        docs = docs[key]
                        break
                else:
                    docs = [docs]
    parser = JSONParser()
    papers: list[ParsedPaper] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing documents", total=len(docs))
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                text = doc
                source_file = f"corpus_doc_{i:04d}"
            else:
                text = doc.get("text") or doc.get("content") or ""
                source_file = doc.get("source_file") or doc.get("id") or f"corpus_doc_{i:04d}"
            if text.strip():
                papers.append(parser.parse(text, source_file=str(source_file)))
            progress.advance(task)
    return papers


def load_pdf_papers(pdfs_dir: str) -> list[ParsedPaper]:
    """Parse all PDF files in pdfs_dir."""
    pdf_paths = sorted(Path(pdfs_dir).glob("*.pdf"))
    if not pdf_paths:
        console.print(f"[yellow]Warning:[/] No PDF files found in {pdfs_dir}")
        return []
    console.print(f"[cyan]Parsing {len(pdf_paths)} PDFs from:[/] {pdfs_dir}")
    parser = PDFParser()
    papers: list[ParsedPaper] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing PDFs", total=len(pdf_paths))
        for pdf_path in pdf_paths:
            papers.append(parser.parse(str(pdf_path)))
            progress.advance(task)
    return papers


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    chunks_cache_path = paths["chunks_cache"]
    bm25_path = paths["bm25_index"]

    console.print(Rule("[bold blue]NetBench-RAG Indexer[/]"))

    # ── Initialise storage objects ────────────────────────────────────────────
    vector_store = vector_store_from_config(cfg)
    chunker = chunker_from_config(cfg)

    # ── Force rebuild: wipe everything ───────────────────────────────────────
    if args.force:
        console.print("[yellow]--force: dropping existing index artefacts[/]")
        vector_store.drop_collection()
        for p in [chunks_cache_path, bm25_path]:
            if Path(p).exists():
                Path(p).unlink()

    # ── Ensure collection exists ──────────────────────────────────────────────
    vector_store.create_collection_if_missing()

    # ── Load existing chunks cache (for resumability) ─────────────────────────
    if Path(chunks_cache_path).exists() and not args.force:
        existing_chunks = load_chunks_cache(chunks_cache_path)
        console.print(
            f"[green]Loaded existing chunks cache:[/] {len(existing_chunks):,} chunks"
        )
    else:
        existing_chunks = []

    # Verify Qdrant count matches cache length
    qdrant_count = vector_store.count()
    if qdrant_count != len(existing_chunks):
        console.print(
            f"[yellow]Warning:[/] Qdrant ({qdrant_count:,}) vs cache ({len(existing_chunks):,}) "
            "mismatch. The larger artefact may be stale. Consider --force to rebuild."
        )
        # Trust the smaller count — use the cache length for ID assignment
        existing_chunks = existing_chunks[:qdrant_count]

    already_indexed: set[str] = {c.source_file for c in existing_chunks}
    console.print(
        f"Already indexed: [bold]{len(already_indexed):,}[/] source files "
        f"({qdrant_count:,} chunks)"
    )

    # ── Parse papers ─────────────────────────────────────────────────────────
    all_papers: list[ParsedPaper] = []
    if args.source in ("json", "both"):
        all_papers.extend(load_json_papers(paths["corpus_json"]))
    if args.source in ("pdf", "both"):
        all_papers.extend(load_pdf_papers(paths["pdfs_dir"]))

    if not all_papers:
        console.print("[red]No papers to index. Exiting.[/]")
        sys.exit(1)

    # ── Filter to new papers only ─────────────────────────────────────────────
    new_papers = [p for p in all_papers if p.source_file not in already_indexed]
    skipped = len(all_papers) - len(new_papers)
    console.print(
        f"Papers to index: [bold]{len(new_papers):,}[/] new  "
        f"(skipping {skipped:,} already indexed)"
    )

    if not new_papers:
        console.print("[green]All papers already indexed. Nothing to do.[/]")
        _print_summary(existing_chunks, qdrant_count)
        return

    # ── Chunk new papers ──────────────────────────────────────────────────────
    console.print("\n[cyan]Chunking new papers...[/]")
    t0 = time.perf_counter()
    new_chunks = chunker.chunk_corpus(new_papers)
    console.print(
        f"  {len(new_chunks):,} new chunks in {time.perf_counter() - t0:.1f}s"
    )

    # ── Embed ─────────────────────────────────────────────────────────────────
    console.print("\n[cyan]Loading embedding model...[/]")
    embedder = embedder_from_config(cfg)
    console.print(
        f"  Model: [bold]{cfg['embedding']['model']}[/]  "
        f"device: {cfg['embedding']['device']}  "
        f"batch: {cfg['embedding']['batch_size']}"
    )
    console.print(f"[cyan]Embedding {len(new_chunks):,} chunks...[/]")
    t0 = time.perf_counter()
    new_vectors = embedder.embed_chunks(new_chunks, show_progress=True)
    console.print(
        f"  Done in {time.perf_counter() - t0:.1f}s  "
        f"shape: {new_vectors.shape}"
    )

    # ── Upsert to Qdrant ──────────────────────────────────────────────────────
    console.print(f"\n[cyan]Upserting to Qdrant (start_id={len(existing_chunks)})...[/]")
    t0 = time.perf_counter()
    vector_store.upsert_chunks(
        chunks=new_chunks,
        vectors=new_vectors,
        start_id=len(existing_chunks),
    )
    console.print(f"  Upserted {len(new_chunks):,} points in {time.perf_counter() - t0:.1f}s")

    # ── Save updated chunks cache ─────────────────────────────────────────────
    all_chunks = existing_chunks + new_chunks
    console.print(f"\n[cyan]Saving chunks cache ({len(all_chunks):,} chunks)...[/]")
    save_chunks_cache(all_chunks, chunks_cache_path)
    console.print(f"  Saved → {chunks_cache_path}")

    # ── Build and save BM25 ───────────────────────────────────────────────────
    lowercase = cfg.get("retrieval", {}).get("bm25_lowercase", True)
    console.print(f"\n[cyan]Building BM25 index over {len(all_chunks):,} chunks...[/]")
    t0 = time.perf_counter()
    bm25 = BM25Index.build(all_chunks, lowercase=lowercase)
    bm25.save(bm25_path)
    console.print(
        f"  Built in {time.perf_counter() - t0:.1f}s  → {bm25_path}"
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    final_count = vector_store.count()
    _print_summary(all_chunks, final_count)


def _print_summary(chunks, qdrant_count: int) -> None:
    """Print a rich summary table of the indexed corpus."""
    console.print()
    console.print(Rule("[bold green]Index Complete[/]"))

    abstract_count = sum(1 for c in chunks if c.is_abstract)
    body_count = len(chunks) - abstract_count
    unique_papers = len({c.source_file for c in chunks})

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Total chunks", f"{len(chunks):,}")
    table.add_row("  Abstract chunks", f"{abstract_count:,}")
    table.add_row("  Body chunks", f"{body_count:,}")
    table.add_row("Unique papers", f"{unique_papers:,}")
    table.add_row("Qdrant points", f"{qdrant_count:,}")
    console.print(table)


if __name__ == "__main__":
    main()
