#!/usr/bin/env python3
"""
query.py — Interactive RAG query interface for NetBench-RAG.

Loads the full pipeline (retriever + generator) once, then accepts questions
in a read-eval-print loop.

Usage
-----
  python query.py                          # use backend from config.yaml
  python query.py --backend openai         # force OpenAI GPT-4o
  python query.py --backend local          # force local Llama
  python query.py --top-k 3               # override reranker_top_n
  python query.py --question "How does BBR work?"  # single-shot (no REPL)
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.generator import Generator, GenerationResult, generator_from_config
from src.retriever import Retriever, retriever_from_config

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive RAG query interface.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--backend",
        choices=["openai", "local"],
        default=None,
        help="LLM backend to use (default: value from config.yaml)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Override reranker_top_n (number of chunks passed to the LLM)",
    )
    p.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question and exit (no interactive REPL)",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_pipeline(
    cfg: dict,
    backend_override: str | None,
    top_k_override: int | None,
) -> tuple[Retriever, Generator]:
    """Load retriever and generator, printing progress to the console."""
    console.print(Rule("[bold blue]NetBench-RAG[/]"))

    console.print("[cyan]Loading retrieval models...[/]", end=" ")
    t0 = time.perf_counter()
    retriever = retriever_from_config(cfg)

    # Apply top-k override if requested
    if top_k_override is not None:
        retriever.reranker_top_n = top_k_override

    console.print(f"[green]done[/] ({time.perf_counter() - t0:.1f}s)")

    backend_name = backend_override or cfg.get("generation", {}).get("backend", "openai")
    console.print(f"[cyan]Loading generator ({backend_name})...[/]", end=" ")
    t0 = time.perf_counter()
    generator = generator_from_config(cfg, backend_override=backend_override)
    console.print(f"[green]done[/] ({time.perf_counter() - t0:.1f}s)")

    model_label = generator._backend.model
    console.print(
        f"\n  Retriever : vector top-{retriever.vector_top_k} + BM25 top-{retriever.bm25_top_k}"
        f" → RRF top-{retriever.rrf_top_n} → reranker top-{retriever.reranker_top_n}"
    )
    console.print(f"  Generator : [bold]{model_label}[/] ({backend_name})")
    console.print()
    return retriever, generator


# ═══════════════════════════════════════════════════════════════════════════════
# Single query
# ═══════════════════════════════════════════════════════════════════════════════

def run_query(
    question: str,
    retriever: Retriever,
    generator: Generator,
) -> GenerationResult:
    """Run one question through the full pipeline and print results."""

    # ── Retrieve ─────────────────────────────────────────────────────────────
    console.print(f"[dim]Retrieving…[/]", end="\r")
    t0 = time.perf_counter()
    retrieval = retriever.retrieve(question)
    ret_time = time.perf_counter() - t0
    console.print(
        f"[dim]Retrieved {len(retrieval.chunks)} chunks "
        f"({len(retrieval.rrf_candidates)} RRF candidates) in {ret_time:.2f}s[/]"
    )

    # ── Generate ─────────────────────────────────────────────────────────────
    console.print(f"[dim]Generating…[/]", end="\r")
    t0 = time.perf_counter()
    result = generator.generate(question, retrieval.chunks)
    gen_time = time.perf_counter() - t0
    console.print(
        f"[dim]Generated in {gen_time:.2f}s"
        + (f"  ({result.prompt_tokens:,} prompt tokens)" if result.prompt_tokens else "")
        + "[/]"
    )
    console.print()

    # ── Print answer ─────────────────────────────────────────────────────────
    console.print(Panel(
        result.answer,
        title=f"[bold green]Answer[/]  [dim]({result.model})[/]",
        border_style="green",
        padding=(1, 2),
    ))

    # ── Print sources ─────────────────────────────────────────────────────────
    console.print()
    console.print("[bold]Sources[/]")
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("#",  style="bold cyan", no_wrap=True, width=3)
    table.add_column("Paper",  no_wrap=False, max_width=52)
    table.add_column("Section", no_wrap=True, max_width=24)
    table.add_column("Score",   no_wrap=True, justify="right", style="yellow")

    for i, (chunk, score) in enumerate(
        zip(retrieval.chunks, retrieval.reranker_scores), start=1
    ):
        title = chunk.paper_title
        if len(title) > 52:
            title = title[:49] + "…"
        if chunk.section_number:
            sec = f"§{chunk.section_number} {chunk.section_heading}".strip()
        else:
            sec = f"§{chunk.section_heading}" if chunk.section_heading else "§Body"
        table.add_row(str(i), title, sec[:24], f"{score:+.3f}")

    console.print(table)
    console.print()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retriever, generator = load_pipeline(cfg, args.backend, args.top_k)

    # ── Single-shot mode ──────────────────────────────────────────────────────
    if args.question:
        run_query(args.question, retriever, generator)
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    console.print("[bold]Ask a question[/]  [dim](type 'quit' or Ctrl-C to exit)[/]\n")

    while True:
        try:
            question = console.input("[bold cyan]>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/]")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Bye.[/]")
            break

        run_query(question, retriever, generator)


if __name__ == "__main__":
    main()
