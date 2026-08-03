#!/usr/bin/env python3
"""
RAG Pipeline Benchmark — Answer Generation (Phase 1)
=====================================================

Runs all HPN benchmark questions through the full NetBench-RAG pipeline
(retrieval + generation) and saves the answers to Excel for LLM-as-Judge
scoring in Phase 2.

Three model options via --model:
  openai           — GPT-4o via the OpenAI API
  local            — Local model at paths.local_model in config.yaml
  /path/to/model   — Any local model directory (overrides config.yaml)

Output:
    outputs/answers/hpn_answers_RAG-{model_name}_{benchmark_tag}.xlsx

The Excel workbook has two sheets:
    • Metadata — pipeline config, retriever settings, timing
    • Answers  — one row per question (same schema as LLM-Training Phase 1,
                 plus RAG-specific columns: retrieval_time_sec, prompt_tokens,
                 retrieved_chunks, reranker_scores)

Usage:
    # OpenAI GPT-4o
    python evaluation/evaluate_rag.py --model openai

    # Local model from config.yaml
    python evaluation/evaluate_rag.py --model local

    # Specific local model by path
    python evaluation/evaluate_rag.py --model /path/to/LLM-Training/models/ModelA

    # Multiple models sequentially (names or paths)
    python evaluation/evaluate_rag.py --model openai local
    python evaluation/evaluate_rag.py --model /path/to/ModelA /path/to/ModelB

    # Override top-k chunks passed to the LLM
    python evaluation/evaluate_rag.py --model openai --top_k 3

    # Custom config / benchmark
    python evaluation/evaluate_rag.py --model openai \\
        --config config.yaml \\
        --benchmark data/prompts/hpn_qa_benchmark_v5_general_skills.json
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

from src.generator import generator_from_config
from src.retriever import retriever_from_config

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = str(ROOT / "config.yaml")
DEFAULT_OUTPUT_DIR = str(ROOT / "outputs" / "answers")


def _benchmark_tag(benchmark_path: str) -> str:
    """Extract a concise tag from the benchmark filename.

    Examples:
        'hpn_qa_benchmark_v4_general_skills.json' → 'v4_general_skills'
        'hpn_qa_benchmark.json'                   → ''  (no tag)
    """
    stem = Path(benchmark_path).stem
    prefix = "hpn_qa_benchmark"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix) + 1:]   # strip prefix + underscore
    return ""

# XML 1.0 disallows: \x00-\x08, \x0b, \x0c, \x0e-\x1f  (tab/LF/CR are fine)
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", re.UNICODE)


def _cell(value: object) -> object:
    """Sanitize a value before writing it to an Excel cell.

    Strips XML-illegal control characters from strings so openpyxl never
    raises ``IllegalCharacterError``.  Non-string values are passed through
    unchanged.
    """
    if isinstance(value, str):
        return _ILLEGAL_XML_RE.sub("", value)
    return value


# ═══════════════════════════════════════════════════════════════════════
# Excel style constants (identical to LLM-Training Phase 1)
# ═══════════════════════════════════════════════════════════════════════

_HEADER_FONT = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
_HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
_HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
_WRAP_ALIGN = Alignment(vertical="top", wrap_text=True)
_THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
_META_KEY_FONT = Font(name="Calibri", bold=True, size=11)


def _style_header_row(ws, num_cols: int):
    """Apply header styling to row 1."""
    for col in range(1, num_cols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _HEADER_ALIGN
        cell.border = _THIN_BORDER


# ═══════════════════════════════════════════════════════════════════════
# Benchmark loader
# ═══════════════════════════════════════════════════════════════════════

def load_benchmark(benchmark_path: str) -> dict:
    """Load the HPN Q&A benchmark JSON file."""
    path = Path(benchmark_path)
    if not path.exists():
        print(f"❌ Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    metadata = data.get("metadata", {})
    print(f"📋 Loaded benchmark: {metadata.get('title', benchmark_path)}")
    print(f"   Questions : {len(questions)}")
    print(f"   Categories: {len(metadata.get('categories', []))}")
    print(f"   Difficulty: {metadata.get('difficulty_levels', [])}")
    return data


# ═══════════════════════════════════════════════════════════════════════
# Excel writer
# ═══════════════════════════════════════════════════════════════════════

def save_answers_to_excel(
    output_path: str,
    model_name: str,
    backend: str,
    benchmark_file: str,
    cfg: dict,
    answers: list[dict],
    total_time: float,
):
    """Save RAG answers and metadata to an Excel workbook.

    Schema is identical to LLM-Training hpn_answers_{model}.xlsx, plus
    RAG-specific columns: retrieval_time_sec, prompt_tokens,
    retrieved_chunks (semicolon-separated titles), reranker_scores.
    """
    wb = Workbook()
    r = cfg.get("retrieval", {})
    gen_cfg = cfg.get("generation", {})

    # ── Metadata sheet ──────────────────────────────────────────────
    ws_meta = wb.active
    ws_meta.title = "Metadata"

    meta_rows = [
        ("Key", "Value"),
        ("model_name", model_name),
        ("backend", backend),
        ("benchmark_file", benchmark_file),
        ("timestamp", datetime.now().isoformat()),
        ("total_questions", len(answers)),
        ("vector_top_k", r.get("vector_top_k", 30)),
        ("bm25_top_k", r.get("bm25_top_k", 30)),
        ("rrf_top_n", r.get("rrf_top_n", 20)),
        ("reranker_top_n", r.get("reranker_top_n", 5)),
        ("reranker_model", r.get("reranker_model", "BAAI/bge-reranker-v2-m3")),
        ("generation_max_tokens",
         gen_cfg.get(backend, {}).get("max_tokens",
         gen_cfg.get(backend, {}).get("max_new_tokens", 512))),
        ("generation_temperature",
         gen_cfg.get(backend, {}).get("temperature", 0.3)),
        ("total_pipeline_time_sec", f"{total_time:.1f}"),
        ("avg_pipeline_time_sec",
         f"{total_time / max(len(answers), 1):.2f}"),
    ]

    for row_idx, (key, value) in enumerate(meta_rows, start=1):
        ws_meta.cell(row=row_idx, column=1, value=key)
        ws_meta.cell(row=row_idx, column=2, value=str(value))
        if row_idx == 1:
            for col in (1, 2):
                ws_meta.cell(row=row_idx, column=col).font = _HEADER_FONT
                ws_meta.cell(row=row_idx, column=col).fill = _HEADER_FILL
                ws_meta.cell(row=row_idx, column=col).alignment = _HEADER_ALIGN
        else:
            ws_meta.cell(row=row_idx, column=1).font = _META_KEY_FONT
            ws_meta.cell(row=row_idx, column=1).border = _THIN_BORDER
            ws_meta.cell(row=row_idx, column=2).border = _THIN_BORDER

    ws_meta.column_dimensions["A"].width = 28
    ws_meta.column_dimensions["B"].width = 80

    # ── Answers sheet ───────────────────────────────────────────────
    ws_ans = wb.create_sheet("Answers")

    headers = [
        "id",
        "category",
        "difficulty",
        "question_type",
        "question",
        "reference_answer",
        "model_answer",
        "keywords",
        "retrieval_time_sec",
        "generation_time_sec",
        "prompt_tokens",
        "retrieved_chunks",
        "reranker_scores",
    ]

    for col_idx, header in enumerate(headers, start=1):
        ws_ans.cell(row=1, column=col_idx, value=header)
    _style_header_row(ws_ans, len(headers))

    for row_idx, ans in enumerate(answers, start=2):
        ws_ans.cell(row=row_idx, column=1,  value=_cell(ans.get("id")))
        ws_ans.cell(row=row_idx, column=2,  value=_cell(ans.get("category")))
        ws_ans.cell(row=row_idx, column=3,  value=_cell(ans.get("difficulty")))
        ws_ans.cell(row=row_idx, column=4,  value=_cell(ans.get("question_type", "")))
        ws_ans.cell(row=row_idx, column=5,  value=_cell(ans.get("question")))
        ws_ans.cell(row=row_idx, column=6,  value=_cell(ans.get("reference_answer")))
        ws_ans.cell(row=row_idx, column=7,  value=_cell(ans.get("model_answer")))
        ws_ans.cell(
            row=row_idx, column=8,
            value=_cell(
                ", ".join(ans.get("keywords", []))
                if isinstance(ans.get("keywords"), list)
                else str(ans.get("keywords", ""))
            ),
        )
        ws_ans.cell(
            row=row_idx, column=9,
            value=round(ans.get("retrieval_time_sec", 0.0), 3),
        )
        ws_ans.cell(
            row=row_idx, column=10,
            value=round(ans.get("generation_time_sec", 0.0), 3),
        )
        ws_ans.cell(row=row_idx, column=11, value=ans.get("prompt_tokens", 0))
        ws_ans.cell(row=row_idx, column=12, value=_cell(ans.get("retrieved_chunks", "")))
        ws_ans.cell(row=row_idx, column=13, value=_cell(ans.get("reranker_scores", "")))

        for col in range(1, len(headers) + 1):
            cell = ws_ans.cell(row=row_idx, column=col)
            cell.alignment = _WRAP_ALIGN
            cell.border = _THIN_BORDER

    col_widths = {
        "A": 8,   # id
        "B": 30,  # category
        "C": 12,  # difficulty
        "D": 16,  # question_type
        "E": 60,  # question
        "F": 80,  # reference_answer
        "G": 80,  # model_answer
        "H": 40,  # keywords
        "I": 18,  # retrieval_time_sec
        "J": 18,  # generation_time_sec
        "K": 14,  # prompt_tokens
        "L": 60,  # retrieved_chunks
        "M": 40,  # reranker_scores
    }
    for col_letter, width in col_widths.items():
        ws_ans.column_dimensions[col_letter].width = width

    ws_ans.freeze_panes = "A2"
    ws_ans.auto_filter.ref = (
        f"A1:{get_column_letter(len(headers))}{len(answers) + 1}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"💾 Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════

def _checkpoint_path(output_dir: str, model_name: str) -> Path:
    """Return the path of the JSON checkpoint file for *model_name*."""
    return Path(output_dir) / f"hpn_answers_{model_name}.checkpoint.json"


def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set[str]]:
    """Load a checkpoint file and return (answers_so_far, answered_ids)."""
    if not ckpt_path.exists():
        return [], set()
    with open(ckpt_path, encoding="utf-8") as f:
        answers = json.load(f)
    answered_ids = {a["id"] for a in answers}
    return answers, answered_ids


def _save_checkpoint(ckpt_path: Path, answers: list[dict]) -> None:
    """Atomically write *answers* to *ckpt_path*."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    tmp.replace(ckpt_path)


# ═══════════════════════════════════════════════════════════════════════
# Core pipeline runner
# ═══════════════════════════════════════════════════════════════════════

def run_rag_benchmark(
    model_spec: str,
    questions: list[dict],
    cfg: dict,
    benchmark_file: str,
    output_dir: str,
    top_k_override: int | None,
):
    """Run the full benchmark for a single model spec.

    ``model_spec`` is one of:
      - ``"openai"``        — GPT-4o via the OpenAI API
      - ``"local"``         — local model at ``paths.local_model`` in config
      - ``"/path/to/dir"``  — filesystem path to a local model directory

    1. Resolve model spec → backend name + optional path override
    2. Load retriever + generator
    3. Run each question through the RAG pipeline
    4. Save to Excel
    """
    # ── Resolve model spec ────────────────────────────────────────────────
    model_spec = model_spec.strip()
    if model_spec in ("openai", "local"):
        backend = model_spec
        model_path_override = None
    else:
        # Treat any other value as a filesystem path to a local model directory
        backend = "local"
        model_path_override = model_spec

    # Patch cfg with the path override (shallow copy; does not modify original)
    if model_path_override is not None:
        cfg = {**cfg, "paths": {**cfg.get("paths", {}), "local_model": model_path_override}}

    print(f"\n{'=' * 70}")
    print(f"🔄 Model: {model_spec}")
    print(f"{'=' * 70}")

    # ── Load pipeline ───────────────────────────────────────────────
    print("⚙️  Loading retrieval models...", end=" ", flush=True)
    t0 = time.perf_counter()
    retriever = retriever_from_config(cfg)
    if top_k_override is not None:
        retriever.reranker_top_n = top_k_override
        print(f"(top_k overridden to {top_k_override})", end=" ")
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    print(f"⚙️  Loading generator ({backend})...", end=" ", flush=True)
    t0 = time.perf_counter()
    generator = generator_from_config(cfg, backend_override=backend)
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    # Derive a clean model name for file naming
    raw_model = generator._backend.model   # e.g. "gpt-4o" or dir basename
    model_name = f"RAG-{raw_model}"

    print(f"\n  Retriever : vector top-{retriever.vector_top_k} + "
          f"BM25 top-{retriever.bm25_top_k} "
          f"→ RRF top-{retriever.rrf_top_n} "
          f"→ reranker top-{retriever.reranker_top_n}")
    print(f"  Generator : {raw_model} ({backend})")
    print(f"  Model tag : {model_name}\n")

    # ── Checkpoint: load any previously answered questions ───────────
    tag = _benchmark_tag(benchmark_file)
    tag_suffix = f"_{tag}" if tag else ""
    output_path = os.path.join(output_dir, f"hpn_answers_{model_name}{tag_suffix}.xlsx")
    ckpt_path = _checkpoint_path(output_dir, model_name)
    answers, answered_ids = _load_checkpoint(ckpt_path)

    if answered_ids:
        print(f"♻️  Resuming from checkpoint — {len(answered_ids)} questions already done.")

    remaining = [q for q in questions if q.get("id", "") not in answered_ids]

    # ── Generate answers ─────────────────────────────────────────────
    total_start = time.perf_counter()
    already_done = len(answers)
    total_q = len(questions)

    print(f"📝 Running {len(remaining)}/{total_q} questions through RAG pipeline...")
    for idx, q in enumerate(remaining, start=already_done + 1):
        q_id = q.get("id", f"Q{idx:03d}")
        print(f"  [{idx:3d}/{total_q}] {q_id} ", end="", flush=True)

        question_text = q["question"]

        # ── Retrieve ────────────────────────────────────────────────
        t_ret = time.perf_counter()
        retrieval = retriever.retrieve(question_text)
        ret_time = time.perf_counter() - t_ret

        # ── Generate ────────────────────────────────────────────────
        t_gen = time.perf_counter()
        result = generator.generate(question_text, retrieval.chunks)
        gen_time = time.perf_counter() - t_gen

        # ── Chunk metadata for storage ───────────────────────────────
        chunk_labels = []
        for chunk in retrieval.chunks:
            title = chunk.paper_title[:40] + "…" if len(chunk.paper_title) > 40 else chunk.paper_title
            sec = f"§{chunk.section_number} {chunk.section_heading}".strip() if chunk.section_number else f"§{chunk.section_heading}"
            chunk_labels.append(f"{title} {sec}")
        retrieved_chunks_str = " | ".join(chunk_labels)

        scores_str = ", ".join(
            f"{s:+.3f}" for s in retrieval.reranker_scores
        )

        print(f"ret={ret_time:.1f}s gen={gen_time:.1f}s "
              f"tok={result.prompt_tokens}")

        answers.append({
            "id": q_id,
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "question_type": q.get("question_type", ""),
            "question": question_text,
            "reference_answer": q.get("reference_answer", ""),
            "model_answer": result.answer,
            "keywords": q.get("keywords", []),
            "retrieval_time_sec": ret_time,
            "generation_time_sec": gen_time,
            "prompt_tokens": result.prompt_tokens,
            "retrieved_chunks": retrieved_chunks_str,
            "reranker_scores": scores_str,
        })

        # Persist progress after every question
        _save_checkpoint(ckpt_path, answers)

    total_time = time.perf_counter() - total_start

    # ── Save to Excel ────────────────────────────────────────────────
    save_answers_to_excel(
        output_path=output_path,
        model_name=model_name,
        backend=backend,
        benchmark_file=benchmark_file,
        cfg=cfg,
        answers=answers,
        total_time=total_time,
    )

    # Checkpoint is no longer needed once Excel is written successfully
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"🗑️  Checkpoint removed: {ckpt_path.name}")

    avg_time = total_time / max(len(answers), 1)
    avg_ret = sum(a["retrieval_time_sec"] for a in answers) / max(len(answers), 1)
    avg_gen = sum(a["generation_time_sec"] for a in answers) / max(len(answers), 1)

    print(f"\n{'─' * 70}")
    print(f"✅ {model_name} — {len(answers)} answers in {total_time:.1f}s "
          f"(avg {avg_time:.2f}s/question)")
    print(f"   Avg retrieval : {avg_ret:.2f}s")
    print(f"   Avg generation: {avg_gen:.2f}s")
    print(f"   Output: {output_path}")
    print(f"{'─' * 70}\n")

    return model_name


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Benchmark — Answer Generation (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # OpenAI GPT-4o\n"
            "  python evaluation/evaluate_rag.py --model openai\n"
            "\n"
            "  # Local model (path from config.yaml)\n"
            "  python evaluation/evaluate_rag.py --model local\n"
            "\n"
            "  # Specific local model by path\n"
            "  python evaluation/evaluate_rag.py --model /path/to/ModelA\n"
            "\n"
            "  # Multiple models sequentially\n"
            "  python evaluation/evaluate_rag.py --model openai /path/to/ModelA\n"
            "\n"
            "  # Override top-k chunks passed to the LLM\n"
            "  python evaluation/evaluate_rag.py --model openai --top_k 3\n"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["openai"],
        help=(
            "Model(s) to benchmark. Accepts: 'openai', 'local', or a filesystem "
            "path to a local model directory. Multiple values allowed, e.g. "
            "--model openai /path/to/ModelA /path/to/ModelB (default: openai)"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark JSON — overrides config.yaml paths.benchmark",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        metavar="N",
        help="Override reranker_top_n (chunks passed to LLM). Default: from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output Excel files (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🏁  NetBench-RAG — Answer Generation (Phase 1)")
    print("=" * 70)

    # ── Load config ─────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Change working directory to project root so relative paths in
    # config.yaml resolve correctly
    os.chdir(str(ROOT))

    # ── Resolve benchmark path ──────────────────────────────────────
    benchmark_file = args.benchmark or cfg.get("paths", {}).get(
        "benchmark", "data/prompts/hpn_qa_benchmark_v4_general_skills.json"
    )

    benchmark = load_benchmark(benchmark_file)
    questions = benchmark["questions"]

    print(f"\n⚙️  Configuration:")
    print(f"   Config file   : {args.config}")
    print(f"   Benchmark     : {benchmark_file}")
    print(f"   Models        : {args.model}")
    print(f"   Top-k override: {args.top_k if args.top_k else 'from config'}")
    print(f"   Output dir    : {args.output_dir}")

    # ── Run benchmark for each backend ──────────────────────────────
    overall_start = time.perf_counter()
    model_names = []

    for model_spec in args.model:
        name = run_rag_benchmark(
            model_spec=model_spec,
            questions=questions,
            cfg=cfg,
            benchmark_file=benchmark_file,
            output_dir=args.output_dir,
            top_k_override=args.top_k,
        )
        model_names.append(name)

    overall_time = time.perf_counter() - overall_start

    # ── Final summary ───────────────────────────────────────────────
    tag = _benchmark_tag(benchmark_file)
    tag_suffix = f"_{tag}" if tag else ""
    print("=" * 70)
    print(f"🏁 All done! {len(args.model)} model(s) benchmarked "
          f"in {overall_time:.1f}s")
    print(f"   Output files in: {args.output_dir}/")
    for name in model_names:
        print(f"     • hpn_answers_{name}{tag_suffix}.xlsx")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
