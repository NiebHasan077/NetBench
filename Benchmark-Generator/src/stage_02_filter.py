"""Stage 02 — auto-relevance filter.

Score every paper by max-passage cosine similarity to the v4 benchmark seed
(question + reference_answer text), keep the top-K above a minimum threshold.

Inputs:
  data/passages.jsonl                   (Stage 01 output)
  ../NetBench-LLM/data/prompts/hpn_qa_benchmark_v4_general_skills.json

Outputs:
  data/seed_text.txt                    seed text for reproducibility
  data/paper_scores.jsonl               {paper_id, score} for ALL papers
  data/filtered_papers.jsonl            {paper_id, score} for survivors
  data/_cache/passage_embeddings.npy    (cached) passage embeddings
  data/_cache/passage_embedding_ids.json (paired) passage ID list
  reports/stage_02_filter.md            top/bottom spot-check report

Usage:
    .venv/bin/python -m src.stage_02_filter
    .venv/bin/python -m src.stage_02_filter --device cuda:0 --batch-size 64
    .venv/bin/python -m src.stage_02_filter --refresh-embeddings
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken

from src.config import load_config
from src.embeddings import Embedder, detect_device
from src.io_utils import read_jsonl, write_jsonl

logger = logging.getLogger("stage_02")


def build_seed_text(v4_path: Path) -> str:
    with v4_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    parts: list[str] = []
    for q in data.get("questions", []):
        for key in ("question", "reference_answer"):
            t = q.get(key, "")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n\n".join(parts)


def chunk_seed(text: str, encoder: tiktoken.Encoding, window: int = 256, overlap: int = 32) -> list[str]:
    tokens = encoder.encode(text, disallowed_special=())
    if not tokens:
        return []
    stride = window - overlap
    if stride <= 0:
        raise ValueError(f"window ({window}) must exceed overlap ({overlap})")
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + window, len(tokens))
        chunks.append(encoder.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start += stride
    return chunks


def first_passage_preview(passages_by_paper: dict, paper_id: str, max_chars: int = 240) -> str:
    plist = passages_by_paper.get(paper_id, [])
    if not plist:
        return ""
    return plist[0]["text"][:max_chars].replace("\n", " ").strip()


def write_report(
    report_path: Path,
    survivors: list[tuple[str, float]],
    sorted_scores: list[tuple[str, float]],
    passages_by_paper: dict,
    n_passages: int,
    n_papers: int,
    n_seed_chunks: int,
    top_k: int,
    min_score: float,
    rejected_below_min: int,
) -> None:
    lines = [
        "# Stage 02 relevance filter — spot-check report",
        "",
        f"- corpus: {n_passages} passages over {n_papers} papers",
        f"- seed: {n_seed_chunks} chunks from v4 benchmark",
        f"- config: top_k={top_k}, min_score={min_score}",
        f"- survivors: **{len(survivors)}**",
    ]
    if rejected_below_min:
        lines.append(f"- {rejected_below_min} papers in top-K but below min_score (rejected)")
    lines.append("")

    lines.append("## Top 20 by score")
    for pid, score in survivors[:20]:
        lines.append(f"- **{pid}** ({score:.4f}) — {first_passage_preview(passages_by_paper, pid)}")

    if len(survivors) > 20:
        lines.append("\n## Bottom 20 of survivors")
        for pid, score in survivors[-20:]:
            lines.append(f"- **{pid}** ({score:.4f}) — {first_passage_preview(passages_by_paper, pid)}")

    lines.append("\n## 10 just below the survivor cutoff")
    cutoff_idx = len(survivors)
    for pid, score in sorted_scores[cutoff_idx : cutoff_idx + 10]:
        lines.append(f"- **{pid}** ({score:.4f}) — {first_passage_preview(passages_by_paper, pid)}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--passages", default=None, help="passages JSONL (default: data/passages.jsonl)")
    parser.add_argument("--v4", default=None, help="path to v4 benchmark JSON (default from config)")
    parser.add_argument("--top-k", type=int, default=None, help="papers to keep (default from config)")
    parser.add_argument("--min-score", type=float, default=None, help="minimum cosine score (default from config)")
    parser.add_argument("--device", default=None, help="torch device (cuda:0, cpu, etc.)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed-window", type=int, default=256, help="seed chunk size in tokens")
    parser.add_argument("--seed-overlap", type=int, default=32)
    parser.add_argument("--refresh-embeddings", action="store_true", help="ignore cached passage embeddings")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    data_dir = cfg.resolve(cfg.paths.data_dir)
    cache_dir = data_dir / "_cache"
    reports_dir = cfg.resolve(cfg.paths.reports_dir)

    passages_path = Path(args.passages) if args.passages else data_dir / "passages.jsonl"
    v4_path = Path(args.v4) if args.v4 else cfg.resolve(cfg.paths.v4_benchmark)
    scores_path = data_dir / "paper_scores.jsonl"
    filtered_path = data_dir / "filtered_papers.jsonl"
    seed_text_path = data_dir / "seed_text.txt"
    cache_emb_path = cache_dir / "passage_embeddings.npy"
    cache_ids_path = cache_dir / "passage_embedding_ids.json"
    report_path = reports_dir / "stage_02_filter.md"

    top_k = args.top_k if args.top_k is not None else cfg.filter.top_k_papers
    min_score = args.min_score if args.min_score is not None else cfg.filter.min_score

    if not passages_path.exists():
        logger.error("passages not found: %s. Run stage_01 first.", passages_path)
        return 1
    if not v4_path.exists():
        logger.error("v4 benchmark not found: %s", v4_path)
        return 1

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading passages: %s", passages_path)
    passages = list(read_jsonl(passages_path))
    n_passages = len(passages)
    if not n_passages:
        logger.error("no passages found")
        return 1

    paper_ids_unique = {p["paper_id"] for p in passages}
    n_papers = len(paper_ids_unique)
    logger.info("loaded %d passages over %d papers", n_passages, n_papers)

    # --- seed ---
    logger.info("building seed text from %s", v4_path)
    seed_text = build_seed_text(v4_path)
    if not seed_text.strip():
        logger.error("seed text is empty; check v4 benchmark file")
        return 1
    seed_text_path.write_text(seed_text, encoding="utf-8")

    encoder = tiktoken.get_encoding("cl100k_base")
    seed_chunks = chunk_seed(seed_text, encoder, window=args.seed_window, overlap=args.seed_overlap)
    logger.info(
        "seed: %d chars, %d chunks (window=%d, overlap=%d)",
        len(seed_text), len(seed_chunks), args.seed_window, args.seed_overlap,
    )

    # --- embedder ---
    device = detect_device(args.device)
    embedder = Embedder(cfg.models.embedding, device=device, batch_size=args.batch_size)

    # --- passage embeddings (cached) ---
    passage_emb: Optional[np.ndarray] = None
    cur_ids = [p["passage_id"] for p in passages]
    if not args.refresh_embeddings and cache_emb_path.exists() and cache_ids_path.exists():
        try:
            cached_ids = json.loads(cache_ids_path.read_text(encoding="utf-8"))
            if cached_ids == cur_ids:
                arr = np.load(cache_emb_path)
                if arr.shape[0] == n_passages:
                    passage_emb = arr
                    logger.info("loaded cached embeddings: %s", arr.shape)
                else:
                    logger.warning("cached array length mismatch (%d vs %d); recomputing", arr.shape[0], n_passages)
            else:
                logger.warning("cached IDs differ from current passages; recomputing")
        except Exception as e:
            logger.warning("could not load cache (%s); recomputing", e)

    if passage_emb is None:
        logger.info("embedding %d passages on %s", n_passages, device)
        passage_texts = [p["text"] for p in passages]
        passage_emb = embedder.embed(passage_texts, normalize=True, show_progress=True)
        np.save(cache_emb_path, passage_emb)
        cache_ids_path.write_text(json.dumps(cur_ids), encoding="utf-8")
        logger.info("saved cache: %s (shape=%s)", cache_emb_path, passage_emb.shape)

    # --- seed embeddings ---
    logger.info("embedding %d seed chunks", len(seed_chunks))
    seed_emb = embedder.embed(seed_chunks, normalize=True, show_progress=False)

    if passage_emb.shape[1] != seed_emb.shape[1]:
        logger.error(
            "embedding dim mismatch: passages=%d, seed=%d",
            passage_emb.shape[1], seed_emb.shape[1],
        )
        return 2

    # --- similarity ---
    logger.info(
        "computing similarity: passages=%s @ seed=%s.T",
        passage_emb.shape, seed_emb.shape,
    )
    sims = passage_emb @ seed_emb.T  # (N_passages, N_seed_chunks)
    passage_max = sims.max(axis=1)

    # Aggregate to paper-level (max passage score)
    paper_scores: dict[str, float] = {}
    for p, s in zip(passages, passage_max):
        pid = p["paper_id"]
        v = float(s)
        if pid not in paper_scores or v > paper_scores[pid]:
            paper_scores[pid] = v

    sorted_scores = sorted(paper_scores.items(), key=lambda x: -x[1])
    top_slice = sorted_scores[:top_k]
    survivors = [(pid, s) for pid, s in top_slice if s >= min_score]
    rejected_below_min = sum(1 for _, s in top_slice if s < min_score)

    cutoff_score = top_slice[-1][1] if top_slice else float("-inf")
    logger.info(
        "top_k=%d cutoff_score=%.4f; survivors above min_score=%.2f: %d (%d in top-K rejected by min_score)",
        top_k, cutoff_score, min_score, len(survivors), rejected_below_min,
    )

    # --- write outputs ---
    write_jsonl(scores_path, ({"paper_id": pid, "score": s} for pid, s in sorted_scores))
    write_jsonl(filtered_path, ({"paper_id": pid, "score": s} for pid, s in survivors))

    passages_by_paper: dict[str, list[dict]] = {}
    for p in passages:
        passages_by_paper.setdefault(p["paper_id"], []).append(p)

    write_report(
        report_path=report_path,
        survivors=survivors,
        sorted_scores=sorted_scores,
        passages_by_paper=passages_by_paper,
        n_passages=n_passages,
        n_papers=n_papers,
        n_seed_chunks=len(seed_chunks),
        top_k=top_k,
        min_score=min_score,
        rejected_below_min=rejected_below_min,
    )
    logger.info("report: %s", report_path)
    logger.info("scores: %s", scores_path)
    logger.info("filtered: %s", filtered_path)

    # --- stdout summary for quick eyeball ---
    print("\n--- top 5 ---")
    for pid, score in survivors[:5]:
        print(f"  {pid} ({score:.4f}) — {first_passage_preview(passages_by_paper, pid)}")
    print("\n--- bottom 5 of survivors ---")
    for pid, score in survivors[-5:]:
        print(f"  {pid} ({score:.4f}) — {first_passage_preview(passages_by_paper, pid)}")
    print("\n--- 5 just below cutoff ---")
    for pid, score in sorted_scores[len(survivors) : len(survivors) + 5]:
        print(f"  {pid} ({score:.4f}) — {first_passage_preview(passages_by_paper, pid)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
