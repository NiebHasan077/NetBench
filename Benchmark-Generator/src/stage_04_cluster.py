"""Stage 04 — topic clustering.

Embed each paper card's topical signature (main_problem + key_concepts + topics),
cluster with HDBSCAN, and write back cards with `cluster_id` populated. Cluster -1
is HDBSCAN's "noise" bucket — kept on each card but excluded from the synthesis-pair
sampling in Phase 5b.

Inputs:
  data/paper_cards.jsonl

Outputs:
  data/paper_cards_clustered.jsonl  cards with cluster_id added
  reports/clusters.md               cluster sizes + top-10 TF-IDF keywords per cluster
  data/_cache/card_embeddings.npy   (cached) card embeddings
  data/_cache/card_embedding_ids.json (paired) paper_id list

Usage:
    .venv/bin/python -m src.stage_04_cluster
    .venv/bin/python -m src.stage_04_cluster --min-cluster-size 4
    .venv/bin/python -m src.stage_04_cluster --refresh-embeddings
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import hdbscan
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import load_config
from src.embeddings import Embedder, detect_device
from src.io_utils import read_jsonl, write_jsonl

logger = logging.getLogger("stage_04")


def card_text_for_embedding(card: dict) -> str:
    """Build the topical signature used for clustering."""
    parts = [
        card.get("main_problem", ""),
        " ".join(card.get("key_concepts", [])),
        " ".join(card.get("possible_question_topics", [])),
    ]
    return " ".join(p for p in parts if p).strip()


def card_text_for_tfidf(card: dict) -> str:
    """Pooled text used per-card for TF-IDF — slightly broader than the embedding text."""
    parts = [
        card.get("main_problem", ""),
        card.get("system_context", ""),
        " ".join(card.get("key_concepts", [])),
        " ".join(card.get("failure_modes", [])),
        " ".join(card.get("possible_question_topics", [])),
    ]
    return " ".join(p for p in parts if p).strip()


def top_terms_per_cluster(
    cards: list[dict],
    cluster_ids: np.ndarray,
    top_n: int = 10,
) -> dict[int, list[str]]:
    """Pool card text by cluster, run TF-IDF across cluster pools, return top-N per cluster."""
    by_cluster: dict[int, list[str]] = {}
    for card, cid in zip(cards, cluster_ids):
        by_cluster.setdefault(int(cid), []).append(card_text_for_tfidf(card))
    cluster_keys = sorted(by_cluster.keys())
    cluster_docs = [" ".join(by_cluster[k]) for k in cluster_keys]
    if not cluster_docs:
        return {}
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=1,
        stop_words="english",
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_/-]{2,}\b",
    )
    try:
        m = vec.fit_transform(cluster_docs)
    except ValueError:
        return {k: [] for k in cluster_keys}
    terms = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for i, k in enumerate(cluster_keys):
        row = m.getrow(i).toarray().ravel()
        if not row.any():
            out[k] = []
            continue
        top_idx = row.argsort()[::-1][:top_n]
        out[k] = [terms[j] for j in top_idx if row[j] > 0]
    return out


def write_report(
    report_path: Path,
    cards: list[dict],
    cluster_ids: np.ndarray,
    keywords: dict[int, list[str]],
    min_cluster_size: int,
) -> None:
    counts = Counter(int(c) for c in cluster_ids)
    n_total = len(cards)
    n_noise = counts.get(-1, 0)
    n_clusters = sum(1 for k in counts if k != -1)
    lines = [
        "# Stage 04 clustering — report",
        "",
        f"- cards: {n_total}",
        f"- clusters (excluding noise): **{n_clusters}**",
        f"- noise (cluster_id=-1): {n_noise} cards",
        f"- min_cluster_size: {min_cluster_size}",
        "",
        "## Clusters by size",
        "",
    ]
    # Cluster_id by descending size; noise last
    ordered = sorted(
        (k for k in counts if k != -1),
        key=lambda k: counts[k],
        reverse=True,
    )
    if -1 in counts:
        ordered.append(-1)

    paper_ids_by_cluster: dict[int, list[str]] = {}
    for card, cid in zip(cards, cluster_ids):
        paper_ids_by_cluster.setdefault(int(cid), []).append(card["paper_id"])

    for cid in ordered:
        size = counts[cid]
        kws = keywords.get(cid, [])
        label = f"cluster {cid}" if cid != -1 else "cluster -1 (noise)"
        lines.append(f"### {label} — {size} cards")
        if kws:
            lines.append(f"top terms: {', '.join(kws)}")
        else:
            lines.append("top terms: (none)")
        sample = paper_ids_by_cluster[cid][:6]
        lines.append(f"sample paper_ids: {', '.join(sample)}{'…' if size > len(sample) else ''}")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--cards", default=None, help="paper_cards.jsonl (default from config)")
    parser.add_argument("--output", default=None, help="paper_cards_clustered.jsonl (default from config)")
    parser.add_argument("--report", default=None, help="report path (default: reports/clusters.md)")
    parser.add_argument("--device", default=None, help="torch device (cuda:0, cpu, etc.)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-cluster-size", type=int, default=None, help="HDBSCAN min_cluster_size (default from config)")
    parser.add_argument("--selection-method", default=None, choices=["eom", "leaf"], help="HDBSCAN cluster_selection_method (default from config)")
    parser.add_argument("--refresh-embeddings", action="store_true")
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
    cache_dir.mkdir(parents=True, exist_ok=True)

    cards_path = Path(args.cards) if args.cards else data_dir / "paper_cards.jsonl"
    output_path = Path(args.output) if args.output else data_dir / "paper_cards_clustered.jsonl"
    report_path = Path(args.report) if args.report else reports_dir / "clusters.md"
    cache_emb_path = cache_dir / "card_embeddings.npy"
    cache_ids_path = cache_dir / "card_embedding_ids.json"

    min_cluster_size = args.min_cluster_size or cfg.clustering.min_cluster_size
    selection_method = args.selection_method or cfg.clustering.cluster_selection_method

    if not cards_path.exists():
        logger.error("paper_cards not found: %s. Run stage_03 first.", cards_path)
        return 1

    cards = list(read_jsonl(cards_path))
    if not cards:
        logger.error("no cards loaded")
        return 1
    logger.info("loaded %d cards", len(cards))

    paper_ids = [c["paper_id"] for c in cards]
    texts = [card_text_for_embedding(c) for c in cards]

    # Embed (cached)
    emb: Optional[np.ndarray] = None
    if not args.refresh_embeddings and cache_emb_path.exists() and cache_ids_path.exists():
        try:
            cached_ids = json.loads(cache_ids_path.read_text(encoding="utf-8"))
            if cached_ids == paper_ids:
                arr = np.load(cache_emb_path)
                if arr.shape[0] == len(cards):
                    emb = arr
                    logger.info("loaded cached card embeddings: %s", arr.shape)
        except Exception as e:
            logger.warning("could not load cache (%s); recomputing", e)

    if emb is None:
        device = detect_device(args.device)
        embedder = Embedder(cfg.models.embedding, device=device, batch_size=args.batch_size)
        logger.info("embedding %d cards on %s", len(cards), device)
        emb = embedder.embed(texts, normalize=True, show_progress=True)
        np.save(cache_emb_path, emb)
        cache_ids_path.write_text(json.dumps(paper_ids), encoding="utf-8")
        logger.info("saved cache: %s shape=%s", cache_emb_path, emb.shape)

    # HDBSCAN — embeddings are already L2-normalized so euclidean ≡ cosine on the unit sphere.
    logger.info("HDBSCAN: min_cluster_size=%d, selection_method=%s", min_cluster_size, selection_method)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method=selection_method,
        prediction_data=False,
    )
    cluster_ids = clusterer.fit_predict(emb.astype(np.float64))

    n_clusters = len(set(int(c) for c in cluster_ids if c != -1))
    n_noise = int((cluster_ids == -1).sum())
    logger.info("clusters: %d (excluding noise), noise: %d / %d cards", n_clusters, n_noise, len(cards))

    # Write enriched cards
    write_jsonl(
        output_path,
        ({**c, "cluster_id": int(cid)} for c, cid in zip(cards, cluster_ids)),
    )
    logger.info("wrote %s", output_path)

    # TF-IDF top terms per cluster
    logger.info("computing top TF-IDF terms per cluster")
    keywords = top_terms_per_cluster(cards, cluster_ids, top_n=10)

    write_report(report_path, cards, cluster_ids, keywords, min_cluster_size)
    logger.info("wrote %s", report_path)

    # stdout summary
    counts = Counter(int(c) for c in cluster_ids)
    print("\n--- cluster sizes (top 10) ---")
    for cid, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:10]:
        kws = keywords.get(cid, [])[:6]
        label = f"cluster {cid}" if cid != -1 else "noise"
        print(f"  {label}: {n} cards | {', '.join(kws)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
