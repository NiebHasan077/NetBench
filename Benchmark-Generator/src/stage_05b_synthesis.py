"""Stage 05b — cross-paper synthesis question generation.

For each within-cluster card pair (and for supplementary noise-pool pairs ranked
by embedding similarity), generate 1 HPN benchmark question that requires
knowledge from BOTH papers simultaneously.

Inputs:
  data/paper_cards_clustered.jsonl
  data/passages.jsonl
  data/_cache/card_embeddings.npy       (from Stage 04, reused without recompute)
  data/_cache/card_embedding_ids.json
  prompts/synthesis_generation.txt

Outputs:
  data/synthesis_candidates.jsonl    one record per accepted synthesis question
  data/_logs/stage_05b.jsonl         per-pair run log

Usage:
    .venv/bin/python -m src.stage_05b_synthesis --limit 5     # pilot
    .venv/bin/python -m src.stage_05b_synthesis               # full run
    .venv/bin/python -m src.stage_05b_synthesis --resume      # skip already-done pairs
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config import load_config
from src.io_utils import append_jsonl, read_jsonl
from src.ollama_client import OllamaClient, OllamaError
from src.schema import DifficultyT, Evidence, Question

logger = logging.getLogger("stage_05b")
PROMPT_VERSION = "v5.0-synthesis"

CATEGORIES = [
    "Transfer Parameters: Definitions and Roles",
    "Concurrency Tuning and Scaling",
    "Pipelining and Small-File Optimization",
    "Parallelism and Large-File Optimization",
    "Dataset Partitioning and Mixed Workloads",
    "BDP-Based Reasoning and Window Sizing",
    "Bottleneck Diagnosis and End-to-End Reasoning",
    "Adaptive and Online Optimization",
    "Fairness, Stability, and Shared Networks",
    "Practical HPN Scenarios and Design",
]


# ---- Output schema (sent to Ollama as `format`) --------------------------------

class _DraftEvidence(BaseModel):
    paper_id: str
    passage_id: str
    quote: str


class _DraftSynthesisOutput(BaseModel):
    question: str
    reference_answer: str
    evidence: list[_DraftEvidence] = Field(default_factory=list)
    difficulty: DifficultyT
    category: str
    keywords: list[str] = Field(default_factory=list)


# ---- Pair utilities ------------------------------------------------------------

def _pair_id(paper_a_id: str, paper_b_id: str) -> str:
    """Deterministic pair ID — sorted so order of (a, b) doesn't matter."""
    nums = sorted([
        paper_a_id.replace("paper_", ""),
        paper_b_id.replace("paper_", ""),
    ])
    return f"NB-SYN-{nums[0]}-{nums[1]}"


def build_pairs(
    cards: list[dict],
    emb: np.ndarray,
    emb_ids: list[str],
    pairs_per_cluster: int,
    synthesis_target: int,
    seed: int,
) -> list[tuple[dict, dict, str]]:
    """Return list of (card_a, card_b, source_label) up to synthesis_target pairs.

    Priority order:
      1. Within-cluster pairs (for each non-noise cluster)
      2. Noise-pool similarity pairs (top-25% cosine, random sample) to fill
         remaining slots up to synthesis_target.
    """
    rng = random.Random(seed)
    id_to_idx: dict[str, int] = {pid: i for i, pid in enumerate(emb_ids)}

    pairs: list[tuple[dict, dict, str]] = []
    pair_ids_seen: set[str] = set()

    # 1. Within-cluster pairs
    by_cluster: dict[int, list[dict]] = {}
    for c in cards:
        cid = c.get("cluster_id", -1)
        if cid != -1:
            by_cluster.setdefault(cid, []).append(c)

    for cid, cluster_cards in sorted(by_cluster.items()):
        all_pairs = list(itertools.combinations(cluster_cards, 2))
        sampled = rng.sample(all_pairs, min(pairs_per_cluster, len(all_pairs)))
        for a, b in sampled:
            pid = _pair_id(a["paper_id"], b["paper_id"])
            if pid not in pair_ids_seen:
                pair_ids_seen.add(pid)
                pairs.append((a, b, f"cluster_{cid}"))

    # 2. Noise-pool similarity pairs
    noise_needed = max(0, synthesis_target - len(pairs))
    if noise_needed > 0:
        noise_cards = [c for c in cards if c.get("cluster_id", -1) == -1]
        # Keep only cards whose embeddings exist in the cache
        indexed = [(c, id_to_idx[c["paper_id"]]) for c in noise_cards if c["paper_id"] in id_to_idx]
        if len(indexed) >= 2:
            valid_noise_cards = [c for c, _ in indexed]
            emb_idx = [i for _, i in indexed]
            ne = emb[emb_idx]                    # (N_noise, D)
            sim_matrix = ne @ ne.T               # cosine sim (embeddings are L2-normalised)
            rows, cols = np.triu_indices(len(valid_noise_cards), k=1)
            sims = sim_matrix[rows, cols]

            # Focus on top-25% most-similar pairs (meaningfully related topics)
            threshold = float(np.percentile(sims, 75))
            mask = sims >= threshold
            hi_rows, hi_cols, hi_sims = rows[mask], cols[mask], sims[mask]

            sort_order = np.argsort(-hi_sims)
            # Oversample candidates 4× to allow dedup filtering
            oversample = min(len(sort_order), noise_needed * 4)
            pool = list(sort_order[:oversample])
            selected = rng.sample(pool, min(noise_needed, len(pool)))

            for idx in selected:
                a = valid_noise_cards[int(hi_rows[idx])]
                b = valid_noise_cards[int(hi_cols[idx])]
                pid = _pair_id(a["paper_id"], b["paper_id"])
                if pid not in pair_ids_seen:
                    pair_ids_seen.add(pid)
                    pairs.append((a, b, "noise_sim"))

    return pairs


# ---- Prompt building -----------------------------------------------------------

def build_user_prompt(
    card_a: dict,
    card_b: dict,
    passage_text_by_id: dict[str, str],
    max_passages_per_card: int = 3,
    passage_max_chars: int = 1000,
) -> str:
    def card_section(card: dict, label: str) -> list[str]:
        anchors = card.get("evidence_anchors", []) or []
        paper_id = card["paper_id"]
        lines = [
            f"=== {label}: {paper_id} ===",
            f"main_problem: {card.get('main_problem', '')}",
            f"key_concepts: {card.get('key_concepts', [])}",
            f"failure_modes: {card.get('failure_modes', [])}",
            "",
            f"ALLOWED PASSAGE IDs for {label} (paper_id={paper_id}):",
        ]
        for a in anchors:
            apid = a.get("passage_id", "")
            claim = a.get("claim", "")
            lines.append(f"  - {apid}  (claim: {claim})")
        lines.append("")
        lines.append(f"PASSAGE TEXTS for {label}:")
        lines.append("")
        shown = 0
        for a in anchors:
            if shown >= max_passages_per_card:
                break
            apid = a.get("passage_id", "")
            text = passage_text_by_id.get(apid, "")
            if not text:
                continue
            lines.append(f"--- {apid} ---")
            lines.append(text[:passage_max_chars])
            lines.append("")
            shown += 1
        return lines

    parts: list[str] = []
    parts.extend(card_section(card_a, "PAPER A"))
    parts.append("")
    parts.extend(card_section(card_b, "PAPER B"))
    parts.append("")
    parts.append(
        f"Generate exactly one synthesis question. Evidence MUST include at least "
        f"1 passage from PAPER A ({card_a['paper_id']}) and at least 1 from "
        f"PAPER B ({card_b['paper_id']}). JSON only."
    )
    return "\n".join(parts)


# ---- Validation ----------------------------------------------------------------

def _validate_draft(
    draft: _DraftSynthesisOutput,
    card_a: dict,
    card_b: dict,
    passage_text_by_id: dict[str, str],
    valid_ids_a: set[str],
    valid_ids_b: set[str],
    pair_id: str,
    benchmark_version: str,
    generator_model: str,
    seed: int,
) -> tuple[Optional[Question], list[str]]:
    warnings: list[str] = []

    if not draft.question.strip() or not draft.reference_answer.strip():
        return None, ["empty question or reference_answer; dropped"]

    paper_a_id = card_a["paper_id"]
    paper_b_id = card_b["paper_id"]
    all_valid = valid_ids_a | valid_ids_b

    good_evidence: list[Evidence] = []
    has_paper_a = False
    has_paper_b = False

    for ev in draft.evidence:
        pid = ev.passage_id
        if pid not in all_valid:
            warnings.append(f"dropped bogus passage_id {pid!r}")
            continue
        # Infer paper_id from passage_id (paper_NNNN_chunk_NN → paper_NNNN)
        inferred_paper = pid.rsplit("_chunk_", 1)[0] if "_chunk_" in pid else None
        if inferred_paper in (paper_a_id, paper_b_id):
            paper_of_ev = inferred_paper
        elif ev.paper_id in (paper_a_id, paper_b_id):
            paper_of_ev = ev.paper_id
        else:
            paper_of_ev = inferred_paper or paper_a_id
            warnings.append(
                f"evidence paper_id {ev.paper_id!r} invalid; inferred {paper_of_ev!r} from passage_id"
            )
        good_evidence.append(
            Evidence(paper_id=paper_of_ev, passage_id=pid, quote=ev.quote.strip()[:480])
        )
        if paper_of_ev == paper_a_id:
            has_paper_a = True
        else:
            has_paper_b = True

    # Backfill missing coverage from card anchors
    if not has_paper_a and valid_ids_a:
        fallback = sorted(valid_ids_a)[0]
        good_evidence.append(Evidence(
            paper_id=paper_a_id, passage_id=fallback,
            quote=passage_text_by_id.get(fallback, "")[:240],
        ))
        warnings.append("backfilled evidence for PAPER A from anchor[0]")
        has_paper_a = True
    if not has_paper_b and valid_ids_b:
        fallback = sorted(valid_ids_b)[0]
        good_evidence.append(Evidence(
            paper_id=paper_b_id, passage_id=fallback,
            quote=passage_text_by_id.get(fallback, "")[:240],
        ))
        warnings.append("backfilled evidence for PAPER B from anchor[0]")
        has_paper_b = True

    if not good_evidence or not has_paper_a or not has_paper_b:
        return None, warnings + ["insufficient dual-paper evidence; dropped"]

    category = draft.category if draft.category in CATEGORIES else "Practical HPN Scenarios and Design"
    if category != draft.category:
        warnings.append(f"category {draft.category!r} remapped to default")

    q = Question(
        id=pair_id,
        benchmark_version=benchmark_version,
        category=category,
        difficulty=draft.difficulty,
        question_type="comparison",
        question=draft.question.strip(),
        reference_answer=draft.reference_answer.strip(),
        source_papers=[paper_a_id, paper_b_id],
        evidence=good_evidence,
        keywords=[k.strip() for k in draft.keywords if k and k.strip()][:7],
        requires_calculation=False,
        synthetic_scenario=False,
        inspired_by_papers=[paper_a_id, paper_b_id],
        generator_model=generator_model,
        critic_model="",
        prompt_version=PROMPT_VERSION,
        seed=seed,
    )
    return q, warnings


# ---- Process one pair ----------------------------------------------------------

def process_pair(
    client: OllamaClient,
    card_a: dict,
    card_b: dict,
    passage_text_by_id: dict[str, str],
    valid_ids_a: set[str],
    valid_ids_b: set[str],
    system_prompt: str,
    schema: dict,
    model: str,
    temperature: float,
    num_predict: int,
    benchmark_version: str,
    seed: int,
    pair_id: str,
) -> tuple[Optional[dict], list[str], float]:
    user_prompt = build_user_prompt(card_a, card_b, passage_text_by_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.time()
    try:
        out = client.chat_json(
            model=model,
            messages=messages,
            temperature=temperature,
            max_retries=1,
            num_predict=num_predict,
            schema=schema,
            repeat_penalty=1.15,
            think=False,
        )
    except OllamaError as e:
        return None, [f"ollama: {e}"], time.time() - t0

    if isinstance(out, list):
        if len(out) == 1 and isinstance(out[0], dict):
            out = out[0]
        else:
            return None, [f"output not a dict (got list of {len(out)})"], time.time() - t0
    if not isinstance(out, dict):
        return None, [f"output not a dict: {type(out).__name__}"], time.time() - t0

    # Fix hallucinated evidence field names (same pattern as stage_05a)
    for ev in out.get("evidence", []):
        if isinstance(ev, dict) and "passage_id" not in ev:
            for key in list(ev.keys()):
                if "passage" in key.lower() and key != "paper_id":
                    ev["passage_id"] = ev.pop(key)
                    break

    try:
        draft = _DraftSynthesisOutput(**out)
    except Exception as e:
        return None, [f"draft schema: {type(e).__name__}: {e}"], time.time() - t0

    question, warnings = _validate_draft(
        draft=draft,
        card_a=card_a,
        card_b=card_b,
        passage_text_by_id=passage_text_by_id,
        valid_ids_a=valid_ids_a,
        valid_ids_b=valid_ids_b,
        pair_id=pair_id,
        benchmark_version=benchmark_version,
        generator_model=model,
        seed=seed,
    )

    elapsed = time.time() - t0
    if question is None:
        return None, warnings, elapsed
    return question.model_dump(), warnings, elapsed


# ---- CLI -----------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--cards", default=None)
    parser.add_argument("--passages", default=None)
    parser.add_argument("--embeddings", default=None, help="card_embeddings.npy (default from cache)")
    parser.add_argument("--emb-ids", default=None, help="card_embedding_ids.json (default from cache)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    data_dir = cfg.resolve(cfg.paths.data_dir)
    cache_dir = data_dir / "_cache"
    prompts_dir = cfg.resolve(cfg.paths.prompts_dir)
    logs_dir = cfg.resolve(cfg.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cards_path = Path(args.cards) if args.cards else data_dir / "paper_cards_clustered.jsonl"
    passages_path = Path(args.passages) if args.passages else data_dir / "passages.jsonl"
    emb_path = Path(args.embeddings) if args.embeddings else cache_dir / "card_embeddings.npy"
    emb_ids_path = Path(args.emb_ids) if args.emb_ids else cache_dir / "card_embedding_ids.json"
    output_path = Path(args.output) if args.output else data_dir / "synthesis_candidates.jsonl"
    prompt_path = Path(args.prompt) if args.prompt else prompts_dir / "synthesis_generation.txt"
    log_path = logs_dir / "stage_05b.jsonl"
    model = args.model or cfg.models.question_generator

    for label, p in [
        ("cards", cards_path), ("passages", passages_path),
        ("embeddings", emb_path), ("emb-ids", emb_ids_path),
        ("prompt", prompt_path),
    ]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return 1

    system_prompt = prompt_path.read_text(encoding="utf-8")
    cards = list(read_jsonl(cards_path))
    if not cards:
        logger.error("no cards loaded")
        return 1
    logger.info("loaded %d cards from %s", len(cards), cards_path)

    emb = np.load(emb_path)
    emb_ids: list[str] = json.loads(emb_ids_path.read_text(encoding="utf-8"))
    logger.info("loaded embeddings: shape=%s, %d IDs", emb.shape, len(emb_ids))

    # Passage lookup
    passage_text_by_id: dict[str, str] = {}
    valid_ids_by_paper: dict[str, set[str]] = {}
    for p in read_jsonl(passages_path):
        passage_text_by_id[p["passage_id"]] = p["text"]
        valid_ids_by_paper.setdefault(p["paper_id"], set()).add(p["passage_id"])
    logger.info("loaded passages for %d papers", len(valid_ids_by_paper))

    # Generate all pairs deterministically
    all_pairs = build_pairs(
        cards=cards,
        emb=emb,
        emb_ids=emb_ids,
        pairs_per_cluster=cfg.generation.synthesis_pairs_per_cluster,
        synthesis_target=cfg.generation.synthesis_target,
        seed=cfg.seed,
    )
    n_cluster_pairs = sum(1 for _, _, s in all_pairs if s.startswith("cluster"))
    n_noise_pairs = sum(1 for _, _, s in all_pairs if s == "noise_sim")
    logger.info(
        "total pairs: %d (cluster: %d, noise_sim: %d)",
        len(all_pairs), n_cluster_pairs, n_noise_pairs,
    )

    # Resume: skip pairs whose ID already exists in output
    if args.resume:
        existing_ids: set[str] = set()
        if output_path.exists():
            for r in read_jsonl(output_path):
                existing_ids.add(r.get("id", ""))
        skip = existing_ids
        logger.info("resume: %d pairs already done", len(skip))
    else:
        skip = set()
        if output_path.exists():
            logger.info("clearing existing output: %s", output_path)
            output_path.unlink()

    todo_pairs = [
        (a, b, src) for a, b, src in all_pairs
        if _pair_id(a["paper_id"], b["paper_id"]) not in skip
    ]
    if args.limit:
        todo_pairs = todo_pairs[: args.limit]

    if not todo_pairs:
        logger.info("nothing to do")
        return 0

    schema = _DraftSynthesisOutput.model_json_schema()
    client = OllamaClient(host=cfg.models.ollama_host, timeout=cfg.ollama.request_timeout_seconds)
    seed = cfg.seed
    bv = cfg.splits.benchmark_version

    logger.info("model=%s, workers=%d, todo=%d pairs", model, args.workers, len(todo_pairs))

    n_ok = n_fail = 0
    times: list[float] = []

    temperature = args.temperature
    num_predict = args.num_predict

    def task(pair: tuple[dict, dict, str]) -> tuple[str, str, str, str, Optional[dict], list[str], float]:
        card_a, card_b, source_label = pair
        pa_id = card_a["paper_id"]
        pb_id = card_b["paper_id"]
        pid = _pair_id(pa_id, pb_id)
        valid_a = valid_ids_by_paper.get(pa_id, set())
        valid_b = valid_ids_by_paper.get(pb_id, set())
        record, warnings, elapsed = process_pair(
            client=client,
            card_a=card_a,
            card_b=card_b,
            passage_text_by_id=passage_text_by_id,
            valid_ids_a=valid_a,
            valid_ids_b=valid_b,
            system_prompt=system_prompt,
            schema=schema,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            benchmark_version=bv,
            seed=seed,
            pair_id=pid,
        )
        return pid, pa_id, pb_id, source_label, record, warnings, elapsed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, p): p for p in todo_pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="pairs", unit="pair"):
            pid, pa_id, pb_id, source_label, record, warnings, elapsed = fut.result()
            times.append(elapsed)
            ok = record is not None
            log_rec = {
                "pair_id": pid,
                "paper_a": pa_id,
                "paper_b": pb_id,
                "source": source_label,
                "model": model,
                "ok": ok,
                "warnings": warnings,
                "latency_ms": int(elapsed * 1000),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_jsonl(log_path, log_rec)
            if record is not None:
                append_jsonl(output_path, record)
                n_ok += 1
            else:
                n_fail += 1
                logger.warning("FAIL %s: %s", pid, warnings)

    avg = sum(times) / max(1, len(times))
    pass_rate = n_ok / max(1, n_ok + n_fail)
    logger.info(
        "done: %d ok, %d fail (pass_rate=%.1f%%), avg %.1f s/pair",
        n_ok, n_fail, pass_rate * 100, avg,
    )
    return 0 if pass_rate >= 0.5 else 3


if __name__ == "__main__":
    sys.exit(main())
