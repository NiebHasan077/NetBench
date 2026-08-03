"""Stage 05c — adversarial / misconception question generation.

For each HPN misconception in prompts/misconceptions.yaml, find the top-K
most relevant paper cards (by embedding cosine similarity to the misconception
statement), then generate 1 adversarial benchmark question per
(misconception, card) pair.

The question is phrased so a naive reader would assert the misconception;
the reference_answer explicitly corrects it using evidence from the card.

Inputs:
  prompts/misconceptions.yaml
  prompts/adversarial_generation.txt
  data/paper_cards_clustered.jsonl
  data/passages.jsonl
  data/_cache/card_embeddings.npy       (from Stage 04, reused)
  data/_cache/card_embedding_ids.json

Outputs:
  data/adversarial_candidates.jsonl    one record per accepted question
  data/_logs/stage_05c.jsonl           per-item run log

Usage:
    .venv/bin/python -m src.stage_05c_adversarial --limit 5     # pilot
    .venv/bin/python -m src.stage_05c_adversarial               # full run
    .venv/bin/python -m src.stage_05c_adversarial --resume      # skip already-done
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config import load_config
from src.embeddings import Embedder, detect_device
from src.io_utils import append_jsonl, read_jsonl
from src.ollama_client import OllamaClient, OllamaError
from src.schema import DifficultyT, Evidence, Question, QuestionTypeT

logger = logging.getLogger("stage_05c")
PROMPT_VERSION = "v5.0-adversarial"

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

# Valid question types for adversarial questions — all adversarial questions are "scenario" type
_ADV_QUESTION_TYPE: QuestionTypeT = "scenario"


# ---- Output schema (sent to Ollama as `format`) --------------------------------

class _DraftEvidence(BaseModel):
    passage_id: str
    excerpt: str  # renamed from 'quote' — gemma4 emits "quote: "text"" (colon inside key) causing parse failures


class _DraftAdversarialOutput(BaseModel):
    """Flat schema — no Literal enum fields and no default-valued fields to avoid gemma4 parse failures."""
    question: str
    reference_answer: str
    evidence: list[_DraftEvidence] = Field(default_factory=list)
    difficulty: DifficultyT
    category: str
    keywords: list[str] = Field(default_factory=list)


# ---- Misconception + card utilities --------------------------------------------

def load_misconceptions(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data.get("misconceptions", [])
    if not items:
        raise ValueError(f"no misconceptions found in {path}")
    return items


def _item_id(misconception_id: str, paper_id: str) -> str:
    paper_num = paper_id.replace("paper_", "")
    return f"NB-ADV-{misconception_id}-{paper_num}"


def find_pairs(
    misconceptions: list[dict],
    cards: list[dict],
    card_emb: np.ndarray,
    emb_ids: list[str],
    top_k: int,
    device: str,
    embedding_model: str,
) -> list[tuple[dict, dict]]:
    """Embed each misconception statement and return top-k (misconception, card) pairs.

    Cosine similarity = dot product because card embeddings are L2-normalised.
    """
    id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}

    # Embed misconception statements using the same model as Stage 04
    misconception_texts = [m["statement"] for m in misconceptions]
    embedder = Embedder(embedding_model, device=device, batch_size=32)
    misc_emb = embedder.embed(misconception_texts, normalize=True, show_progress=False)
    # misc_emb: (n_misconceptions, D)

    # Only consider cards whose embeddings exist in the cache
    valid_cards = [c for c in cards if c["paper_id"] in id_to_idx]
    valid_emb_idx = [id_to_idx[c["paper_id"]] for c in valid_cards]
    ce = card_emb[valid_emb_idx]  # (n_cards, D)

    # cosine sim: (n_misconceptions, n_cards)
    sim = misc_emb @ ce.T

    pairs: list[tuple[dict, dict]] = []
    seen: set[str] = set()  # dedup (misconception_id, paper_id)

    for m_idx, misc in enumerate(misconceptions):
        sims_for_m = sim[m_idx]
        top_indices = np.argsort(-sims_for_m)[:top_k]
        for c_idx in top_indices:
            card = valid_cards[int(c_idx)]
            key = (misc["id"], card["paper_id"])
            if key not in seen:
                seen.add(key)
                pairs.append((misc, card))

    return pairs


# ---- Prompt building -----------------------------------------------------------

def build_user_prompt(
    misconception: dict,
    card: dict,
    passage_text_by_id: dict[str, str],
    max_passages: int = 4,
    passage_max_chars: int = 1000,
) -> str:
    anchors = card.get("evidence_anchors", []) or []
    paper_id = card["paper_id"]

    lines = [
        "TARGET MISCONCEPTION",
        f"ID: {misconception['id']}",
        f"Statement: \"{misconception['statement']}\"",
        f"Why it's wrong: \"{misconception['why_wrong']}\"",
        "",
        f"PAPER CARD: {paper_id}",
        f"main_problem: {card.get('main_problem', '')}",
        f"key_concepts: {card.get('key_concepts', [])}",
        f"failure_modes: {card.get('failure_modes', [])}",
        f"important_numbers: {card.get('important_numbers', [])}",
        "",
        "ALLOWED PASSAGE IDs (the only IDs you may use):",
    ]
    for a in anchors:
        apid = a.get("passage_id", "")
        claim = a.get("claim", "")
        lines.append(f"  - {apid}  (claim: {claim})")
    lines.append("")
    lines.append("PASSAGE TEXTS:")
    lines.append("")
    shown = 0
    for a in anchors:
        if shown >= max_passages:
            break
        apid = a.get("passage_id", "")
        text = passage_text_by_id.get(apid, "")
        if not text:
            continue
        lines.append(f"--- {apid} ---")
        lines.append(text[:passage_max_chars])
        lines.append("")
        shown += 1

    lines.append(
        "Generate exactly one adversarial question targeting the above misconception. "
        "The question must not reveal that it is testing a misconception. JSON only."
    )
    return "\n".join(lines)


# ---- Validation ----------------------------------------------------------------

def _validate_draft(
    draft: _DraftAdversarialOutput,
    misconception: dict,
    card: dict,
    passage_text_by_id: dict[str, str],
    valid_passage_ids: set[str],
    item_id: str,
    benchmark_version: str,
    generator_model: str,
    seed: int,
) -> tuple[Optional[Question], list[str]]:
    warnings: list[str] = []

    if not draft.question.strip() or not draft.reference_answer.strip():
        return None, ["empty question or reference_answer; dropped"]

    paper_id = card["paper_id"]

    good_evidence: list[Evidence] = []
    for ev in draft.evidence:
        pid = ev.passage_id
        if pid not in valid_passage_ids:
            warnings.append(f"dropped bogus passage_id {pid!r}")
            continue
        good_evidence.append(
            Evidence(paper_id=paper_id, passage_id=pid, quote=ev.excerpt.strip()[:480])
        )

    if not good_evidence:
        if valid_passage_ids:
            fallback = sorted(valid_passage_ids)[0]
            good_evidence = [Evidence(
                paper_id=paper_id, passage_id=fallback,
                quote=passage_text_by_id.get(fallback, "")[:240],
            )]
            warnings.append("no valid evidence; backfilled from anchor[0]")
        else:
            return None, warnings + ["no valid passage IDs; dropped"]

    category = draft.category if draft.category in CATEGORIES else "Practical HPN Scenarios and Design"
    if category != draft.category:
        warnings.append(f"category {draft.category!r} remapped")

    q = Question(
        id=item_id,
        benchmark_version=benchmark_version,
        category=category,
        difficulty=draft.difficulty,
        question_type=_ADV_QUESTION_TYPE,
        question=draft.question.strip(),
        reference_answer=draft.reference_answer.strip(),
        source_papers=[paper_id],
        evidence=good_evidence,
        keywords=[k.strip() for k in draft.keywords if k and k.strip()][:7],
        requires_calculation=False,
        synthetic_scenario=True,   # scenario is (partially) constructed to elicit misconception
        inspired_by_papers=[paper_id],
        generator_model=generator_model,
        critic_model="",
        prompt_version=PROMPT_VERSION,
        seed=seed,
    )
    return q, warnings


# ---- Process one item ----------------------------------------------------------

def process_item(
    client: OllamaClient,
    misconception: dict,
    card: dict,
    passage_text_by_id: dict[str, str],
    valid_passage_ids: set[str],
    system_prompt: str,
    schema: dict,
    model: str,
    temperature: float,
    num_predict: int,
    benchmark_version: str,
    seed: int,
    item_id: str,
) -> tuple[Optional[dict], list[str], float]:
    user_prompt = build_user_prompt(misconception, card, passage_text_by_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.time()
    last_err: Optional[str] = None
    out: Optional[dict] = None
    for attempt in range(2):
        t = temperature if attempt == 0 else 0.2
        try:
            raw = client.chat(
                model=model,
                messages=messages,
                temperature=t,
                num_predict=num_predict,
                format=schema,
                repeat_penalty=1.15,
                think=False,
            )
        except OllamaError as e:
            return None, [f"ollama: {e}"], time.time() - t0

        # Fix gemma4 colon-inside-key-name bug:
        # model writes '"excerpt: "text"' instead of '"excerpt": "text"'
        # Safe to fix because these words don't naturally appear as '"word: "' in passage text.
        for field in ("excerpt", "quote"):
            raw = raw.replace(f'"{field}: "', f'"{field}": "')

        try:
            out = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_err = str(e)
            if attempt == 0:
                time.sleep(0.5)
                continue
    if out is None:
        return None, [f"ollama: could not parse chat JSON after 2 attempt(s): {last_err}"], time.time() - t0

    if isinstance(out, list):
        if len(out) == 1 and isinstance(out[0], dict):
            out = out[0]
        else:
            return None, [f"output not a dict (got list of {len(out)})"], time.time() - t0
    if not isinstance(out, dict):
        return None, [f"output not a dict: {type(out).__name__}"], time.time() - t0

    # Fix hallucinated evidence field names (same pattern as 05a/05b)
    for ev in out.get("evidence", []):
        if isinstance(ev, dict) and "passage_id" not in ev:
            for key in list(ev.keys()):
                if "passage" in key.lower():
                    ev["passage_id"] = ev.pop(key)
                    break
        # Normalize: model may emit "quote" instead of "excerpt"
        if isinstance(ev, dict) and "excerpt" not in ev and "quote" in ev:
            ev["excerpt"] = ev.pop("quote")

    try:
        draft = _DraftAdversarialOutput(**out)
    except Exception as e:
        return None, [f"draft schema: {type(e).__name__}: {e}"], time.time() - t0

    question, warnings = _validate_draft(
        draft=draft,
        misconception=misconception,
        card=card,
        passage_text_by_id=passage_text_by_id,
        valid_passage_ids=valid_passage_ids,
        item_id=item_id,
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
    parser.add_argument("--misconceptions", default=None, help="misconceptions.yaml path")
    parser.add_argument("--cards", default=None)
    parser.add_argument("--passages", default=None)
    parser.add_argument("--embeddings", default=None, help="card_embeddings.npy")
    parser.add_argument("--emb-ids", default=None, help="card_embedding_ids.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None, help="embedding device (default: cuda:0)")
    parser.add_argument("--top-k-cards", type=int, default=0,
                        help="cards per misconception (default: ceil(adversarial_target/n_misconceptions))")
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

    misc_path = Path(args.misconceptions) if args.misconceptions else prompts_dir / "misconceptions.yaml"
    cards_path = Path(args.cards) if args.cards else data_dir / "paper_cards_clustered.jsonl"
    passages_path = Path(args.passages) if args.passages else data_dir / "passages.jsonl"
    emb_path = Path(args.embeddings) if args.embeddings else cache_dir / "card_embeddings.npy"
    emb_ids_path = Path(args.emb_ids) if args.emb_ids else cache_dir / "card_embedding_ids.json"
    output_path = Path(args.output) if args.output else data_dir / "adversarial_candidates.jsonl"
    prompt_path = Path(args.prompt) if args.prompt else prompts_dir / "adversarial_generation.txt"
    log_path = logs_dir / "stage_05c.jsonl"
    model = args.model or cfg.models.question_generator
    device = detect_device(args.device)

    for label, p in [
        ("misconceptions", misc_path), ("cards", cards_path),
        ("passages", passages_path), ("embeddings", emb_path),
        ("emb-ids", emb_ids_path), ("prompt", prompt_path),
    ]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return 1

    system_prompt = prompt_path.read_text(encoding="utf-8")
    misconceptions = load_misconceptions(misc_path)
    logger.info("loaded %d misconceptions", len(misconceptions))

    cards = list(read_jsonl(cards_path))
    if not cards:
        logger.error("no cards loaded")
        return 1
    logger.info("loaded %d cards", len(cards))

    card_emb = np.load(emb_path)
    emb_ids: list[str] = json.loads(emb_ids_path.read_text(encoding="utf-8"))
    logger.info("loaded card embeddings: shape=%s", card_emb.shape)

    passage_text_by_id: dict[str, str] = {}
    valid_ids_by_paper: dict[str, set[str]] = {}
    for p in read_jsonl(passages_path):
        passage_text_by_id[p["passage_id"]] = p["text"]
        valid_ids_by_paper.setdefault(p["paper_id"], set()).add(p["passage_id"])
    logger.info("loaded passages for %d papers", len(valid_ids_by_paper))

    # Determine cards per misconception
    top_k = args.top_k_cards or math.ceil(cfg.generation.adversarial_target / len(misconceptions))
    top_k = max(1, top_k)
    logger.info("top_k_cards=%d per misconception, device=%s", top_k, device)

    # Build pairs via embedding similarity
    all_pairs = find_pairs(
        misconceptions=misconceptions,
        cards=cards,
        card_emb=card_emb,
        emb_ids=emb_ids,
        top_k=top_k,
        device=device,
        embedding_model=cfg.models.embedding,
    )
    logger.info("total (misconception, card) pairs: %d", len(all_pairs))

    # Resume: skip already-generated IDs
    if args.resume:
        existing_ids: set[str] = set()
        if output_path.exists():
            for r in read_jsonl(output_path):
                existing_ids.add(r.get("id", ""))
        skip = existing_ids
        logger.info("resume: %d items already done", len(skip))
    else:
        skip = set()
        if output_path.exists():
            logger.info("clearing existing output: %s", output_path)
            output_path.unlink()

    todo_pairs = [
        (m, c) for m, c in all_pairs
        if _item_id(m["id"], c["paper_id"]) not in skip
    ]
    if args.limit:
        todo_pairs = todo_pairs[: args.limit]

    if not todo_pairs:
        logger.info("nothing to do")
        return 0

    schema = _DraftAdversarialOutput.model_json_schema()
    client = OllamaClient(host=cfg.models.ollama_host, timeout=cfg.ollama.request_timeout_seconds)
    seed = cfg.seed
    bv = cfg.splits.benchmark_version

    logger.info("model=%s, workers=%d, todo=%d items", model, args.workers, len(todo_pairs))

    n_ok = n_fail = 0
    times: list[float] = []

    temperature = args.temperature
    num_predict = args.num_predict

    def task(pair: tuple[dict, dict]) -> tuple[str, str, str, Optional[dict], list[str], float]:
        misc, card = pair
        iid = _item_id(misc["id"], card["paper_id"])
        valid_ids = valid_ids_by_paper.get(card["paper_id"], set())
        record, warnings, elapsed = process_item(
            client=client,
            misconception=misc,
            card=card,
            passage_text_by_id=passage_text_by_id,
            valid_passage_ids=valid_ids,
            system_prompt=system_prompt,
            schema=schema,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            benchmark_version=bv,
            seed=seed,
            item_id=iid,
        )
        return iid, misc["id"], card["paper_id"], record, warnings, elapsed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, p): p for p in todo_pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="items", unit="item"):
            iid, misc_id, paper_id, record, warnings, elapsed = fut.result()
            times.append(elapsed)
            ok = record is not None
            log_rec = {
                "item_id": iid,
                "misconception_id": misc_id,
                "paper_id": paper_id,
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
                logger.warning("FAIL %s: %s", iid, warnings)

    avg = sum(times) / max(1, len(times))
    pass_rate = n_ok / max(1, n_ok + n_fail)
    logger.info(
        "done: %d ok, %d fail (pass_rate=%.1f%%), avg %.1f s/item",
        n_ok, n_fail, pass_rate * 100, avg,
    )
    return 0 if pass_rate >= 0.5 else 3


if __name__ == "__main__":
    sys.exit(main())
