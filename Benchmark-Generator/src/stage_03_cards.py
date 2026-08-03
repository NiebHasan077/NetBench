"""Stage 03 — paper cards.

For each surviving paper from Stage 02, build a structured paper card via the
configured Ollama generator (default: qwen3.5:9b via /api/chat, think=False)
using the first N passages as context. Each card carries `evidence_anchors[]`
whose `passage_id` values are validated to actually exist in this paper's
passages.

Inputs:
  data/filtered_papers.jsonl   (Stage 02 output)
  data/passages.jsonl          (Stage 01 output)
  prompts/card_generation.txt

Outputs:
  data/paper_cards.jsonl       one card per surviving paper
  data/_logs/stage_03.jsonl    per-paper log line (status, latency, warnings)

Usage:
    .venv/bin/python -m src.stage_03_cards --limit 5     # pilot
    .venv/bin/python -m src.stage_03_cards               # full run on filtered
    .venv/bin/python -m src.stage_03_cards --resume      # skip already-done
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config import load_config
from src.io_utils import append_jsonl, existing_ids, read_jsonl
from src.ollama_client import OllamaClient, OllamaError
from src.schema import PaperCard

logger = logging.getLogger("stage_03")
PROMPT_VERSION = "v5.0-cards"


def build_user_prompt(paper_id: str, passages: list[dict], n_passages: int) -> str:
    selected = passages[:n_passages]
    lines = [f"PAPER_ID: {paper_id}", ""]
    lines.append(f"PASSAGES (use these passage_ids verbatim in evidence_anchors[]):")
    lines.append("")
    for p in selected:
        lines.append(f"--- passage {p['passage_id']} ---")
        lines.append(p["text"])
        lines.append("")
    lines.append(f"Now produce the JSON card for paper_id={paper_id}.")
    return "\n".join(lines)


def validate_card(
    raw: dict,
    paper_id: str,
    valid_passage_ids: set[str],
) -> tuple[Optional[PaperCard], list[str]]:
    """Validate schema + passage-id grounding. Returns (card_or_None, warnings)."""
    warnings: list[str] = []

    # Always force paper_id to the truth (LLM sometimes drops or restyles it).
    raw_paper_id = raw.get("paper_id")
    if raw_paper_id != paper_id:
        if raw_paper_id:
            warnings.append(f"paper_id rewritten ({raw_paper_id} -> {paper_id})")
        raw["paper_id"] = paper_id

    try:
        card = PaperCard(**raw)
    except Exception as e:
        return None, [f"schema: {type(e).__name__}: {e}"]

    if "not hpn" in card.main_problem.lower() or "not hpn-relevant" in card.main_problem.lower():
        return None, ["self-reported non-HPN"]

    if len(card.main_problem.strip()) < 15:
        return None, [f"main_problem too short ({len(card.main_problem)} chars)"]

    bad = [a.passage_id for a in card.evidence_anchors if a.passage_id not in valid_passage_ids]
    if bad:
        warnings.append(f"dropped {len(bad)} bogus passage_id(s): {bad[:3]}")
        card.evidence_anchors = [a for a in card.evidence_anchors if a.passage_id in valid_passage_ids]

    # Deduplicate by passage_id, keeping the first anchor per unique passage.
    seen: set[str] = set()
    deduped = []
    for a in card.evidence_anchors:
        if a.passage_id not in seen:
            seen.add(a.passage_id)
            deduped.append(a)
    if len(deduped) < len(card.evidence_anchors):
        warnings.append(f"deduped evidence_anchors: {len(card.evidence_anchors)} -> {len(deduped)}")
    card.evidence_anchors = deduped

    if not card.evidence_anchors:
        return None, warnings + ["no valid evidence_anchors after filtering"]

    if len(card.key_concepts) == 0:
        warnings.append("empty key_concepts")

    return card, warnings


def process_paper(
    client: OllamaClient,
    paper_id: str,
    passages: list[dict],
    system_prompt: str,
    model: str,
    n_passages: int,
    temperature: float,
    num_predict: int,
    schema: dict,
) -> tuple[Optional[dict], list[str], float]:
    # Accept any real passage_id of this paper as a valid evidence anchor —
    # not only the passages we sent. The model may extrapolate to plausible
    # neighbouring chunk IDs; we keep them iff they actually exist.
    valid_ids = {p["passage_id"] for p in passages}
    user_prompt = build_user_prompt(paper_id, passages, n_passages)
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

    # Some models occasionally wrap the object in a singleton list under
    # schema enforcement — unwrap it.
    warnings_pre: list[str] = []
    if isinstance(out, list):
        if len(out) == 1 and isinstance(out[0], dict):
            out = out[0]
            warnings_pre.append("unwrapped singleton-list output")
        else:
            return None, [f"output not a dict (got list of {len(out)})"], time.time() - t0
    if not isinstance(out, dict):
        return None, [f"output not a dict: {type(out).__name__}"], time.time() - t0

    card, warnings = validate_card(out, paper_id, valid_ids)
    warnings = warnings_pre + warnings
    if card is None:
        return None, warnings, time.time() - t0

    rec = card.model_dump()
    if warnings:
        rec["_validation_warnings"] = warnings
    return rec, warnings, time.time() - t0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--filtered", default=None, help="filtered_papers.jsonl (default from config)")
    parser.add_argument("--passages", default=None, help="passages.jsonl (default from config)")
    parser.add_argument("--output", default=None, help="paper_cards.jsonl (default from config)")
    parser.add_argument("--prompt", default=None, help="prompt template path")
    parser.add_argument("--model", default=None, help="ollama model (default from config.models.card_generator)")
    parser.add_argument("--n-passages", type=int, default=6, help="passages sent to LLM per paper")
    parser.add_argument("--workers", type=int, default=2, help="concurrent ollama calls (set OLLAMA_NUM_PARALLEL accordingly)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=4096, help="max output tokens (must be enough for a full card; truncated JSON will fail to parse)")
    parser.add_argument("--limit", type=int, default=0, help="process only first N papers (after resume filter)")
    parser.add_argument("--resume", action="store_true", help="skip papers already in output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    data_dir = cfg.resolve(cfg.paths.data_dir)
    prompts_dir = cfg.resolve(cfg.paths.prompts_dir)
    logs_dir = cfg.resolve(cfg.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = Path(args.filtered) if args.filtered else data_dir / "filtered_papers.jsonl"
    passages_path = Path(args.passages) if args.passages else data_dir / "passages.jsonl"
    output_path = Path(args.output) if args.output else data_dir / "paper_cards.jsonl"
    prompt_path = Path(args.prompt) if args.prompt else prompts_dir / "card_generation.txt"
    log_path = logs_dir / "stage_03.jsonl"
    model = args.model or cfg.models.card_generator

    for label, p in [("filtered", filtered_path), ("passages", passages_path), ("prompt", prompt_path)]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return 1

    system_prompt = prompt_path.read_text(encoding="utf-8")

    filtered = list(read_jsonl(filtered_path))
    target_paper_ids = [p["paper_id"] for p in filtered]
    logger.info("loaded %d filtered paper IDs", len(target_paper_ids))

    passages_by_paper: dict[str, list[dict]] = {}
    for p in read_jsonl(passages_path):
        passages_by_paper.setdefault(p["paper_id"], []).append(p)
    for plist in passages_by_paper.values():
        plist.sort(key=lambda x: x["passage_id"])

    skip = existing_ids(output_path, key="paper_id") if args.resume else set()
    if not args.resume and output_path.exists():
        logger.info("clearing existing output: %s", output_path)
        output_path.unlink()
    if args.resume and skip:
        logger.info("resume: %d cards already in output", len(skip))

    todo = [pid for pid in target_paper_ids if pid not in skip]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        logger.info("nothing to do")
        return 0

    client = OllamaClient(host=cfg.models.ollama_host, timeout=cfg.ollama.request_timeout_seconds)
    schema = PaperCard.model_json_schema()
    logger.info(
        "model=%s, workers=%d, n_passages=%d, schema-enforced JSON, todo=%d papers",
        model, args.workers, args.n_passages, len(todo),
    )

    n_ok = 0
    n_fail = 0
    times: list[float] = []

    def task(paper_id: str):
        plist = passages_by_paper.get(paper_id, [])
        if not plist:
            return paper_id, None, ["no passages"], 0.0
        rec, warnings, elapsed = process_paper(
            client=client,
            paper_id=paper_id,
            passages=plist,
            system_prompt=system_prompt,
            model=model,
            n_passages=args.n_passages,
            temperature=args.temperature,
            num_predict=args.num_predict,
            schema=schema,
        )
        return paper_id, rec, warnings, elapsed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, pid): pid for pid in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="cards", unit="paper"):
            paper_id, rec, warnings, elapsed = fut.result()
            times.append(elapsed)
            log_rec = {
                "paper_id": paper_id,
                "model": model,
                "ok": rec is not None,
                "warnings": warnings,
                "latency_ms": int(elapsed * 1000),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_jsonl(log_path, log_rec)
            if rec is not None:
                rec["generator_model"] = model
                rec["prompt_version"] = PROMPT_VERSION
                append_jsonl(output_path, rec)
                n_ok += 1
            else:
                n_fail += 1
                logger.warning("FAIL %s: %s", paper_id, warnings)

    avg = sum(times) / max(1, len(times))
    pass_rate = n_ok / max(1, n_ok + n_fail)
    logger.info(
        "done: %d ok, %d fail (pass_rate=%.1f%%), avg %.1f s/paper",
        n_ok, n_fail, pass_rate * 100, avg,
    )

    return 0 if pass_rate >= 0.5 else 3


if __name__ == "__main__":
    sys.exit(main())
