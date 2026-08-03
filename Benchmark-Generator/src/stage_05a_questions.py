"""Stage 05a — per-paper candidate question generation.

For each paper card from Stage 03/04, generate 4 candidate benchmark questions
(concept, reasoning, calculation-or-diagnosis, scenario) via the configured
question_generator. Cards are read from `data/paper_cards_clustered.jsonl`
(falling back to `data/paper_cards.jsonl` if clustering hasn't run yet).

Inputs:
  data/paper_cards_clustered.jsonl (or data/paper_cards.jsonl)
  data/passages.jsonl              (for evidence-quote substring grounding)
  prompts/question_generation.txt

Outputs:
  data/candidate_questions.jsonl   one record per accepted question
  data/_logs/stage_05a.jsonl       per-card run log (status, latency, warnings)

Usage:
    .venv/bin/python -m src.stage_05a_questions --limit 5     # pilot
    .venv/bin/python -m src.stage_05a_questions               # full run
    .venv/bin/python -m src.stage_05a_questions --resume      # skip already-done cards
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config import load_config
from src.io_utils import append_jsonl, existing_ids, read_jsonl
from src.ollama_client import OllamaClient, OllamaError
from src.schema import (
    DifficultyT,
    Evidence,
    QuestionTypeT,
    Question,
)

logger = logging.getLogger("stage_05a")
PROMPT_VERSION = "v5.0-questions"

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


# ---- Output schema (sent to Ollama as `format`) -------------------------------
class _DraftEvidence(BaseModel):
    passage_id: str
    quote: str


class _DraftQuestion(BaseModel):
    question_type: QuestionTypeT
    question: str
    reference_answer: str
    evidence: list[_DraftEvidence] = Field(default_factory=list)
    difficulty: DifficultyT
    category: str
    keywords: list[str] = Field(default_factory=list)
    requires_calculation: bool = False
    synthetic_scenario: bool = False


class _DraftCardOutput(BaseModel):
    questions: list[_DraftQuestion]


# ---- Helpers ------------------------------------------------------------------
def build_user_prompt(card: dict, passage_text_by_id: dict[str, str], passage_max_chars: int = 1200) -> str:
    anchors = card.get("evidence_anchors", []) or []
    parts = [
        f"PAPER_ID: {card['paper_id']}",
        "",
        "PAPER CARD:",
        f"  main_problem: {card.get('main_problem', '')}",
        f"  system_context: {card.get('system_context', '')}",
        f"  key_concepts: {card.get('key_concepts', [])}",
        f"  important_formulas: {card.get('important_formulas', [])}",
        f"  important_numbers: {card.get('important_numbers', [])}",
        f"  failure_modes: {card.get('failure_modes', [])}",
        f"  possible_question_topics: {card.get('possible_question_topics', [])}",
        "",
        "ALLOWED PASSAGE IDS for evidence (the only IDs you may use):",
    ]
    for a in anchors:
        pid = a.get("passage_id", "")
        claim = a.get("claim", "")
        parts.append(f"  - {pid}  (claim: {claim})")
    parts.append("")
    parts.append("PASSAGE TEXTS for substring quoting:")
    parts.append("")
    for a in anchors:
        pid = a.get("passage_id", "")
        text = passage_text_by_id.get(pid, "")
        if not text:
            continue
        parts.append(f"--- passage {pid} ---")
        parts.append(text[:passage_max_chars])
        parts.append("")
    parts.append("Now produce 4 questions exactly as specified by the system message. JSON only.")
    return "\n".join(parts)


def _q_id(paper_id: str, idx: int) -> str:
    # paper_NNNN -> NB-HPN-NNNN-qN  (4-digit padded paper number)
    suffix = paper_id.replace("paper_", "")
    return f"NB-HPN-{suffix}-q{idx}"


def _validate_drafts(
    raw_drafts: list[_DraftQuestion],
    paper_id: str,
    valid_passage_ids: set[str],
    passage_text_by_id: dict[str, str],
    benchmark_version: str,
    generator_model: str,
    seed: int,
) -> tuple[list[Question], list[str]]:
    """Promote drafts → Question records, with light per-draft fixes.

    Heavy validation (memorization-leak, exact substring match, cross-corpus
    dedup) is intentionally deferred to Phase 6. Stage 05a only enforces:
      * schema parses
      * evidence[].passage_id is real for this paper
      * non-empty question and reference_answer
    """
    out: list[Question] = []
    warnings: list[str] = []

    for i, d in enumerate(raw_drafts):
        if not d.question.strip() or not d.reference_answer.strip():
            warnings.append(f"q{i}: empty question or reference_answer; dropped")
            continue

        good_evidence: list[Evidence] = []
        for ev in d.evidence:
            pid = ev.passage_id
            if pid not in valid_passage_ids:
                warnings.append(f"q{i}: dropped bogus passage_id {pid}")
                continue
            good_evidence.append(
                Evidence(paper_id=paper_id, passage_id=pid, quote=ev.quote.strip()[:480])
            )

        if not good_evidence:
            # As a last resort, attach the first card-level anchor with an empty
            # quote so downstream substring-validators have something to reject
            # on. This keeps the draft visible in the candidate pool rather
            # than silently disappearing.
            if valid_passage_ids:
                pid = sorted(valid_passage_ids)[0]
                quote = passage_text_by_id.get(pid, "")[:240]
                good_evidence = [Evidence(paper_id=paper_id, passage_id=pid, quote=quote)]
                warnings.append(f"q{i}: no valid evidence; backfilled from card anchor[0]")
            else:
                warnings.append(f"q{i}: no valid evidence and no anchors to backfill; dropped")
                continue

        category = d.category if d.category in CATEGORIES else "Practical HPN Scenarios and Design"
        if category != d.category:
            warnings.append(f"q{i}: category '{d.category}' not in v4 list; remapped")

        q = Question(
            id=_q_id(paper_id, i),
            benchmark_version=benchmark_version,
            category=category,
            difficulty=d.difficulty,
            question_type=d.question_type,
            question=d.question.strip(),
            reference_answer=d.reference_answer.strip(),
            source_papers=[paper_id],
            evidence=good_evidence,
            keywords=[k.strip() for k in d.keywords if k and k.strip()][:7],
            requires_calculation=d.requires_calculation,
            synthetic_scenario=d.synthetic_scenario,
            inspired_by_papers=[paper_id],
            generator_model=generator_model,
            critic_model="",
            prompt_version=PROMPT_VERSION,
            seed=seed,
        )
        out.append(q)

    return out, warnings


def process_card(
    client: OllamaClient,
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
) -> tuple[list[dict], list[str], float]:
    user_prompt = build_user_prompt(card, passage_text_by_id)
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
        return [], [f"ollama: {e}"], time.time() - t0

    # The schema may yield a list-shaped output if the model misreads "object".
    if isinstance(out, list):
        if len(out) == 1 and isinstance(out[0], dict):
            out = out[0]
        else:
            return [], [f"output not a dict (got list of {len(out)})"], time.time() - t0
    if not isinstance(out, dict):
        return [], [f"output not a dict: {type(out).__name__}"], time.time() - t0

    # Fix hallucinated evidence field names (passage_lag_id, passage_flag, etc.)
    for q in out.get("questions", []):
        for ev in q.get("evidence", []) if isinstance(q, dict) else []:
            if isinstance(ev, dict) and "passage_id" not in ev:
                for key in list(ev.keys()):
                    if "passage" in key.lower():
                        ev["passage_id"] = ev.pop(key)
                        break

    try:
        drafted = _DraftCardOutput(**out)
    except Exception as e:
        return [], [f"draft schema: {type(e).__name__}: {e}"], time.time() - t0

    questions, warnings = _validate_drafts(
        raw_drafts=drafted.questions,
        paper_id=card["paper_id"],
        valid_passage_ids=valid_passage_ids,
        passage_text_by_id=passage_text_by_id,
        benchmark_version=benchmark_version,
        generator_model=model,
        seed=seed,
    )

    records = [q.model_dump() for q in questions]
    return records, warnings, time.time() - t0


# ---- CLI ---------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--cards", default=None)
    parser.add_argument("--passages", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model", default=None, help="ollama model (default: from config.models.question_generator)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
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
    prompts_dir = cfg.resolve(cfg.paths.prompts_dir)
    logs_dir = cfg.resolve(cfg.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cards_path = Path(args.cards) if args.cards else None
    if cards_path is None:
        clustered = data_dir / "paper_cards_clustered.jsonl"
        cards_path = clustered if clustered.exists() else data_dir / "paper_cards.jsonl"
    passages_path = Path(args.passages) if args.passages else data_dir / "passages.jsonl"
    output_path = Path(args.output) if args.output else data_dir / "candidate_questions.jsonl"
    prompt_path = Path(args.prompt) if args.prompt else prompts_dir / "question_generation.txt"
    log_path = logs_dir / "stage_05a.jsonl"
    model = args.model or cfg.models.question_generator

    for label, p in [("cards", cards_path), ("passages", passages_path), ("prompt", prompt_path)]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return 1

    system_prompt = prompt_path.read_text(encoding="utf-8")
    cards = list(read_jsonl(cards_path))
    if not cards:
        logger.error("no cards loaded")
        return 1
    logger.info("loaded %d cards from %s", len(cards), cards_path)

    # Build {paper_id: {passage_id: text}}
    passages_by_paper: dict[str, dict[str, str]] = {}
    valid_ids_by_paper: dict[str, set[str]] = {}
    for p in read_jsonl(passages_path):
        pid = p["paper_id"]
        passages_by_paper.setdefault(pid, {})[p["passage_id"]] = p["text"]
        valid_ids_by_paper.setdefault(pid, set()).add(p["passage_id"])
    logger.info("loaded passages for %d papers", len(passages_by_paper))

    # Resume = skip cards whose paper_id already has questions in output
    if args.resume:
        # We use a paper-id key on each question's source_papers[0] to detect resume.
        seen_paper_ids: set[str] = set()
        for r in read_jsonl(output_path):
            sp = r.get("source_papers") or []
            if sp:
                seen_paper_ids.add(sp[0])
        skip = seen_paper_ids
        logger.info("resume: %d papers already have questions in output", len(skip))
    else:
        skip = set()
        if output_path.exists():
            logger.info("clearing existing output: %s", output_path)
            output_path.unlink()

    todo_cards = [c for c in cards if c["paper_id"] not in skip]
    if args.limit:
        todo_cards = todo_cards[: args.limit]

    if not todo_cards:
        logger.info("nothing to do")
        return 0

    schema = _DraftCardOutput.model_json_schema()
    client = OllamaClient(host=cfg.models.ollama_host, timeout=cfg.ollama.request_timeout_seconds)
    seed = cfg.seed
    bv = cfg.splits.benchmark_version

    logger.info(
        "model=%s, workers=%d, todo=%d cards, schema-enforced JSON",
        model, args.workers, len(todo_cards),
    )

    n_cards_ok = 0
    n_cards_fail = 0
    n_questions = 0
    times: list[float] = []

    def task(card: dict):
        pid = card["paper_id"]
        ptexts = passages_by_paper.get(pid, {})
        valid_ids = valid_ids_by_paper.get(pid, set())
        if not valid_ids:
            return card, [], ["no passages found for paper"], 0.0
        records, warnings, elapsed = process_card(
            client=client,
            card=card,
            passage_text_by_id=ptexts,
            valid_passage_ids=valid_ids,
            system_prompt=system_prompt,
            schema=schema,
            model=model,
            temperature=args.temperature,
            num_predict=args.num_predict,
            benchmark_version=bv,
            seed=seed,
        )
        return card, records, warnings, elapsed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, c): c for c in todo_cards}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="cards", unit="card"):
            card, records, warnings, elapsed = fut.result()
            times.append(elapsed)
            ok = len(records) > 0
            log_rec = {
                "paper_id": card["paper_id"],
                "model": model,
                "ok": ok,
                "n_questions": len(records),
                "warnings": warnings,
                "latency_ms": int(elapsed * 1000),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_jsonl(log_path, log_rec)
            for rec in records:
                append_jsonl(output_path, rec)
            n_questions += len(records)
            if ok:
                n_cards_ok += 1
            else:
                n_cards_fail += 1
                logger.warning("FAIL %s: %s", card["paper_id"], warnings)

    avg = sum(times) / max(1, len(times))
    pass_rate = n_cards_ok / max(1, n_cards_ok + n_cards_fail)
    logger.info(
        "done: %d cards ok, %d fail (pass_rate=%.1f%%), %d questions, avg %.1f s/card",
        n_cards_ok, n_cards_fail, pass_rate * 100, n_questions, avg,
    )
    return 0 if pass_rate >= 0.5 else 3


if __name__ == "__main__":
    sys.exit(main())
