"""Stage 06 — Validation gauntlet.

Runs five checks on every candidate question (cheapest first):
  6.1 Schema           — Pydantic
  6.2 Evidence quotes  — quote is exact substring of passage text
  6.3 Memorization     — 8-gram + cosine vs training passages
  6.4 Embedding dedup  — cosine vs already-accepted questions
  6.5 Critic LLM       — grounding / answerability / specificity

One rewrite attempt for items that fail only 6.5 (critic rejected but
suggests a rewrite). Rewrites are re-run through 6.1-6.4 only.

Inputs:
  data/candidate_questions.jsonl
  data/synthesis_candidates.jsonl
  data/adversarial_candidates.jsonl
  data/passages.jsonl
  data/_cache/passage_embeddings.npy
  data/_cache/passage_embedding_ids.json
  prompts/critic_generation.txt

Outputs:
  data/validated_questions.jsonl
  data/_logs/stage_06.jsonl
  reports/stage_stats.md

Usage:
    .venv/bin/python -m src.stage_06_validate
    .venv/bin/python -m src.stage_06_validate --limit 100   # pilot
    .venv/bin/python -m src.stage_06_validate --resume
    .venv/bin/python -m src.stage_06_validate --skip-critic
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config import load_config
from src.embeddings import Embedder, detect_device
from src.io_utils import append_jsonl, read_jsonl
from src.ollama_client import OllamaClient, OllamaError
from src.validators import (
    build_ngram_set,
    check_dedup,
    check_evidence_quotes,
    check_memorization_cosine,
    check_memorization_ngram,
    check_schema,
)

logger = logging.getLogger("stage_06")
PROMPT_VERSION = "v5.0-critic"

# ── Critic output schema ─────────────────────────────────────────────────────

class _DraftCriticOutput(BaseModel):
    """Flat schema — no Literal enum fields, no string defaults (avoid gemma4 bugs)."""
    grounded: bool
    answerable: bool
    non_generic: bool
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    rewrite_suggestion: str


# ── Critic call ──────────────────────────────────────────────────────────────

def _build_critic_user_prompt(
    record: dict,
    passage_text_by_id: dict[str, str],
    passage_max_chars: int = 600,
) -> str:
    lines = [
        f"QUESTION: {record['question']}",
        "",
        f"REFERENCE ANSWER: {record['reference_answer']}",
        "",
        "EVIDENCE:",
    ]
    for ev in record.get("evidence", []):
        pid = ev.get("passage_id", "")
        quote = ev.get("quote", "")
        passage = passage_text_by_id.get(pid, "")[:passage_max_chars]
        lines.append(f"  passage_id: {pid}")
        lines.append(f"  quote: {quote!r}")
        lines.append(f"  passage_text: {passage!r}")
        lines.append("")
    lines.append("Evaluate the question against the three criteria and output JSON.")
    return "\n".join(lines)


def run_critic(
    client: OllamaClient,
    record: dict,
    passage_text_by_id: dict[str, str],
    system_prompt: str,
    schema: dict,
    model: str,
    temperature: float,
    num_predict: int,
) -> tuple[Optional[_DraftCriticOutput], list[str], float]:
    """Call the critic LLM. Returns (output | None, warnings, elapsed_seconds)."""
    user_prompt = _build_critic_user_prompt(record, passage_text_by_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    t0 = time.time()
    last_err: Optional[str] = None
    for attempt in range(2):
        t = temperature if attempt == 0 else 0.2
        try:
            raw = client.chat(
                model=model,
                messages=messages,
                temperature=t,
                num_predict=num_predict,
                format=schema,
                think=False,
            )
        except OllamaError as e:
            return None, [f"ollama: {e}"], time.time() - t0
        # Fix gemma4 colon-inside-key-name bug for known field names
        for field in ("rewrite_suggestion",):
            raw = raw.replace(f'"{field}: "', f'"{field}": "')
        try:
            out_dict = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_err = str(e)
            if attempt == 0:
                time.sleep(0.5)
                continue
    else:
        return None, [f"json parse failed: {last_err}"], time.time() - t0

    if not isinstance(out_dict, dict):
        return None, [f"output not a dict: {type(out_dict).__name__}"], time.time() - t0

    try:
        draft = _DraftCriticOutput(**out_dict)
    except Exception as e:
        return None, [f"critic schema: {e}"], time.time() - t0

    return draft, [], time.time() - t0


# ── Stats tracker ────────────────────────────────────────────────────────────

class Stats:
    def __init__(self) -> None:
        self.total = 0
        self.by_source: dict[str, int] = defaultdict(int)
        self.fail: dict[str, int] = defaultdict(int)    # key → count
        self.fail_by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.accepted = 0
        self.rewrite_accepted = 0
        self.rewrite_fail = 0

    def record_candidate(self, source: str) -> None:
        self.total += 1
        self.by_source[source] += 1

    def record_fail(self, source: str, check: str) -> None:
        self.fail[check] += 1
        self.fail_by_source[source][check] += 1

    def record_accept(self, rewrite: bool = False) -> None:
        if rewrite:
            self.rewrite_accepted += 1
        else:
            self.accepted += 1

    def record_rewrite_fail(self) -> None:
        self.rewrite_fail += 1


def _source_tag(record: dict) -> str:
    iid = record.get("id", "")
    if iid.startswith("NB-ADV-"):
        return "5c"
    if iid.startswith("NB-SYN-"):
        return "5b"
    return "5a"


# ── Stats report ─────────────────────────────────────────────────────────────

def write_stats_report(stats: Stats, path: Path, n_survivors: int) -> None:
    checks = ["6.1", "6.2", "6.3a", "6.3b", "6.4", "6.5"]
    check_names = {
        "6.1": "Schema (Pydantic)",
        "6.2": "Evidence quote substring",
        "6.3a": "Memorization n-gram",
        "6.3b": "Memorization cosine",
        "6.4": "Embedding dedup",
        "6.5": "Critic LLM",
    }

    lines = [
        "# Stage 06 Validation — Reject Rate Report",
        "",
        f"**Total candidates:** {stats.total}",
        f"**Accepted (critic):** {stats.accepted}",
        f"**Accepted (rewrite):** {stats.rewrite_accepted}",
        f"**Total accepted:** {n_survivors}",
        f"**Overall pass rate:** {n_survivors / max(1, stats.total) * 100:.1f}%",
        "",
        "## Reject Rates per Validator",
        "",
        "| Check | Name | Rejected | Rate |",
        "|---|---|---|---|",
    ]
    for ck in checks:
        n = stats.fail.get(ck, 0)
        rate = n / max(1, stats.total) * 100
        lines.append(f"| {ck} | {check_names[ck]} | {n} | {rate:.1f}% |")

    rewrite_total = stats.rewrite_accepted + stats.rewrite_fail
    lines += [
        "",
        f"**Rewrite attempts:** {rewrite_total}  "
        f"(accepted {stats.rewrite_accepted}, dropped {stats.rewrite_fail})",
        "",
        "## By Source",
        "",
        "| Source | Candidates | Accepted | Rate |",
        "|---|---|---|---|",
    ]
    for src in ["5a", "5b", "5c"]:
        n_cand = stats.by_source.get(src, 0)
        n_fail = sum(stats.fail_by_source[src].values())
        n_acc = n_cand - n_fail
        # subtract rewrites that failed from the same source (approximate)
        rate = n_acc / max(1, n_cand) * 100
        lines.append(f"| {src} | {n_cand} | ≥{max(0, n_acc)} | ≈{rate:.0f}% |")

    lines += [
        "",
        "## Per-Source Reject Breakdown",
        "",
    ]
    for src in ["5a", "5b", "5c"]:
        if stats.by_source.get(src, 0) == 0:
            continue
        lines.append(f"### Source {src}")
        lines.append("")
        for ck in checks:
            n = stats.fail_by_source[src].get(ck, 0)
            if n:
                lines.append(f"- {ck} {check_names[ck]}: {n}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("stats report written to %s", path)


# ── Deterministic validation helper ─────────────────────────────────────────

def run_deterministic_checks(
    record: dict,
    q_emb: np.ndarray,
    r_emb: np.ndarray,
    passage_text_by_id: dict[str, str],
    training_ngrams: frozenset[str],
    passage_embeddings: np.ndarray,
    accepted_dedup_arr: Optional[np.ndarray],
    leak_cosine_threshold: float,
    dedup_cosine_threshold: float,
    ngram_n: int,
) -> tuple[bool, str, str]:
    """Run checks 6.1-6.4. Returns (passed, failed_check_key, reason)."""
    ok, reason = check_schema(record)
    if not ok:
        return False, "6.1", reason

    ok, reason = check_evidence_quotes(record, passage_text_by_id)
    if not ok:
        return False, "6.2", reason

    ok, reason = check_memorization_ngram(record.get("reference_answer", ""), training_ngrams, ngram_n)
    if not ok:
        return False, "6.3a", reason

    ok, reason = check_memorization_cosine(r_emb, passage_embeddings, leak_cosine_threshold)
    if not ok:
        return False, "6.3b", reason

    ok, reason = check_dedup(q_emb, accepted_dedup_arr, dedup_cosine_threshold)
    if not ok:
        return False, "6.4", reason

    return True, "", ""


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--candidates-5a", default=None)
    parser.add_argument("--candidates-5b", default=None)
    parser.add_argument("--candidates-5c", default=None)
    parser.add_argument("--passages", default=None)
    parser.add_argument("--emb-cache", default=None, help="passage_embeddings.npy")
    parser.add_argument("--emb-ids", default=None, help="passage_embedding_ids.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--critic-prompt", default=None)
    parser.add_argument("--model", default=None, help="critic model (default: cfg.models.critic)")
    parser.add_argument("--device", default=None, help="embedding device")
    parser.add_argument("--workers", type=int, default=2, help="parallel critic workers")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="pilot: only process first N candidates")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--sources", nargs="+", choices=["5a", "5b", "5c"], default=None,
        help="only load candidates from these sources (default: all)",
    )
    parser.add_argument("--skip-critic", action="store_true", help="skip 6.5 LLM critic")
    parser.add_argument("--no-rewrite", action="store_true", help="skip rewrite attempts")
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
    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    path_5a = Path(args.candidates_5a) if args.candidates_5a else data_dir / "candidate_questions.jsonl"
    path_5b = Path(args.candidates_5b) if args.candidates_5b else data_dir / "synthesis_candidates.jsonl"
    path_5c = Path(args.candidates_5c) if args.candidates_5c else data_dir / "adversarial_candidates.jsonl"
    passages_path = Path(args.passages) if args.passages else data_dir / "passages.jsonl"
    emb_path = Path(args.emb_cache) if args.emb_cache else cache_dir / "passage_embeddings.npy"
    emb_ids_path = Path(args.emb_ids) if args.emb_ids else cache_dir / "passage_embedding_ids.json"
    output_path = Path(args.output) if args.output else data_dir / "validated_questions.jsonl"
    log_path = Path(args.log) if args.log else logs_dir / "stage_06.jsonl"
    stats_path = Path(args.stats) if args.stats else reports_dir / "stage_stats.md"
    critic_prompt_path = Path(args.critic_prompt) if args.critic_prompt else prompts_dir / "critic_generation.txt"

    model = args.model or cfg.models.critic
    device = detect_device(args.device)

    # ── Validate inputs ──────────────────────────────────────────────────────
    for label, p in [
        ("passages", passages_path), ("emb-cache", emb_path),
        ("emb-ids", emb_ids_path), ("critic-prompt", critic_prompt_path),
    ]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return 1

    # ── Load candidates ──────────────────────────────────────────────────────
    sources_filter = set(args.sources) if args.sources else {"5a", "5b", "5c"}
    candidates: list[dict] = []
    for src_path, label in [(path_5a, "5a"), (path_5b, "5b"), (path_5c, "5c")]:
        if label not in sources_filter:
            logger.info("skipping source %s (not in --sources)", label)
            continue
        if src_path.exists():
            recs = list(read_jsonl(src_path))
            logger.info("loaded %d candidates from %s (%s)", len(recs), src_path.name, label)
            candidates.extend(recs)
        else:
            logger.warning("%s not found: %s — skipping", label, src_path)

    if not candidates:
        logger.error("no candidates loaded")
        return 1

    if args.limit:
        candidates = candidates[: args.limit]
        logger.info("pilot mode: limited to %d candidates", len(candidates))

    logger.info("total candidates: %d", len(candidates))

    # ── Resume ───────────────────────────────────────────────────────────────
    done_ids: set[str] = set()
    initial_accepted_records: list[dict] = []
    if args.resume and output_path.exists():
        for r in read_jsonl(output_path):
            done_ids.add(r.get("id", ""))
            initial_accepted_records.append(r)
        logger.info("resume: %d items already accepted", len(done_ids))

    todo = [c for c in candidates if c.get("id", "") not in done_ids]
    logger.info("todo: %d candidates to validate", len(todo))

    if not todo:
        logger.info("nothing to do")
        return 0

    # ── Load passage data ────────────────────────────────────────────────────
    logger.info("loading passages …")
    passage_text_by_id: dict[str, str] = {}
    passage_texts_all: list[str] = []
    for p in read_jsonl(passages_path):
        passage_text_by_id[p["passage_id"]] = p["text"]
        passage_texts_all.append(p["text"])

    logger.info("building %d-gram set from %d passages …", cfg.validation.ngram_leak_size, len(passage_texts_all))
    t_ngram = time.time()
    training_ngrams = build_ngram_set(passage_texts_all, n=cfg.validation.ngram_leak_size)
    logger.info("n-gram set built: %d entries (%.1f s)", len(training_ngrams), time.time() - t_ngram)

    # ── Load passage embeddings ──────────────────────────────────────────────
    logger.info("loading passage embeddings from cache …")
    passage_embeddings = np.load(emb_path).astype(np.float32)
    passage_emb_ids: list[str] = json.loads(emb_ids_path.read_text())
    logger.info("passage embeddings: shape=%s", passage_embeddings.shape)

    # ── Embed candidates upfront ─────────────────────────────────────────────
    logger.info("embedding candidate questions and reference answers …")
    embedder = Embedder(cfg.models.embedding, device=device, batch_size=64)

    question_texts = [c.get("question", "") for c in todo]
    ref_texts = [c.get("reference_answer", "") for c in todo]

    q_embeddings = embedder.embed(question_texts, normalize=True, show_progress=True)
    r_embeddings = embedder.embed(ref_texts, normalize=True, show_progress=True)

    # Map id → embedding
    q_emb_by_id: dict[str, np.ndarray] = {c["id"]: q_embeddings[i] for i, c in enumerate(todo)}
    r_emb_by_id: dict[str, np.ndarray] = {c["id"]: r_embeddings[i] for i, c in enumerate(todo)}

    # ── Initialize dedup pool from resume records ────────────────────────────
    accepted_dedup_pool: list[np.ndarray] = []
    if initial_accepted_records:
        logger.info("embedding %d resumed records for dedup pool …", len(initial_accepted_records))
        init_texts = [r.get("question", "") for r in initial_accepted_records]
        init_embs = embedder.embed(init_texts, normalize=True, show_progress=False)
        accepted_dedup_pool = list(init_embs)
    accepted_dedup_arr: Optional[np.ndarray] = (
        np.stack(accepted_dedup_pool) if accepted_dedup_pool else None
    )

    # ── Stats + logging ──────────────────────────────────────────────────────
    stats = Stats()
    for c in todo:   # count only items to process (not resumed/skipped)
        stats.record_candidate(_source_tag(c))

    def log_item(record: dict, check: str, passed: bool, reason: str, latency_ms: int = 0) -> None:
        src = _source_tag(record)
        entry = {
            "item_id": record.get("id", ""),
            "source": src,
            "check": check,
            "passed": passed,
            "reason": reason,
            "latency_ms": latency_ms,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(log_path, entry)
        if not passed:
            stats.record_fail(src, check)

    # ── Phase 1: Deterministic checks (6.1-6.4) ─────────────────────────────
    logger.info("Phase 1: running deterministic checks (6.1-6.4) …")
    pending_critic: list[dict] = []  # candidates that passed 6.1-6.4

    for record in tqdm(todo, desc="det-checks", unit="q"):
        iid = record.get("id", "")
        if not iid:
            continue
        src = _source_tag(record)
        q_emb = q_emb_by_id[iid]
        r_emb = r_emb_by_id[iid]

        passed, check_key, reason = run_deterministic_checks(
            record=record,
            q_emb=q_emb,
            r_emb=r_emb,
            passage_text_by_id=passage_text_by_id,
            training_ngrams=training_ngrams,
            passage_embeddings=passage_embeddings,
            accepted_dedup_arr=accepted_dedup_arr,
            leak_cosine_threshold=cfg.validation.leak_cosine_threshold,
            dedup_cosine_threshold=cfg.validation.dedup_cosine_threshold,
            ngram_n=cfg.validation.ngram_leak_size,
        )

        if not passed:
            log_item(record, check_key, False, reason)
            logger.debug("FAIL %s (%s): %s", iid, check_key, reason)
            continue

        log_item(record, "6.4", True, "")
        # Add to dedup pool optimistically (before critic)
        accepted_dedup_pool.append(q_emb)
        accepted_dedup_arr = np.stack(accepted_dedup_pool)
        pending_critic.append(record)

    logger.info(
        "Phase 1 done: %d/%d passed deterministic checks",
        len(pending_critic), len(todo),
    )

    # ── Phase 2: Critic (6.5) in parallel ───────────────────────────────────
    if args.skip_critic:
        logger.info("--skip-critic: accepting all %d items that passed 6.1-6.4", len(pending_critic))
        critic_results: dict[str, tuple[Optional[_DraftCriticOutput], list[str], float]] = {
            r["id"]: (_DraftCriticOutput(
                grounded=True, answerable=True, non_generic=True,
                accepted=True, reasons=[], rewrite_suggestion="",
            ), [], 0.0)
            for r in pending_critic
        }
    else:
        logger.info(
            "Phase 2: running critic on %d candidates (workers=%d, model=%s) …",
            len(pending_critic), args.workers, model,
        )
        critic_system_prompt = critic_prompt_path.read_text(encoding="utf-8")
        critic_schema = _DraftCriticOutput.model_json_schema()
        client = OllamaClient(host=cfg.models.ollama_host, timeout=cfg.ollama.request_timeout_seconds)

        critic_results = {}

        def _critic_task(record: dict) -> tuple[str, Optional[_DraftCriticOutput], list[str], float]:
            draft, warnings, elapsed = run_critic(
                client=client,
                record=record,
                passage_text_by_id=passage_text_by_id,
                system_prompt=critic_system_prompt,
                schema=critic_schema,
                model=model,
                temperature=args.temperature,
                num_predict=args.num_predict,
            )
            return record["id"], draft, warnings, elapsed

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_critic_task, r): r for r in pending_critic}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="critic", unit="q"):
                iid, draft, warnings, elapsed = fut.result()
                critic_results[iid] = (draft, warnings, elapsed)

    # ── Phase 3: Process critic results in submission order ──────────────────
    logger.info("Phase 3: processing critic results …")
    rewrite_queue: list[tuple[dict, str]] = []  # (original_record, rewrite_suggestion)

    for record in pending_critic:
        iid = record["id"]
        src = _source_tag(record)
        draft, warnings, elapsed = critic_results.get(iid, (None, ["no result"], 0.0))
        latency_ms = int(elapsed * 1000)

        if draft is None:
            log_item(record, "6.5", False, f"critic error: {warnings}", latency_ms)
            logger.warning("CRITIC-ERROR %s: %s", iid, warnings)
            continue

        if draft.accepted:
            # Stamp critic_model and write
            out_record = dict(record)
            out_record["critic_model"] = model
            append_jsonl(output_path, out_record)
            stats.record_accept(rewrite=False)
            log_item(record, "6.5", True, "", latency_ms)
            logger.debug("ACCEPT %s", iid)
        else:
            log_item(record, "6.5", False, "; ".join(draft.reasons), latency_ms)
            logger.debug("CRITIC-REJECT %s: %s", iid, draft.reasons)
            if not args.no_rewrite and draft.rewrite_suggestion and draft.rewrite_suggestion.strip():
                rewrite_queue.append((record, draft.rewrite_suggestion.strip()))

    # ── Phase 4: Rewrite attempts ────────────────────────────────────────────
    if rewrite_queue:
        logger.info("Phase 4: attempting %d rewrites …", len(rewrite_queue))

        for orig_record, suggestion in tqdm(rewrite_queue, desc="rewrites", unit="q"):
            iid = orig_record["id"]
            src = _source_tag(orig_record)

            # Build rewritten record
            rewritten = dict(orig_record)
            rewritten["question"] = suggestion

            # Re-embed new question text
            new_q_emb = embedder.embed([suggestion], normalize=True, show_progress=False)[0]

            # Re-run deterministic checks (6.1-6.4)
            # Note: reference_answer and evidence unchanged → 6.3 results are same
            # Still re-run all for correctness
            r_emb = r_emb_by_id[iid]
            passed, check_key, reason = run_deterministic_checks(
                record=rewritten,
                q_emb=new_q_emb,
                r_emb=r_emb,
                passage_text_by_id=passage_text_by_id,
                training_ngrams=training_ngrams,
                passage_embeddings=passage_embeddings,
                accepted_dedup_arr=accepted_dedup_arr,
                leak_cosine_threshold=cfg.validation.leak_cosine_threshold,
                dedup_cosine_threshold=cfg.validation.dedup_cosine_threshold,
                ngram_n=cfg.validation.ngram_leak_size,
            )

            if not passed:
                log_item(rewritten, f"{check_key}-rewrite", False, reason)
                stats.record_rewrite_fail()
                logger.debug("REWRITE-FAIL %s (%s): %s", iid, check_key, reason)
                continue

            # Accept rewrite — update dedup pool
            accepted_dedup_pool.append(new_q_emb)
            accepted_dedup_arr = np.stack(accepted_dedup_pool)

            rewritten["critic_model"] = model
            append_jsonl(output_path, rewritten)
            stats.record_accept(rewrite=True)
            log_item(rewritten, "rewrite", True, "accepted after rewrite")
            logger.debug("REWRITE-ACCEPT %s", iid)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_survivors = stats.accepted + stats.rewrite_accepted
    pass_rate = n_survivors / max(1, len(todo))
    logger.info(
        "done: %d accepted (%d direct + %d rewrite), %d total candidates processed",
        n_survivors, stats.accepted, stats.rewrite_accepted, len(todo),
    )
    logger.info("pass rate: %.1f%%", pass_rate * 100)

    # ── Write stats report ───────────────────────────────────────────────────
    write_stats_report(stats, stats_path, n_survivors + len(initial_accepted_records))
    logger.info("reject rate breakdown:")
    for check, count in sorted(stats.fail.items()):
        logger.info("  %s: %d rejected (%.1f%%)", check, count, count / max(1, len(todo)) * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
