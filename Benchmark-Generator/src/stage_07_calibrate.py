"""Stage 07 — Difficulty calibration.

For each validated question, generates one answer per model tier (weak / mid / strong),
judges each answer against the reference with gemma4:26b, then assigns an empirical
difficulty tag:

  easy   — all 3 models correct
  medium — strong ✓, mid ✓, weak ✗
  hard   — strong ✓, mid ✗  (regardless of weak)
  drop   — strong ✗  (deemed ambiguous or unanswerable)

One rewrite attempt is NOT performed here; questions dropped at Phase 6 already
passed the critic. Drop = truly hard for even the strongest model.

Inputs:
  data/validated_questions.jsonl
  prompts/calibration_judge.txt

Outputs:
  data/calibrated_questions.jsonl  — validated questions with updated difficulty
  data/_logs/stage_07.jsonl        — per-question calibration log
  reports/difficulty_distribution.md

Usage:
    .venv/bin/python -m src.stage_07_calibrate
    .venv/bin/python -m src.stage_07_calibrate --limit 10   # pilot
    .venv/bin/python -m src.stage_07_calibrate --resume --workers 2
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.config import load_config
from src.io_utils import append_jsonl, read_jsonl
from src.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger("stage_07")

PROMPT_VERSION = "v5.0-calibration"

_ANSWER_SYSTEM = (
    "You are an expert in high-performance networking (HPN). "
    "Answer the question concisely and accurately."
)


# ── Model helpers ────────────────────────────────────────────────────────────

def _think_param(model: str) -> Optional[bool]:
    """Return think=False for models that use hybrid thinking (qwen3.x, gemma4.x).

    Both qwen3 and gemma4 can enter a silent thinking mode where output goes to
    message.thinking instead of message.content, resulting in empty responses.
    Passing think=False suppresses this and ensures text appears in message.content.
    """
    if model.startswith(("qwen3", "gemma4")):
        return False
    return None


# ── LLM helpers ─────────────────────────────────────────────────────────────

def generate_answer(
    client: OllamaClient,
    model: str,
    question: str,
    temperature: float,
    num_predict: int,
) -> tuple[str, int]:
    """Generate a free-form answer. Returns (answer_text, latency_ms)."""
    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    try:
        answer = client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            num_predict=num_predict,
            think=_think_param(model),
        )
    except OllamaError as e:
        logger.warning("generate_answer failed for %s: %s", model, e)
        answer = ""
    latency_ms = int((time.time() - t0) * 1000)
    return answer.strip(), latency_ms


def run_judge(
    client: OllamaClient,
    judge_model: str,
    judge_prompt: str,
    question: str,
    reference: str,
    model_answer: str,
    temperature: float,
    num_predict: int,
) -> tuple[bool, str, int]:
    """Judge model_answer against reference. Returns (correct, reason, latency_ms).

    The judge prompt requests free-text "CORRECT" / "INCORRECT" response (no JSON),
    avoiding gemma4:26b's tendency to return empty content for schema-enforced calls.
    Two attempts; on both failures returns (False, "judge-error", latency_ms).
    """
    user_msg = (
        f"Question: {question}\n\n"
        f"Reference Answer: {reference}\n\n"
        f"Model Answer: {model_answer if model_answer else '(no answer provided)'}"
    )
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_msg},
    ]

    t0 = time.time()
    for attempt in range(2):
        t = temperature if attempt == 0 else 0.1
        try:
            raw = client.chat(
                judge_model, messages, temperature=t, num_predict=num_predict,
                think=_think_param(judge_model),
            )
        except OllamaError as e:
            logger.warning("judge chat failed (attempt %d): %s", attempt, e)
            continue

        raw = raw.strip()
        if not raw:
            logger.debug("judge returned empty (attempt %d)", attempt)
            continue

        # Parse: first word on first non-empty line must be CORRECT or INCORRECT
        first_word = raw.split()[0].upper().rstrip(".")
        if first_word in ("CORRECT", "INCORRECT"):
            correct = first_word == "CORRECT"
            # Collect reason from remaining text
            lines = raw.split("\n")
            reason_lines = [l.strip() for l in lines[1:] if l.strip()]
            reason = " ".join(reason_lines) if reason_lines else raw
            latency_ms = int((time.time() - t0) * 1000)
            return correct, reason, latency_ms

        # Fallback: scan response for CORRECT/INCORRECT keyword anywhere
        upper = raw.upper()
        if "INCORRECT" in upper:
            latency_ms = int((time.time() - t0) * 1000)
            return False, raw[:200], latency_ms
        if "CORRECT" in upper:
            latency_ms = int((time.time() - t0) * 1000)
            return True, raw[:200], latency_ms

        logger.debug("judge response unparseable (attempt %d): %r", attempt, raw[:100])

    latency_ms = int((time.time() - t0) * 1000)
    logger.warning("judge failed after 2 attempts")
    return False, "judge-error", latency_ms


# ── Difficulty assignment ────────────────────────────────────────────────────

def assign_difficulty(
    weak_ok: bool,
    mid_ok: bool,
    strong_ok: bool,
) -> Optional[str]:
    """Return 'easy' / 'medium' / 'hard' or None (drop).

    Tagging rules (from DEV_PLAN Phase 7):
      easy   — all 3 correct
      medium — strong ✓, mid ✓, weak ✗
      hard   — strong ✓, mid ✗  (regardless of weak)
      drop   — strong ✗  (regardless of weak/mid)
    """
    if not strong_ok:
        return None  # drop
    if mid_ok and weak_ok:
        return "easy"
    if mid_ok and not weak_ok:
        return "medium"
    # strong ✓, mid ✗ (regardless of weak)
    return "hard"


# ── Per-question worker ──────────────────────────────────────────────────────

def process_question(
    record: dict,
    client: OllamaClient,
    judge_prompt: str,
    weak_model: str,
    mid_model: str,
    strong_model: str,
    judge_model: str,
    temperature: float,
    num_predict_answer: int,
    num_predict_judge: int,
) -> dict:
    """Run 3 answer + 3 judge calls for one question. Returns result dict."""
    qid = record.get("id", "")
    question = record.get("question", "")
    reference = record.get("reference_answer", "")

    judge_errors = 0
    ans_latencies: dict[str, int] = {}
    judge_latencies: dict[str, int] = {}
    correct: dict[str, bool] = {}

    for role, model in [("weak", weak_model), ("mid", mid_model), ("strong", strong_model)]:
        answer, ans_ms = generate_answer(client, model, question, temperature, num_predict_answer)
        ok, reason, judge_ms = run_judge(
            client, judge_model, judge_prompt, question, reference,
            answer, temperature, num_predict_judge,
        )
        if reason == "judge-error":
            judge_errors += 1
        ans_latencies[role] = ans_ms
        judge_latencies[role] = judge_ms
        correct[role] = ok

        logger.debug("%s %s %s: correct=%s (%d ms ans, %d ms judge)",
                     qid, role, model, ok, ans_ms, judge_ms)

    difficulty = assign_difficulty(correct["weak"], correct["mid"], correct["strong"])

    return {
        "id": qid,
        "difficulty": difficulty,
        "difficulty_original": record.get("difficulty", ""),
        "weak_correct": correct["weak"],
        "mid_correct": correct["mid"],
        "strong_correct": correct["strong"],
        "weak_latency_ms": ans_latencies["weak"],
        "mid_latency_ms": ans_latencies["mid"],
        "strong_latency_ms": ans_latencies["strong"],
        "judge_latency_ms": sum(judge_latencies.values()),
        "judge_errors": judge_errors,
    }


# ── Report ───────────────────────────────────────────────────────────────────

def write_report(
    results: list[dict],
    report_path: Path,
    orig_dist: Counter,
) -> None:
    accepted = [r for r in results if r["difficulty"] is not None]
    dropped = [r for r in results if r["difficulty"] is None]

    new_dist: Counter = Counter(r["difficulty"] for r in accepted)
    total_new = len(accepted)

    lines = [
        "# Stage 07 — Difficulty Calibration Report",
        "",
        f"**Input:** {len(results)} questions",
        f"**Accepted:** {total_new}",
        f"**Dropped (strong model wrong):** {len(dropped)}",
        f"**Drop rate:** {len(dropped) / max(1, len(results)) * 100:.1f}%",
        "",
        "## Generator-Assigned Difficulty (before calibration)",
        "",
        "| Difficulty | Count | % |",
        "|---|---|---|",
    ]
    for d in ["easy", "medium", "hard"]:
        n = orig_dist.get(d, 0)
        total_orig = sum(orig_dist.values())
        lines.append(f"| {d} | {n} | {n / max(1, total_orig) * 100:.1f}% |")

    lines += [
        "",
        "## Empirical Difficulty (after calibration)",
        "",
        "| Difficulty | Count | % | Target % |",
        "|---|---|---|---|",
    ]
    targets = {"easy": 0.25, "medium": 0.45, "hard": 0.30}
    for d in ["easy", "medium", "hard"]:
        n = new_dist.get(d, 0)
        pct = n / max(1, total_new) * 100
        target_pct = targets[d] * 100
        gap = pct - target_pct
        gap_str = f"{gap:+.1f}%"
        lines.append(f"| {d} | {n} | {pct:.1f}% ({gap_str} vs target) | {target_pct:.0f}% |")

    # Balancing note
    lines += [""]
    needs_balance = any(
        abs(new_dist.get(d, 0) / max(1, total_new) - targets[d]) > 0.15
        for d in ["easy", "medium", "hard"]
    )
    if needs_balance:
        lines.append(
            "**Note:** Distribution is >15% off target in at least one bucket. "
            "Phase 8 should apply stratified sampling to balance the splits."
        )
    else:
        lines.append(
            "Distribution is within 15% of target in all buckets. "
            "Phase 8 stratified sampling should meet the 25/45/30 target."
        )

    lines += [
        "",
        "## Model Correctness by Tier",
        "",
        "| Tier | Correct | % |",
        "|---|---|---|",
    ]
    total = len(results)
    for role in ["weak", "mid", "strong"]:
        n_correct = sum(1 for r in results if r[f"{role}_correct"])
        lines.append(f"| {role} | {n_correct} | {n_correct / max(1, total) * 100:.1f}% |")

    lines += [
        "",
        "## Judge Error Rate",
        "",
        f"Total judge errors (parse failures): {sum(r['judge_errors'] for r in results)}",
        f"Items with ≥1 judge error: {sum(1 for r in results if r['judge_errors'] > 0)}",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("report written to %s", report_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--input", default=None, help="validated_questions.jsonl")
    parser.add_argument("--output", default=None, help="calibrated_questions.jsonl")
    parser.add_argument("--log", default=None, help="stage_07.jsonl log path")
    parser.add_argument("--report", default=None, help="difficulty_distribution.md")
    parser.add_argument("--judge-prompt", default=None, help="calibration_judge.txt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="generation temperature (default 0.1 for near-determinism)")
    parser.add_argument("--num-predict-answer", type=int, default=512,
                        help="max tokens for answer generation")
    parser.add_argument("--num-predict-judge", type=int, default=256,
                        help="max tokens for judge output")
    parser.add_argument("--limit", type=int, default=0, help="pilot: only first N questions")
    parser.add_argument("--resume", action="store_true",
                        help="skip items already in stage_07 log")
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
    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input) if args.input else data_dir / "validated_questions.jsonl"
    output_path = Path(args.output) if args.output else data_dir / "calibrated_questions.jsonl"
    log_path = Path(args.log) if args.log else logs_dir / "stage_07.jsonl"
    report_path = Path(args.report) if args.report else reports_dir / "difficulty_distribution.md"
    judge_prompt_path = Path(args.judge_prompt) if args.judge_prompt else prompts_dir / "calibration_judge.txt"

    for label, p in [("input", input_path), ("judge-prompt", judge_prompt_path)]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return 1

    weak_model = cfg.models.calibration.weak
    mid_model = cfg.models.calibration.mid
    strong_model = cfg.models.calibration.strong
    judge_model = strong_model  # judge uses the strongest local model

    logger.info("calibration models: weak=%s mid=%s strong=%s judge=%s",
                weak_model, mid_model, strong_model, judge_model)

    # ── Load inputs ──────────────────────────────────────────────────────────
    questions = list(read_jsonl(input_path))
    logger.info("loaded %d questions from %s", len(questions), input_path.name)

    orig_dist: Counter = Counter(q.get("difficulty", "") for q in questions)

    # ── Resume: skip already-logged item IDs ─────────────────────────────────
    done_ids: set[str] = set()
    if args.resume and log_path.exists():
        for entry in read_jsonl(log_path):
            done_ids.add(entry.get("item_id", ""))
        logger.info("resume: %d items already processed", len(done_ids))

    todo = [q for q in questions if q.get("id", "") not in done_ids]

    if args.limit:
        todo = todo[: args.limit]
        logger.info("pilot mode: limited to %d questions", len(todo))

    if not todo:
        logger.info("nothing to do")
        return 0

    logger.info("todo: %d questions to calibrate", len(todo))

    # ── Load prompt ───────────────────────────────────────────────────────────
    judge_prompt = judge_prompt_path.read_text(encoding="utf-8").strip()

    # ── Ollama client ─────────────────────────────────────────────────────────
    client = OllamaClient(
        host=cfg.models.ollama_host,
        timeout=cfg.ollama.request_timeout_seconds,
    )

    # ── Parallel calibration ─────────────────────────────────────────────────
    logger.info("starting calibration with %d workers …", args.workers)

    all_results: list[dict] = []
    accepted = 0
    dropped = 0

    from tqdm import tqdm  # local import so module loads without tqdm for schema-only tests

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_question,
                rec,
                client,
                judge_prompt,
                weak_model,
                mid_model,
                strong_model,
                judge_model,
                args.temperature,
                args.num_predict_answer,
                args.num_predict_judge,
            ): rec
            for rec in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="calibrate", unit="q"):
            rec = futures[fut]
            qid = rec.get("id", "?")
            try:
                result = fut.result()
            except Exception as e:
                logger.error("process_question crashed for %s: %s", qid, e)
                result = {
                    "id": qid,
                    "difficulty": None,
                    "difficulty_original": rec.get("difficulty", ""),
                    "weak_correct": False,
                    "mid_correct": False,
                    "strong_correct": False,
                    "weak_latency_ms": 0,
                    "mid_latency_ms": 0,
                    "strong_latency_ms": 0,
                    "judge_latency_ms": 0,
                    "judge_errors": 3,
                }

            all_results.append(result)

            # Write log entry immediately (crash-safe: --resume reads this log)
            log_entry = {
                "item_id": result["id"],
                "difficulty_assigned": result["difficulty"] if result["difficulty"] else "dropped",
                "difficulty_original": result["difficulty_original"],
                "weak_correct": result["weak_correct"],
                "mid_correct": result["mid_correct"],
                "strong_correct": result["strong_correct"],
                "weak_latency_ms": result["weak_latency_ms"],
                "mid_latency_ms": result["mid_latency_ms"],
                "strong_latency_ms": result["strong_latency_ms"],
                "judge_latency_ms": result["judge_latency_ms"],
                "judge_errors": result["judge_errors"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_jsonl(log_path, log_entry)

            if result["difficulty"] is not None:
                out_record = dict(rec)
                out_record["difficulty"] = result["difficulty"]
                out_record["calibration_model"] = judge_model
                append_jsonl(output_path, out_record)
                accepted += 1
                logger.debug("ACCEPT %s → %s (was %s)", result["id"],
                             result["difficulty"], result["difficulty_original"])
            else:
                dropped += 1
                logger.debug("DROP %s (strong wrong; was %s)", result["id"],
                             result["difficulty_original"])

    # ── Final stats ───────────────────────────────────────────────────────────
    new_dist: Counter = Counter(r["difficulty"] for r in all_results if r["difficulty"])
    logger.info(
        "done: %d accepted, %d dropped (drop rate %.1f%%)",
        accepted, dropped, dropped / max(1, accepted + dropped) * 100,
    )
    logger.info("empirical distribution: %s", dict(new_dist))

    # Combine with any already-accepted items (resume case) for the full report
    all_log_results: list[dict] = all_results
    if args.resume and log_path.exists():
        # Re-read full log to get complete picture for the report
        all_log_results = []
        for entry in read_jsonl(log_path):
            all_log_results.append({
                "difficulty": entry.get("difficulty_assigned") if entry.get("difficulty_assigned") != "dropped" else None,
                "difficulty_original": entry.get("difficulty_original", ""),
                "weak_correct": entry.get("weak_correct", False),
                "mid_correct": entry.get("mid_correct", False),
                "strong_correct": entry.get("strong_correct", False),
                "judge_errors": entry.get("judge_errors", 0),
            })

    # ── For the report, use full orig_dist (all validated questions, not just todo) ──
    full_orig_dist: Counter = Counter(q.get("difficulty", "") for q in questions)
    write_report(all_log_results, report_path, full_orig_dist)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
