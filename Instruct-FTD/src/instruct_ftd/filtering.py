"""Phase 7 filtering, judging, and deduplication for candidate examples."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .ollama_client import OllamaClient
from .progress import append_jsonl, ensure_resume_safe, fmt_elapsed, load_checkpoint_map, log, read_jsonl, write_json


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_LOW_QUALITY_RE = re.compile(
    r"(table of contents|acknowledgements|organization of thesis|chapter 1|chapter 2|references\s+\d)",
    re.IGNORECASE,
)
_BAD_RESPONSE_RE = re.compile(
    r"(as an ai|i do not have access|i cannot browse|language model)",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "why",
}


@dataclass
class ReviewRecord:
    """Internal review state for one candidate example."""

    example: dict
    source_path: str
    source_index: int
    heuristic_score: float
    reject_reason: str = ""
    judge_result: dict | None = None


def load_candidate_examples(hpn_path: Path, rag_path: Path) -> list[dict]:
    """Load Phase 6 candidate examples from HPN and RAG files."""

    rows: list[dict] = []
    rows.extend(read_jsonl(hpn_path))
    rows.extend(read_jsonl(rag_path))
    return rows


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _normalize_text(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", (text or "").lower()).strip()


def _signature(text: str) -> str:
    return " ".join(_normalize_text(text).split())


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall((text or "").lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _evidence_text(example: dict) -> str:
    evidence = example.get("evidence") or []
    return "\n".join(str(item.get("text") or "") for item in evidence)


def _lexical_overlap(example: dict) -> float:
    evidence_tokens = _token_set(_evidence_text(example))
    response_tokens = _token_set(str(example.get("response") or ""))
    if not response_tokens or not evidence_tokens:
        return 0.0
    return len(response_tokens & evidence_tokens) / len(response_tokens)


def _simhash_bits(text: str) -> int:
    tokens = list(_token_set(text))
    if not tokens:
        return 0

    weights = [0] * 64
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1

    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _response_length_bounds(example: dict) -> tuple[int, int]:
    category = str(example.get("category") or "")
    if category == "rag_unanswerable":
        return (40, 240)
    if str(example.get("source_split")) == "rag":
        return (100, 420)
    return (80, 420)


def _question_format_valid(example: dict) -> bool:
    question = str(example.get("question") or "")
    source_split = str(example.get("source_split") or "")
    if source_split == "rag":
        return question.startswith("Excerpts:\n\n") and "\n\nQuestion: " in question
    return not question.startswith("Excerpts:\n\n")


def _required_fields_valid(example: dict) -> bool:
    required = ["system", "question", "response", "category"]
    return all(str(example.get(field) or "").strip() for field in required)


def _heuristic_quality_score(example: dict) -> float:
    response_words = _word_count(str(example.get("response") or ""))
    min_words, max_words = _response_length_bounds(example)
    overlap = _lexical_overlap(example)
    within_range = 1.0 if min_words <= response_words <= max_words else 0.0
    cite_bonus = 0.2 if "[1]" in str(example.get("response") or "") or "[2]" in str(example.get("response") or "") else 0.0
    return round((within_range * 2.0) + overlap + cite_bonus, 4)


def heuristic_reject_reason(example: dict) -> str:
    """Return a heuristic rejection reason, or empty string if the example passes."""

    if not _required_fields_valid(example):
        return "missing_required_fields"

    response = str(example.get("response") or "")
    question = str(example.get("question") or "")
    evidence_text = _evidence_text(example)
    response_words = _word_count(response)
    min_words, max_words = _response_length_bounds(example)

    if not _question_format_valid(example):
        return "bad_question_format"
    if not example.get("evidence"):
        return "missing_evidence"
    if _LOW_QUALITY_RE.search(question) or _LOW_QUALITY_RE.search(response) or _LOW_QUALITY_RE.search(evidence_text):
        return "low_quality_front_matter"
    if _BAD_RESPONSE_RE.search(response):
        return "assistant_meta_language"
    if response_words < min_words:
        return "response_too_short"
    if response_words > max_words:
        return "response_too_long"
    if _lexical_overlap(example) < 0.05:
        return "low_evidence_overlap"
    if str(example.get("paper_title") or "") == "Unknown":
        return "unknown_paper_title"
    return ""


def heuristic_filter(examples: list[dict], source_path: str) -> tuple[list[ReviewRecord], list[dict]]:
    """Apply structural and lexical heuristics before optional judging."""

    kept: list[ReviewRecord] = []
    rejects: list[dict] = []

    for index, example in enumerate(examples):
        reason = heuristic_reject_reason(example)
        score = _heuristic_quality_score(example)
        record = ReviewRecord(
            example=example,
            source_path=source_path,
            source_index=index,
            heuristic_score=score,
            reject_reason=reason,
        )
        if reason:
            rejects.append(
                {
                    "phase": "heuristic_filter",
                    "reason": reason,
                    "source_path": source_path,
                    "source_index": index,
                    "category": example.get("category"),
                    "paper_id": example.get("paper_id"),
                    "request_id": example.get("request_id"),
                    "question": example.get("question"),
                }
            )
        else:
            kept.append(record)

    return kept, rejects


def deduplicate(records: list[ReviewRecord]) -> tuple[list[ReviewRecord], list[dict]]:
    """Remove exact and near-duplicate examples, keeping the best heuristic candidate."""

    exact_seen: dict[tuple[str, str], ReviewRecord] = {}
    exact_question_seen: dict[tuple[str, str, str], ReviewRecord] = {}
    kept: list[ReviewRecord] = []
    rejects: list[dict] = []

    sorted_records = sorted(
        records,
        key=lambda record: (
            -record.heuristic_score,
            -(record.example.get("word_count") or 0),
            record.example.get("request_id") or "",
        ),
    )

    simhash_buckets: dict[tuple[str, str], list[tuple[int, ReviewRecord]]] = {}

    for record in sorted_records:
        example = record.example
        category = str(example.get("category") or "")
        source_split = str(example.get("source_split") or "")
        q_sig = _signature(str(example.get("question") or ""))
        qr_sig = (
            q_sig,
            _signature(str(example.get("response") or "")),
        )
        exact_key = (source_split, json.dumps(qr_sig))
        question_key = (source_split, category, q_sig)

        if exact_key in exact_seen:
            rejects.append(
                {
                    "phase": "dedup_exact",
                    "reason": "duplicate_question_response",
                    "paper_id": example.get("paper_id"),
                    "category": category,
                    "request_id": example.get("request_id"),
                    "kept_request_id": exact_seen[exact_key].example.get("request_id"),
                    "question": example.get("question"),
                }
            )
            continue

        if question_key in exact_question_seen:
            rejects.append(
                {
                    "phase": "dedup_exact",
                    "reason": "duplicate_question",
                    "paper_id": example.get("paper_id"),
                    "category": category,
                    "request_id": example.get("request_id"),
                    "kept_request_id": exact_question_seen[question_key].example.get("request_id"),
                    "question": example.get("question"),
                }
            )
            continue

        simhash = _simhash_bits(q_sig)
        bucket_key = (source_split, category)
        near_dupe = None
        for other_hash, other_record in simhash_buckets.get(bucket_key, []):
            if _hamming_distance(simhash, other_hash) <= 3:
                near_dupe = other_record
                break

        if near_dupe is not None:
            rejects.append(
                {
                    "phase": "dedup_near",
                    "reason": "near_duplicate_question",
                    "paper_id": example.get("paper_id"),
                    "category": category,
                    "request_id": example.get("request_id"),
                    "kept_request_id": near_dupe.example.get("request_id"),
                    "question": example.get("question"),
                }
            )
            continue

        exact_seen[exact_key] = record
        exact_question_seen[question_key] = record
        simhash_buckets.setdefault(bucket_key, []).append((simhash, record))
        kept.append(record)

    return kept, rejects


def _judge_prompt(example: dict) -> list[dict[str, str]]:
    source_split = str(example.get("source_split") or "")
    judge_focus = (
        "Pay special attention to groundedness, citation behavior, and proper abstention for unanswerable RAG cases."
        if source_split == "rag"
        else "Pay special attention to technical specificity, utility for instruction tuning, and whether the answer is supported by the evidence."
    )
    user_prompt = (
        "Review one synthetic instruction-tuning example.\n\n"
        f"Category: {example.get('category')}\n"
        f"Source split: {source_split}\n"
        f"Task type: {example.get('task_type')}\n"
        f"Difficulty: {example.get('difficulty')}\n\n"
        "Question:\n"
        f"{example.get('question')}\n\n"
        "Response:\n"
        f"{example.get('response')}\n\n"
        "Evidence:\n"
        f"{_evidence_text(example)}\n\n"
        f"{judge_focus}\n\n"
        "Return exactly one JSON object with this shape:\n"
        "{\n"
        '  "keep": true,\n'
        '  "groundedness": 1,\n'
        '  "specificity": 1,\n'
        '  "usefulness": 1,\n'
        '  "reason": "short explanation"\n'
        "}\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a strict dataset quality reviewer for synthetic instruction-tuning data. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _parse_judge_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    match = _JSON_RE.search(cleaned)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def judge_records(
    records: list[ReviewRecord],
    *,
    model: str,
    base_url: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_predict: int = 250,
    checkpoint_path: Path | None = None,
    existing_judge_rows: list[dict] | None = None,
) -> tuple[list[ReviewRecord], list[dict], list[dict]]:
    """Optionally run a local Ollama judge over post-dedup records."""

    client = OllamaClient(base_url=base_url)
    kept: list[ReviewRecord] = []
    rejects: list[dict] = []
    judge_rows: list[dict] = []
    checkpoint_map = load_checkpoint_map(checkpoint_path) if checkpoint_path else {}

    for row in existing_judge_rows or []:
        request_id = str(row.get("request_id") or "").strip()
        if request_id:
            checkpoint_map[request_id] = row

    total = len(records)
    skipped = sum(
        1 for r in records
        if str(r.example.get("request_id") or "").strip() in checkpoint_map
    )
    t_judge = time.monotonic()
    log(f"Phase 7 judge  |  model={model}  |  records={total} ({skipped} already done, {total - skipped} pending)")
    done_count = skipped

    for record in records:
        request_id = str(record.example.get("request_id") or "").strip()
        if request_id and request_id in checkpoint_map:
            existing_row = checkpoint_map[request_id]
            judge_rows.append(existing_row)
            if existing_row.get("error"):
                rejects.append(
                    {
                        "phase": "judge",
                        "reason": f"judge_failed: {existing_row['error']}",
                        "paper_id": record.example.get("paper_id"),
                        "category": record.example.get("category"),
                        "request_id": record.example.get("request_id"),
                        "question": record.example.get("question"),
                    }
                )
                continue
            parsed = existing_row.get("judge_result") or {}
            record.judge_result = parsed
            if not parsed.get("keep", False):
                rejects.append(
                    {
                        "phase": "judge",
                        "reason": parsed.get("reason", "judge_rejected"),
                        "paper_id": record.example.get("paper_id"),
                        "category": record.example.get("category"),
                        "request_id": record.example.get("request_id"),
                        "question": record.example.get("question"),
                    }
                )
                continue
            kept.append(record)
            continue

        done_count += 1

        messages = _judge_prompt(record.example)
        try:
            api_response = client.chat_json(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                num_predict=num_predict,
                think=False,
            )

            raw_text = str((api_response.get("message") or {}).get("content") or "").strip()
            if not raw_text:
                raw_text = str((api_response.get("message") or {}).get("thinking") or "").strip()

            parsed = _parse_judge_json(raw_text)
            record.judge_result = parsed
            judge_row = {
                "request_id": record.example.get("request_id"),
                "paper_id": record.example.get("paper_id"),
                "category": record.example.get("category"),
                "judge_model": model,
                "judge_result": parsed,
                "raw_text": raw_text,
                "error": "",
            }
            judge_rows.append(judge_row)
            if checkpoint_path:
                append_jsonl(checkpoint_path, judge_row)
                if request_id:
                    checkpoint_map[request_id] = judge_row
        except Exception as exc:  # noqa: BLE001
            judge_row = {
                "request_id": record.example.get("request_id"),
                "paper_id": record.example.get("paper_id"),
                "category": record.example.get("category"),
                "judge_model": model,
                "judge_result": {},
                "raw_text": "",
                "error": str(exc),
            }
            judge_rows.append(judge_row)
            if checkpoint_path:
                append_jsonl(checkpoint_path, judge_row)
                if request_id:
                    checkpoint_map[request_id] = judge_row
            rejects.append(
                {
                    "phase": "judge",
                    "reason": f"judge_failed: {exc}",
                    "paper_id": record.example.get("paper_id"),
                    "category": record.example.get("category"),
                    "request_id": record.example.get("request_id"),
                    "question": record.example.get("question"),
                }
            )
            log(f"  [{done_count}/{total}] {request_id} | {record.example.get('category')} | {record.example.get('paper_id')}  ->  FAIL (judge error: {exc})")
            continue

        if not parsed.get("keep", False):
            rejects.append(
                {
                    "phase": "judge",
                    "reason": parsed.get("reason", "judge_rejected"),
                    "paper_id": record.example.get("paper_id"),
                    "category": record.example.get("category"),
                    "request_id": record.example.get("request_id"),
                    "question": record.example.get("question"),
                }
            )
            log(f"  [{done_count}/{total}] {request_id} | {record.example.get('category')} | {record.example.get('paper_id')}  ->  rejected ({parsed.get('reason', 'judge_rejected')})")
            continue

        log(f"  [{done_count}/{total}] {request_id} | {record.example.get('category')} | {record.example.get('paper_id')}  ->  keep")
        kept.append(record)

    log(f"Phase 7 judge done  |  elapsed={fmt_elapsed(t_judge)}  |  kept={len(kept)}  rejected={len(rejects)}")
    return kept, rejects, judge_rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_run_dir(project_root: Path) -> Path:
    """Return a timestamped default run directory for filtered outputs."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root / "data" / "intermediate" / "filtered" / f"run_{timestamp}"


def run_filtering(
    *,
    input_hpn_path: Path,
    input_rag_path: Path,
    output_dir: Path,
    judge_model: str | None = None,
    base_url: str = "http://127.0.0.1:11434",
    resume: bool = False,
) -> dict:
    """Run the full Phase 7 filtering pipeline."""

    t_start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_resume_safe(
        output_dir,
        [
            "filtered_hpn.jsonl",
            "filtered_rag.jsonl",
            "reject_log.jsonl",
            "filtering_report.json",
            "judge_results.jsonl",
        ],
        resume=resume,
    )

    hpn_examples = read_jsonl(input_hpn_path)
    rag_examples = read_jsonl(input_rag_path)
    log(
        f"Phase 7 filtering  |  hpn_in={len(hpn_examples)}  rag_in={len(rag_examples)}"
        + (f"  judge={judge_model}" if judge_model else "  judge=disabled")
    )

    hpn_records, heuristic_hpn_rejects = heuristic_filter(hpn_examples, str(input_hpn_path))
    rag_records, heuristic_rag_rejects = heuristic_filter(rag_examples, str(input_rag_path))
    log(
        f"Heuristic filter done  |  "
        f"hpn: {len(hpn_examples)} -> {len(hpn_records)} kept ({len(heuristic_hpn_rejects)} rejected)  |  "
        f"rag: {len(rag_examples)} -> {len(rag_records)} kept ({len(heuristic_rag_rejects)} rejected)"
    )

    post_heuristic = hpn_records + rag_records
    deduped_records, dedup_rejects = deduplicate(post_heuristic)
    log(f"Deduplication done  |  {len(post_heuristic)} -> {len(deduped_records)} unique ({len(dedup_rejects)} removed)")

    judge_rows: list[dict] = []
    judge_rejects: list[dict] = []
    final_records = deduped_records
    if judge_model:
        judge_results_path = output_dir / "judge_results.jsonl"
        existing_judge_rows = read_jsonl(judge_results_path) if resume else []
        final_records, judge_rejects, judge_rows = judge_records(
            deduped_records,
            model=judge_model,
            base_url=base_url,
            checkpoint_path=judge_results_path,
            existing_judge_rows=existing_judge_rows,
        )

    filtered_hpn = [record.example for record in final_records if record.example.get("source_split") == "hpn"]
    filtered_rag = [record.example for record in final_records if record.example.get("source_split") == "rag"]

    reject_rows = heuristic_hpn_rejects + heuristic_rag_rejects + dedup_rejects + judge_rejects

    _write_jsonl(output_dir / "filtered_hpn.jsonl", filtered_hpn)
    _write_jsonl(output_dir / "filtered_rag.jsonl", filtered_rag)
    _write_jsonl(output_dir / "reject_log.jsonl", reject_rows)
    if judge_rows:
        _write_jsonl(output_dir / "judge_results.jsonl", judge_rows)

    report = {
        "run_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_hpn_count": len(hpn_examples),
        "input_rag_count": len(rag_examples),
        "post_heuristic_count": len(post_heuristic),
        "post_dedup_count": len(deduped_records),
        "final_hpn_count": len(filtered_hpn),
        "final_rag_count": len(filtered_rag),
        "reject_count": len(reject_rows),
        "judge_enabled": bool(judge_model),
        "judge_model": judge_model or "",
        "resume_mode": resume,
        "reject_reasons": dict(Counter(row["reason"] for row in reject_rows)),
        "category_counts": dict(Counter(example.get("category") for example in filtered_hpn + filtered_rag)),
    }
    write_json(output_dir / "filtering_report.json", report)
    log(
        f"Phase 7 filtering done  |  elapsed={fmt_elapsed(t_start)}"
        f"  |  hpn={len(filtered_hpn)}  rag={len(filtered_rag)}  total_rejected={len(reject_rows)}"
    )
    return report
