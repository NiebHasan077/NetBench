"""JSON-response generic anchors and append-only final dataset augmentation."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


HERMES_DATASET_NAME = "NousResearch/hermes-function-calling-v1"
DEFAULT_HERMES_JSON_CONFIGS = ("json_mode_singleturn", "json_mode_agentic")
DEFAULT_GENERIC_SYSTEM = "You are a helpful assistant."


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _word_count(text: str) -> int:
    return len((text or "").split())


def _prompt_length_bucket(question: str) -> str:
    words = _word_count(question)
    if words < 80:
        return "short"
    if words < 220:
        return "medium"
    return "long"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_local_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [dict(row) for row in data]
    raise ValueError(f"Unsupported local Hermes input format: {path}")


def _message_role(message: dict) -> str:
    return _clean_text(message.get("from") or message.get("role")).lower()


def _message_value(message: dict) -> str:
    return str(message.get("value") or message.get("content") or "").strip()


def _conversation_messages(row: dict) -> list[dict]:
    value = row.get("conversations") or row.get("messages") or []
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, dict)]


def _extract_turns(row: dict) -> tuple[str, str, str] | None:
    system = ""
    user_messages: list[str] = []
    assistant_messages: list[str] = []

    for message in _conversation_messages(row):
        role = _message_role(message)
        value = _message_value(message)
        if not value:
            continue
        if role == "system":
            system = value
        elif role in {"human", "user"}:
            user_messages.append(value)
        elif role in {"gpt", "assistant", "model"}:
            assistant_messages.append(value)

    if not user_messages or not assistant_messages:
        return None

    return system or DEFAULT_GENERIC_SYSTEM, user_messages[0], assistant_messages[-1]


def normalize_json_response(response: str) -> str | None:
    """Return a canonical JSON string if response is exactly JSON-like."""

    candidate = response.strip()
    match = _FENCE_RE.match(candidate)
    if match:
        candidate = match.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return json.dumps(parsed, ensure_ascii=False, indent=2)


def normalize_hermes_json_rows(
    rows: Iterable[dict],
    *,
    source_category: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Normalize Hermes ShareGPT-style JSON-mode rows into generic anchors."""

    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        if max_samples is not None and len(normalized) >= max_samples:
            break

        turns = _extract_turns(row)
        if not turns:
            continue

        system, question, raw_response = turns
        question = _clean_text(question)
        response = normalize_json_response(raw_response)
        if not question or response is None:
            continue

        signature = (question.lower(), response)
        if signature in seen:
            continue
        seen.add(signature)

        normalized.append(
            {
                "system": system,
                "question": question,
                "response": response,
                "category": "generic_json",
                "source_split": "generic",
                "task_type": "json_response",
                "difficulty": "",
                "paper_id": "",
                "paper_title": "",
                "source_pdf": "",
                "question_type": "structured_output",
                "prompt_length_bucket": _prompt_length_bucket(question),
                "generation_family": "generic_anchor",
                "teacher_model": "",
                "bundle_id": str(row.get("id") or ""),
                "word_count": _word_count(response),
                "evidence": [],
                "anchor_source": HERMES_DATASET_NAME,
                "anchor_source_category": source_category,
                "hermes_category": str(row.get("category") or ""),
                "hermes_subcategory": str(row.get("subcategory") or ""),
                "hermes_task": str(row.get("task") or ""),
            }
        )

    return normalized


def load_hermes_json_anchors(
    *,
    configs: Sequence[str] = DEFAULT_HERMES_JSON_CONFIGS,
    split: str = "train",
    target_count: int | None = None,
    max_samples_per_config: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load and normalize Hermes JSON-mode anchors from Hugging Face datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "The 'datasets' package is required to import Hermes JSON anchors."
        ) from exc

    rows: list[dict] = []
    for config in configs:
        dataset = load_dataset(HERMES_DATASET_NAME, config, split=split)
        rows.extend(
            normalize_hermes_json_rows(
                (dict(row) for row in dataset),
                source_category=config,
                max_samples=max_samples_per_config,
            )
        )

    rows = select_rows(rows, target_count, seed=seed)
    return rows


def load_local_hermes_json_anchors(
    path: Path,
    *,
    source_category: str = "local_hermes_json",
    target_count: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load and normalize local Hermes-shaped JSON/JSONL rows."""

    rows = normalize_hermes_json_rows(_load_local_rows(path), source_category=source_category)
    return select_rows(rows, target_count, seed=seed)


def select_rows(rows: Sequence[dict], count: int | None, *, seed: int) -> list[dict]:
    """Select up to count rows deterministically."""

    selected = list(rows)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if count is None:
        return selected
    if count < 0:
        raise ValueError("count must be non-negative")
    if len(selected) < count:
        raise ValueError(f"Requested {count} rows but only {len(selected)} are available")
    return selected[:count]


def validate_json_anchor_rows(rows: Iterable[dict]) -> list[dict]:
    """Return rows whose response is valid JSON."""

    valid: list[dict] = []
    for row in rows:
        response = str(row.get("response") or "")
        if normalize_json_response(response) is None:
            continue
        valid.append(row)
    return valid


def _counts(rows: Iterable[dict], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows))


def _signature(row: dict) -> tuple[str, str]:
    return (_clean_text(row.get("question")).lower(), _clean_text(row.get("response")).lower())


def append_json_anchors_to_final(
    *,
    input_train: Path,
    input_validation: Path,
    json_anchors: Path,
    output_dir: Path,
    count: int,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """Append exactly count JSON anchors to an existing final dataset."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1")

    train_rows = _read_jsonl(input_train)
    val_rows = _read_jsonl(input_validation)
    anchor_rows = validate_json_anchor_rows(_read_jsonl(json_anchors))

    existing_signatures = {_signature(row) for row in [*train_rows, *val_rows]}
    deduped_anchors = [
        row
        for row in anchor_rows
        if _signature(row) not in existing_signatures
    ]
    selected_anchors = select_rows(deduped_anchors, count, seed=seed)

    val_count = int(round(count * val_ratio))
    selected_for_val = selected_anchors[:val_count]
    selected_for_train = selected_anchors[val_count:]

    output_train_rows = [*train_rows, *selected_for_train]
    output_val_rows = [*val_rows, *selected_for_val]

    train_rng = random.Random(seed)
    val_rng = random.Random(seed + 1)
    train_rng.shuffle(output_train_rows)
    val_rng.shuffle(output_val_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", output_train_rows)
    write_jsonl(output_dir / "validation.jsonl", output_val_rows)

    report = {
        "run_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "append_json_anchors",
        "input_train_count": len(train_rows),
        "input_validation_count": len(val_rows),
        "input_total_count": len(train_rows) + len(val_rows),
        "available_json_anchor_count": len(anchor_rows),
        "available_json_anchor_count_after_existing_dedup": len(deduped_anchors),
        "requested_json_count": count,
        "selected_json_count": len(selected_anchors),
        "train_json_count": len(selected_for_train),
        "validation_json_count": len(selected_for_val),
        "output_train_count": len(output_train_rows),
        "output_validation_count": len(output_val_rows),
        "output_total_count": len(output_train_rows) + len(output_val_rows),
        "train_category_counts": _counts(output_train_rows, "category"),
        "validation_category_counts": _counts(output_val_rows, "category"),
        "train_task_type_counts": _counts(output_train_rows, "task_type"),
        "validation_task_type_counts": _counts(output_val_rows, "task_type"),
    }
    (output_dir / "json_append_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
