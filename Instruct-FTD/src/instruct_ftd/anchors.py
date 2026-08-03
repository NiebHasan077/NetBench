"""Phase 8 generic anchor import and normalization."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_GENERIC_SYSTEM = "You are a helpful assistant."


@dataclass(frozen=True)
class AnchorRecord:
    """Canonical generic-anchor example."""

    system: str
    question: str
    response: str
    category: str
    source_split: str = "generic"
    task_type: str = "generic_anchor"
    difficulty: str = ""
    paper_id: str = ""
    paper_title: str = ""
    source_pdf: str = ""
    question_type: str = ""
    prompt_length_bucket: str = "short"
    generation_family: str = "generic_anchor"
    teacher_model: str = ""
    bundle_id: str = ""
    word_count: int = 0
    evidence: list | None = None
    anchor_source: str = ""
    anchor_source_category: str = ""

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        if data["evidence"] is None:
            data["evidence"] = []
        return data


def _word_count(text: str) -> int:
    return len((text or "").split())


def _clean_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def _default_system(system_text: str | None) -> str:
    cleaned = _clean_text(system_text)
    return cleaned if cleaned else DEFAULT_GENERIC_SYSTEM


def _row_signature(record: AnchorRecord) -> tuple[str, str, str]:
    return (
        record.category,
        _clean_text(record.question).lower(),
        _clean_text(record.response).lower(),
    )


def _read_json_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported local anchor file format: {path}")


def _filter_quality(rows: Iterable[AnchorRecord], min_response_words: int) -> list[AnchorRecord]:
    filtered: list[AnchorRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not row.question.strip() or not row.response.strip():
            continue
        if _word_count(row.response) < min_response_words:
            continue
        signature = _row_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        filtered.append(row)
    return filtered


def normalize_openorca_rows(rows: Iterable[dict], max_samples: int | None = None) -> list[AnchorRecord]:
    """Normalize OpenOrca rows into the canonical anchor contract."""

    normalized: list[AnchorRecord] = []
    for row in rows:
        if max_samples is not None and len(normalized) >= max_samples:
            break

        question = _clean_text(row.get("question"))
        response = _clean_text(row.get("response"))
        system = _default_system(row.get("system_prompt") or row.get("system"))
        if not question or not response:
            continue

        normalized.append(
            AnchorRecord(
                system=system,
                question=question,
                response=response,
                category="generic_orca",
                question_type="reasoning",
                word_count=_word_count(response),
                anchor_source="Open-Orca/OpenOrca",
                anchor_source_category="orca_reasoning",
            )
        )
    return normalized


def normalize_dolly_rows(rows: Iterable[dict], max_samples: int | None = None) -> list[AnchorRecord]:
    """Normalize Dolly rows into the canonical anchor contract."""

    normalized: list[AnchorRecord] = []
    for row in rows:
        if max_samples is not None and len(normalized) >= max_samples:
            break

        question = _clean_text(row.get("instruction"))
        response = _clean_text(row.get("response"))
        system = _default_system(row.get("context"))
        source_category = _clean_text(row.get("category")) or "dolly"
        if not question or not response:
            continue

        normalized.append(
            AnchorRecord(
                system=system,
                question=question,
                response=response,
                category="generic_dolly",
                question_type="task",
                word_count=_word_count(response),
                anchor_source="databricks/databricks-dolly-15k",
                anchor_source_category=source_category,
            )
        )
    return normalized


def load_hf_dataset_rows(dataset_name: str, split: str) -> list[dict]:
    """Load rows from Hugging Face datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "The 'datasets' package is required for Hugging Face anchor import."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def import_generic_anchors(
    *,
    orca_local: Path | None = None,
    dolly_local: Path | None = None,
    orca_hf_split: str = "train[:10000]",
    dolly_hf_split: str = "train",
    max_orca_samples: int | None = None,
    max_dolly_samples: int | None = None,
    min_response_words: int = 20,
) -> list[AnchorRecord]:
    """Import and normalize generic anchors from local files or Hugging Face."""

    if orca_local:
        orca_rows = _read_json_rows(orca_local)
    else:
        orca_rows = load_hf_dataset_rows("Open-Orca/OpenOrca", orca_hf_split)

    if dolly_local:
        dolly_rows = _read_json_rows(dolly_local)
    else:
        dolly_rows = load_hf_dataset_rows("databricks/databricks-dolly-15k", dolly_hf_split)

    orca_records = normalize_openorca_rows(orca_rows, max_samples=max_orca_samples)
    dolly_records = normalize_dolly_rows(dolly_rows, max_samples=max_dolly_samples)
    combined = _filter_quality([*orca_records, *dolly_records], min_response_words=min_response_words)
    return combined


def build_anchor_report(records: list[AnchorRecord]) -> dict:
    """Build a compact Phase 8 import report."""

    category_counts = Counter(record.category for record in records)
    source_counts = Counter(record.anchor_source for record in records)
    source_category_counts = Counter(record.anchor_source_category for record in records)
    return {
        "record_count": len(records),
        "category_counts": dict(category_counts),
        "anchor_source_counts": dict(source_counts),
        "anchor_source_category_counts": dict(source_category_counts),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
