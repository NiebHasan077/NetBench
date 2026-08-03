"""JSONL streaming reader/writer with --resume support.

Every stage uses these helpers so a SIGKILL never loses lines and re-runs
skip already-processed inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Union

PathLike = Union[Path, str]


def read_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file. Returns nothing if file is missing."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} invalid JSON: {e}") from e


def write_jsonl(path: PathLike, records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file, overwriting. Returns count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    return n


def append_jsonl(path: PathLike, record: dict[str, Any]) -> None:
    """Append a single record. Flushes after write so SIGKILL is safe."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def existing_ids(path: PathLike, key: str = "id") -> set[str]:
    """IDs already present in the output JSONL — used by stage runners for --resume."""
    return {rec[key] for rec in read_jsonl(path) if key in rec}


def count_lines(path: PathLike) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
