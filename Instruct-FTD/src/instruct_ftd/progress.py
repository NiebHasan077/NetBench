"""Helpers for append-only JSONL checkpoints, resume-safe runs, and progress logging."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_resume_safe(output_dir: Path, artifact_names: Iterable[str], *, resume: bool) -> None:
    if resume:
        return
    existing = [name for name in artifact_names if (output_dir / name).exists()]
    if existing:
        raise ValueError(
            "Output directory already contains artifacts. "
            f"Use --resume to continue or choose a new output directory. Existing: {', '.join(sorted(existing))}"
        )


def load_checkpoint_map(path: Path, *, key_field: str = "request_id") -> dict[str, dict]:
    checkpoint: dict[str, dict] = {}
    for row in read_jsonl(path):
        key = str(row.get(key_field) or "").strip()
        if key:
            checkpoint[key] = row
    return checkpoint


def log(msg: str) -> None:
    """Print a timestamped status line to stderr."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def fmt_elapsed(start: float) -> str:
    """Format elapsed seconds since *start* as a human-readable string."""
    secs = time.monotonic() - start
    if secs < 60:
        return f"{secs:.1f}s"
    mins, secs = divmod(int(secs), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"
