"""Phase 9 final mixing and train/validation splitting."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TARGET_RATIOS = {
    "hpn": 0.6,
    "rag": 0.2,
    "generic": 0.2,
}


@dataclass
class Group:
    """Leakage-control grouping unit for split assignment."""

    key: str
    rows: list[dict]
    hash_value: int


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _signature(example: dict) -> str:
    source_split = str(example.get("source_split") or "")
    if source_split in {"hpn", "rag"}:
        return str(example.get("paper_id") or example.get("request_id") or "")
    question = " ".join(str(example.get("question") or "").lower().split())
    return question


def _group_rows(rows: list[dict], source_split: str) -> list[Group]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        if source_split in {"hpn", "rag"}:
            key = str(row.get("paper_id") or row.get("request_id") or "")
        else:
            key = _signature(row)
        buckets.setdefault(key, []).append(row)

    groups = [
        Group(key=key, rows=group_rows, hash_value=_stable_hash(f"{source_split}:{key}"))
        for key, group_rows in buckets.items()
    ]
    groups.sort(key=lambda group: group.hash_value)
    return groups


def _largest_feasible_total(counts: dict[str, int]) -> int:
    feasible = []
    for split, ratio in TARGET_RATIOS.items():
        available = counts.get(split, 0)
        if ratio <= 0:
            continue
        feasible.append(int(available / ratio))
    return min(feasible) if feasible else 0


def _target_counts(total: int) -> dict[str, int]:
    base = {split: int(total * ratio) for split, ratio in TARGET_RATIOS.items()}
    remainder = total - sum(base.values())
    order = sorted(TARGET_RATIOS.items(), key=lambda item: item[1], reverse=True)
    for index in range(remainder):
        split = order[index % len(order)][0]
        base[split] += 1
    return base


def _select_groups(groups: list[Group], target_count: int) -> list[Group]:
    selected: list[Group] = []
    total = 0
    for group in groups:
        size = len(group.rows)
        if total >= target_count:
            break
        if total + size <= target_count:
            selected.append(group)
            total += size
            continue

        remaining = target_count - total
        if remaining > 0 and not selected:
            selected.append(Group(key=group.key, rows=group.rows[:remaining], hash_value=group.hash_value))
            total += remaining
            break

    return selected


def _split_selected_groups(
    groups: list[Group],
    *,
    val_ratio: float,
) -> tuple[list[dict], list[dict]]:
    if not groups:
        return [], []

    total_count = sum(len(group.rows) for group in groups)
    val_target = int(round(total_count * val_ratio))

    groups = sorted(groups, key=lambda group: group.hash_value)
    val_groups: list[Group] = []
    train_groups: list[Group] = []
    val_count = 0

    for group in groups:
        size = len(group.rows)
        if val_count < val_target and val_count + size <= val_target:
            val_groups.append(group)
            val_count += size
        else:
            train_groups.append(group)

    if not val_groups and groups and val_target > 0:
        moved = train_groups.pop(0)
        val_groups.append(moved)

    train_rows = [row for group in train_groups for row in group.rows]
    val_rows = [row for group in val_groups for row in group.rows]
    return train_rows, val_rows


def _shuffle_rows(rows: list[dict], seed: int) -> list[dict]:
    cloned = list(rows)
    rng = random.Random(seed)
    rng.shuffle(cloned)
    return cloned


def _counts(rows: Iterable[dict], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows))


def _merge_paper_groups(hpn_groups: list[Group], rag_groups: list[Group]) -> list[Group]:
    """Merge selected HPN and RAG groups so a paper stays in one split."""

    merged: dict[str, list[dict]] = {}
    hash_values: dict[str, int] = {}

    for group in hpn_groups + rag_groups:
        key = group.key
        merged.setdefault(key, []).extend(group.rows)
        hash_values.setdefault(key, group.hash_value)

    combined = [
        Group(key=key, rows=rows, hash_value=hash_values[key])
        for key, rows in merged.items()
    ]
    combined.sort(key=lambda group: group.hash_value)
    return combined


def mix_and_split(
    *,
    filtered_hpn: list[dict],
    filtered_rag: list[dict],
    generic_anchors: list[dict],
    output_dir: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
    strict_mix: bool = True,
) -> dict:
    """Mix source splits and build final train/validation datasets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    available = {
        "hpn": len(filtered_hpn),
        "rag": len(filtered_rag),
        "generic": len(generic_anchors),
    }

    if strict_mix and any(count == 0 for count in available.values()):
        missing = [split for split, count in available.items() if count == 0]
        raise ValueError(f"Strict mix requested but missing source splits: {', '.join(missing)}")

    if strict_mix:
        total_examples = _largest_feasible_total(available)
        target_counts = _target_counts(total_examples)
    else:
        total_examples = sum(available.values())
        target_counts = available.copy()

    grouped = {
        "hpn": _group_rows(filtered_hpn, "hpn"),
        "rag": _group_rows(filtered_rag, "rag"),
        "generic": _group_rows(generic_anchors, "generic"),
    }

    selected_groups = {
        split: _select_groups(groups, target_counts.get(split, 0))
        for split, groups in grouped.items()
    }

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    paper_groups = _merge_paper_groups(selected_groups["hpn"], selected_groups["rag"])
    paper_train, paper_val = _split_selected_groups(paper_groups, val_ratio=val_ratio)
    train_rows.extend(paper_train)
    val_rows.extend(paper_val)

    generic_train, generic_val = _split_selected_groups(selected_groups["generic"], val_ratio=val_ratio)
    train_rows.extend(generic_train)
    val_rows.extend(generic_val)

    train_rows = _shuffle_rows(train_rows, seed)
    val_rows = _shuffle_rows(val_rows, seed + 1)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as handle:
        for row in val_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    leak_papers = sorted(
        set(row.get("paper_id") for row in train_rows if row.get("paper_id"))
        & set(row.get("paper_id") for row in val_rows if row.get("paper_id"))
    )

    report = {
        "run_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_mix": strict_mix,
        "available_counts": available,
        "target_counts": target_counts,
        "selected_counts": {
            split: sum(len(group.rows) for group in groups)
            for split, groups in selected_groups.items()
        },
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "train_source_split_counts": _counts(train_rows, "source_split"),
        "validation_source_split_counts": _counts(val_rows, "source_split"),
        "train_category_counts": _counts(train_rows, "category"),
        "validation_category_counts": _counts(val_rows, "category"),
        "train_task_type_counts": _counts(train_rows, "task_type"),
        "validation_task_type_counts": _counts(val_rows, "task_type"),
        "paper_leakage_count": len(leak_papers),
        "paper_leakage_examples": leak_papers[:20],
    }
    (output_dir / "mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
