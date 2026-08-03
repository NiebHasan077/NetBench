"""Stage 08 — Splits + manifest.

Partitions calibrated_questions.jsonl into the four benchmark splits:
  - dev          (random sample of 5a per-paper Qs, target 30)
  - test         (random sample of 5a per-paper Qs, target 200, no dev overlap)
  - synthesis    (all 5b synthesis Qs that survived calibration)
  - adversarial  (all 5c adversarial Qs that survived calibration)

dev + test are stratified by (category, difficulty) using largest-remainder
allocation so both splits look like the full 5a distribution. Synthesis and
adversarial pull from their dedicated streams (id-prefix routing). When a
target is short of supply, we ship what we have and record the shortfall in
the manifest.

Inputs:
  data/calibrated_questions.jsonl

Outputs:
  benchmark/hpn_benchmark_v5_dev.jsonl
  benchmark/hpn_benchmark_v5_test.jsonl
  benchmark/hpn_benchmark_v5_synthesis.jsonl
  benchmark/hpn_benchmark_v5_adversarial.jsonl
  benchmark/manifest.json

Usage:
    .venv/bin/python -m src.stage_08_split
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from src.config import load_config
from src.io_utils import read_jsonl, write_jsonl

logger = logging.getLogger("stage_08")


# ── Manifest schema ──────────────────────────────────────────────────────────

class _SplitCounts(BaseModel):
    dev: int
    test: int
    synthesis: int
    adversarial: int
    total: int


class _SplitTargets(BaseModel):
    dev: int
    test: int
    synthesis: int
    adversarial: int
    total: int


class _SourceCounts(BaseModel):
    per_paper_5a: int
    synthesis_5b: int
    adversarial_5c: int


class _Models(BaseModel):
    card_generator: str
    question_generator: str
    critic: str
    calibration_weak: str
    calibration_mid: str
    calibration_strong: str
    calibration_judge: str
    embedding: str


class Manifest(BaseModel):
    benchmark_version: str
    generated_at: str
    seed: int
    counts: _SplitCounts
    targets: _SplitTargets
    sources: _SourceCounts
    models: _Models
    prompt_versions: dict[str, str]
    notes: list[str]


# ── Stratified sampling ──────────────────────────────────────────────────────

def _largest_remainder(raw: list[float], target: int) -> list[int]:
    """Round floats to ints summing exactly to `target` (largest-remainder method)."""
    floors = [int(x) for x in raw]
    leftover = target - sum(floors)
    if leftover == 0:
        return floors
    # Sort indices by descending fractional remainder
    remainders = sorted(
        range(len(raw)), key=lambda i: -(raw[i] - floors[i])
    )
    for i in remainders[:leftover]:
        floors[i] += 1
    return floors


def stratified_split(
    items: list[dict],
    n_dev: int,
    n_test: int,
    key_fn: Callable[[dict], tuple],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratified sample into dev (size n_dev) and test (size n_test) buckets.

    Caps each stratum to its actual size; if total supply < n_dev+n_test, the
    returned splits are correspondingly smaller (caller must check).
    """
    rng = random.Random(seed)
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        by_stratum[key_fn(it)].append(it)

    strata = sorted(by_stratum.keys(), key=lambda k: (str(k[0]), str(k[1])))
    for k in strata:
        rng.shuffle(by_stratum[k])

    total_supply = len(items)
    target_total = n_dev + n_test

    # Allocate target_total across strata proportional to stratum size
    raw_alloc = [
        len(by_stratum[k]) * target_total / total_supply for k in strata
    ]
    alloc = _largest_remainder(raw_alloc, target_total)
    # Cap each to its stratum's actual size (won't trip if supply >= target)
    for i, k in enumerate(strata):
        alloc[i] = min(alloc[i], len(by_stratum[k]))

    # Within each stratum's allocation, split into dev / test proportionally
    raw_dev = [a * n_dev / target_total for a in alloc]
    dev_per = _largest_remainder(raw_dev, min(n_dev, sum(alloc)))
    # Cap each dev count to its stratum allocation
    for i in range(len(dev_per)):
        if dev_per[i] > alloc[i]:
            dev_per[i] = alloc[i]

    dev: list[dict] = []
    test: list[dict] = []
    for i, k in enumerate(strata):
        bucket = by_stratum[k]
        n_pick = alloc[i]
        n_d = dev_per[i]
        dev.extend(bucket[:n_d])
        test.extend(bucket[n_d:n_pick])
    return dev, test


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--input", default=None, help="calibrated_questions.jsonl")
    parser.add_argument("--benchmark-dir", default=None, help="output directory")
    parser.add_argument("--manifest", default=None, help="manifest.json path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    data_dir = cfg.resolve(cfg.paths.data_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else cfg.resolve(cfg.paths.benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input) if args.input else data_dir / "calibrated_questions.jsonl"
    manifest_path = Path(args.manifest) if args.manifest else benchmark_dir / "manifest.json"

    if not input_path.exists():
        logger.error("input not found: %s", input_path)
        return 1

    version = cfg.splits.benchmark_version
    target_dev = cfg.splits.dev
    target_test = cfg.splits.test
    target_synth = cfg.splits.synthesis
    target_adv = cfg.splits.adversarial
    seed = cfg.seed

    questions = list(read_jsonl(input_path))
    logger.info("loaded %d calibrated questions", len(questions))

    # Partition by id prefix
    pool_5a = [q for q in questions if q["id"].startswith("NB-HPN-")]
    pool_5b = [q for q in questions if q["id"].startswith("NB-SYN-")]
    pool_5c = [q for q in questions if q["id"].startswith("NB-ADV-")]
    leftover = [
        q for q in questions
        if not (q["id"].startswith(("NB-HPN-", "NB-SYN-", "NB-ADV-")))
    ]
    if leftover:
        logger.warning("%d questions have unrecognized ID prefix; ignored", len(leftover))

    logger.info(
        "source pools: 5a=%d  5b=%d  5c=%d", len(pool_5a), len(pool_5b), len(pool_5c)
    )

    # Synthesis & adversarial: take everything (cap at target)
    rng = random.Random(seed)
    rng.shuffle(pool_5b)
    rng.shuffle(pool_5c)
    synthesis_split = pool_5b[:target_synth]
    adversarial_split = pool_5c[:target_adv]

    # Dev+test: stratified by (category, difficulty) from 5a
    if len(pool_5a) < target_dev + target_test:
        logger.warning(
            "5a pool (%d) < dev+test target (%d); will undershoot",
            len(pool_5a), target_dev + target_test,
        )
    dev_split, test_split = stratified_split(
        pool_5a,
        n_dev=target_dev,
        n_test=target_test,
        key_fn=lambda r: (r.get("category", ""), r.get("difficulty", "")),
        seed=seed,
    )

    # Sanity: no ID overlap across splits
    all_ids: list[str] = []
    for split_name, split in [
        ("dev", dev_split),
        ("test", test_split),
        ("synthesis", synthesis_split),
        ("adversarial", adversarial_split),
    ]:
        for q in split:
            all_ids.append(q["id"])
    dups = [i for i, c in Counter(all_ids).items() if c > 1]
    if dups:
        logger.error("FATAL: duplicate IDs across splits: %s", dups[:5])
        return 2

    # Write splits
    paths = {
        "dev": benchmark_dir / f"hpn_benchmark_{version}_dev.jsonl",
        "test": benchmark_dir / f"hpn_benchmark_{version}_test.jsonl",
        "synthesis": benchmark_dir / f"hpn_benchmark_{version}_synthesis.jsonl",
        "adversarial": benchmark_dir / f"hpn_benchmark_{version}_adversarial.jsonl",
    }
    counts = {
        "dev": write_jsonl(paths["dev"], dev_split),
        "test": write_jsonl(paths["test"], test_split),
        "synthesis": write_jsonl(paths["synthesis"], synthesis_split),
        "adversarial": write_jsonl(paths["adversarial"], adversarial_split),
    }
    for name, n in counts.items():
        logger.info("%s: %d → %s", name, n, paths[name].name)

    # Build manifest
    notes: list[str] = []
    if counts["synthesis"] < target_synth:
        notes.append(
            f"Synthesis split short: shipped {counts['synthesis']} vs target {target_synth} "
            f"(only {len(pool_5b)} 5b candidates survived calibration)."
        )
    if counts["adversarial"] < target_adv:
        notes.append(
            f"Adversarial split short: shipped {counts['adversarial']} vs target {target_adv} "
            f"(only {len(pool_5c)} 5c candidates survived calibration)."
        )
    if counts["dev"] < target_dev or counts["test"] < target_test:
        notes.append(
            f"Dev+test short: shipped {counts['dev']}+{counts['test']} "
            f"vs target {target_dev}+{target_test} (5a pool {len(pool_5a)})."
        )

    # Collect prompt versions and model attribution from records
    prompt_versions: dict[str, str] = {}
    for q in questions:
        pv = q.get("prompt_version")
        if pv:
            # Bucket by stream
            if q["id"].startswith("NB-HPN-"):
                prompt_versions.setdefault("per_paper_5a", pv)
            elif q["id"].startswith("NB-SYN-"):
                prompt_versions.setdefault("synthesis_5b", pv)
            elif q["id"].startswith("NB-ADV-"):
                prompt_versions.setdefault("adversarial_5c", pv)
    prompt_versions["calibration"] = "v5.0-calibration"

    manifest = Manifest(
        benchmark_version=version,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seed=seed,
        counts=_SplitCounts(
            dev=counts["dev"],
            test=counts["test"],
            synthesis=counts["synthesis"],
            adversarial=counts["adversarial"],
            total=sum(counts.values()),
        ),
        targets=_SplitTargets(
            dev=target_dev,
            test=target_test,
            synthesis=target_synth,
            adversarial=target_adv,
            total=cfg.splits.total,
        ),
        sources=_SourceCounts(
            per_paper_5a=len(pool_5a),
            synthesis_5b=len(pool_5b),
            adversarial_5c=len(pool_5c),
        ),
        models=_Models(
            card_generator=cfg.models.card_generator,
            question_generator=cfg.models.question_generator,
            critic=cfg.models.critic,
            calibration_weak=cfg.models.calibration.weak,
            calibration_mid=cfg.models.calibration.mid,
            calibration_strong=cfg.models.calibration.strong,
            calibration_judge=cfg.models.calibration.strong,
            embedding=cfg.models.embedding,
        ),
        prompt_versions=prompt_versions,
        notes=notes,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("manifest written to %s", manifest_path)

    # Summary
    total = sum(counts.values())
    logger.info(
        "done: dev=%d  test=%d  synthesis=%d  adversarial=%d  total=%d",
        counts["dev"], counts["test"], counts["synthesis"], counts["adversarial"], total,
    )
    if notes:
        logger.warning("manifest notes: %d", len(notes))
        for n in notes:
            logger.warning("  - %s", n)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
