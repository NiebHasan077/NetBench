"""Stage 09 — Provenance enrichment (offline join, no model calls).

Adds per-item provenance to the benchmark by joining artifacts the pipeline
already produced, and writes a benchmark-level provenance sidecar:

  calibration ladder  <- data/_logs/stage_07.jsonl        (key: item_id)
  validation results  <- data/_logs/stage_06.jsonl        (key: item_id)
  sidecar provenance  <- benchmark/manifest.json + corpus SHA-256

The join is deterministic, idempotent, and strictly additive: ``question`` and
``reference_answer`` are never modified, only new keys are added, so re-running
reproduces byte-identical output and downstream scoring is unaffected.

This stage refuses to write a partial result: every shipped item must join on
all three sources, or it aborts (non-zero exit) without touching outputs.

Inputs:
  benchmark/hpn_benchmark_v5.0_all.jsonl
  data/_logs/stage_07.jsonl, data/_logs/stage_06.jsonl
  benchmark/manifest.json

Outputs:
  benchmark/hpn_benchmark_v5.0_all.jsonl          (enriched, in place)
  benchmark/hpn_benchmark_v5.0_provenance.json

Usage:
  .venv/bin/python -m src.stage_09_enrich
  .venv/bin/python -m src.stage_09_enrich --dry-run   # report coverage, write nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from src.config import load_config
from src.io_utils import read_jsonl, write_jsonl

logger = logging.getLogger("stage_09")

BENCHMARK_FILE = "hpn_benchmark_v5.0_all.jsonl"
PROVENANCE_FILE = "hpn_benchmark_v5.0_provenance.json"
# Keys this stage owns: stripped before re-adding so re-runs are idempotent.
# (cluster_id was emitted by an earlier revision; kept here so stale values are cleaned.)
ENRICH_KEYS = ("calibration", "validation", "cluster_id")


def _calibration_map(log_path: Path) -> dict[str, dict]:
    """item_id -> {ladder:{weak,mid,strong}, difficulty_original}. One row per item."""
    out: dict[str, dict] = {}
    for r in read_jsonl(log_path):
        out[r["item_id"]] = {
            "ladder": {
                "weak": bool(r["weak_correct"]),
                "mid": bool(r["mid_correct"]),
                "strong": bool(r["strong_correct"]),
            },
            "difficulty_original": r.get("difficulty_original", ""),
        }
    return out


def _validation_map(log_path: Path) -> dict[str, dict]:
    """item_id -> {checks_passed:[...], issues:[{check,reason}]} aggregated over rows."""
    rows: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
    for r in read_jsonl(log_path):
        rows[r["item_id"]].append((r["check"], bool(r["passed"]), r.get("reason", "")))
    out: dict[str, dict] = {}
    for item_id, recs in rows.items():
        out[item_id] = {
            "checks_passed": sorted({c for c, ok, _ in recs if ok}),
            "issues": [{"check": c, "reason": reason} for c, ok, reason in recs if not ok],
        }
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Report coverage, write nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    data_dir = cfg.resolve(cfg.paths.data_dir)
    bench_dir = cfg.resolve(cfg.paths.benchmark_dir)
    bench_path = bench_dir / BENCHMARK_FILE
    if not bench_path.exists():
        logger.error("benchmark not found: %s", bench_path)
        return 1

    items = list(read_jsonl(bench_path))
    logger.info("loaded %d benchmark items", len(items))

    calib = _calibration_map(data_dir / "_logs" / "stage_07.jsonl")
    valid = _validation_map(data_dir / "_logs" / "stage_06.jsonl")

    # --- coverage gate: refuse to emit anything partial ---
    miss_cal = [q["id"] for q in items if q["id"] not in calib]
    miss_val = [q["id"] for q in items if q["id"] not in valid]
    for name, miss in (("calibration", miss_cal), ("validation", miss_val)):
        logger.info("%s coverage: %d/%d", name, len(items) - len(miss), len(items))
        if miss:
            logger.error("ABORT: %d items lack %s, e.g. %s", len(miss), name, miss[:5])
            return 2

    enriched = []
    for q in items:
        out = {k: v for k, v in q.items() if k not in ENRICH_KEYS}  # drop stale enrichment
        out["calibration"] = calib[q["id"]]
        out["validation"] = valid[q["id"]]
        # invariant: never mutate the scored text
        assert out["question"] == q["question"] and out["reference_answer"] == q["reference_answer"]
        enriched.append(out)

    # --- benchmark-level provenance sidecar ---
    manifest = json.loads((bench_dir / "manifest.json").read_text())
    corpus_path = cfg.resolve(cfg.paths.corpus)
    provenance = {
        "benchmark_version": manifest.get("benchmark_version"),
        "generated_at": manifest.get("generated_at"),
        "seed": manifest.get("seed"),
        "counts": manifest.get("counts"),
        "models": manifest.get("models"),
        "prompt_versions": manifest.get("prompt_versions"),
        "corpus": {
            "path": cfg.paths.corpus,
            "sha256": _sha256(corpus_path) if corpus_path.exists() else None,
        },
        "enrichment": {
            "stage": "stage_09_enrich",
            "fields_added": ["calibration", "validation"],
            "enriched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    if args.dry_run:
        logger.info("dry-run: would enrich %d items + write %s", len(enriched), PROVENANCE_FILE)
        return 0

    n = write_jsonl(bench_path, enriched)
    (bench_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info("wrote %d enriched items -> %s", n, bench_path)
    logger.info("wrote provenance sidecar -> %s", bench_dir / PROVENANCE_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
