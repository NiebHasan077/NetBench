#!/usr/bin/env python3
"""Annotate the released benchmark so it is self-describing on its own.

Two facts about HPN-QA live outside the benchmark file and have to be carried
alongside it to be usable:

  * **which items are excluded from scoring** -- the audit in
    ``analysis/excluded_items.csv`` removes 9 items from every reported score,
    but the flag existed only in the answer/judged workbooks, so a reader of
    the JSONL alone could not reproduce the scored set;
  * **where each evidence quote comes from** -- items cite sources as
    ``paper_NNNN``, an index into a corpus that is not redistributed, so the
    verbatim quotes could not be traced to a publication or to the licence
    they are used under.

This stage joins both onto the benchmark. It adds ``excluded_from_scoring``
and ``exclusion_reason`` per item, and ``source_doi`` and ``source_license``
per evidence quote. The join is deterministic, idempotent, and strictly
additive: ``question``, ``reference_answer``, and ``quote`` are never touched,
and re-running reproduces byte-identical output.

Like ``stage_09_enrich``, this refuses to write a partial result: every
evidence ``paper_id`` must resolve and every excluded id must exist in the
benchmark, or it aborts without touching the outputs.

Usage:
    python tools/annotate_benchmark_release.py
    python tools/annotate_benchmark_release.py --dry-run   # report coverage only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "NetBench-LLM" / "data" / "prompts"

# Both files are the same benchmark: the generator writes `_all`, the evaluator
# and analysis code read the short name. They are kept byte-identical.
BENCHMARKS = [PROMPTS / "hpn_benchmark_v5.0_all.jsonl", PROMPTS / "hpn_benchmark_v5.0.jsonl"]
PROVENANCE = PROMPTS / "hpn_benchmark_v5.0_provenance.json"
EXCLUSIONS = ROOT / "analysis" / "excluded_items.csv"
SOURCES = ROOT / "analysis" / "outputs" / "source_papers.csv"

# Keys this stage owns: stripped before re-adding so re-runs are idempotent.
ITEM_KEYS = ("excluded_from_scoring", "exclusion_reason")
EVIDENCE_KEYS = ("source_doi", "source_license")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def annotate(items: list[dict], excl: dict[str, str], src: dict[str, dict]) -> list[dict]:
    out = []
    for q in items:
        rec = {k: v for k, v in q.items() if k not in ITEM_KEYS}
        evidence = []
        for e in rec.get("evidence") or []:
            row = src[e["paper_id"]]
            ev = {k: v for k, v in e.items() if k not in EVIDENCE_KEYS}
            ev["source_doi"] = row["doi"]
            ev["source_license"] = row["license"]
            evidence.append(ev)
        rec["evidence"] = evidence
        rec["excluded_from_scoring"] = q["id"] in excl
        rec["exclusion_reason"] = excl.get(q["id"], "")
        # invariant: never mutate scored text or the verbatim quotes
        assert rec["question"] == q["question"]
        assert rec["reference_answer"] == q["reference_answer"]
        assert [e["quote"] for e in evidence] == [e["quote"] for e in (q.get("evidence") or [])]
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="Report coverage, write nothing")
    a = ap.parse_args(argv)

    items = [json.loads(l) for l in BENCHMARKS[0].read_text(encoding="utf-8").splitlines() if l.strip()]
    excl = {r["id"]: r["reason"] for r in read_csv(EXCLUSIONS)}
    src = {r["paper_id"]: r for r in read_csv(SOURCES)}
    print(f"benchmark: {len(items)} items | exclusions: {len(excl)} | source papers: {len(src)}")

    # --- coverage gates: refuse to emit anything partial ---
    ids = {q["id"] for q in items}
    if stray := sorted(excl.keys() - ids):
        print(f"ABORT: {len(stray)} excluded ids are not in the benchmark, e.g. {stray[:5]}", file=sys.stderr)
        return 2
    cited = {e["paper_id"] for q in items for e in (q.get("evidence") or [])}
    if unresolved := sorted(cited - src.keys()):
        print(f"ABORT: {len(unresolved)} cited papers do not resolve, e.g. {unresolved[:5]}", file=sys.stderr)
        return 2
    print(f"coverage: {len(cited)}/{len(cited)} cited papers resolve; "
          f"{len(excl)}/{len(excl)} exclusions matched")

    annotated = annotate(items, excl, src)
    quotes = sum(len(q["evidence"]) for q in annotated)
    with_doi = sum(1 for q in annotated for e in q["evidence"] if e["source_doi"])
    print(f"annotated: {sum(q['excluded_from_scoring'] for q in annotated)} items flagged excluded; "
          f"{with_doi}/{quotes} quotes carry a DOI")

    if a.dry_run:
        print("dry-run: wrote nothing")
        return 0

    payload = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in annotated)
    for path in BENCHMARKS:
        path.write_text(payload, encoding="utf-8")
        print(f"wrote {len(annotated)} items -> {path.relative_to(ROOT)}")

    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prov["release_annotation"] = {
        "stage": "tools/annotate_benchmark_release.py",
        "item_fields_added": list(ITEM_KEYS),
        "evidence_fields_added": list(EVIDENCE_KEYS),
        "sources": ["analysis/excluded_items.csv", "analysis/outputs/source_papers.csv"],
        "annotated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    PROVENANCE.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"updated provenance sidecar -> {PROVENANCE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
