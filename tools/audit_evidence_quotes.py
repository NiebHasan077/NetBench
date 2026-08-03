#!/usr/bin/env python3
"""Check every benchmark evidence quote against the corpus text it cites.

Stage 6.2 of the generator verifies that each ``evidence.quote`` is an exact
substring of its source passage, and paraphrased quotes from the synthesis and
adversarial streams are repaired before validation. This is the independent
end-of-pipeline check on the shipped benchmark: it re-derives, from the corpus
of record, whether each quote as released is verbatim in the paper it cites.

The corpus is not redistributed, so this cannot be re-run from the released
artifact alone -- like ``recover_source_papers.py``, it produces a committed
CSV that travels with the benchmark as the audit of record. The corpus SHA-256
is checked against the benchmark provenance manifest first, so the result
always refers to the exact input the benchmark was generated from.

Matching normalises whitespace only. Near-misses are reported with a similarity
ratio so a truncation can be told apart from a paraphrase.

Usage:
    python tools/audit_evidence_quotes.py --corpus /path/to/research_corpus_v3.json
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "NetBench-LLM" / "data" / "prompts" / "hpn_benchmark_v5.0_all.jsonl"
PROVENANCE = ROOT / "NetBench-LLM" / "data" / "prompts" / "hpn_benchmark_v5.0_provenance.json"
OUT = ROOT / "analysis" / "outputs" / "evidence_quote_audit.csv"


def norm(s: str) -> str:
    return " ".join(s.split())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def closest(quote: str, text: str) -> tuple[float, str]:
    """Best-aligned window in ``text``, anchored on the quote's first word."""
    if not quote:
        return 0.0, ""
    best = (0.0, "")
    for m in re.finditer(re.escape(quote.split()[0]), text):
        cand = text[m.start(): m.start() + len(quote) + 40]
        r = difflib.SequenceMatcher(None, quote, cand).ratio()
        if r > best[0]:
            best = (r, cand)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--corpus", required=True, help="research_corpus_v3.json (not redistributed)")
    a = ap.parse_args(argv)

    corpus_path = Path(a.corpus)
    expected = json.loads(PROVENANCE.read_text(encoding="utf-8"))["corpus"]["sha256"]
    actual = sha256(corpus_path)
    if actual != expected:
        print(f"ABORT: corpus SHA-256 {actual[:16]}... does not match the manifest "
              f"({expected[:16]}...); the audit would not describe the benchmark of record",
              file=sys.stderr)
        return 2
    print(f"corpus verified against the manifest ({expected[:16]}...)")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    items = [json.loads(l) for l in BENCHMARK.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = {i: norm(r["text"]) for i, r in enumerate(corpus)}

    rows, exact = [], 0
    for it in items:
        for e in it.get("evidence") or []:
            idx = int(e["paper_id"].split("_")[1])
            text = texts.get(idx, "")
            q = norm(e["quote"])
            hit = bool(text) and q in text
            exact += hit
            ratio, window = (1.0, q) if hit else closest(q, text)
            rows.append({
                "item_id": it["id"],
                "split": it["split"],
                "paper_id": e["paper_id"],
                "passage_id": e["passage_id"],
                "verbatim": hit,
                "similarity": round(ratio, 3),
                "quote": e["quote"],
                "closest_corpus_window": "" if hit else window,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{exact}/{len(rows)} quotes are verbatim in the paper they cite "
          f"({100 * exact / len(rows):.1f}%)")
    for r in sorted((r for r in rows if not r["verbatim"]), key=lambda r: -r["similarity"]):
        print(f"  [{r['similarity']:.3f}] {r['item_id']:16s} {r['paper_id']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
