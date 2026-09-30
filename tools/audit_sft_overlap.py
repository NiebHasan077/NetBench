#!/usr/bin/env python3
"""Audit the final instruction (SFT) data against the scored benchmark items.

The benchmark and the SFT set are generated from the same corpus by separate
pipelines, and benchmark outputs are never inputs to SFT generation. That keeps
the benchmark out of the training data by construction, but it cannot rule out
the two pipelines independently writing the same question, or an SFT answer
restating a reference answer. This measures both, and counts how many benchmark
source papers also contributed SFT examples.

The SFT records are not redistributed, so -- like ``audit_evidence_quotes.py``
-- this produces a committed CSV that is the audit of record. Both JSONL files
are checked against the SHA-256 recorded in ``DATA_PROVENANCE.md`` first, so
the result always refers to the exact dataset the models were trained on.

Each scored item (the 233 not ``excluded_from_scoring``) is compared with the
generated SFT examples; the OpenOrca/Dolly anchors are not from the corpus and
are left out.

  shared paper   one of the item's source papers also has SFT examples. The
                 two pipelines number corpus papers differently, so papers are
                 matched on the first 40 characters of their normalised text
                 (benchmark: ``source_papers.csv`` corpus_title_guess; SFT:
                 paper_title). A paper whose text has fewer usable characters
                 stays unmatched, so the count is a lower bound.
  exact dup      the normalised question equals an SFT question.
  8-gram         the share of the item's word 8-grams -- question and reference
                 answer separately -- found anywhere in SFT questions or
                 responses. Same n as the generator's memorisation-leak check.

Usage:
    python tools/audit_sft_overlap.py --sft /path/to/Instruct-FTD/v3_run_plus_json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "NetBench-LLM" / "data" / "prompts" / "hpn_benchmark_v5.0.jsonl"
SOURCE_PAPERS = ROOT / "analysis" / "outputs" / "source_papers.csv"
OUT = ROOT / "analysis" / "outputs" / "sft_overlap_audit.csv"

# DATA_PROVENANCE.md, "Recovered Local Evidence".
EXPECTED_SHA256 = {
    "train.jsonl": "a7d01645b3f35c78a74da8f944ecbbf371bf7d805bb8e0e57d384295feafa2e6",
    "validation.jsonl": "5abd7b48d08a59e1cdf98cb075ed01d9c27f7c6006bec72d39d9e74e535b08bf",
}
PREFIX = 40
NGRAM = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split()


def grams(s: str) -> set[str]:
    t = words(s)
    return {" ".join(t[i:i + NGRAM]) for i in range(len(t) - NGRAM + 1)}


def prefix(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())[:PREFIX]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sft", required=True,
                    help="directory holding the final train.jsonl + validation.jsonl (not redistributed)")
    args = ap.parse_args(argv)

    sft_dir = Path(args.sft)
    sft = []
    for name, expected in EXPECTED_SHA256.items():
        path = sft_dir / name
        actual = sha256(path)
        if actual != expected:
            print(f"{path}: SHA-256 {actual} does not match DATA_PROVENANCE.md ({expected})",
                  file=sys.stderr)
            return 1
        with path.open(encoding="utf-8") as f:
            sft.extend(json.loads(line) for line in f)
    generated = [r for r in sft if r.get("source_split") in ("hpn", "rag")]

    sft_prefixes = {p for p in (prefix(r.get("paper_title")) for r in generated) if len(p) == PREFIX}
    sft_questions = {" ".join(words(r["question"])) for r in generated}
    sft_grams: set[str] = set()
    for r in generated:
        sft_grams |= grams(r["question"]) | grams(r["response"])

    with SOURCE_PAPERS.open(encoding="utf-8") as f:
        paper_prefix = {row["paper_id"]: prefix(row["corpus_title_guess"]) for row in csv.DictReader(f)}

    def containment(s: str) -> float:
        g = grams(s)
        return len(g & sft_grams) / len(g) if g else 0.0

    rows = []
    with BENCHMARK.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["excluded_from_scoring"]:
                continue
            papers = item["source_papers"]
            shared = any(len(paper_prefix.get(p, "")) == PREFIX and paper_prefix[p] in sft_prefixes
                         for p in papers)
            rows.append({
                "id": item["id"],
                "split": item["split"],
                "source_papers": ";".join(papers),
                "shared_source_paper": int(shared),
                "exact_question_dup": int(" ".join(words(item["question"])) in sft_questions),
                "question_8gram_containment": round(containment(item["question"]), 3),
                "reference_8gram_containment": round(containment(item["reference_answer"]), 3),
            })
    rows.sort(key=lambda r: r["id"])

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    papers = {p for r in rows for p in r["source_papers"].split(";")}
    matchable = {p for p in papers if len(paper_prefix.get(p, "")) == PREFIX}
    shared_papers = {p for p in matchable if paper_prefix[p] in sft_prefixes}
    q = [r["question_8gram_containment"] for r in rows]
    a = [r["reference_8gram_containment"] for r in rows]
    print(f"SFT: {len(sft)} rows, {len(generated)} generated (hash-verified)")
    print(f"scored items: {len(rows)} | source papers: {len(papers)} "
          f"({len(matchable)} matchable) | also SFT sources: {len(shared_papers)} "
          f"| items citing one: {sum(r['shared_source_paper'] for r in rows)}")
    print(f"exact duplicate questions: {sum(r['exact_question_dup'] for r in rows)}")
    print(f"question 8-gram containment: max {max(q):.3f}, items > 0: {sum(x > 0 for x in q)}")
    print(f"reference 8-gram containment: max {max(a):.3f}, items > 0: {sum(x > 0 for x in a)}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
