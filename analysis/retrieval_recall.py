"""R5: gold-evidence retrieval recall, resolved from the committed answer logs.

Phase 12, answering the reviewer's "the RAG result is a best-case retrieval
experiment... the paper does not report gold-evidence recall@4 or recall@20".
The charge is that because the index contains the passages the benchmark was
generated from, retrieval operates under near-oracle coverage -- and we never
measured whether it actually finds them.

An earlier reading of this thought `retrieved_chunks` logged the first ~40
characters of chunk *text*, which would have made resolution a fuzzy-matching
problem over extraction boilerplate. It does not. `evaluate_rag.py:499-503`
builds each label as

    paper_title[:40] + "…"   (or the whole title if it is shorter)
    + " §" + section_number + " " + section_heading

and every one of those is a field on the chunk. So the same label can be built
for all 33,503 cached chunks and resolution becomes an exact dictionary lookup.
What remains is genuine collision: distinct papers whose titles agree in the
first 40 characters -- common here, because the PDF extractor sometimes puts
licence lines or arXiv stamps in the title field. Collisions are not guessed
away. They are carried through as a set of candidate papers, which turns the
answer into a **bounded interval** rather than a point:

  lower bound   a question counts as a hit only if some retrieved slot resolves
                *unambiguously* to one of its gold papers
  upper bound   a slot counts if its candidate set contains a gold paper at all

The true recall lies between, and the width of the interval is itself the
report on how much the logging cost us.

Two facts make the chain trustworthy, both measured rather than assumed:

  * `paper_NNNN` maps to `corpus_doc_NNNN`. Verified against verbatim evidence:
    254 of 291 gold quotes (87.3%) appear in the identity-mapped document, and
    of the 37 that do not, exactly **one** appears in any other gold document.
    The residue is extraction noise between two PDF cleaners, not misalignment
    -- a wrong mapping would scatter quotes across documents, not lose them.
  * retrieval is deterministic and generator-independent: all RAG runs return
    byte-identical `retrieved_chunks` per question, so pooling runs adds
    coverage of the roster, not independent retrieval samples.

Reads only committed artifacts plus the local chunk cache. Nothing is retrieved,
generated, or judged here; `retrieval_rerun.py` is the exact measurement this
one brackets.

    analysis/.venv/bin/python analysis/retrieval_recall.py
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import pickle
import re
import sys
import types

import pandas as pd

CHUNKS = "NetBench-RAG/data/chunks_cache.pkl"
BENCHMARK = "NetBench-LLM/data/prompts/hpn_benchmark_v5.0.jsonl"
ANSWER_GLOBS = ("NetBench-LLM/outputs/**/hpn_answers_RAG*.xlsx",
                "NetBench-RAG/outputs/answers/*.xlsx")


def load_chunks(path: str) -> list:
    """Unpickle the chunk cache without the NetBench-RAG package on the path.

    The cache holds `src.chunker.Chunk` instances. Pickle only needs a class
    object to attach each `__dict__` to, so a stub with the right qualified name
    is enough and keeps this script runnable from the analysis venv, which is
    where every other number in the paper is produced.
    """
    if "src.chunker" not in sys.modules:
        pkg = sys.modules.setdefault("src", types.ModuleType("src"))
        mod = types.ModuleType("src.chunker")
        mod.Chunk = type("Chunk", (), {})
        pkg.chunker = mod
        sys.modules["src.chunker"] = mod
    with open(path, "rb") as fh:
        return pickle.load(fh)


def chunk_label(title: str, number: str, heading: str) -> str:
    """Rebuild evaluate_rag.py's label byte for byte (evaluate_rag.py:499-503)."""
    t = title[:40] + "…" if len(title) > 40 else title
    sec = f"§{number} {heading}".strip() if number else f"§{heading}"
    return f"{t} {sec}"


def label_index(chunks: list) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for c in chunks:
        lab = chunk_label(str(c.paper_title), str(c.section_number),
                          str(c.section_heading))
        idx.setdefault(lab, set()).add(paper_of(c.source_file))
    return idx


def paper_of(source_file: str) -> str:
    """corpus_doc_NNNN -> paper_NNNN (validated against verbatim gold quotes)."""
    return "paper_" + str(source_file).rsplit("_", 1)[-1]


def gold_papers(path: str) -> dict[str, set[str]]:
    """question_id -> the papers its evidence was quoted from."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("excluded_from_scoring"):
                continue
            gold = {e["paper_id"] for e in (r.get("evidence") or [])}
            if not gold:
                ps = r["source_papers"]
                gold = set(ast.literal_eval(ps) if isinstance(ps, str) else ps or [])
            out[r["id"]] = gold
    return out


def read_runs() -> dict[str, pd.DataFrame]:
    runs, seen = {}, set()
    for pat in ANSWER_GLOBS:
        for path in sorted(glob.glob(pat, recursive=True)):
            name = os.path.basename(path)
            if name in seen:
                continue
            d = pd.read_excel(path, sheet_name="Answers")
            if "retrieved_chunks" not in d:
                continue
            seen.add(name)
            runs[name] = d[["id", "difficulty", "retrieved_chunks"]]
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--provenance",
                    default="NetBench-RAG/outputs/retrieval_provenance.csv",
                    help="per-rank log from NetBench-RAG/evaluation/"
                         "retrieval_only.py; absent -> bounds only")
    ap.add_argument("--out", default="analysis/outputs")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    idx = label_index(chunks)
    gold = gold_papers(BENCHMARK)
    runs = read_runs()
    print(f"{len(idx):,} distinct chunk labels over the cache; "
          f"{len(gold)} scored questions; {len(runs)} RAG runs\n")

    rows, slots_tot, slots_uni, slots_unk = [], 0, 0, 0
    for name, d in runs.items():
        lo = hi = n = 0
        for _, r in d.iterrows():
            g = gold.get(r["id"])
            if not g or not isinstance(r["retrieved_chunks"], str):
                continue
            n += 1
            hit_lo = hit_hi = False
            for lab in r["retrieved_chunks"].split(" | "):
                cand = idx.get(lab.strip())
                slots_tot += 1
                if cand is None:
                    slots_unk += 1
                    continue
                if len(cand) == 1:
                    slots_uni += 1
                    hit_lo |= bool(cand & g)
                hit_hi |= bool(cand & g)
            lo += hit_lo
            hi += hit_hi
        rows.append(dict(run=name, n=n,
                         recall_at_4_lower=round(lo / n, 4) if n else None,
                         recall_at_4_upper=round(hi / n, 4) if n else None))
        print(f"{name[:52]:52s} n={n:3d}  recall@4 "
              f"{100 * lo / n:5.1f}%--{100 * hi / n:5.1f}%")

    df = pd.DataFrame(rows)
    lo_p, hi_p = df.recall_at_4_lower.mean(), df.recall_at_4_upper.mean()
    print(f"\nretrieved slots resolved: {slots_tot:,} total, "
          f"{100 * slots_uni / slots_tot:.1f}% to exactly one paper, "
          f"{100 * (slots_tot - slots_uni - slots_unk) / slots_tot:.1f}% ambiguous, "
          f"{100 * slots_unk / slots_tot:.1f}% not in the cache")
    print(f"pooled recall@4: {100 * lo_p:.1f}%--{100 * hi_p:.1f}% "
          f"(identical across runs by construction: retrieval does not depend "
          f"on the generator)")

    exact = exact_metrics(args.provenance, gold)
    if exact:
        print(f"\nexact, from the retrieval rerun ({args.provenance}):")
        for k in ("recall_at_1", "recall_at_4", "recall_at_10", "recall_at_20"):
            print(f"  {k:13s} {100 * exact[k]:5.1f}%")
        print(f"  {'MRR@20':13s} {exact['mrr_at_20']:5.3f}   "
              f"median gold rank {exact['median_gold_rank']:.0f} of "
              f"{exact['depth']:.0f}, over {exact['n']:.0f} questions")
        inside = lo_p - 1e-9 <= exact["recall_at_4"] <= hi_p + 1e-9
        print(f"  exact recall@4 falls {'INSIDE' if inside else 'OUTSIDE'} the "
              f"bracket the logs alone support "
              f"({100 * lo_p:.1f}--{100 * hi_p:.1f}%), which is the check on "
              f"that method")
        rows.append(dict(run="EXACT (retrieval rerun)", n=exact["n"],
                         recall_at_4_lower=round(exact["recall_at_4"], 4),
                         recall_at_4_upper=round(exact["recall_at_4"], 4)))
        df = pd.DataFrame(rows)

        eva = evidence_metrics(args.provenance, chunks, BENCHMARK)
        print("\npassage level -- did a retrieved chunk contain the quote?")
        for k in (1, 4, 10, 20):
            print(f"  evidence_recall_at_{k:<2d} {100 * eva[f'evidence_recall_at_{k}']:5.1f}%")
        print(f"  {'MRR@20':19s} {eva['evidence_mrr_at_20']:5.3f}   "
              f"median rank {eva['median_evidence_rank']:.0f}, over "
              f"{eva['n_scored']:.0f} questions "
              f"({eva['n_excluded_unmatchable']:.0f} excluded: quote not "
              f"verbatim-findable in its own source document)")
        pd.DataFrame([exact | eva]).to_csv(
            os.path.join(args.out, "retrieval_recall_exact.csv"), index=False)

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "retrieval_recall_bounds.csv"), index=False)
    print(f"\nwrote {args.out}/retrieval_recall_bounds.csv"
          + (" and retrieval_recall_exact.csv" if exact else ""))


def evidence_metrics(path: str, chunks: list, bench: str) -> dict[str, float] | None:
    """Passage-level recall: did a retrieved chunk contain the quoted evidence?

    Paper-level recall is a proxy -- the right paper reaching the generator does
    not mean the right paragraph did -- and the reviewer asked for gold-evidence
    recall, so measure the evidence. The benchmark's `passage_id` cannot be used
    directly: it indexes the *generator's* segmentation of the paper, which is
    not the retriever's chunking. The quote itself is the common currency, so a
    hit is a retrieved chunk whose normalized text contains the item's
    normalized quote.

    Questions whose quote is not verbatim-findable anywhere in its own source
    document are **excluded, not counted as misses**: the two pipelines cleaned
    the same PDFs differently (ligatures, hyphenation, column order), and
    failing to match there is a fact about the extractors, not about retrieval.
    That exclusion is reported, since it bounds what this number covers.
    """
    if not os.path.exists(path):
        return None
    norm = lambda s: re.sub(r"\W+", " ", str(s)).lower().strip()   # noqa: E731
    d = pd.read_csv(path)
    d["chunk_index"] = d.chunk_index.astype(str)
    want = set(zip(d.source_file, d.chunk_index))
    text = {(c.source_file, str(c.chunk_index)): norm(c.text)
            for c in chunks if (c.source_file, str(c.chunk_index)) in want}
    doc = {}
    for c in chunks:
        doc.setdefault(c.source_file, []).append(norm(c.text))
    doc = {k: " ".join(v) for k, v in doc.items()}

    quotes: dict[str, list[tuple[str, str]]] = {}
    with open(bench, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("excluded_from_scoring"):
                continue
            quotes[r["id"]] = [(e["paper_id"], norm(e["quote"])[:180])
                               for e in (r.get("evidence") or [])]

    ranks, skipped = [], 0
    for qid, g in d.groupby("question_id"):
        if qid not in quotes:
            continue          # excluded from scoring by the benchmark, not by us
        qs = [(p, q) for p, q in quotes[qid]
              if q and q in doc.get("corpus_doc_" + p.split("_")[1], "")]
        if not qs:
            skipped += 1
            continue
        rank = 0
        for _, row in g.sort_values("rank").iterrows():
            if any(q in text.get((row.source_file, row.chunk_index), "")
                   for _, q in qs):
                rank = int(row["rank"])
                break
        ranks.append(rank)
    s = pd.Series(ranks)
    found = s[s > 0]
    depth = int(d["rank"].max())
    return dict(
        n_scored=float(len(s)), n_excluded_unmatchable=float(skipped),
        depth=float(depth),
        **{f"evidence_recall_at_{k}": float(((s > 0) & (s <= k)).mean())
           for k in (1, 4, 10, 20) if k <= depth},
        evidence_mrr_at_20=float((1.0 / s.where(s > 0)).fillna(0.0).mean()),
        median_evidence_rank=float(found.median()) if len(found) else float("nan"),
    )


def exact_metrics(path: str, gold: dict[str, set[str]]) -> dict[str, float] | None:
    """recall@k, MRR and gold rank from the rerun's per-rank provenance log.

    A question counts at k if any of its top-k chunks comes from a paper its
    evidence was quoted from. Questions whose gold papers are absent from the
    benchmark's scored set are skipped, not scored as misses.
    """
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    depth = int(d["rank"].max())
    ranks = []
    for qid, g in d.groupby("question_id"):
        want = gold.get(qid)
        if not want:
            continue
        hits = g.loc[g.paper_id.isin(want), "rank"]
        ranks.append(int(hits.min()) if len(hits) else 0)   # 0 = never retrieved
    ranks = pd.Series(ranks)
    found = ranks[ranks > 0]
    return dict(
        n=float(len(ranks)), depth=float(depth),
        **{f"recall_at_{k}": float(((ranks > 0) & (ranks <= k)).mean())
           for k in (1, 4, 10, 20) if k <= depth},
        mrr_at_20=float((1.0 / ranks.where(ranks > 0)).fillna(0.0).mean()),
        median_gold_rank=float(found.median()) if len(found) else float("nan"),
    )


if __name__ == "__main__":
    main()
