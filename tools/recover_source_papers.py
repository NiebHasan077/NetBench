#!/usr/bin/env python3
"""Recover resolvable metadata for every source paper cited by HPN-QA.

Benchmark items cite their sources as ``paper_NNNN``, an index assigned by
``Benchmark-Generator/src/stage_01_chunk.py`` over the corpus record order.
That id is meaningless outside the private corpus, so the released benchmark
cannot be traced back to real publications, and the licence status of the
verbatim evidence quotes cannot be established.

This script rebuilds the mapping:

    paper_NNNN -> corpus[NNNN].text -> leading title -> Crossref -> DOI, licence

Each cleaned-up corpus text begins with the paper's title (PDF extraction runs
it into the author list), so the first tokens are sent to Crossref's
``query.bibliographic`` endpoint, which is purpose-built for reference
resolution. OpenAlex ``title.search`` was tried first and returns 0 hits for
these records, so Crossref is used instead. The best hit is scored by token
overlap; where the paper prints its own DOI, that is used to confirm the match
outright. Low-scoring rows are emitted as ``match=review`` rather than
silently accepted.

Usage:
    python tools/recover_source_papers.py \\
        --corpus /path/to/research_corpus_v3.json \\
        --benchmark NetBench-LLM/data/prompts/hpn_benchmark_v5.0_all.jsonl \\
        --out source_papers.csv --mailto you@example.org

``--mailto`` only joins Crossref's polite pool; it is not written to the CSV.
Network access is required. Re-runs are cheap: pass ``--cache`` to reuse a
previous JSON response dump.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import urlopen, Request

CROSSREF = "https://api.crossref.org/works"
OPENALEX = "https://api.openalex.org/works"
STOP = {"a", "an", "the", "of", "for", "and", "on", "in", "to", "with", "via"}


def leading_title(text: str, max_words: int = 16) -> str:
    """Best-effort title from the head of a corpus record."""
    head = " ".join(text.strip().split())[:400]
    head = re.sub(r"^\d+\s+", "", head)             # leading page/line number
    head = re.sub(r"\b[Ee]mail:.*$", "", head)
    words = head.split()[:max_words]
    return " ".join(words).strip(" .,;:-")


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})", re.I)


def in_text_ids(text: str) -> tuple[str, str]:
    """DOI / arXiv id printed in the paper itself (exact, no matching needed)."""
    head = text[:20000]
    d = DOI_RE.search(head)
    a = ARXIV_RE.search(head)
    return (d.group(0).rstrip(".,;)") if d else ""), (a.group(1) if a else "")


def query(title: str, mailto: str, retries: int = 3) -> dict | None:
    """Crossref bibliographic match -- built for reference resolution, and far
    more reliable here than OpenAlex title.search (which returns 0 for these)."""
    url = (f"{CROSSREF}?query.bibliographic={quote_plus(title)}"
           f"&rows=5&mailto={quote_plus(mailto)}")
    for attempt in range(retries):
        try:
            req = Request(url, headers={
                "User-Agent": f"NetBench-artifact/1.0 (mailto:{mailto})"})
            with urlopen(req, timeout=30) as r:
                return json.load(r)["message"]
        except Exception as e:                       # noqa: BLE001 - report and retry
            if attempt == retries - 1:
                print(f"    ! {type(e).__name__}: {e}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _title_of(w: dict) -> str:
    t = w.get("title")
    return (t[0] if isinstance(t, list) and t else (t or "")) or ""


def best_hit(cand: str, results: list[dict]) -> tuple[dict | None, float]:
    ct = tokens(cand)
    best, score = None, 0.0
    for w in results:
        wt = tokens(_title_of(w))
        if not wt or not ct:
            continue
        j = len(ct & wt) / len(ct | wt)
        cover = len(ct & wt) / len(wt)               # title fully inside our head?
        s = max(j, cover * 0.95)
        if s > score:
            best, score = w, s
    return best, score


CC = re.compile(r"creativecommons\.org/licenses/([a-z-]+)", re.I)


def licence_of(work: dict) -> str:
    """Normalise Crossref licence URLs. Publisher TDM/copyright URLs are NOT
    open redistribution grants and are reported verbatim so they stand out."""
    urls = [l.get("URL", "") for l in (work.get("license") or []) if l.get("URL")]
    for u in urls:
        m = CC.search(u)
        if m:
            return "cc-" + m.group(1).lower()
    if any("elsevier.com/tdm" in u for u in urls):
        return "publisher-tdm (elsevier)"
    if any("acm.org" in u for u in urls):
        return "publisher-copyright (acm)"
    if urls:
        return "publisher-specific"
    return "not-stated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", default="source_papers.csv")
    ap.add_argument("--mailto", required=True, help="Crossref polite-pool contact")
    ap.add_argument("--cache", default=None, help="JSON cache of API responses")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N only")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.benchmark, encoding="utf-8") if l.strip()]
    ids: set[str] = set()
    for it in items:
        ids.update(it.get("source_papers") or [])
        ids.update(it.get("inspired_by_papers") or [])
        for e in it.get("evidence") or []:
            if isinstance(e, dict) and e.get("paper_id"):
                ids.add(e["paper_id"])
    order = sorted(ids, key=lambda s: int(s.split("_")[1]))
    if a.limit:
        order = order[: a.limit]

    corpus = json.load(open(a.corpus, encoding="utf-8"))
    print(f"corpus records: {len(corpus)}; source ids to resolve: {len(order)}")

    cache: dict[str, dict] = {}
    cpath = Path(a.cache) if a.cache else None
    if cpath and cpath.exists():
        cache = json.loads(cpath.read_text())
        print(f"cache: {len(cache)} responses")

    rows, stats = [], {"ok": 0, "review": 0, "miss": 0}
    for i, pid in enumerate(order, 1):
        idx = int(pid.split("_")[1])
        if idx >= len(corpus):
            rows.append({"paper_id": pid, "match": "miss", "note": "index out of range"})
            stats["miss"] += 1
            continue
        text = corpus[idx]["text"]
        cand = leading_title(text)
        doi_t, arx_t = in_text_ids(text)
        if pid in cache:
            data = cache[pid]
        else:
            data = query(cand, a.mailto) or {"items": []}
            cache[pid] = data
            time.sleep(0.35)                          # Crossref polite pool
        hit, score = best_hit(cand, data.get("items") or [])
        if hit is None:
            rows.append({"paper_id": pid, "corpus_title_guess": cand,
                         "doi_in_text": doi_t, "arxiv_in_text": arx_t,
                         "match": "miss", "score": 0.0})
            stats["miss"] += 1
        else:
            m = "ok" if score >= a.threshold else "review"
            stats[m] += 1
            doi = (hit.get("DOI") or "").lower()
            if doi_t and doi and doi_t.lower().rstrip("/") == doi:
                m = "ok"                              # confirmed by in-text DOI
                score = 1.0
            rows.append({
                "paper_id": pid,
                "title": _title_of(hit),
                "doi": doi,
                "license": licence_of(hit),
                "year": ((hit.get("issued") or {}).get("date-parts") or [[""]])[0][0] or "",
                "doi_in_text": doi_t,
                "arxiv_in_text": arx_t,
                "match": m,
                "score": round(score, 3),
                "corpus_title_guess": cand,
            })
        if i % 25 == 0 or i == len(order):
            print(f"  {i}/{len(order)}  ok={stats['ok']} review={stats['review']} miss={stats['miss']}")

    if cpath:
        cpath.write_text(json.dumps(cache))

    cols = ["paper_id", "title", "doi", "license", "year", "doi_in_text",
            "arxiv_in_text", "match", "score", "corpus_title_guess"]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}: {len(rows)} rows  "
          f"(ok={stats['ok']} review={stats['review']} miss={stats['miss']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
