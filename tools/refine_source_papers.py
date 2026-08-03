#!/usr/bin/env python3
"""Second pass over rows that ``recover_source_papers.py`` could not confirm.

The first pass takes the head of each corpus record as the title. That fails
when a repository or publisher banner precedes the real title -- ``HAL Id:``,
``Scholars' Mine``, ``CERN-THESIS-...``, ``See discussions, stats``, IEEE/ACM
running headers, ``Received ... accepted ...`` -- which is why rows matched
things like "Hal-hal yang dapat merasa akidah" or "NUCLEAR MINERALS IN PAKISTAN".

This pass resolves each unconfirmed row in three tiers, strongest first:

  1. DOI printed in the paper      -> Crossref /works/{doi}          (exact)
  2. arXiv id printed in the paper -> arXiv API for the exact title,
                                      then Crossref by that title    (exact-ish)
  3. boilerplate-stripped title    -> Crossref query.bibliographic   (fuzzy)

Rows still below threshold keep ``match=review`` and are listed for a human.

Usage:
    python tools/refine_source_papers.py --corpus PATH --csv analysis/outputs/source_papers.csv \\
        --mailto you@example.org
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from urllib.parse import quote_plus, quote
from urllib.request import urlopen, Request

CROSSREF = "https://api.crossref.org/works"
ARXIV = "http://export.arxiv.org/api/query"
STOP = {"a", "an", "the", "of", "for", "and", "on", "in", "to", "with", "via"}

# Banners that precede the real title in archive / publisher PDFs.
BOILER = [
    r"HAL Id:\s*\S+.*?archive for the deposit and dissemination of scientific research documents[^.]*\.",
    r"HAL Id:\s*\S+\s*(?:https?://\S+\s*)?Submitted on [^A-Z]*",
    r"See discussions,? stats,? and author profiles for this publication at:?\s*\S*",
    r"CERN-THESIS-\d{4}-\d+\s*[\d/]*",
    r"Scholars'? Mine.*?(?=[A-Z][a-z])",
    r"Missouri (?:University )?(?:of Science )?and Technology\s*",
    r"Received [A-Z][a-z]+ \d{1,2},? \d{4},? accepted [A-Z][a-z]+ \d{1,2},? \d{4}[^A-Z]*",
    r"(?:IEEE|ACM)[/A-Z ]* TRANSACTIONS ON [A-Z ,]+VOL\.?\s*\d+\s*,?\s*NO\.?\s*\d+[^A-Z]*",
    r"COMMUNICATIONS OF THE ACM[^|]*?practice\s*",
    r"Journal of [A-Za-z ]+ Vol\.?\s*\d+[^A-Z]*(?:Regular Paper\s*)?",
    r"\d{4} IEEE International Conference on [A-Za-z ()]+\s*",
    r"IEEE International Conference on Big Data \(?Big Data\)?\s*[\d\-/.$]*\s*\d*\s*",
    r"Copyright\s*\(?c?\)?\s*\d{4}[^.]*?\.\s*",
    r"This work is licensed under a Creative Commons[^.]*\.\s*",
    r"Preprint\.?\s*Accepted paper at [^.]*\.\s*(?:\d{4} IEEE\.[^.]*\.\s*)?",
    r"arXiv:\d{4}\.\d{4,5}v?\d*\s*\[?[a-z\-]+\.?[A-Z]{2}\]?\s*\d{1,2} [A-Z][a-z]{2} \d{4}\s*",
    r"SUBMITTED TO [A-Z/ ]+\s*\d*\s*",
    r"Contents lists available at ScienceDirect[^A-Z]*",
    r"Future Generation Computer Systems\s*",
    r"ARGONNE NATIONAL LABORATORY[^A-Z]*(?:\d+ South Cass Avenue[^A-Z]*)?",
    r"KSII TRANSACTIONS ON [A-Z ]+VOL\.?\s*\d+[^A-Z]*",
    r"The Nucleus \d+[^a-z]*www\.\S+\s*\d*\s*(?:Pakistan\s*)?(?:The Nucleus\s*)*I?\s*S?\s*",
    r"Title\s+(?=[A-Z])",
]
BOILER_RE = [re.compile(p, re.I | re.S) for p in BOILER]


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def strip_boiler(text: str) -> str:
    head = " ".join(text.strip().split())[:1500]
    for _ in range(4):                       # banners can stack
        before = head
        for rx in BOILER_RE:
            head = rx.sub(" ", head, count=1)
        head = re.sub(r"^\W*\d+\s+", "", head).strip()
        if head == before:
            break
    return " ".join(head.split())


def title_guess(text: str, max_words: int = 16) -> str:
    head = strip_boiler(text)
    head = re.sub(r"\b[Ee]mail:.*$", "", head)
    return " ".join(head.split()[:max_words]).strip(" .,;:-")


def _get(url: str, mailto: str, retries: int = 3):
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": f"NetBench-artifact/1.0 (mailto:{mailto})"})
            with urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:                                  # noqa: BLE001
            if i == retries - 1:
                print(f"      ! {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
                return None
            time.sleep(1.5 * (i + 1))
    return None


def cr_by_doi(doi: str, mailto: str):
    b = _get(f"{CROSSREF}/{quote(doi)}?mailto={quote_plus(mailto)}", mailto)
    if not b:
        return None
    try:
        return json.loads(b)["message"]
    except Exception:
        return None


def cr_search(q: str, mailto: str):
    b = _get(f"{CROSSREF}?query.bibliographic={quote_plus(q)}&rows=5&mailto={quote_plus(mailto)}", mailto)
    if not b:
        return []
    try:
        return json.loads(b)["message"].get("items") or []
    except Exception:
        return []


def arxiv_title(aid: str, mailto: str) -> str:
    b = _get(f"{ARXIV}?id_list={quote(aid)}&max_results=1", mailto)
    if not b:
        return ""
    m = re.search(rb"<entry>.*?<title>(.*?)</title>", b, re.S)
    return " ".join(m.group(1).decode("utf-8", "replace").split()) if m else ""


def title_of(w: dict) -> str:
    t = w.get("title")
    return (t[0] if isinstance(t, list) and t else (t or "")) or ""


CC = re.compile(r"creativecommons\.org/licenses/([a-z-]+)", re.I)


def licence_of(w: dict) -> str:
    urls = [l.get("URL", "") for l in (w.get("license") or []) if l.get("URL")]
    for u in urls:
        m = CC.search(u)
        if m:
            return "cc-" + m.group(1).lower()
    if any("elsevier.com/tdm" in u for u in urls):
        return "publisher-tdm (elsevier)"
    if any("acm.org" in u for u in urls):
        return "publisher-copyright (acm)"
    return "publisher-specific" if urls else "not-stated"


def score(cand: str, t: str) -> float:
    ct, wt = tokens(cand), tokens(t)
    if not ct or not wt:
        return 0.0
    return max(len(ct & wt) / len(ct | wt), (len(ct & wt) / len(wt)) * 0.95)


def fill(row: dict, w: dict, how: str, sc: float) -> None:
    row["title"] = title_of(w)
    row["doi"] = (w.get("DOI") or "").lower()
    row["license"] = licence_of(w)
    dp = ((w.get("issued") or {}).get("date-parts") or [[""]])[0]
    row["year"] = dp[0] if dp else ""
    row["match"] = how
    row["score"] = round(sc, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--csv", default="analysis/outputs/source_papers.csv")
    ap.add_argument("--mailto", required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    cols = list(rows[0].keys())
    corpus = json.load(open(a.corpus, encoding="utf-8"))
    todo = [r for r in rows if r["match"] != "ok"]
    print(f"refining {len(todo)} unconfirmed rows of {len(rows)}")

    fixed = {"doi": 0, "arxiv": 0, "title": 0}
    for i, r in enumerate(todo, 1):
        idx = int(r["paper_id"].split("_")[1])
        text = corpus[idx]["text"] if idx < len(corpus) else ""
        cand = title_guess(text)

        # tier 1 -- DOI printed in the paper
        if r.get("doi_in_text"):
            w = cr_by_doi(r["doi_in_text"], a.mailto)
            time.sleep(0.35)
            if w and title_of(w):
                fill(r, w, "ok", 1.0)
                r["corpus_title_guess"] = cand
                fixed["doi"] += 1
                continue

        # tier 2 -- arXiv id printed in the paper
        if r.get("arxiv_in_text"):
            at = arxiv_title(r["arxiv_in_text"], a.mailto)
            time.sleep(0.35)
            if at:
                hits = cr_search(at, a.mailto)
                time.sleep(0.35)
                best, bs = None, 0.0
                for w in hits:
                    s = score(at, title_of(w))
                    if s > bs:
                        best, bs = w, s
                if best is not None and bs >= 0.7:
                    fill(r, best, "ok", bs)
                else:                       # arXiv-only, no published record
                    r["title"] = at
                    r["doi"] = ""
                    r["license"] = "arxiv (per-paper: see arXiv abs page)"
                    r["match"] = "ok"
                    r["score"] = 1.0
                r["corpus_title_guess"] = cand
                fixed["arxiv"] += 1
                continue

        # tier 3 -- boilerplate-stripped title
        if cand:
            hits = cr_search(cand, a.mailto)
            time.sleep(0.35)
            best, bs = None, 0.0
            for w in hits:
                s = score(cand, title_of(w))
                if s > bs:
                    best, bs = w, s
            if best is not None and bs > float(r["score"] or 0):
                fill(r, best, "ok" if bs >= a.threshold else "review", bs)
                if bs >= a.threshold:
                    fixed["title"] += 1
            r["corpus_title_guess"] = cand
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  doi={fixed['doi']} arxiv={fixed['arxiv']} title={fixed['title']}")

    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["match"] == "ok")
    print(f"\n{a.csv}: ok={ok}/{len(rows)}  still-review={len(rows)-ok}")
    for r in rows:
        if r["match"] != "ok":
            print(f"   [{r['score']:>5}] {r['paper_id']}  {r['corpus_title_guess'][:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
