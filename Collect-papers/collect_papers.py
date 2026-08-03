#!/usr/bin/env python3
"""
collect_papers.py — Generic research paper downloader powered by OpenAlex
and Semantic Scholar.

Domain configuration is driven entirely by a YAML profile file.
See profiles/template.yaml for all available fields, and profiles/hpn.yaml
for a working High-Performance Networking example.

Two search strategies (configured per-profile):
  journal — fetch OA papers from journals with stable OpenAlex source IDs
  keyword — search by focused topic buckets across all venues

Two API sources (selected via --sources):
  openalex — OpenAlex API (default)
  s2       — Semantic Scholar Academic Graph API (open-access only)
  all      — both sources merged and deduplicated

Features:
  • Profile-driven: journals, keywords, year range, paths — all in YAML
  • Multi-source: OpenAlex + Semantic Scholar with cross-source DOI dedup
  • Per-profile output directories (downloads, metadata, failed log)
  • Per-profile venue cache (avoids cross-domain source-ID collisions)
  • Deduplicates against a per-profile paperbase index (auto-updated after each download)
  • Internal dedup by DOI + normalized title within each run
  • PDF validation (checks %%PDF magic bytes)
  • Resume support (skips already-downloaded source IDs)
  • Retry with exponential backoff for downloads

Usage:
    python collect_papers.py                                    # uses profiles/hpn.yaml
    python collect_papers.py --profile profiles/hpn.yaml        # explicit profile
    python collect_papers.py --profile profiles/ml-systems.yaml # different domain
    python collect_papers.py --profile profiles/hpn.yaml --mode journal
    python collect_papers.py --profile profiles/hpn.yaml --sources all
    python collect_papers.py --profile profiles/hpn.yaml --sources s2
    python collect_papers.py --profile profiles/hpn.yaml --dry-run
    python collect_papers.py --profile profiles/hpn.yaml --mailto you@example.com
"""

import argparse
import json
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from profile import Profile, load_profile
from llm_validator import LLMValidator, LLMConnectionError
from paperbase.index_manager import normalize as _paperbase_normalize, IndexManager
from s2_client import S2Client

# ─────────────────────────────────────────────────────────────────────────
# Constants (not domain-specific — never override these via profiles)
# ─────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = SCRIPT_DIR / "profiles" / "hpn.yaml"

OPENALEX_API = "https://api.openalex.org"

# Rate limiting  (polite pool = 10 req/s with mailto)
API_DELAY = 0.15        # seconds between API calls
DOWNLOAD_DELAY = 0.5    # seconds between PDF downloads

# Retry
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0     # exponential multiplier

PER_PAGE = 200           # OpenAlex maximum per-page

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def normalize_title(title: str) -> str:
    """Canonical title normalization — delegates to paperbase.index_manager.normalize.

    Kept as a named function so call-sites throughout this module are unchanged.
    """
    return _paperbase_normalize(title)


def safe_filename(title: str, max_len: int = 150) -> str:
    """Sanitize a title string into a safe filename (no extension)."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title)
    name = name.strip(". ")
    if len(name) > max_len:
        name = name[:max_len].rsplit(" ", 1)[0]
    return name or "untitled"


def reconstruct_abstract(
    inverted_index: Optional[Dict[str, List[int]]],
) -> Optional[str]:
    """Rebuild abstract text from OpenAlex's inverted-index format.

    Returns None if inverted_index is None (no abstract provided).
    Returns an empty string if inverted_index is an empty dict.
    """
    if inverted_index is None:
        return None
    if not inverted_index:
        return ""
    positions: List[Tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(w for _, w in positions)


# Publishers that block automated PDF downloads (HTTP 403/paywall)
_BLOCKED_HOSTS = {
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "www.sciencedirect.com",
    "onlinelibrary.wiley.com",
    "link.springer.com",
    "www.tandfonline.com",
    "journals.sagepub.com",
}


def _is_repository(loc: Dict) -> bool:
    """Return True if this location is a freely-downloadable repository (not a publisher paywall)."""
    if loc.get("host_type") == "repository":
        return True
    url = loc.get("pdf_url") or ""
    host = url.split("/")[2] if url.startswith("http") else ""
    return host not in _BLOCKED_HOSTS


def extract_pdf_url(work: Dict) -> Optional[str]:
    """Return the best freely-downloadable PDF URL from a work record, or None.

    Preference order:
    1. Any repository-hosted PDF (arxiv, PMC, institutional repo, etc.)
    2. Publisher PDF as a last resort (may return 403 for bot requests)
    """
    all_locations = []
    best = work.get("best_oa_location") or {}
    if best.get("pdf_url"):
        all_locations.append(best)
    for loc in work.get("locations", []):
        if loc.get("pdf_url") and loc is not best:
            all_locations.append(loc)

    # Prefer repository-hosted PDFs
    for loc in all_locations:
        if _is_repository(loc):
            return loc["pdf_url"]
    # Fall back to any publisher URL
    for loc in all_locations:
        return loc["pdf_url"]
    return None


def extract_license(work: Dict) -> Optional[str]:
    best = work.get("best_oa_location") or {}
    if best.get("license"):
        return best["license"]
    for loc in work.get("locations", []):
        if loc.get("license"):
            return loc["license"]
    return None


def extract_authors(work: Dict) -> List[str]:
    out: List[str] = []
    for auth in work.get("authorships", []):
        name = (auth.get("author") or {}).get("display_name")
        if name:
            out.append(name)
    return out


def extract_venue(work: Dict) -> Optional[str]:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name")


# ─────────────────────────────────────────────────────────────────────────
# Source-agnostic normalization
# ─────────────────────────────────────────────────────────────────────────

def normalize_openalex_work(work: Dict, source_tag: str) -> Dict:
    """Convert a raw OpenAlex work dict into a source-agnostic paper record."""
    return {
        "source_id": work.get("id", ""),
        "title": work.get("display_name") or "",
        "doi": work.get("doi"),
        "year": work.get("publication_year"),
        "authors": extract_authors(work),
        "venue": extract_venue(work),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "pdf_url": extract_pdf_url(work),
        "license": extract_license(work),
        "source_tag": source_tag,
    }


def normalize_s2_paper(paper: Dict) -> Dict:
    """Convert a raw Semantic Scholar paper dict into a source-agnostic record.

    Papers without an open-access PDF URL get ``pdf_url=None``, which causes
    the main pipeline loop to skip them (open-access only).
    """
    # PDF — only open-access
    pdf_url = None
    oa = paper.get("openAccessPdf")
    if oa and isinstance(oa, dict):
        pdf_url = oa.get("url")

    # DOI
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")

    # Authors
    authors: List[str] = []
    for a in paper.get("authors") or []:
        name = a.get("name")
        if name:
            authors.append(name)

    return {
        "source_id": f"s2:{paper.get('paperId', '')}",
        "title": paper.get("title") or "",
        "doi": doi,
        "year": paper.get("year"),
        "authors": authors,
        "venue": paper.get("venue") or None,
        "abstract": paper.get("abstract"),
        "pdf_url": pdf_url,
        "license": None,  # S2 API does not provide license info
        "source_tag": "s2",
    }


# ─────────────────────────────────────────────────────────────────────────
# OpenAlex API client
# ─────────────────────────────────────────────────────────────────────────

class OpenAlexClient:
    """Thin wrapper with retry, rate-limiting, and venue resolution."""

    def __init__(self, mailto: str, venue_cache_path: Path) -> None:
        self.mailto = mailto
        self.venue_cache_path = venue_cache_path
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            f"collect-papers/1.0 (mailto:{mailto})"
        )

    # ── low-level GET ────────────────────────────────────────────────

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        params["mailto"] = self.mailto
        url = f"{OPENALEX_API}/{endpoint}"
        for attempt in range(MAX_RETRIES):
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", 5))
                    wait = max(retry_after, RETRY_BACKOFF ** attempt)
                    print(f"  ⏳ Rate-limited — waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                time.sleep(API_DELAY)
                return r.json()
            except requests.RequestException as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF ** attempt
                    print(f"  API error ({exc}), retrying in {wait:.1f}s …")
                    time.sleep(wait)
                else:
                    raise
        return {}

    # ── venue resolution ─────────────────────────────────────────────

    def resolve_source_id(
        self, search_term: str, match_substr: str
    ) -> Optional[str]:
        """Search OpenAlex sources; return the short ID of the best match."""
        data = self._get("sources", {"search": search_term, "per-page": 10})
        candidates: List[Tuple[int, str]] = []
        for src in data.get("results", []):
            name = src.get("display_name", "")
            if match_substr.lower() in name.lower():
                candidates.append((src.get("works_count", 0), src["id"]))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1].rstrip("/").split("/")[-1]  # e.g. "S123456"

    def resolve_all_venues(
        self, venues: List[Tuple[str, str]]
    ) -> Dict[str, str]:
        """Resolve every venue name → source ID, with local JSON cache."""
        cache: Dict[str, str] = {}
        if self.venue_cache_path.exists():
            cache = json.loads(self.venue_cache_path.read_text("utf-8"))

        resolved: Dict[str, str] = {}
        for search_term, match_key in venues:
            if match_key in cache:
                resolved[match_key] = cache[match_key]
                continue
            sid = self.resolve_source_id(search_term, match_key)
            if sid:
                resolved[match_key] = sid
                cache[match_key] = sid
                print(f"  Resolved '{match_key}' -> {sid}")
            else:
                print(f"  WARNING: could not resolve venue '{search_term}'")

        self.venue_cache_path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return resolved

    # ── paginated work fetch ─────────────────────────────────────────

    def fetch_works(
        self,
        filter_str: str,
        search: Optional[str] = None,
        max_results: int = 150,
    ) -> List[Dict]:
        works: List[Dict] = []
        cursor = "*"
        while len(works) < max_results:
            params: Dict[str, Any] = {
                "filter": filter_str,
                "per-page": min(PER_PAGE, max_results - len(works)),
                "cursor": cursor,
            }
            if search:
                params["search"] = search

            data = self._get("works", params)
            results = data.get("results", [])
            if not results:
                break
            works.extend(results)

            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return works[:max_results]


# ─────────────────────────────────────────────────────────────────────────
# Download + validation
# ─────────────────────────────────────────────────────────────────────────

def download_pdf(url: str, out_path: Path, mailto: str) -> "tuple[bool, str]":
    """Download with retry + exponential backoff.

    Returns (True, "") on success, or (False, reason) on failure.
    """
    headers = {"User-Agent": f"collect-papers/1.0 (mailto:{mailto})"}
    last_reason = "unknown error"
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(
                url, stream=True, timeout=60, headers=headers
            ) as r:
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", 5))
                    wait = max(retry_after, RETRY_BACKOFF ** attempt)
                    print(f"  ⏳ PDF rate-limited — waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                if r.status_code == 403:
                    return False, f"HTTP 403 Forbidden (publisher blocks automated access)"
                if r.status_code == 401:
                    return False, f"HTTP 401 Unauthorized"
                r.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            return True, ""
        except requests.exceptions.HTTPError as exc:
            last_reason = f"HTTP {exc.response.status_code if exc.response is not None else '?'}"
            break  # no point retrying 4xx/5xx
        except Exception as exc:
            last_reason = str(exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF ** attempt)
    return False, last_reason


def validate_pdf(path: Path) -> bool:
    """Return True if the file starts with the %PDF magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────
# Dedup / integration with Check-Exists
# ─────────────────────────────────────────────────────────────────────────

def load_existing_titles(index_path: Optional[Path]) -> Set[str]:
    """Load normalized titles from a Check-Exists title_index.json.

    Returns an empty set (with a notice) if index_path is None or missing.
    """
    if index_path is None:
        print("  No paperbase_dir configured — skipping paperbase dedup")
        return set()
    if not index_path.exists():
        print(f"  No existing index at {index_path} — skipping paperbase dedup")
        return set()
    index = json.loads(index_path.read_text("utf-8"))
    print(f"  Loaded {len(index)} titles from {index_path}")
    return set(index.keys())


def load_downloaded_ids(metadata_path: Path) -> Set[str]:
    """Load source IDs already in metadata.jsonl (resume support).

    Reads ``source_id`` with fallback to ``openalex_id`` for backward
    compatibility with metadata written before the S2 integration.
    """
    if not metadata_path.exists():
        return set()
    ids: Set[str] = set()
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("source_id") or rec.get("openalex_id", "")
                if sid:
                    ids.add(sid)
            except json.JSONDecodeError:
                pass
    ids.discard("")
    print(f"  Resuming — {len(ids)} papers already downloaded")
    return ids


def append_jsonl(path: Path, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────

def collect(
    profile: Profile,
    *,
    mode: str,
    sources: str = "openalex",
    dry_run: bool,
    no_llm: bool = False,
    mailto_override: Optional[str] = None,
    index_override: Optional[Path] = None,
) -> None:
    """Run the full collection pipeline for the given profile.

    Parameters
    ----------
    profile:
        Loaded and validated Profile object.
    mode:
        Search strategy — "journal", "keyword", or "both".
    sources:
        API sources — "openalex" (default), "s2", or "all".
    dry_run:
        If True, print candidates but do not write any files.
    no_llm:
        If True, skip LLM validation even if enabled in the profile.
    mailto_override:
        When set, overrides profile.mailto (useful for quick CLI testing).
    index_override:
        When set, overrides the paperbase index path from the profile.
    """
    mailto = mailto_override or profile.mailto
    paperbase_index = (
        index_override
        if index_override is not None
        else (profile.paperbase_dir / "title_index.json" if profile.paperbase_dir else None)
    )

    profile.downloads_dir.mkdir(parents=True, exist_ok=True)
    profile.papers_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dedup context ───────────────────────────────────────────
    print(f"\nProfile : {profile.name}")
    print(f"Output  : {profile.downloads_dir}")
    print("Loading dedup context …")
    existing_titles = load_existing_titles(paperbase_index)
    downloaded_ids = load_downloaded_ids(profile.metadata_path)

    # Incremental index — kept live during the run, flushed to disk at the end
    index_manager: Optional[IndexManager] = None
    if paperbase_index is not None and not dry_run:
        index_manager = IndexManager.load(paperbase_index)

    seen_dois: Set[str] = set()
    seen_titles: Set[str] = set(existing_titles)   # pre-seed with paperbase

    # ── Initialize LLM validator (optional) ───────────────────────────────
    validator: Optional[LLMValidator] = None
    if profile.llm_validation is not None and not no_llm:
        validator = LLMValidator(
            config=profile.llm_validation,
            profile_name=profile.name,
            domain_description=profile.domain_description,
            cache_path=profile.validation_cache_path,
        )
        print(
            f"  LLM validation          : "
            f"{profile.llm_validation.backend}/{profile.llm_validation.model}"
        )

    # ── Fetch from APIs ──────────────────────────────────────────────────
    all_papers: List[Dict] = []  # normalized paper records

    if sources in ("openalex", "all"):
        client = OpenAlexClient(mailto, profile.venue_cache_path)
        year_filter = (
            f"publication_year:>{profile.year_min - 1},"
            f"publication_year:<{profile.year_max + 1}"
        )
        base_filter = f"{year_filter},open_access.is_oa:true,has_pdf_url:true"

        if mode in ("journal", "both") and profile.journals:
            print("\n── Journal-based search ───────────────────────────")
            source_map = client.resolve_all_venues(profile.journals)
            for journal_name, source_id in source_map.items():
                filt = f"primary_location.source.id:{source_id},{base_filter}"
                print(f"  {journal_name} ({source_id}) …", end=" ", flush=True)
                works = client.fetch_works(filt, max_results=profile.max_per_query)
                print(f"→ {len(works)} works")
                all_papers.extend(
                    normalize_openalex_work(w, "journal") for w in works
                )

        if mode in ("keyword", "both") and profile.keywords:
            print("\n── Keyword-based search ───────────────────────────")
            keyword_filter = base_filter
            if profile.subfield_filter:
                keyword_filter = f"{base_filter},{profile.subfield_filter}"
            for bucket in profile.keywords:
                print(f"  {bucket} …", end=" ", flush=True)
                works = client.fetch_works(
                    keyword_filter, search=bucket, max_results=profile.max_per_query
                )
                print(f"→ {len(works)} works")
                all_papers.extend(
                    normalize_openalex_work(w, "keyword") for w in works
                )

    if sources in ("s2", "all") and profile.s2_config is not None:
        s2cfg = profile.s2_config
        s2_client = S2Client(api_key=s2cfg.api_key)
        year_range = f"{profile.year_min}-{profile.year_max}"

        print("\n── Semantic Scholar search ────────────────────────")
        for bucket in s2cfg.keywords:
            print(f"  {bucket} …", end=" ", flush=True)
            papers = s2_client.search(
                bucket,
                year_range=year_range,
                fields_of_study=s2cfg.fields_of_study or None,
                open_access_only=True,
                max_results=s2cfg.max_per_query,
            )
            print(f"→ {len(papers)} papers")
            all_papers.extend(normalize_s2_paper(p) for p in papers)

    print(f"\nTotal candidate works: {len(all_papers)}")

    # ── Deduplicate + download ───────────────────────────────────────
    stats = dict.fromkeys(
        [
            "fetched",
            "skip_resumed",
            "skip_dup_doi",
            "skip_dup_title",
            "skip_in_paperbase",
            "skip_no_pdf",
            "skip_llm_rejected",
            "downloaded",
            "download_failed",
            "invalid_pdf",
        ],
        0,
    )
    stats["fetched"] = len(all_papers)

    print("\nProcessing …\n")

    for i, paper in enumerate(all_papers, 1):
        source_id = paper["source_id"]
        title = paper["title"]
        doi = paper["doi"]
        norm = normalize_title(title)

        # 1. Resume: skip already-downloaded
        if source_id in downloaded_ids:
            stats["skip_resumed"] += 1
            continue

        # 2. DOI dedup
        if doi:
            if doi in seen_dois:
                stats["skip_dup_doi"] += 1
                continue
            seen_dois.add(doi)

        # 3. Title dedup (includes existing paperbase titles)
        if norm:
            if norm in seen_titles:
                if norm in existing_titles:
                    stats["skip_in_paperbase"] += 1
                else:
                    stats["skip_dup_title"] += 1
                continue
            seen_titles.add(norm)

        # 4. Must have a PDF URL
        pdf_url = paper["pdf_url"]
        if not pdf_url:
            stats["skip_no_pdf"] += 1
            continue

        # 5. LLM abstract validation (optional)
        abstract = paper["abstract"]
        if validator is not None:
            llm_result = validator.validate(
                source_id,
                title,
                abstract,
                is_journal_paper=(paper["source_tag"] == "journal"),
            )
            if not llm_result.relevant:
                stats["skip_llm_rejected"] += 1
                print(f"  [LLM SKIP] {title}")
                continue

        # ── Build metadata record ────────────────────────────────────────────
        if source_id.startswith("s2:"):
            short_id = source_id[3:]
        elif source_id:
            short_id = source_id.rstrip("/").split("/")[-1]
        else:
            short_id = "unknown"

        record: Dict[str, Any] = {
            "source_id": source_id,
            "short_id": short_id,
            "title": title,
            "normalized_title": norm,
            "doi": doi,
            "year": paper["year"],
            "authors": paper["authors"],
            "venue": paper["venue"],
            "abstract": abstract,
            "license": paper["license"],
            "pdf_url": pdf_url,
            "local_filename": None,  # set after download
        }
        # Backward compat: keep openalex_id for OpenAlex-sourced papers
        if not source_id.startswith("s2:"):
            record["openalex_id"] = source_id

        # ── Dry-run: just print ──────────────────────────────────
        if dry_run:
            lic = f" [{record['license']}]" if record["license"] else ""
            ven = f" @ {record['venue']}" if record["venue"] else ""
            print(f"  [{i}] {title}{ven}{lic}")
            stats["downloaded"] += 1
            continue

        # ── Download ─────────────────────────────────────────────
        filename = f"{safe_filename(title)}.pdf"
        pdf_path = profile.papers_dir / filename
        if pdf_path.exists():
            filename = f"{safe_filename(title)}_{short_id}.pdf"
            pdf_path = profile.papers_dir / filename

        ok, fail_reason = download_pdf(pdf_url, pdf_path, mailto)
        if not ok:
            record["error"] = "download_failed"
            record["fail_reason"] = fail_reason
            append_jsonl(profile.failed_path, record)
            stats["download_failed"] += 1
            print(f"  [FAIL] {title}  ({fail_reason})")
            continue

        # ── Validate ─────────────────────────────────────────────
        if not validate_pdf(pdf_path):
            pdf_path.unlink(missing_ok=True)
            record["error"] = "invalid_pdf"
            append_jsonl(profile.failed_path, record)
            stats["invalid_pdf"] += 1
            print(f"  [BAD PDF] {title}")
            continue

        # ── Record success ───────────────────────────────────────
        record["local_filename"] = filename
        append_jsonl(profile.metadata_path, record)
        downloaded_ids.add(source_id)
        stats["downloaded"] += 1

        # Incrementally update the title index so subsequent runs skip this paper
        if index_manager is not None:
            index_manager.add_pdf(pdf_path, root=profile.paperbase_dir)

        lic = f" [{record['license']}]" if record["license"] else ""
        print(f"  [OK] {title}{lic}")
        time.sleep(DOWNLOAD_DELAY)

    if validator is not None:
        validator.close()

    # Flush incremental index to disk (no-op if nothing was downloaded)
    if index_manager is not None:
        index_manager.save(paperbase_index)

    # ── Summary ──────────────────────────────────────────────────────
    tag = "DRY RUN — " if dry_run else ""
    print(f"\n{'=' * 60}")
    print(f"{tag}Collection summary  [{profile.name}]")
    print(f"  Fetched from API        : {stats['fetched']}")
    print(f"  Already downloaded      : {stats['skip_resumed']}")
    print(f"  Duplicate DOI           : {stats['skip_dup_doi']}")
    print(f"  Duplicate title (new)   : {stats['skip_dup_title']}")
    print(f"  In existing paperbase   : {stats['skip_in_paperbase']}")
    print(f"  No PDF URL              : {stats['skip_no_pdf']}")
    if profile.llm_validation is not None:
        print(f"  LLM rejected            : {stats['skip_llm_rejected']}")
    verb = "Would download" if dry_run else "Downloaded"
    print(f"  {verb:24s}: {stats['downloaded']}")
    if not dry_run:
        print(f"  Download failed         : {stats['download_failed']}")
        print(f"  Invalid PDF (removed)   : {stats['invalid_pdf']}")
        print(f"\n  Metadata  : {profile.metadata_path}")
        print(f"  Failures  : {profile.failed_path}")
        print(f"  PDFs      : {profile.downloads_dir}/")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generic research paper downloader powered by OpenAlex and Semantic Scholar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python collect_papers.py                                    # uses profiles/hpn.yaml
              python collect_papers.py --profile profiles/hpn.yaml        # explicit profile
              python collect_papers.py --profile profiles/ml-systems.yaml # different domain
              python collect_papers.py --profile profiles/hpn.yaml --dry-run
              python collect_papers.py --profile profiles/hpn.yaml --mode journal
              python collect_papers.py --profile profiles/hpn.yaml --sources all
              python collect_papers.py --profile profiles/hpn.yaml --sources s2
        """),
    )
    ap.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help=f"Path to a profile YAML file (default: {DEFAULT_PROFILE_PATH})",
    )
    ap.add_argument(
        "--mode",
        choices=["journal", "keyword", "both"],
        default="both",
        help="Search strategy: journal, keyword, or both (default: both)",
    )
    ap.add_argument(
        "--sources",
        choices=["openalex", "s2", "all"],
        default="openalex",
        help="API sources: openalex (default), s2 (Semantic Scholar), or all.",
    )
    ap.add_argument(
        "--mailto",
        default=None,
        help="Override profile mailto for OpenAlex polite pool",
    )
    ap.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Override the paperbase title_index.json path from the profile",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be downloaded — no files written.",
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM validation even if enabled in the profile.",
    )
    args = ap.parse_args()

    try:
        profile = load_profile(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        collect(
            profile,
            mode=args.mode,
            sources=args.sources,
            dry_run=args.dry_run,
            no_llm=args.no_llm,
            mailto_override=args.mailto,
            index_override=args.index,
        )
    except LLMConnectionError as exc:
        print(f"ERROR: LLM connection failed \u2014 {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
