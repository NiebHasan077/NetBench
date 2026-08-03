"""
profile.py — Profile dataclass and YAML loader for collect_papers.

A profile is a YAML file that fully describes a research domain to collect:
what journals to fetch, what keyword buckets to search, which OpenAlex
subfield IDs to restrict results to, where the local paperbase index lives,
and where to write downloaded PDFs and metadata.

Required profile fields
-----------------------
  name                : Human-readable name for the domain (str)
  mailto              : Email for OpenAlex polite pool (str)

Optional profile fields (with defaults)
----------------------------------------
  journals            : list of {search, match} dicts   (default: [])
  keywords            : list of query strings            (default: [])
  subfield_filter     : OpenAlex filter string           (default: None)
  year_min            : oldest publication year          (default: 2018)
  year_max            : newest publication year          (default: 2026)
  max_per_query       : max results per bucket/venue     (default: 150)
  domain_description  : plain-English domain summary     (default: "")
  paperbase_dir       : path to a Check-Exists directory (default: None)
  downloads_dir       : path to write PDFs               (default: downloads/<name>/)
  llm_validation      : LLM validation config block      (default: None)

At least one of `journals` or `keywords` must be non-empty.

Path resolution
---------------
  `paperbase_dir` and `downloads_dir` are resolved relative to the
  directory that contains the profile YAML file, unless they are absolute.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyyaml is required: pip install pyyaml>=6.0"
    ) from exc

# ── Sentinel used to detect a placeholder mailto ────────────────────────
_PLACEHOLDER_MAILTO = "your_email@example.com"

# ── Valid values for LLM config fields ───────────────────────────────────
_VALID_LLM_BACKENDS = frozenset({"ollama", "openai"})
_VALID_LLM_FALLBACKS = frozenset({"download", "skip"})


# ── LLM config ────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for optional LLM-based abstract relevance validation.

    Fields
    ------
    backend:
        LLM transport — "ollama" (local) or "openai" (OpenAI-compatible API).
    model:
        Model name as recognised by the backend (e.g. "llama3", "gpt-4o-mini").
    base_url:
        Base URL of the backend.  Ollama default: "http://localhost:11434".
    api_key:
        Bearer token.  Leave empty string for Ollama.
    temperature:
        Sampling temperature.  0 = deterministic (recommended).
    timeout:
        Per-request timeout in seconds.
    fallback_on_no_abstract:
        Policy when a paper has no abstract:
          "download" — allow the paper through (safe default).
          "skip"     — reject without an LLM call.
    skip_journal_papers:
        If True, journal-sourced papers bypass validation entirely.
        They are already domain-specific by venue filter.
    """
    backend: str                        # "ollama" | "openai"
    model: str
    base_url: str
    api_key: str                        # empty string for Ollama
    temperature: float                  # 0 = deterministic
    timeout: int                        # seconds
    fallback_on_no_abstract: str        # "download" | "skip"
    skip_journal_papers: bool


# ── Semantic Scholar config ────────────────────────────────────────────────

@dataclass
class S2Config:
    """Configuration for Semantic Scholar API integration.

    Fields
    ------
    api_key:
        Semantic Scholar API key.  Free registration at semanticscholar.org.
        Empty string for unauthenticated access (heavily rate-limited).
    fields_of_study:
        Filter results to these fields, e.g. ["Computer Science"].
    keywords:
        S2-specific keyword buckets (separate from OpenAlex keywords
        because S2 uses different search syntax).
    max_per_query:
        Maximum results per keyword query.
    """
    api_key: str
    fields_of_study: List[str]
    keywords: List[str]
    max_per_query: int


# ── Profile dataclass ─────────────────────────────────────────────────────

@dataclass
class Profile:
    """Fully resolved configuration for one paper-collection domain."""

    # Identity
    name: str

    # OpenAlex search config
    journals: List[Tuple[str, str]]   # (search_term, match_key)
    keywords: List[str]               # keyword bucket strings
    subfield_filter: Optional[str]    # None = no subfield restriction

    # Year range
    year_min: int
    year_max: int

    # API config
    max_per_query: int
    mailto: str

    # LLM placeholder (populated in Phase 3)
    domain_description: str

    # Paths (all resolved to absolute at load time)
    paperbase_dir: Optional[Path]   # None = skip paperbase dedup
    downloads_dir: Path
    papers_dir: Path                # derived: downloads_dir / papers/  — where PDFs live
    metadata_path: Path             # derived: downloads_dir / metadata.jsonl
    failed_path: Path               # derived: downloads_dir / failed_downloads.jsonl
    venue_cache_path: Path          # derived: profile file dir / <name>_venue_cache.json
    llm_validation: Optional[LLMConfig]   # None = disabled
    validation_cache_path: Path           # derived: downloads_dir / validation_cache.json
    s2_config: Optional[S2Config]         # None = Semantic Scholar disabled


# ── Loader ───────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a profile name to a filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_") or "profile"


def load_profile(path: Path) -> Profile:
    """Load and validate a profile YAML file, returning a resolved Profile.

    Parameters
    ----------
    path:
        Absolute or relative path to the profile YAML file.

    Returns
    -------
    Profile
        Fully resolved profile with all paths as absolute Path objects.

    Raises
    ------
    FileNotFoundError
        If the profile YAML file does not exist.
    ValueError
        If required fields are missing or values are logically invalid.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    raw = yaml.safe_load(path.read_text("utf-8")) or {}
    profile_dir = path.parent

    # ── Required fields ──────────────────────────────────────────────
    for required in ("name", "mailto"):
        if required not in raw:
            raise ValueError(
                f"Profile '{path}' is missing required field: '{required}'"
            )

    name: str = str(raw["name"]).strip()
    if not name:
        raise ValueError(f"Profile '{path}': 'name' must not be empty")

    mailto: str = str(raw["mailto"]).strip()

    # ── Optional fields with defaults ────────────────────────────────

    # journals: list of {search: ..., match: ...} dicts
    raw_journals = raw.get("journals") or []
    journals: List[Tuple[str, str]] = []
    for i, entry in enumerate(raw_journals):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Profile '{path}': journals[{i}] must be a dict with "
                f"'search' and 'match' keys"
            )
        if "search" not in entry or "match" not in entry:
            raise ValueError(
                f"Profile '{path}': journals[{i}] must have both "
                f"'search' and 'match' keys"
            )
        journals.append((str(entry["search"]), str(entry["match"])))

    # keywords: list of strings
    raw_keywords = raw.get("keywords") or []
    if not isinstance(raw_keywords, list):
        raise ValueError(f"Profile '{path}': 'keywords' must be a list")
    keywords: List[str] = [str(k) for k in raw_keywords]

    # At least one search strategy must be defined
    if not journals and not keywords:
        raise ValueError(
            f"Profile '{path}': at least one of 'journals' or 'keywords' "
            f"must be non-empty"
        )

    subfield_filter: Optional[str] = raw.get("subfield_filter") or None

    year_min: int = int(raw.get("year_min", 2018))
    year_max: int = int(raw.get("year_max", 2026))
    if year_min > year_max:
        raise ValueError(
            f"Profile '{path}': year_min ({year_min}) must be <= "
            f"year_max ({year_max})"
        )

    max_per_query: int = int(raw.get("max_per_query", 150))
    if max_per_query < 1:
        raise ValueError(
            f"Profile '{path}': max_per_query must be >= 1, got {max_per_query}"
        )

    domain_description: str = str(raw.get("domain_description") or "").strip()

    # ── Path resolution ───────────────────────────────────────────────
    def _resolve(value: Optional[str], fallback: Path) -> Optional[Path]:
        """Resolve a path value relative to profile_dir, or return fallback."""
        if value is None:
            return fallback
        p = Path(value)
        if not p.is_absolute():
            p = (profile_dir / p).resolve()
        return p

    paperbase_raw = raw.get("paperbase_dir")
    paperbase_dir: Optional[Path] = None
    if paperbase_raw is not None:
        p = Path(paperbase_raw)
        if not p.is_absolute():
            p = (profile_dir / p).resolve()
        paperbase_dir = p

    slug = _slugify(name)

    downloads_raw = raw.get("downloads_dir")
    if downloads_raw is not None:
        p = Path(downloads_raw)
        if not p.is_absolute():
            downloads_dir = (profile_dir / p).resolve()
        else:
            downloads_dir = p
    else:
        # Default: <profile_dir>/../downloads/<slug>/
        # i.e. Collect-papers/downloads/hpn/
        downloads_dir = (profile_dir.parent / "downloads" / slug).resolve()

    metadata_path = downloads_dir / "metadata.jsonl"
    failed_path = downloads_dir / "failed_downloads.jsonl"
    papers_dir = downloads_dir / "papers"
    venue_cache_path = profile_dir / f"{slug}_venue_cache.json"

    # ── LLM validation config ─────────────────────────────────────────
    llm_validation: Optional[LLMConfig] = None
    raw_llm = raw.get("llm_validation")
    if raw_llm and isinstance(raw_llm, dict) and raw_llm.get("enabled", True):
        backend = str(raw_llm.get("backend", "ollama")).strip()
        if backend not in _VALID_LLM_BACKENDS:
            raise ValueError(
                f"Profile '{path}': llm_validation.backend must be one of "
                f"{sorted(_VALID_LLM_BACKENDS)}, got {backend!r}"
            )
        fallback = str(raw_llm.get("fallback_on_no_abstract", "download")).strip()
        if fallback not in _VALID_LLM_FALLBACKS:
            raise ValueError(
                f"Profile '{path}': llm_validation.fallback_on_no_abstract "
                f"must be one of {sorted(_VALID_LLM_FALLBACKS)}, got {fallback!r}"
            )
        llm_validation = LLMConfig(
            backend=backend,
            model=str(raw_llm.get("model", "llama3")).strip(),
            base_url=str(raw_llm.get("base_url", "http://localhost:11434")).strip(),
            api_key=str(raw_llm.get("api_key") or "").strip(),
            temperature=float(raw_llm.get("temperature", 0.0)),
            timeout=int(raw_llm.get("timeout", 30)),
            fallback_on_no_abstract=fallback,
            skip_journal_papers=bool(raw_llm.get("skip_journal_papers", True)),
        )

    validation_cache_path = downloads_dir / "validation_cache.json"

    # ── Semantic Scholar config ─────────────────────────────────────────
    s2_config: Optional[S2Config] = None
    raw_s2 = raw.get("semantic_scholar")
    if raw_s2 and isinstance(raw_s2, dict) and raw_s2.get("enabled", False):
        s2_keywords = raw_s2.get("keywords") or []
        if not isinstance(s2_keywords, list):
            raise ValueError(
                f"Profile '{path}': semantic_scholar.keywords must be a list"
            )
        if not s2_keywords:
            raise ValueError(
                f"Profile '{path}': semantic_scholar.keywords must not be "
                f"empty when enabled"
            )
        s2_fos = raw_s2.get("fields_of_study") or ["Computer Science"]
        if not isinstance(s2_fos, list):
            raise ValueError(
                f"Profile '{path}': semantic_scholar.fields_of_study must be a list"
            )
        s2_api_key = str(raw_s2.get("api_key") or "").strip()
        s2_api_key_env = str(raw_s2.get("api_key_env") or "").strip()
        if s2_api_key_env:
            s2_api_key = os.environ.get(s2_api_key_env, s2_api_key).strip()
        s2_config = S2Config(
            api_key=s2_api_key,
            fields_of_study=[str(f) for f in s2_fos],
            keywords=[str(k) for k in s2_keywords],
            max_per_query=int(raw_s2.get("max_per_query", 100)),
        )

    # ── Warn about placeholder mailto ────────────────────────────────
    if mailto == _PLACEHOLDER_MAILTO:
        print(
            f"WARNING: Profile '{name}' uses the placeholder email. "
            f"Set 'mailto' to your real address for OpenAlex polite-pool access."
        )

    return Profile(
        name=name,
        journals=journals,
        keywords=keywords,
        subfield_filter=subfield_filter,
        year_min=year_min,
        year_max=year_max,
        max_per_query=max_per_query,
        mailto=mailto,
        domain_description=domain_description,
        paperbase_dir=paperbase_dir,
        downloads_dir=downloads_dir,
        papers_dir=papers_dir,
        metadata_path=metadata_path,
        failed_path=failed_path,
        venue_cache_path=venue_cache_path,
        llm_validation=llm_validation,
        validation_cache_path=validation_cache_path,
        s2_config=s2_config,
    )
