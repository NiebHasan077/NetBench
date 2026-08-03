"""
paperbase/index_manager.py
--------------------------
Single source of truth for title normalization, PDF title extraction, and
incremental index management.

Used by:
  • paperbase/build_index.py  — full-scan CLI
  • paperbase/check_paper.py  — lookup CLI
  • collect_papers.py         — incremental update after each download

Public API
----------
  normalize(title)             Pure function: canonical string → comparable key
  extract_title(pdf_path)      → (raw_title: str, source: str)
  IndexManager                 Load / query / update / save the title index

IndexManager contract
---------------------
  im = IndexManager.load(path)           # load from JSON, or empty if missing
  im.contains(norm_title)                # bool — exact key lookup
  im.all_normalized_titles()             # frozenset[str] — for seeding dedup
  result = im.add_pdf(pdf_path, root)    # add one PDF; returns AddResult
  im.save(path)                          # flush index + candidate list to disk

AddResult dataclass
-------------------
  status : "added" | "strong_dup" | "weak_dup" | "extract_failed"
  raw_title  : str
  norm_title : str
  source     : str  ("metadata", "font-heuristic", "no-text", "error: …")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyMuPDF is required: pip install 'PyMuPDF>=1.24'"
    ) from exc


# ── Normalization ─────────────────────────────────────────────────────────

def normalize(title: str) -> str:
    """Canonical normalization for cross-index title comparison.

    Lowercase → strip punctuation → collapse whitespace.

    This is the single authoritative implementation shared by collect_papers,
    build_index, and check_paper.  All three previously had subtly different
    versions (e.g. ``[^a-z0-9\\s]`` vs ``[^\\w\\s]``); this fixes the
    inconsistency so dedup hits are never missed due to normalization drift.
    """
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)   # replace punctuation with space (keeps unicode)
    t = re.sub(r"\s+", " ", t)        # collapse whitespace
    return t.strip()


# ── PDF title extraction ──────────────────────────────────────────────────

_BAD_METADATA = {"untitled", "microsoft word", ".doc", ".docx", ".tex", "unknown"}

_MIN_TITLE_LEN = 15

_BOILERPLATE = re.compile(
    r"(contents\s+lists?\s+available"
    r"|journal\s+homepage"
    r"|all\s+rights\s+reserved"
    r"|published\s+by\s+elsevier"
    r"|doi\s*:\s*10\.\d{4,}"
    r")",
    re.IGNORECASE,
)

_TOP_BAND_FRACTION = 0.10


def _is_good_metadata_title(title: str) -> bool:
    if len(title) < 8:
        return False
    t = title.lower()
    return not any(bad in t for bad in _BAD_METADATA)


def _is_valid_title(text: str) -> bool:
    if len(text) < _MIN_TITLE_LEN:
        return False
    return not _BOILERPLATE.search(text)


def _spans_from_page(page, min_y: float = 0.0) -> list:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    spans = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if len(text) <= 1:
                    continue
                y = span["origin"][1]
                if y < min_y:
                    continue
                spans.append((span["size"], y, text))
    return spans


def _best_title_from_spans(spans: list) -> str:
    if not spans:
        return ""
    sizes = sorted({s[0] for s in spans}, reverse=True)
    for size in sizes:
        tier = [s for s in spans if s[0] >= size - 1.0]
        tier.sort(key=lambda s: s[1])
        candidate = " ".join(s[2] for s in tier).strip()
        if _is_valid_title(candidate):
            return candidate
    return ""


def _extract_by_font_heuristic(page) -> str:
    page_height = page.rect.height
    top_band = page_height * _TOP_BAND_FRACTION
    body_spans = _spans_from_page(page, min_y=top_band)
    candidate = _best_title_from_spans(body_spans)
    if candidate:
        return candidate
    all_spans = _spans_from_page(page, min_y=0.0)
    return _best_title_from_spans(all_spans)


def extract_title(pdf_path: Path) -> tuple[str, str]:
    """Return ``(raw_title, source)`` or ``("", "error: …")`` on failure.

    source is one of: "metadata", "font-heuristic", "no-text", "error: <msg>"
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        return "", f"error: {exc}"

    try:
        meta = (doc.metadata or {}).get("title", "").strip()
        if _is_good_metadata_title(meta):
            return meta, "metadata"
        if doc.page_count > 0:
            heuristic = _extract_by_font_heuristic(doc[0]).strip()
            if heuristic:
                return heuristic, "font-heuristic"
        return "", "no-text"
    finally:
        doc.close()


# ── Duplicate detection helpers ───────────────────────────────────────────

_FILE_SIZE_TOLERANCE = 0.05
_TEXT_SIM_THRESHOLD  = 0.80
_MIN_TOKENS          = 20


def _page_count(pdf_path: Path) -> Optional[int]:
    try:
        doc = fitz.open(str(pdf_path))
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return None


def _first_page_tokens(pdf_path: Path) -> list[str]:
    try:
        doc = fitz.open(str(pdf_path))
        if doc.page_count == 0:
            doc.close()
            return []
        page = doc[0]
        min_y = page.rect.height * _TOP_BAND_FRACTION
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        tokens: list[str] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["origin"][1] < min_y:
                        continue
                    tokens.extend(re.findall(r"\w+", span["text"].lower()))
        doc.close()
        return tokens
    except Exception:
        return []


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_strong_duplicate(path_a: Path, path_b: Path) -> tuple[bool, dict]:
    """Return (is_strong, checks_dict).

    All three checks must pass (True or None) for a STRONG classification.
    None means the check was inconclusive (unreadable file) — not a failure.
    """
    checks: dict = {}

    pc_a, pc_b = _page_count(path_a), _page_count(path_b)
    checks["page_count"] = (pc_a == pc_b) if (pc_a is not None and pc_b is not None) else None

    try:
        sa, sb = path_a.stat().st_size, path_b.stat().st_size
        ratio = abs(sa - sb) / max(sa, sb) if max(sa, sb) > 0 else 0.0
        checks["file_size"] = ratio <= _FILE_SIZE_TOLERANCE
    except OSError:
        checks["file_size"] = None

    tok_a = _first_page_tokens(path_a)
    tok_b = _first_page_tokens(path_b)
    if len(tok_a) >= _MIN_TOKENS and len(tok_b) >= _MIN_TOKENS:
        checks["text_sim"] = _jaccard(tok_a, tok_b) >= _TEXT_SIM_THRESHOLD
    else:
        checks["text_sim"] = None

    is_strong = all(v is not False for v in checks.values())
    return is_strong, checks


# ── AddResult ─────────────────────────────────────────────────────────────

@dataclass
class AddResult:
    """Result of ``IndexManager.add_pdf()``."""
    status: str          # "added" | "strong_dup" | "weak_dup" | "extract_failed"
    raw_title: str
    norm_title: str
    source: str          # "metadata", "font-heuristic", "no-text", "error: …"
    existing_path: str = ""  # populated for strong_dup / weak_dup


# ── IndexManager ──────────────────────────────────────────────────────────

class IndexManager:
    """In-memory title index with incremental add and JSON persistence.

    The internal index maps ``normalized_title → {raw_title, path, source}``.
    A separate ``_candidates`` list accumulates duplicate pairs (STRONG and
    WEAK) for later review via ``verify_duplicates.py``.

    Parameters are never passed to ``__init__`` directly; always use the
    ``load()`` class method.
    """

    def __init__(self) -> None:
        self._index: dict[str, dict] = {}      # norm → {raw_title, path, source}
        self._candidates: list[dict] = []       # duplicate pairs

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def load(cls, index_path: Optional[Path]) -> "IndexManager":
        """Load an existing ``title_index.json``, or return an empty manager.

        Parameters
        ----------
        index_path:
            Path to ``title_index.json``.  If ``None`` or the file does not
            exist, returns an empty manager (no-op on all calls).
        """
        im = cls()
        if index_path is None or not index_path.exists():
            return im
        try:
            raw = json.loads(index_path.read_text("utf-8"))
            if isinstance(raw, dict):
                im._index = raw
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file → start fresh
        return im

    # ── Queries ───────────────────────────────────────────────────────

    def contains(self, norm_title: str) -> bool:
        """Return True if the normalized title is in the index."""
        return norm_title in self._index

    def all_normalized_titles(self) -> frozenset:
        """Return all normalized titles currently in the index."""
        return frozenset(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)

    # ── Mutation ──────────────────────────────────────────────────────

    def add_pdf(self, pdf_path: Path, root: Optional[Path] = None) -> AddResult:
        """Extract a title from ``pdf_path`` and add it to the index.

        Parameters
        ----------
        pdf_path:
            Absolute path to the newly downloaded / validated PDF.
        root:
            If provided, paths stored in the index are recorded relative to
            ``root.parent`` (same convention as the original ``build_index.py``
            so old and new index entries are compatible).  If ``None``, the
            absolute path is stored.

        Returns
        -------
        AddResult
            Always returns a result; never raises.

        Behaviour
        ---------
        - Title extraction fails → status ``"extract_failed"``, no index change.
        - Title not in index → add entry, status ``"added"``.
        - Title already in index, STRONG dup → record candidate, status ``"strong_dup"``, no index change.
        - Title already in index, WEAK  dup → record candidate, status ``"weak_dup"``,  no index change.
        """
        raw_title, source = extract_title(pdf_path)

        if not raw_title:
            return AddResult(
                status="extract_failed",
                raw_title="",
                norm_title="",
                source=source,
            )

        norm = normalize(raw_title)

        # Compute stored path
        if root is not None:
            try:
                rel = str(pdf_path.relative_to(root.parent))
            except ValueError:
                rel = str(pdf_path)
        else:
            rel = str(pdf_path)

        if norm not in self._index:
            self._index[norm] = {"raw_title": raw_title, "path": rel, "source": source}
            return AddResult(
                status="added",
                raw_title=raw_title,
                norm_title=norm,
                source=source,
            )

        # Collision — run duplicate checks
        existing_rel = self._index[norm]["path"]
        if root is not None:
            existing_abs = (root.parent / existing_rel).resolve()
        else:
            existing_abs = Path(existing_rel)

        is_strong, checks = _is_strong_duplicate(existing_abs, pdf_path)
        status = "strong_dup" if is_strong else "weak_dup"
        confidence = "STRONG" if is_strong else "WEAK"

        self._candidates.append({
            "normalized_title": norm,
            "raw_title":        raw_title,
            "existing":         existing_rel,
            "duplicate":        rel,
            "confidence":       confidence,
            "checks":           checks,
        })

        return AddResult(
            status=status,
            raw_title=raw_title,
            norm_title=norm,
            source=source,
            existing_path=existing_rel,
        )

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, index_path: Path) -> None:
        """Write ``title_index.json`` to ``index_path``.

        If any duplicate candidates were accumulated during this session,
        also write/merge ``duplicate_candidates.json`` next to the index.
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if self._candidates:
            candidates_path = index_path.parent / "duplicate_candidates.json"
            existing: list = []
            if candidates_path.exists():
                try:
                    existing = json.loads(candidates_path.read_text("utf-8"))
                except (json.JSONDecodeError, OSError):
                    existing = []
            candidates_path.write_text(
                json.dumps(existing + self._candidates, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._candidates = []  # reset after flush
