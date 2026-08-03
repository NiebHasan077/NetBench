"""Phase 2 corpus normalization for the Instruct-FTD pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


_ABSTRACT_RE = re.compile(r"\bAbstract\b|\bABSTRACT\b", re.UNICODE)

_AFFIL_RE = re.compile(
    r"\b(?:University|Institute|Department|Laboratory|Lab|College|"
    r"School|Center|Centre|Corp|Inc|Ltd|IEEE|ACM|Email|email|@)\b",
    re.IGNORECASE,
)

_KNOWN_HEADINGS = frozenset(
    {
        "introduction",
        "background",
        "related work",
        "related works",
        "motivation",
        "overview",
        "architecture",
        "design",
        "system model",
        "system design",
        "framework",
        "methodology",
        "method",
        "approach",
        "implementation",
        "performance",
        "evaluation",
        "experiments",
        "experiment",
        "results",
        "result",
        "analysis",
        "discussion",
        "conclusion",
        "conclusions",
        "future work",
        "acknowledgment",
        "acknowledgements",
        "acknowledgments",
        "references",
        "appendix",
    }
)

_SECTION_KEYWORDS = (
    r"Introduction|Background|Related Work(?:s)?|Motivation|Overview|"
    r"Architecture|Design|System(?:\s+(?:Model|Design|Overview))?|"
    r"Framework|Methodology|Method|Approach|Implementation|"
    r"Performance|Evaluation|Experiment(?:s|al)?|Results?|Analysis|"
    r"Discussion|Conclusion(?:s)?|Future Work|"
    r"Acknowledgments?|Acknowledgements?|References?|Appendix"
)

_ROMAN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(I{1,3}V?|I?V|V?I{1,3}|I?X|X{1,3})"
    r"\s*\."
    r"\s+"
    r"([A-Z][A-Z\s\-]{1,70}?)"
    r"(?=\s+[A-Z]?[a-z]|\s*\Z)",
    re.UNICODE,
)

_NUMERIC_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(\d+(?:\.\d+)*)\.?\s+"
    r"((?:" + _SECTION_KEYWORDS + r"))",
    re.UNICODE,
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Section:
    """A normalized logical section of a paper."""

    heading: str
    text: str
    section_number: str = ""
    token_estimate: int = 0


@dataclass
class NormalizedPaper:
    """Normalized representation of one OCR-flattened paper."""

    paper_id: str
    corpus_index: int
    source_pdf: str
    pdf_mapping_method: str
    title: str
    authors: str
    abstract: str
    sections: list[Section]
    parse_confidence: float
    parse_method: str
    raw_char_count: int
    abstract_token_estimate: int
    total_section_tokens: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sections"] = [asdict(section) for section in self.sections]
        return data


def _clean(text: str) -> str:
    """Normalize OCR text into single-spaced plain text."""

    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def estimate_tokens(text: str) -> int:
    """Cheap offline token estimate used by normalization and chunking."""

    return len(_TOKEN_RE.findall(text))


def _fallback_abstract(text: str, abs_m: Optional[re.Match[str]]) -> str:
    if not abs_m:
        return ""
    snippet = text[abs_m.end() : abs_m.end() + 1500]
    cut = snippet.rfind(". ", 200, 1200)
    return _clean(snippet[: cut + 1] if cut > 0 else snippet)


def _clean_heading(raw: str) -> str:
    titled = _clean(raw).title()

    def _merge(match: re.Match[str]) -> str:
        frag, rest = match.group(1), match.group(2)
        return frag + rest[0].lower() + rest[1:]

    prev = None
    result = titled
    while result != prev:
        prev = result
        result = re.sub(
            r"(?<!\w)([A-Z]{1,2}) ([A-Z][a-z]{2,})",
            _merge,
            result,
        )

    words = result.split()
    return " ".join(words[:6])


def _extract_title_authors(text: str) -> tuple[str, str]:
    """Heuristically extract title and author block from leading OCR text."""

    head = text[:700]
    abs_m = _ABSTRACT_RE.search(head)
    preamble = head[: abs_m.start()].strip() if abs_m else head.strip()
    preamble_norm = preamble.replace("\x00", "  ")
    parts = [p.strip() for p in re.split(r"\s{2,}", preamble_norm) if p.strip()]

    if not parts:
        fallback = _clean(preamble[:160]) or "Unknown"
        return fallback, "Unknown"

    title_parts: list[str] = []
    author_parts: list[str] = []
    in_authors = False

    for part in parts:
        if in_authors:
            author_parts.append(part)
            continue

        if _AFFIL_RE.search(part) or len(part) < 4:
            in_authors = True
            author_parts.append(part)
            continue

        words = part.split()
        if words:
            cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
            avg_word_len = sum(len(w) for w in words) / len(words)
            if title_parts and cap_ratio > 0.75 and avg_word_len < 9 and len(words) <= 14:
                in_authors = True
                author_parts.append(part)
                continue

        title_parts.append(part)

    title = _clean(" ".join(title_parts))
    authors = _clean(" ".join(author_parts))

    if len(title) > 220:
        title = title[:220].rsplit(" ", 1)[0] + "..."

    return title or "Unknown", authors or "Unknown"


def _split_by_sections(text: str) -> tuple[list[tuple[str, str, str]], str, float]:
    """Split OCR-flattened text into numbered sections plus abstract."""

    abs_m = _ABSTRACT_RE.search(text)
    abs_end = abs_m.end() if abs_m else 0

    roman_hits = list(_ROMAN_RE.finditer(text))
    numeric_hits = list(_NUMERIC_RE.finditer(text))

    if len(roman_hits) >= len(numeric_hits) and len(roman_hits) >= 2:
        hits = roman_hits
    elif len(numeric_hits) >= 2:
        hits = numeric_hits
    else:
        hits = roman_hits if roman_hits else numeric_hits
        if not hits:
            abstract = _fallback_abstract(text, abs_m)
            return [], abstract, 0.0

    first_section = hits[0].start()
    if abs_m and abs_end < first_section:
        abstract = _clean(text[abs_end:first_section])
    else:
        abstract = _fallback_abstract(text, abs_m)

    sections: list[tuple[str, str, str]] = []
    for idx, match in enumerate(hits):
        num = match.group(1).rstrip(".")
        heading = _clean_heading(match.group(2))
        body_start = match.end()
        body_end = hits[idx + 1].start() if idx + 1 < len(hits) else len(text)
        body = _clean(text[body_start:body_end])
        if body:
            sections.append((num, heading, body))

    if not sections:
        return [], abstract, 0.0

    known = sum(1 for _, heading, _ in sections if heading.lower() in _KNOWN_HEADINGS)
    quality = known / len(sections)
    quantity = min(len(sections), 8) / 8
    confidence = round(0.55 * quality + 0.45 * quantity, 3)
    return sections, abstract, confidence


def _fallback_body_section(text: str) -> list[Section]:
    abs_m = _ABSTRACT_RE.search(text)
    body_start = abs_m.end() if abs_m else 0
    body = _clean(text[body_start:]) if body_start else _clean(text)
    return [Section(heading="", text=body, section_number="", token_estimate=estimate_tokens(body))]


def _sorted_pdfs(pdfs_dir: Path) -> list[Path]:
    return sorted(
        [path for path in pdfs_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda path: path.name.lower(),
    )


def _resolve_pdf_mapping(corpus_index: int, pdfs: list[Path], corpus_size: int) -> tuple[str, str]:
    if not pdfs:
        return "", "unavailable"
    if len(pdfs) == corpus_size and corpus_index < len(pdfs):
        return f"pdfs/{pdfs[corpus_index].name}", "positional_sorted"
    return "", "unavailable"


def normalize_record(
    text: str,
    corpus_index: int,
    corpus_size: int,
    pdfs: list[Path],
) -> NormalizedPaper:
    """Normalize a single corpus record."""

    raw = text or ""
    title, authors = _extract_title_authors(raw)
    section_tuples, abstract, confidence = _split_by_sections(raw)

    if section_tuples:
        sections = [
            Section(
                heading=heading,
                text=body,
                section_number=number,
                token_estimate=estimate_tokens(body),
            )
            for number, heading, body in section_tuples
        ]
        parse_method = "json_structured"
    else:
        sections = _fallback_body_section(raw)
        parse_method = "json_fallback"

    source_pdf, pdf_mapping_method = _resolve_pdf_mapping(corpus_index, pdfs, corpus_size)
    paper_id = f"paper_{corpus_index + 1:04d}"

    return NormalizedPaper(
        paper_id=paper_id,
        corpus_index=corpus_index,
        source_pdf=source_pdf,
        pdf_mapping_method=pdf_mapping_method,
        title=title,
        authors=authors,
        abstract=abstract,
        sections=sections,
        parse_confidence=confidence,
        parse_method=parse_method,
        raw_char_count=len(raw),
        abstract_token_estimate=estimate_tokens(abstract),
        total_section_tokens=sum(section.token_estimate for section in sections),
    )


def load_corpus(corpus_path: Path) -> list[dict]:
    """Load the raw OCR corpus JSON."""

    with corpus_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level list in {corpus_path}")
    return data


def normalize_corpus(corpus_path: Path, pdfs_dir: Optional[Path] = None) -> list[NormalizedPaper]:
    """Normalize the whole corpus into structured paper records."""

    docs = load_corpus(corpus_path)
    pdfs = _sorted_pdfs(pdfs_dir) if pdfs_dir and pdfs_dir.exists() else []
    return [
        normalize_record(
            text=str(doc.get("text", "")),
            corpus_index=index,
            corpus_size=len(docs),
            pdfs=pdfs,
        )
        for index, doc in enumerate(docs)
    ]


def build_normalization_report(papers: list[NormalizedPaper]) -> dict:
    """Build a compact corpus-normalization summary for inspection."""

    paper_count = len(papers)
    structured_count = sum(1 for paper in papers if paper.parse_method == "json_structured")
    fallback_count = paper_count - structured_count
    mapped_pdfs = sum(1 for paper in papers if paper.source_pdf)
    avg_sections = round(sum(len(paper.sections) for paper in papers) / paper_count, 2) if paper_count else 0.0
    avg_conf = round(sum(paper.parse_confidence for paper in papers) / paper_count, 3) if paper_count else 0.0
    avg_section_tokens = round(sum(paper.total_section_tokens for paper in papers) / paper_count, 1) if paper_count else 0.0

    return {
        "paper_count": paper_count,
        "structured_count": structured_count,
        "fallback_count": fallback_count,
        "pdf_mapped_count": mapped_pdfs,
        "avg_sections_per_paper": avg_sections,
        "avg_parse_confidence": avg_conf,
        "avg_section_tokens_per_paper": avg_section_tokens,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

