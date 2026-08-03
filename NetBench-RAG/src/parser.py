"""
Document parsers for the NetBench-RAG pipeline.

Two concrete parsers:
  JSONParser  — flat-text corpus (research_corpus_new.json)
                Text has been OCR-flattened: no newlines, spaces injected into
                words (e.g. "I NTRODUCTION"), all whitespace collapsed.
  PDFParser   — source PDF files via PyMuPDF
                Handles single- and two-column IEEE/ACM layouts, strips running
                page headers/footers, uses font-size metadata for structure.

Both produce a unified ParsedPaper → list[Section] structure consumed by
the chunker (src/chunker.py).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Section:
    """One logical section of a research paper."""
    heading: str          # e.g. "Introduction", "Related Work", ""
    text: str             # body text (whitespace-normalised)
    section_number: str = ""  # e.g. "1", "2.1", "III" — empty if unknown


@dataclass
class ParsedPaper:
    """Unified representation of a parsed research paper."""
    source_file: str         # basename of original file
    title: str
    authors: str
    abstract: str            # may be empty string
    sections: list[Section]  # body sections (abstract excluded)
    parse_confidence: float  # 0.0 – 1.0
    parse_method: str        # "json_structured" | "json_fallback" | "pdf"

    def all_sections(self) -> list[Section]:
        """Abstract section (if present) followed by body sections."""
        result: list[Section] = []
        if self.abstract.strip():
            result.append(Section(heading="Abstract", text=self.abstract))
        result.extend(self.sections)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Collapse whitespace runs (including null bytes), strip leading/trailing spaces."""
    text = text.replace("\x00", " ")   # OCR corpus uses null bytes as separators
    return re.sub(r"\s+", " ", text).strip()


# Section headings that confirm confident detection when matched.
_KNOWN_HEADINGS: frozenset[str] = frozenset({
    "introduction", "background", "related work", "related works",
    "motivation", "overview", "architecture", "design", "system model",
    "system design", "framework", "methodology", "method", "approach",
    "implementation", "performance", "evaluation", "experiments",
    "experiment", "results", "result", "analysis", "discussion",
    "conclusion", "conclusions", "future work", "acknowledgment",
    "acknowledgements", "acknowledgments", "references", "appendix",
})


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Parser
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section header regexes for flat OCR text ──────────────────────────────────

# Roman numeral sections: "I. I NTRODUCTION" or "III. SYSTEM DESIGN"
# The OCR injected spaces inside words, so we tolerate that inside the heading.
_ROMAN_RE = re.compile(
    r"(?<![A-Za-z0-9])"                     # not preceded by alphanum
    r"(I{1,3}V?|I?V|V?I{1,3}|I?X|X{1,3})" # Roman numeral
    r"\s*\."                                 # literal period (OCR may insert space: "IV .")
    r"\s+"                                   # whitespace
    r"([A-Z][A-Z\s\-]{1,70}?)"             # ALL-CAPS heading — non-greedy
    r"(?=\s+[A-Z]?[a-z]|\s*\Z)",           # stops at mixed-case body text
    re.UNICODE,
)

# Numeric sections with well-known headings — explicit list prevents matching
# arbitrary numbers in text (e.g. "2.0 Mbps", "1 author").
_SECTION_KEYWORDS = (
    r"Introduction|Background|Related Work(?:s)?|Motivation|Overview|"
    r"Architecture|Design|System(?:\s+(?:Model|Design|Overview))?|"
    r"Framework|Methodology|Method|Approach|Implementation|"
    r"Performance|Evaluation|Experiment(?:s|al)?|Results?|Analysis|"
    r"Discussion|Conclusion(?:s)?|Future Work|"
    r"Acknowledgments?|Acknowledgements?|References?|Appendix"
)
_NUMERIC_RE = re.compile(
    r"(?<![A-Za-z])"               # not inside a word
    r"(\d+(?:\.\d+)*)\.?\s+"       # number (e.g. "1", "2.1", "3.2.1")
    r"((?:" + _SECTION_KEYWORDS + r"))",   # keyword only — no trailing body words
    re.UNICODE,
)

# Abstract marker
_ABSTRACT_RE = re.compile(r"\bAbstract\b|\bABSTRACT\b", re.UNICODE)

# Tokens that suggest we've entered the author/affiliation block
_AFFIL_RE = re.compile(
    r"\b(?:University|Institute|Department|Laboratory|Lab|College|"
    r"School|Center|Centre|Corp|Inc|Ltd|IEEE|ACM|Email|email|@)\b",
    re.IGNORECASE,
)


def _extract_title_authors(text: str) -> tuple[str, str]:
    """
    Heuristically extract title and author string from the leading text.

    Strategy: everything before the 'Abstract' keyword (or first 600 chars) is
    the preamble.  We split the preamble on multi-space runs (common OCR
    separator) then classify parts as title vs. author/affiliation.
    """
    head = text[:600]
    abs_m = _ABSTRACT_RE.search(head)
    preamble = head[: abs_m.start()].strip() if abs_m else head

    # Null bytes (\x00) act as separators in this OCR-extracted corpus;
    # normalise them to double spaces so the split below works uniformly.
    preamble_norm = preamble.replace("\x00", "  ")
    parts = [p.strip() for p in re.split(r"\s{2,}", preamble_norm) if p.strip()]
    if not parts:
        return _clean(preamble[:150]) or "Unknown", "Unknown"

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

        # Looks like an author list: mostly capitalised short words
        words = part.split()
        if words:
            cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
            avg_word_len = sum(len(w) for w in words) / len(words)
            if title_parts and cap_ratio > 0.75 and avg_word_len < 9 and len(words) <= 12:
                in_authors = True
                author_parts.append(part)
                continue

        title_parts.append(part)

    title = _clean(" ".join(title_parts)) or _clean(preamble[:150])
    authors = _clean(" ".join(author_parts))

    if len(title) > 200:
        title = title[:200].rsplit(" ", 1)[0] + "…"

    return title or "Unknown", authors or "Unknown"


def _clean_heading(raw: str) -> str:
    """
    Fix OCR-injected spaces inside capitalised words, convert to Title Case,
    and truncate to 6 words maximum.

    "I NTRODUCTION"              → "Introduction"
    "B ACKGROUND AND MOTIV A TION" → "Background And Motiv Ation"
    "E V ALUA TION"              → "Evaluation"

    Strategy: convert to title case first, then merge 1–2 char standalone
    fragments into the following word (iterative, handles chains).
    """
    # Title case transforms "B ACKGROUND AND MOTIV A TION"
    # → "B Ackground And Motiv A Tion"
    titled = _clean(raw).title()

    # Merge short (1–2 char) OCR fragments into the following word, lowercasing
    # the second group's leading capital so the join reads naturally:
    #   "B Ackground" → "Background",  "I Ntroduction" → "Introduction"
    # Iterate until stable to handle chains like "E V Alua Tion".
    # Capture both the fragment and the following word so we can lowercase
    # the join: "B Ackground" → "Background",  "I Ntroduction" → "Introduction"
    def _merge(match: re.Match) -> str:
        frag, rest = match.group(1), match.group(2)
        return frag + rest[0].lower() + rest[1:]

    prev = None
    result = titled
    while result != prev:
        prev = result
        result = re.sub(
            r"(?<!\w)([A-Z]{1,2}) ([A-Z][a-z]{2,})",   # both groups captured
            _merge,
            result,
        )

    words = result.split()
    return " ".join(words[:6])


def _split_by_sections(
    text: str,
) -> tuple[list[tuple[str, str, str]], str, float]:
    """
    Split flat text into (section_number, heading, body) tuples.

    Returns
    -------
    sections    : list of (number, heading, body) — may be empty
    abstract    : extracted abstract text
    confidence  : 0.0 – 1.0 (higher = more reliable structure detected)
    """
    abs_m = _ABSTRACT_RE.search(text)
    abs_end = abs_m.end() if abs_m else 0

    # ── Find section header candidates ──────────────────────────────────────
    roman_hits = list(_ROMAN_RE.finditer(text))
    numeric_hits = list(_NUMERIC_RE.finditer(text))

    # Prefer the strategy that found more matches
    if len(roman_hits) >= len(numeric_hits) and len(roman_hits) >= 2:
        hits = roman_hits
    elif len(numeric_hits) >= 2:
        hits = numeric_hits
    else:
        # Neither strategy found enough — try fallback: any ≥1 match of either
        hits = roman_hits if roman_hits else numeric_hits
        if not hits:
            abstract = _fallback_abstract(text, abs_m)
            return [], abstract, 0.0

    # ── Extract abstract: between abs_end and first section ─────────────────
    first_sec = hits[0].start()
    if abs_m and abs_end < first_sec:
        abstract = _clean(text[abs_end:first_sec])
    else:
        abstract = _fallback_abstract(text, abs_m)

    # ── Build section list ───────────────────────────────────────────────────
    sections: list[tuple[str, str, str]] = []
    for i, m in enumerate(hits):
        num = m.group(1).rstrip(".")
        heading = _clean_heading(m.group(2))
        body_start = m.end()
        body_end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        body = _clean(text[body_start:body_end])
        sections.append((num, heading, body))

    # ── Confidence score ─────────────────────────────────────────────────────
    if not sections:
        return [], abstract, 0.0

    known = sum(
        1 for _, h, _ in sections if h.lower() in _KNOWN_HEADINGS
    )
    # Blend: fraction of known headings (quality) + section count (quantity)
    quality = known / len(sections)
    quantity = min(len(sections), 8) / 8
    confidence = round(0.55 * quality + 0.45 * quantity, 3)

    return sections, abstract, confidence


def _fallback_abstract(text: str, abs_m: Optional[re.Match]) -> str:
    """Extract abstract when section splitting fails."""
    if not abs_m:
        return ""
    snippet = text[abs_m.end() : abs_m.end() + 1500]
    # Cut at a sentence boundary around 1000–1200 chars
    cut = snippet.rfind(". ", 200, 1200)
    return _clean(snippet[: cut + 1] if cut > 0 else snippet)


class JSONParser:
    """
    Parses documents from the flat-text JSON corpus.

    Each document is a dict ``{"text": "..."}`` where the text has been
    extracted from a PDF and whitespace-collapsed (no newlines).
    """

    def parse(self, text: str, source_file: str) -> ParsedPaper:
        title, authors = _extract_title_authors(text)
        section_tuples, abstract, confidence = _split_by_sections(text)

        if section_tuples:
            sections = [
                Section(heading=h, text=body, section_number=num)
                for num, h, body in section_tuples
            ]
            method = "json_structured"
        else:
            # Fallback: single section containing the full body text
            abs_m = _ABSTRACT_RE.search(text)
            body_start = abs_m.end() if abs_m else 0
            body = _clean(text[body_start:]) if body_start else _clean(text)
            sections = [Section(heading="", text=body, section_number="")]
            method = "json_fallback"

        return ParsedPaper(
            source_file=source_file,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            parse_confidence=confidence,
            parse_method=method,
        )

    def parse_corpus(self, corpus_path: str) -> list[ParsedPaper]:
        """Parse all documents from the JSON corpus file."""
        with open(corpus_path, encoding="utf-8") as f:
            docs = json.load(f)
        return [
            self.parse(doc["text"], source_file=f"corpus_doc_{i:04d}")
            for i, doc in enumerate(docs)
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# PDF Parser
# ═══════════════════════════════════════════════════════════════════════════════

# Header/footer strip margin: ignore blocks in top/bottom N% of page height
_HEADER_FOOTER_MARGIN = 0.07   # 7%

# A block is "full-width" (spanning both columns) if its width exceeds this
# fraction of the page width.
_FULL_WIDTH_THRESHOLD = 0.60

# A section heading span must be shorter than this (chars) — avoids matching
# long sentences that happen to be bold or large.
_MAX_HEADING_CHARS = 120


class PDFParser:
    """
    Parses research paper PDFs using PyMuPDF.

    Features
    --------
    - Font-size-based body text detection (most frequent size = body)
    - Two-column layout detection and correct reading-order reconstruction
    - Page header/footer stripping
    - Title/authors extracted from page 0 font hierarchy
    - Section headers identified by bold flag and/or size relative to body
    - Abstract extracted by 'Abstract' keyword
    """

    def parse(self, pdf_path: str) -> ParsedPaper:
        source_file = Path(pdf_path).name
        try:
            # Suppress non-fatal MuPDF rendering warnings (corrupt fonts, zlib errors)
            # that are common in older academic PDFs and don't affect text extraction.
            fitz.TOOLS.mupdf_display_errors(False)
            doc = fitz.open(pdf_path)
            return self._parse_doc(doc, source_file)
        except Exception as exc:
            # Return a minimal fallback rather than crashing the pipeline
            return ParsedPaper(
                source_file=source_file,
                title="Unknown",
                authors="Unknown",
                abstract="",
                sections=[Section(heading="", text=f"[Parse error: {exc}]")],
                parse_confidence=0.0,
                parse_method="pdf",
            )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _parse_doc(self, doc: fitz.Document, source_file: str) -> ParsedPaper:
        body_size = self._dominant_font_size(doc)
        two_col = self._is_two_column(doc)

        # Collect all spans page by page in reading order
        all_spans: list[dict] = []
        for page in doc:
            all_spans.extend(
                self._page_spans(page, body_size, two_col)
            )

        # Rebuild document as a sequence of tagged tokens:
        # each span carries: text, is_heading, font_size
        title, authors, abstract, sections = self._structure_spans(
            all_spans, body_size
        )

        return ParsedPaper(
            source_file=source_file,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            parse_confidence=self._confidence(title, abstract, sections),
            parse_method="pdf",
        )

    # ── Font size analysis ───────────────────────────────────────────────────

    @staticmethod
    def _dominant_font_size(doc: fitz.Document) -> float:
        """
        Find the most common font size (weighted by character count).
        This is the body text size; headings will be ≥ this value.
        """
        size_chars: Counter[float] = Counter()
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        s = round(span["size"], 1)
                        size_chars[s] += len(span["text"].strip())
        if not size_chars:
            return 10.0
        return size_chars.most_common(1)[0][0]

    # ── Column detection ─────────────────────────────────────────────────────

    @staticmethod
    def _is_two_column(doc: fitz.Document) -> bool:
        """
        Heuristic: if most text blocks occupy less than 60 % of the page width,
        the document uses a two-column layout.
        """
        page = doc[0]
        pw = page.rect.width
        if pw == 0:
            return False
        block_widths = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") == 0:
                w = block["bbox"][2] - block["bbox"][0]
                block_widths.append(w / pw)
        if not block_widths:
            return False
        # Two-column when the median block width is less than 60 % of page width
        block_widths.sort()
        median = block_widths[len(block_widths) // 2]
        return median < _FULL_WIDTH_THRESHOLD

    # ── Per-page span extraction ─────────────────────────────────────────────

    def _page_spans(
        self,
        page: fitz.Page,
        body_size: float,
        two_col: bool,
    ) -> list[dict]:
        """
        Return spans for one page in correct reading order, with header/footer
        blocks stripped and each span annotated with is_heading.
        """
        pw = page.rect.width
        ph = page.rect.height
        top_cut = ph * _HEADER_FOOTER_MARGIN
        bot_cut = ph * (1 - _HEADER_FOOTER_MARGIN)

        blocks = page.get_text("dict")["blocks"]
        text_blocks = [
            b for b in blocks
            if b.get("type") == 0
            and b["bbox"][1] >= top_cut       # not in header strip
            and b["bbox"][3] <= bot_cut       # not in footer strip
        ]

        if not text_blocks:
            return []

        if two_col:
            blocks_ordered = self._two_col_order(text_blocks, pw)
        else:
            blocks_ordered = sorted(text_blocks, key=lambda b: b["bbox"][1])

        spans: list[dict] = []
        for block in blocks_ordered:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span["text"].strip()
                    if not txt:
                        continue
                    size = round(span["size"], 1)
                    is_bold = bool(span.get("flags", 0) & 16)  # bit 4 = bold
                    is_heading = (
                        len(txt) >= 4                         # no single chars/noise
                        and len(txt) <= _MAX_HEADING_CHARS
                        # At least one real word (≥3 alpha chars) — filters "<", "T", "O"
                        and any(len(w) >= 3 and w.isalpha() for w in txt.split())
                        # No mid-sentence period (avoids treating body sentences as headings)
                        and not re.search(r"\.\s+[a-z]", txt)
                        and (
                            # Non-bold: must be much larger (1.5×) to avoid author names
                            # (IEEE titles ~2.4×, arXiv section headers ~1.17× — both pass)
                            size >= body_size * 1.5
                            # Bold: near body size is sufficient for section headers.
                            # 0.92× keeps IEEE 9pt-bold abstract text (< 0.92×10=9.2) out.
                            or (is_bold and size >= body_size * 0.92)
                        )
                    )
                    spans.append({
                        "text": txt,
                        "size": size,
                        "is_bold": is_bold,
                        "is_heading": is_heading,
                        "bbox": span["bbox"],
                    })
        return spans

    @staticmethod
    def _two_col_order(blocks: list[dict], page_width: float) -> list[dict]:
        """
        Re-order blocks for a two-column page into correct reading order:
        full-width blocks (title, abstract header) in y-order first;
        then left-column blocks top-to-bottom, then right-column blocks.
        """
        mid = page_width / 2
        full, left, right = [], [], []

        for b in blocks:
            x0, _, x1, _ = b["bbox"]
            width = x1 - x0
            if width > page_width * _FULL_WIDTH_THRESHOLD:
                full.append(b)
            elif x1 < mid + 20:    # small tolerance for column gutter
                left.append(b)
            else:
                right.append(b)

        full.sort(key=lambda b: b["bbox"][1])
        left.sort(key=lambda b: b["bbox"][1])
        right.sort(key=lambda b: b["bbox"][1])

        return full + left + right

    # ── Document structure assembly ──────────────────────────────────────────

    def _structure_spans(
        self,
        spans: list[dict],
        body_size: float,
    ) -> tuple[str, str, str, list[Section]]:
        """
        Walk the ordered spans and identify:
          1. Title  (largest font size on page 0 — already the first spans)
          2. Authors (smaller text before Abstract)
          3. Abstract (after 'Abstract' heading span)
          4. Sections (each introduced by a heading span)
        """
        if not spans:
            return "Unknown", "Unknown", "", []

        # ── Title: largest font block at the start ───────────────────────────
        max_size = max(s["size"] for s in spans[:30])  # only look at top spans
        title_parts: list[str] = []
        idx = 0
        while idx < len(spans) and spans[idx]["size"] >= max_size * 0.90:
            title_parts.append(spans[idx]["text"])
            idx += 1

        title = _clean(" ".join(title_parts)) or "Unknown"

        # ── Authors and abstract: scan until first real section heading ──────
        pre_section: list[str] = []
        abstract_parts: list[str] = []
        in_abstract = False
        section_start_idx = idx  # will be updated below

        for i in range(idx, len(spans)):
            s = spans[i]
            txt = s["text"]

            # 'Abstract' keyword triggers abstract collection.
            # Catch both standalone "Abstract" heading spans AND IEEE-style
            # "Abstract—" / "Abstract:" inline labels that may not be flagged
            # as is_heading due to their smaller font size.
            if re.match(r"^Abstract[\W]?$|^ABSTRACT$", txt) or (
                txt.lower().startswith("abstract") and len(txt) <= 20
            ):
                in_abstract = True
                section_start_idx = i + 1
                continue

            # A heading span that is NOT 'Abstract' ends the pre-section block
            if s["is_heading"] and not re.match(r"abstract", txt, re.IGNORECASE):
                section_start_idx = i
                break

            if in_abstract:
                abstract_parts.append(txt)
            else:
                pre_section.append(txt)

        authors = _clean(" ".join(pre_section))
        # Trim institution / email junk from authors
        authors = re.split(r"\b(?:Abstract|ABSTRACT|University|Institute)\b", authors)[0].strip()

        abstract = _clean(" ".join(abstract_parts))

        # ── Sections: heading span → body spans until next heading ───────────
        sections: list[Section] = []
        current_heading = ""
        current_num = ""
        current_body: list[str] = []

        for s in spans[section_start_idx:]:
            if s["is_heading"]:
                # Save previous section
                if current_body:
                    body = _clean(" ".join(current_body))
                    if body:
                        sections.append(
                            Section(
                                heading=current_heading,
                                text=body,
                                section_number=current_num,
                            )
                        )
                # Parse number + heading from the heading span
                current_num, current_heading = self._parse_section_label(s["text"])
                current_body = []
            else:
                current_body.append(s["text"])

        # Flush last section
        if current_body:
            body = _clean(" ".join(current_body))
            if body:
                sections.append(
                    Section(
                        heading=current_heading,
                        text=body,
                        section_number=current_num,
                    )
                )

        return title, authors, abstract, sections

    @staticmethod
    def _parse_section_label(text: str) -> tuple[str, str]:
        """
        Extract section number and clean heading from a raw heading span.
        "III. SYSTEM DESIGN" → ("III", "System Design")
        "2.1 Background"     → ("2.1", "Background")
        "Conclusion"         → ("",    "Conclusion")
        """
        m = re.match(
            r"^((?:I{1,3}V?|I?V|V?I{1,3}|I?X|X{1,3})\.?|(\d+(?:\.\d+)*?)\.?)"
            r"\s+(.+)$",
            text.strip(),
        )
        if m:
            num = m.group(1).rstrip(".").strip()
            heading = _clean(m.group(3)).title()
        else:
            num = ""
            heading = _clean(text).title()
        return num, heading

    @staticmethod
    def _confidence(title: str, abstract: str, sections: list[Section]) -> float:
        if title == "Unknown" and not abstract and not sections:
            return 0.0
        score = 0.0
        if title != "Unknown":
            score += 0.2
        if abstract:
            score += 0.2
        known = sum(1 for s in sections if s.heading.lower() in _KNOWN_HEADINGS)
        if sections:
            score += 0.3 * min(len(sections), 6) / 6
            score += 0.3 * (known / len(sections))
        return round(min(score, 1.0), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_document(path: str) -> ParsedPaper:
    """
    Parse a single document (PDF or JSON) and return a ParsedPaper.
    For JSON, the file must contain a single dict with a 'text' key.
    """
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return PDFParser().parse(path)
    elif p.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
        return JSONParser().parse(text, source_file=p.name)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def parse_pdf_directory(pdfs_dir: str) -> list[ParsedPaper]:
    """Parse all PDFs in a directory."""
    parser = PDFParser()
    papers = []
    for pdf_path in sorted(Path(pdfs_dir).glob("*.pdf")):
        papers.append(parser.parse(str(pdf_path)))
    return papers
