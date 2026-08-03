"""
Chunker for the NetBench-RAG pipeline.

Converts ParsedPaper objects (from src/parser.py) into Chunk objects ready
for embedding and storage.

Strategy
--------
1. Always emit a dedicated abstract chunk (if abstract is present).
2. For each section, split at paragraph boundaries first; only fall back to
   sentence-level splitting when a paragraph is still too large.
3. Target ~1200 tokens per chunk (bert-base-uncased tokenizer for counting).
4. 150-token overlap between consecutive chunks within the same section.
5. Discard chunks under 100 tokens.
6. Every chunk carries a context prefix prepended to the text before embedding:
       Paper: '<title>' | Section: <num> <heading> | Authors: <authors>
   The prefix is stored separately so it can be stripped for display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from tokenizers import Tokenizer

from .parser import ParsedPaper, Section


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    One embeddable unit of text from a research paper.

    ``text_for_embedding`` = context_prefix + " " + text
    ``text`` alone is shown to the LLM and in retrieval results.
    """
    # Core content
    text: str                   # body text of the chunk (no prefix)
    context_prefix: str         # "Paper: '...' | Section: ... | Authors: ..."

    # Paper-level metadata
    paper_title: str
    authors: str
    source_file: str
    parse_confidence: float
    parse_method: str

    # Section-level metadata
    section_heading: str        # e.g. "Introduction"
    section_number: str         # e.g. "1", "III", "" for abstract
    is_abstract: bool = False

    # Position metadata
    chunk_index: int = 0        # 0-based index within the paper
    total_chunks: int = 0       # total chunks for this paper (filled in post)

    @property
    def text_for_embedding(self) -> str:
        """Text passed to the embedding model."""
        return f"{self.context_prefix} {self.text}"


# ═══════════════════════════════════════════════════════════════════════════════
# Token counting
# ═══════════════════════════════════════════════════════════════════════════════

class _TokenCounter:
    """
    Thin wrapper around a HuggingFace fast tokenizer for token counting.
    Uses bert-base-uncased (model-agnostic, fast, no special tokens needed).
    Loaded once and reused.
    """

    _instance: Optional[_TokenCounter] = None

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self._tok = Tokenizer.from_pretrained(model_name)

    @classmethod
    def get(cls, model_name: str = "bert-base-uncased") -> "_TokenCounter":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        """
        Return the largest prefix of *text* that fits in *max_tokens*.
        Operates at the token level, then reconstructs from the offsets.
        """
        enc = self._tok.encode(text, add_special_tokens=False)
        if len(enc.ids) <= max_tokens:
            return text
        # Use character offsets from the encoding to slice exactly
        last_offset = enc.offsets[max_tokens - 1][1]  # end of last kept token
        return text[:last_offset]


# ═══════════════════════════════════════════════════════════════════════════════
# Text splitting helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Paragraph boundary: two or more whitespace chars OR double newlines.
# NOTE: both parsers normalise whitespace to single spaces (via _clean()),
# so this pattern never fires on the current corpus.  Long sections fall
# through directly to sentence-level splitting in _text_to_units(), which
# produces correct results.  If the corpus is ever extended with documents
# that preserve newlines, this pattern will activate automatically.
_PARA_SPLIT_RE = re.compile(r"(?<=[.!?])\s{2,}|\n{2,}")

# Sentence boundary — split after . ! ? followed by whitespace + capital.
# Avoids splitting at "Fig. 2" or "e.g. the" by requiring the capital after.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(])")


def _split_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraphs; each non-empty paragraph is a string."""
    parts = _PARA_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences."""
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# Core chunker
# ═══════════════════════════════════════════════════════════════════════════════

class Chunker:
    """
    Converts a ParsedPaper into a list of Chunk objects.

    Parameters
    ----------
    target_tokens   : desired chunk size in tokens (default 1200)
    overlap_tokens  : overlap carried from previous chunk (default 150)
    min_tokens      : chunks shorter than this are discarded (default 100)
    tokenizer_model : HuggingFace tokenizer model name for token counting
    """

    def __init__(
        self,
        target_tokens: int = 1200,
        overlap_tokens: int = 150,
        min_tokens: int = 100,
        tokenizer_model: str = "bert-base-uncased",
    ) -> None:
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.min_tokens = min_tokens
        self._tc = _TokenCounter.get(tokenizer_model)

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_paper(self, paper: ParsedPaper) -> list[Chunk]:
        """
        Chunk a single ParsedPaper.

        Returns a list of Chunk objects with chunk_index and total_chunks filled.
        """
        chunks: list[Chunk] = []

        # 1. Abstract chunk (always a single dedicated chunk)
        if paper.abstract.strip():
            chunks.append(self._make_abstract_chunk(paper))

        # 2. Body section chunks
        for section in paper.sections:
            chunks.extend(self._chunk_section(paper, section))

        # Fill in positional metadata
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_corpus(self, papers: list[ParsedPaper]) -> list[Chunk]:
        """Chunk all papers in a corpus."""
        all_chunks: list[Chunk] = []
        for paper in papers:
            all_chunks.extend(self.chunk_paper(paper))
        return all_chunks

    # ── Abstract chunk ────────────────────────────────────────────────────────

    def _make_abstract_chunk(self, paper: ParsedPaper) -> Chunk:
        """
        Create a single chunk for the paper abstract.

        If the abstract is unusually long (> target_tokens), it is truncated
        at a sentence boundary rather than split — abstracts should stay whole.
        """
        text = paper.abstract.strip()
        if self._tc.count(text) > self.target:
            text = self._truncate_at_sentence(text, self.target)

        prefix = self._build_prefix(
            title=paper.title,
            authors=paper.authors,
            section_number="",
            section_heading="Abstract",
        )
        return Chunk(
            text=text,
            context_prefix=prefix,
            paper_title=paper.title,
            authors=paper.authors,
            source_file=paper.source_file,
            parse_confidence=paper.parse_confidence,
            parse_method=paper.parse_method,
            section_heading="Abstract",
            section_number="",
            is_abstract=True,
        )

    # ── Section chunking ──────────────────────────────────────────────────────

    def _chunk_section(self, paper: ParsedPaper, section: Section) -> list[Chunk]:
        """
        Chunk a single section body into one or more Chunk objects.

        Algorithm
        ---------
        1. Try paragraph-level splitting first.
        2. If a single paragraph still exceeds target, split at sentence level.
        3. Pack units greedily into chunks of ≤ target_tokens.
        4. Carry an overlap tail from the previous chunk into the next.
        5. Discard chunks under min_tokens.
        """
        text = section.text.strip()
        if not text:
            return []

        prefix = self._build_prefix(
            title=paper.title,
            authors=paper.authors,
            section_number=section.section_number,
            section_heading=section.heading,
        )

        units = self._text_to_units(text)
        if not units:
            return []

        raw_chunks = self._pack_units(units)

        result: list[Chunk] = []
        for chunk_text in raw_chunks:
            n = self._tc.count(chunk_text)
            if n < self.min_tokens:
                continue
            result.append(Chunk(
                text=chunk_text,
                context_prefix=prefix,
                paper_title=paper.title,
                authors=paper.authors,
                source_file=paper.source_file,
                parse_confidence=paper.parse_confidence,
                parse_method=paper.parse_method,
                section_heading=section.heading,
                section_number=section.section_number,
                is_abstract=False,
            ))
        return result

    # ── Unit splitting ────────────────────────────────────────────────────────

    def _text_to_units(self, text: str) -> list[str]:
        """
        Break *text* into atomic units for packing.

        A unit is either:
        - A paragraph (if it fits within target_tokens), or
        - Individual sentences within an over-large paragraph.
        """
        units: list[str] = []
        for para in _split_paragraphs(text) or [text]:
            if self._tc.count(para) <= self.target:
                units.append(para)
            else:
                # Paragraph too large — fall back to sentence splitting
                for sent in _split_sentences(para) or [para]:
                    if self._tc.count(sent) <= self.target:
                        units.append(sent)
                    else:
                        # Single sentence exceeds target — hard truncate
                        units.append(
                            self._tc.truncate_tokens(sent, self.target)
                        )
        return units

    # ── Greedy packing with overlap ───────────────────────────────────────────

    def _pack_units(self, units: list[str]) -> list[str]:
        """
        Greedily pack units into chunks of ≤ target_tokens.

        When a new chunk starts, it is seeded with the overlap tail from the
        previous chunk (the last `overlap_tokens` tokens of the previous chunk).
        """
        if not units:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens: int = 0
        overlap_tail: str = ""   # carried overlap from last chunk

        for unit in units:
            unit_tokens = self._tc.count(unit)

            # If adding this unit would overflow, flush first
            if current_tokens + unit_tokens > self.target and current_parts:
                chunk_text = " ".join(current_parts)
                chunks.append(chunk_text)

                # Build overlap tail: take the trailing overlap_tokens from chunk
                overlap_tail = self._trailing_tokens(chunk_text, self.overlap)
                overlap_tail_tokens = self._tc.count(overlap_tail)

                # Start next chunk with the overlap tail + current unit
                current_parts = [overlap_tail, unit] if overlap_tail else [unit]
                current_tokens = overlap_tail_tokens + unit_tokens
            else:
                current_parts.append(unit)
                current_tokens += unit_tokens

        # Flush the last chunk
        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks

    # ── Overlap / truncation helpers ──────────────────────────────────────────

    def _trailing_tokens(self, text: str, n_tokens: int) -> str:
        """
        Return the last *n_tokens* tokens of *text* as a string.
        Tries to start at a sentence boundary to keep the overlap coherent.
        """
        enc = self._tc._tok.encode(text, add_special_tokens=False)
        total = len(enc.ids)
        if total <= n_tokens:
            return text

        start_token = total - n_tokens
        # Character offset of the start token
        start_char = enc.offsets[start_token][0]
        tail = text[start_char:].strip()

        # Prefer to start at a sentence boundary within ±20 % of target
        lower = max(0, start_char - len(text) // 10)
        upper = min(len(text), start_char + len(text) // 10)
        window = text[lower:upper]
        sent_m = list(re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", window))
        if sent_m:
            # Pick the boundary closest to start_char
            best = min(sent_m, key=lambda m: abs((lower + m.end()) - start_char))
            candidate = text[lower + best.end():].strip()
            if self._tc.count(candidate) <= n_tokens + 20:
                tail = candidate

        return tail

    def _truncate_at_sentence(self, text: str, max_tokens: int) -> str:
        """
        Truncate *text* to at most *max_tokens*, ending at a sentence boundary.
        Falls back to hard token truncation if no sentence boundary is found.
        """
        hard = self._tc.truncate_tokens(text, max_tokens)
        # Find the last sentence-ending period before the hard cut
        cut = hard.rfind(". ")
        if cut > len(hard) // 2:
            return hard[: cut + 1]
        return hard

    # ── Context prefix builder ────────────────────────────────────────────────

    @staticmethod
    def _build_prefix(
        title: str,
        authors: str,
        section_number: str,
        section_heading: str,
    ) -> str:
        """
        Build the context prefix string.

        Examples
        --------
        Paper: 'GridFTP Architecture...' | Section: 3.2 Pipelining | Authors: Allcock et al.
        Paper: 'BBR: Congestion-Based...' | Section: Abstract | Authors: Cardwell et al.
        Paper: 'Unknown' | Section: Introduction | Authors: Unknown
        """
        # Trim title to ~80 chars for readability
        display_title = title if len(title) <= 80 else title[:77] + "…"

        if section_number:
            sec_label = f"{section_number} {section_heading}".strip()
        else:
            sec_label = section_heading or "Body"

        # Abbreviate author string: keep up to first comma + "et al." if long
        display_authors = authors
        if len(authors) > 60:
            first_comma = authors.find(",")
            if first_comma > 0:
                display_authors = authors[:first_comma] + " et al."
            else:
                display_authors = authors[:57] + "…"

        return (
            f"Paper: '{display_title}' | "
            f"Section: {sec_label} | "
            f"Authors: {display_authors}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience factory from config
# ═══════════════════════════════════════════════════════════════════════════════

def chunker_from_config(cfg: dict) -> Chunker:
    """
    Instantiate a Chunker from the ``chunking`` section of config.yaml.

    Parameters
    ----------
    cfg : the full parsed config dict (yaml.safe_load output)
    """
    c = cfg.get("chunking", {})
    return Chunker(
        target_tokens=c.get("target_tokens", 1200),
        overlap_tokens=c.get("overlap_tokens", 150),
        min_tokens=c.get("min_tokens", 100),
        tokenizer_model=c.get("tokenizer_model", "bert-base-uncased"),
    )
