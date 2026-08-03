"""Phase 3 evidence chunking for the Instruct-FTD pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .normalize import NormalizedPaper, Section, estimate_tokens


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(])")
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_INFORMATIVE_HEADINGS = {
    "background",
    "motivation",
    "design",
    "architecture",
    "framework",
    "method",
    "methodology",
    "implementation",
    "evaluation",
    "results",
    "analysis",
    "discussion",
    "conclusion",
}


@dataclass
class Chunk:
    """Reusable evidence chunk for HPN and RAG data generation."""

    chunk_id: str
    paper_id: str
    paper_title: str
    source_pdf: str
    chunk_profile: str
    section_heading: str
    section_number: str
    is_abstract: bool
    chunk_index: int
    total_chunks: int
    token_estimate: int
    context_prefix: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagPromptBundle:
    """Bundle of RAG chunks that can later seed grounded QA generation."""

    bundle_id: str
    bundle_type: str
    paper_ids: list[str]
    chunk_ids: list[str]
    prompt_length_bucket: str
    excerpt_count: int
    token_estimate: int

    def to_dict(self) -> dict:
        return asdict(self)


def _section_label(section: Section, is_abstract: bool) -> str:
    if is_abstract:
        return "Abstract"
    if section.section_number:
        return f"{section.section_number} {section.heading}".strip()
    return section.heading or "Body"


def _context_prefix(paper: NormalizedPaper, section: Section, is_abstract: bool) -> str:
    return (
        f"Paper: '{paper.title}' | Section: {_section_label(section, is_abstract)}"
    )


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= max_tokens:
        return text
    end = matches[max_tokens - 1].end()
    return text[:end].strip()


def _tail_by_tokens(text: str, max_tokens: int) -> str:
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= max_tokens:
        return text
    start = matches[-max_tokens].start()
    return text[start:].strip()


class ChunkProfile:
    """Offline chunking profile with approximate token constraints."""

    def __init__(self, name: str, target_tokens: int, overlap_tokens: int, min_tokens: int) -> None:
        self.name = name
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def chunk_text(self, text: str) -> list[str]:
        sentences = _split_sentences(text)
        if not sentences:
            cleaned = text.strip()
            return [cleaned] if cleaned else []

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if estimate_tokens(candidate) <= self.target_tokens:
                current = candidate
                continue

            if current and estimate_tokens(current) >= self.min_tokens:
                chunks.append(current)
                overlap = _tail_by_tokens(current, self.overlap_tokens)
                current = f"{overlap} {sentence}".strip() if overlap else sentence
            else:
                long_sentence = _truncate_to_tokens(sentence, self.target_tokens)
                if estimate_tokens(long_sentence) >= self.min_tokens:
                    chunks.append(long_sentence)
                current = ""

        if current and estimate_tokens(current) >= self.min_tokens:
            chunks.append(current)

        return chunks


def build_chunks_for_profile(papers: list[NormalizedPaper], profile: ChunkProfile) -> list[Chunk]:
    """Chunk all papers for one profile."""

    rows: list[Chunk] = []

    for paper in papers:
        paper_chunks: list[Chunk] = []

        if paper.abstract.strip():
            abstract_section = Section(
                heading="Abstract",
                text=paper.abstract.strip(),
                section_number="",
                token_estimate=estimate_tokens(paper.abstract),
            )
            abstract_text = _truncate_to_tokens(paper.abstract.strip(), profile.target_tokens)
            paper_chunks.append(
                Chunk(
                    chunk_id="",
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    source_pdf=paper.source_pdf,
                    chunk_profile=profile.name,
                    section_heading="Abstract",
                    section_number="",
                    is_abstract=True,
                    chunk_index=0,
                    total_chunks=0,
                    token_estimate=estimate_tokens(abstract_text),
                    context_prefix=_context_prefix(paper, abstract_section, True),
                    text=abstract_text,
                )
            )

        for section in paper.sections:
            for part in profile.chunk_text(section.text):
                paper_chunks.append(
                    Chunk(
                        chunk_id="",
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        source_pdf=paper.source_pdf,
                        chunk_profile=profile.name,
                        section_heading=section.heading,
                        section_number=section.section_number,
                        is_abstract=False,
                        chunk_index=0,
                        total_chunks=0,
                        token_estimate=estimate_tokens(part),
                        context_prefix=_context_prefix(paper, section, False),
                        text=part,
                    )
                )

        for index, chunk in enumerate(paper_chunks):
            chunk.chunk_index = index
            chunk.total_chunks = len(paper_chunks)
            chunk.chunk_id = (
                f"{paper.paper_id}_{profile.name}_{index:03d}"
            )

        rows.extend(paper_chunks)

    return rows


def _length_bucket(token_estimate: int) -> str:
    if token_estimate < 400:
        return "short"
    if token_estimate < 900:
        return "medium"
    if token_estimate < 1600:
        return "long"
    return "xlong"


def build_rag_bundles(chunks: list[Chunk], max_windows_per_paper: int = 4) -> list[RagPromptBundle]:
    """Build deterministic bundle candidates for later grounded QA generation."""

    bundles: list[RagPromptBundle] = []
    by_paper: dict[str, list[Chunk]] = {}
    by_heading: dict[str, list[Chunk]] = {}

    for chunk in chunks:
        by_paper.setdefault(chunk.paper_id, []).append(chunk)
        normalized_heading = (chunk.section_heading or "").strip().lower()
        if normalized_heading in _INFORMATIVE_HEADINGS:
            by_heading.setdefault(normalized_heading, []).append(chunk)

    bundle_index = 0

    for paper_id, paper_chunks in by_paper.items():
        window_count = 0
        for start in range(0, max(0, len(paper_chunks) - 1)):
            if window_count >= max_windows_per_paper:
                break
            window = paper_chunks[start : start + 2]
            if len(window) < 2:
                continue
            token_estimate = sum(chunk.token_estimate for chunk in window)
            bundles.append(
                RagPromptBundle(
                    bundle_id=f"rag_bundle_{bundle_index:05d}",
                    bundle_type="same_paper_adjacent",
                    paper_ids=[paper_id],
                    chunk_ids=[chunk.chunk_id for chunk in window],
                    prompt_length_bucket=_length_bucket(token_estimate),
                    excerpt_count=len(window),
                    token_estimate=token_estimate,
                )
            )
            bundle_index += 1
            window_count += 1

        abstracts = [chunk for chunk in paper_chunks if chunk.is_abstract]
        bodies = [chunk for chunk in paper_chunks if not chunk.is_abstract]
        if abstracts and bodies:
            pair = [abstracts[0], bodies[0]]
            token_estimate = sum(chunk.token_estimate for chunk in pair)
            bundles.append(
                RagPromptBundle(
                    bundle_id=f"rag_bundle_{bundle_index:05d}",
                    bundle_type="abstract_plus_body",
                    paper_ids=[paper_id],
                    chunk_ids=[chunk.chunk_id for chunk in pair],
                    prompt_length_bucket=_length_bucket(token_estimate),
                    excerpt_count=len(pair),
                    token_estimate=token_estimate,
                )
            )
            bundle_index += 1

    for heading, heading_chunks in sorted(by_heading.items()):
        if len(heading_chunks) < 2:
            continue
        first = heading_chunks[0]
        second = next(
            (chunk for chunk in heading_chunks[1:] if chunk.paper_id != first.paper_id),
            None,
        )
        if not second:
            continue
        pair = [first, second]
        token_estimate = sum(chunk.token_estimate for chunk in pair)
        bundles.append(
            RagPromptBundle(
                bundle_id=f"rag_bundle_{bundle_index:05d}",
                bundle_type=f"cross_paper_{heading.replace(' ', '_')}",
                paper_ids=[chunk.paper_id for chunk in pair],
                chunk_ids=[chunk.chunk_id for chunk in pair],
                prompt_length_bucket=_length_bucket(token_estimate),
                excerpt_count=len(pair),
                token_estimate=token_estimate,
            )
        )
        bundle_index += 1

    return bundles


def read_normalized_papers(path: Path) -> list[NormalizedPaper]:
    """Load normalized papers from JSONL."""

    rows: list[NormalizedPaper] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sections = [Section(**section) for section in row.get("sections", [])]
            row["sections"] = sections
            rows.append(NormalizedPaper(**row))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

