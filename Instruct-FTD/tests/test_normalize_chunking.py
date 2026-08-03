from __future__ import annotations

import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.chunking import ChunkProfile, build_chunks_for_profile, build_rag_bundles
from instruct_ftd.normalize import NormalizedPaper, Section, normalize_record


class NormalizeAndChunkingTests(unittest.TestCase):
    def test_normalize_record_extracts_sections_and_pdf_mapping(self) -> None:
        raw_text = (
            "FAST TCP for Long Fat Networks  Jane Doe John Roe  University of Example  "
            "ABSTRACT FAST TCP uses queueing delay to react before severe loss occurs. "
            "1 Introduction Long fat networks need stable congestion control. "
            "2 Design The protocol uses queueing delay as a congestion signal. "
            "3 Evaluation Experiments show better stability and throughput."
        )

        paper = normalize_record(
            text=raw_text,
            corpus_index=0,
            corpus_size=1,
            pdfs=[Path("fast_tcp.pdf")],
        )

        self.assertEqual(paper.paper_id, "paper_0001")
        self.assertEqual(paper.parse_method, "json_structured")
        self.assertEqual(paper.source_pdf, "pdfs/fast_tcp.pdf")
        self.assertGreaterEqual(len(paper.sections), 3)
        self.assertNotEqual(paper.title, "Unknown")

    def test_chunking_builds_rag_bundles(self) -> None:
        papers = [
            NormalizedPaper(
                paper_id="paper_0001",
                corpus_index=0,
                source_pdf="pdfs/a.pdf",
                pdf_mapping_method="positional_sorted",
                title="Paper A",
                authors="A",
                abstract="This abstract explains ECN-based transport tuning for data centers.",
                sections=[
                    Section(heading="Evaluation", text="The evaluation reports a throughput gain and lower latency under load.", section_number="1", token_estimate=20),
                    Section(heading="Discussion", text="The discussion explains why the queue growth stays bounded.", section_number="2", token_estimate=20),
                ],
                parse_confidence=0.8,
                parse_method="json_structured",
                raw_char_count=100,
                abstract_token_estimate=10,
                total_section_tokens=40,
            ),
            NormalizedPaper(
                paper_id="paper_0002",
                corpus_index=1,
                source_pdf="pdfs/b.pdf",
                pdf_mapping_method="positional_sorted",
                title="Paper B",
                authors="B",
                abstract="This abstract covers congestion signaling and telemetry.",
                sections=[
                    Section(heading="Evaluation", text="The evaluation compares telemetry-guided control against ECN-only control.", section_number="1", token_estimate=20),
                ],
                parse_confidence=0.8,
                parse_method="json_structured",
                raw_char_count=100,
                abstract_token_estimate=10,
                total_section_tokens=20,
            ),
        ]

        chunks = build_chunks_for_profile(papers, ChunkProfile("rag", 120, 20, 10))
        bundles = build_rag_bundles(chunks, max_windows_per_paper=2)

        bundle_types = {bundle.bundle_type for bundle in bundles}
        self.assertIn("abstract_plus_body", bundle_types)
        self.assertTrue(any(bundle.bundle_type.startswith("cross_paper_") for bundle in bundles))


if __name__ == "__main__":
    unittest.main()
