from __future__ import annotations

import unittest

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.chunking import Chunk, RagPromptBundle
from instruct_ftd.normalize import NormalizedPaper, Section
from instruct_ftd.pilot import (
    _parse_generated_json,
    _prompt_length_bucket,
    build_pilot_requests,
    choose_rag_bundle,
)


class PilotHelperTests(unittest.TestCase):
    def test_parse_generated_json_handles_fences_and_trailing_text(self) -> None:
        fenced = """```json\n{"question": "Q", "response": "A"}\n```"""
        noisy = '{"question": "Q2", "response": "A2"} trailing text'

        self.assertEqual(_parse_generated_json(fenced)["question"], "Q")
        self.assertEqual(_parse_generated_json(noisy)["response"], "A2")

    def test_choose_rag_bundle_prefers_shorter_abstract_plus_body(self) -> None:
        bundles = [
            RagPromptBundle("b1", "abstract_plus_body", ["paper_1"], ["c1", "c2"], "long", 2, 1200),
            RagPromptBundle("b2", "abstract_plus_body", ["paper_1"], ["c3", "c4"], "short", 2, 300),
            RagPromptBundle("b3", "same_paper_adjacent", ["paper_1"], ["c5", "c6"], "short", 2, 250),
        ]

        self.assertEqual(choose_rag_bundle(bundles).bundle_id, "b2")

    def test_prompt_length_bucket_reflects_chunk_size(self) -> None:
        short_chunk = Chunk("c1", "p1", "Title", "pdfs/p1.pdf", "short", "Intro", "1", False, 0, 1, 120, "", "text")
        rag_chunk = Chunk("c2", "p1", "Title", "pdfs/p1.pdf", "rag", "Results", "2", False, 1, 2, 500, "", "text")

        self.assertEqual(_prompt_length_bucket([short_chunk], "hpn"), "short")
        self.assertEqual(_prompt_length_bucket([short_chunk, rag_chunk], "rag"), "medium")

    def test_build_pilot_requests_assigns_stable_request_ids(self) -> None:
        papers = [
            NormalizedPaper(
                paper_id="paper_0001",
                corpus_index=0,
                source_pdf="pdfs/p1.pdf",
                pdf_mapping_method="positional_sorted",
                title="Paper",
                authors="Author",
                abstract="Abstract about RDMA and congestion control.",
                sections=[Section(heading="Design", text="RDMA congestion control design details.", section_number="1", token_estimate=10)],
                parse_confidence=0.8,
                parse_method="json_structured",
                raw_char_count=50,
                abstract_token_estimate=8,
                total_section_tokens=10,
            )
        ]
        short_chunks = [
            Chunk("paper_0001_short_000", "paper_0001", "Paper", "pdfs/p1.pdf", "short", "Design", "1", False, 0, 1, 120, "", "RDMA congestion control design details.")
        ]
        rag_chunks = [
            Chunk("paper_0001_rag_000", "paper_0001", "Paper", "pdfs/p1.pdf", "rag", "Abstract", "", True, 0, 2, 150, "", "Abstract text."),
            Chunk("paper_0001_rag_001", "paper_0001", "Paper", "pdfs/p1.pdf", "rag", "Design", "1", False, 1, 2, 220, "", "Body text."),
        ]
        rag_bundles = [
            RagPromptBundle("rag_bundle_00001", "abstract_plus_body", ["paper_0001"], ["paper_0001_rag_000", "paper_0001_rag_001"], "short", 2, 370)
        ]

        requests = build_pilot_requests(
            papers=papers,
            short_chunks=short_chunks,
            rag_chunks=rag_chunks,
            rag_bundles=rag_bundles,
            family_names=["hpn_fact_qa", "rag_grounded_qa"],
            difficulty_cycle=["easy", "medium", "hard"],
        )

        self.assertEqual([request.request_id for request in requests], ["pilot_req_000000", "pilot_req_000001"])


if __name__ == "__main__":
    unittest.main()
