from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.candidate_generation import run_candidate_generation
from instruct_ftd.chunking import Chunk, RagPromptBundle
from instruct_ftd.normalize import NormalizedPaper, Section
from instruct_ftd.pilot import load_chunks, run_pilot
from instruct_ftd.progress import read_jsonl


def build_fixture_papers() -> list[NormalizedPaper]:
    return [
        NormalizedPaper(
            paper_id="paper_0001",
            corpus_index=0,
            source_pdf="pdfs/p1.pdf",
            pdf_mapping_method="positional_sorted",
            title="RDMA Congestion Control Paper",
            authors="Author",
            abstract="This paper studies RDMA congestion control and network stability.",
            sections=[
                Section(heading="Design", text="The design explains RDMA congestion control and queue stability.", section_number="1", token_estimate=30),
                Section(heading="Evaluation", text="The evaluation reports latency and throughput gains under load.", section_number="2", token_estimate=30),
            ],
            parse_confidence=0.9,
            parse_method="json_structured",
            raw_char_count=200,
            abstract_token_estimate=12,
            total_section_tokens=60,
        )
    ]


def build_fixture_chunks() -> tuple[list[Chunk], list[Chunk], list[RagPromptBundle]]:
    short_chunks = [
        Chunk("paper_0001_short_000", "paper_0001", "RDMA Congestion Control Paper", "pdfs/p1.pdf", "short", "Design", "1", False, 0, 2, 150, "", "The design explains RDMA congestion control and queue stability."),
        Chunk("paper_0001_short_001", "paper_0001", "RDMA Congestion Control Paper", "pdfs/p1.pdf", "short", "Evaluation", "2", False, 1, 2, 160, "", "The evaluation reports latency and throughput gains under load."),
    ]
    rag_chunks = [
        Chunk("paper_0001_rag_000", "paper_0001", "RDMA Congestion Control Paper", "pdfs/p1.pdf", "rag", "Abstract", "", True, 0, 2, 180, "", "This paper studies RDMA congestion control and network stability."),
        Chunk("paper_0001_rag_001", "paper_0001", "RDMA Congestion Control Paper", "pdfs/p1.pdf", "rag", "Evaluation", "2", False, 1, 2, 220, "", "The evaluation reports latency and throughput gains under load."),
    ]
    rag_bundles = [
        RagPromptBundle("rag_bundle_00001", "abstract_plus_body", ["paper_0001"], ["paper_0001_rag_000", "paper_0001_rag_001"], "short", 2, 400)
    ]
    return short_chunks, rag_chunks, rag_bundles


class ResumeGenerationTests(unittest.TestCase):
    def test_run_pilot_dry_run_can_resume_without_duplicate_work(self) -> None:
        papers = build_fixture_papers()
        short_chunks, rag_chunks, rag_bundles = build_fixture_chunks()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "pilot_run"
            first = run_pilot(
                model="unused",
                output_dir=output_dir,
                papers=papers,
                short_chunks=short_chunks,
                rag_chunks=rag_chunks,
                rag_bundles=rag_bundles,
                family_names=["hpn_fact_qa", "rag_grounded_qa"],
                paper_limit=1,
                min_parse_confidence=0.4,
                temperature=0.0,
                top_p=0.9,
                num_predict=10,
                base_url="http://127.0.0.1:9",
                dry_run=True,
                resume=False,
            )
            second = run_pilot(
                model="unused",
                output_dir=output_dir,
                papers=papers,
                short_chunks=short_chunks,
                rag_chunks=rag_chunks,
                rag_bundles=rag_bundles,
                family_names=["hpn_fact_qa", "rag_grounded_qa"],
                paper_limit=1,
                min_parse_confidence=0.4,
                temperature=0.0,
                top_p=0.9,
                num_predict=10,
                base_url="http://127.0.0.1:9",
                dry_run=True,
                resume=True,
            )

            checkpoint_rows = read_jsonl(output_dir / "pilot_checkpoint.jsonl")
            raw_rows = read_jsonl(output_dir / "pilot_raw_generations.jsonl")

        self.assertEqual(first["request_count"], 2)
        self.assertEqual(second["completed_request_count"], 2)
        self.assertEqual(len(checkpoint_rows), 2)
        self.assertEqual(len(raw_rows), 2)

    def test_run_candidate_generation_dry_run_can_resume_without_duplicate_work(self) -> None:
        papers = build_fixture_papers()
        short_chunks, rag_chunks, rag_bundles = build_fixture_chunks()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "candidate_run"
            first = run_candidate_generation(
                model="unused",
                output_dir=output_dir,
                papers=papers,
                short_chunks=short_chunks,
                rag_chunks=rag_chunks,
                rag_bundles=rag_bundles,
                paper_limit=1,
                min_parse_confidence=0.4,
                min_keyword_score=1,
                families_override=["hpn_fact_qa", "rag_grounded_qa"],
                include_unanswerable_every=0,
                temperature=0.0,
                top_p=0.9,
                num_predict=10,
                base_url="http://127.0.0.1:9",
                dry_run=True,
                resume=False,
            )
            second = run_candidate_generation(
                model="unused",
                output_dir=output_dir,
                papers=papers,
                short_chunks=short_chunks,
                rag_chunks=rag_chunks,
                rag_bundles=rag_bundles,
                paper_limit=1,
                min_parse_confidence=0.4,
                min_keyword_score=1,
                families_override=["hpn_fact_qa", "rag_grounded_qa"],
                include_unanswerable_every=0,
                temperature=0.0,
                top_p=0.9,
                num_predict=10,
                base_url="http://127.0.0.1:9",
                dry_run=True,
                resume=True,
            )

            checkpoint_rows = read_jsonl(output_dir / "candidate_checkpoint.jsonl")
            raw_rows = read_jsonl(output_dir / "candidate_raw_generations.jsonl")

        self.assertEqual(first["request_count"], 2)
        self.assertEqual(second["completed_request_count"], 2)
        self.assertEqual(len(checkpoint_rows), 2)
        self.assertEqual(len(raw_rows), 2)


if __name__ == "__main__":
    unittest.main()
