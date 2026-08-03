from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.mixing import mix_and_split


def build_row(source_split: str, paper_id: str, question: str, category: str) -> dict:
    return {
        "system": "System",
        "question": question,
        "response": "A grounded response with enough words for testing the final dataset pipeline.",
        "category": category,
        "source_split": source_split,
        "task_type": category,
        "paper_id": paper_id,
        "paper_title": f"Title {paper_id}" if paper_id else "",
        "evidence": [{"text": "grounded response final dataset pipeline"}],
    }


class MixingTests(unittest.TestCase):
    def test_strict_mix_requires_all_source_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                mix_and_split(
                    filtered_hpn=[build_row("hpn", "paper_1", "Q1", "hpn_fact_qa")],
                    filtered_rag=[],
                    generic_anchors=[],
                    output_dir=Path(tmp_dir),
                    strict_mix=True,
                )

    def test_mix_and_split_prevents_cross_split_paper_leakage(self) -> None:
        hpn_rows = [
            build_row("hpn", "paper_1", "Q1", "hpn_fact_qa"),
            build_row("hpn", "paper_2", "Q2", "hpn_fact_qa"),
            build_row("hpn", "paper_3", "Q3", "hpn_fact_qa"),
        ]
        rag_rows = [
            build_row("rag", "paper_1", "Excerpts:\n\n[1] text\n\nQuestion: RAG Q1", "rag_grounded_qa"),
        ]
        generic_rows = [
            build_row("generic", "", "Why do deterministic splits matter?", "generic_orca"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            report = mix_and_split(
                filtered_hpn=hpn_rows,
                filtered_rag=rag_rows,
                generic_anchors=generic_rows,
                output_dir=output_dir,
                strict_mix=True,
                val_ratio=0.4,
                seed=7,
            )

            train_rows = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines() if line.strip()]
            val_rows = [json.loads(line) for line in (output_dir / "validation.jsonl").read_text().splitlines() if line.strip()]

        train_papers = {row["paper_id"] for row in train_rows if row.get("paper_id")}
        val_papers = {row["paper_id"] for row in val_rows if row.get("paper_id")}

        self.assertEqual(report["paper_leakage_count"], 0)
        self.assertFalse(train_papers & val_papers)
        self.assertEqual(report["selected_counts"], {"hpn": 3, "rag": 1, "generic": 1})


if __name__ == "__main__":
    unittest.main()
