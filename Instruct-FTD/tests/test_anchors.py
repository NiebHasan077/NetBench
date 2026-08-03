from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.anchors import import_generic_anchors, normalize_openorca_rows


class AnchorImportTests(unittest.TestCase):
    def test_openorca_normalization_accepts_system_field(self) -> None:
        rows = [
            {
                "system": "Custom system prompt",
                "question": "What is congestion control?",
                "response": "Congestion control regulates sending behavior to avoid persistent overload in the network.",
            }
        ]

        records = normalize_openorca_rows(rows)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].system, "Custom system prompt")
        self.assertEqual(records[0].category, "generic_orca")

    def test_import_generic_anchors_deduplicates_exact_rows(self) -> None:
        orca_rows = [
            {
                "system": "Helpful system",
                "question": "Explain deterministic evaluation.",
                "response": "Deterministic evaluation matters because it keeps model comparisons reproducible across runs.",
            },
            {
                "system": "Helpful system",
                "question": "Explain deterministic evaluation.",
                "response": "Deterministic evaluation matters because it keeps model comparisons reproducible across runs.",
            },
        ]
        dolly_rows = [
            {
                "instruction": "Summarize why validation splits matter.",
                "context": "",
                "response": "Validation splits matter because they provide a consistent held-out estimate of generalization.",
                "category": "open_qa",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            orca_path = tmp_path / "orca.jsonl"
            dolly_path = tmp_path / "dolly.jsonl"
            orca_path.write_text("".join(json.dumps(row) + "\n" for row in orca_rows), encoding="utf-8")
            dolly_path.write_text("".join(json.dumps(row) + "\n" for row in dolly_rows), encoding="utf-8")

            records = import_generic_anchors(
                orca_local=orca_path,
                dolly_local=dolly_path,
                min_response_words=5,
            )

        self.assertEqual(len(records), 2)
        self.assertEqual({record.category for record in records}, {"generic_orca", "generic_dolly"})


if __name__ == "__main__":
    unittest.main()
