from __future__ import annotations

import unittest
from pathlib import Path


class CliContractTests(unittest.TestCase):
    def test_expected_scripts_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = [
            "scripts/normalize_corpus.py",
            "scripts/build_chunks.py",
            "scripts/run_pilot_generation.py",
            "scripts/run_candidate_generation.py",
            "scripts/filter_candidates.py",
            "scripts/import_generic_anchors.py",
            "scripts/import_hermes_json_anchors.py",
            "scripts/build_final_dataset.py",
            "scripts/append_json_anchors_to_final.py",
        ]

        for relative_path in expected:
            self.assertTrue((root / relative_path).exists(), relative_path)


if __name__ == "__main__":
    unittest.main()
