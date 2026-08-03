from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.json_anchors import (
    append_json_anchors_to_final,
    normalize_hermes_json_rows,
)


def _hermes_row(row_id: str, question: str, response: str) -> dict:
    return {
        "id": row_id,
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant that answers in JSON."},
            {"from": "human", "value": question},
            {"from": "gpt", "value": response},
        ],
        "category": "json",
        "subcategory": "test",
        "task": "return structured data",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class JsonAnchorTests(unittest.TestCase):
    def test_normalize_hermes_json_rows_keeps_only_valid_json_responses(self) -> None:
        rows = [
            _hermes_row("ok", "Return Maya as JSON.", '{"name": "Maya", "active": true}'),
            _hermes_row("bad", "Return prose.", "The answer is Maya."),
        ]

        records = normalize_hermes_json_rows(rows, source_category="json_mode_singleturn")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["category"], "generic_json")
        self.assertEqual(records[0]["task_type"], "json_response")
        self.assertEqual(records[0]["anchor_source_category"], "json_mode_singleturn")
        self.assertEqual(json.loads(records[0]["response"])["name"], "Maya")

    def test_append_json_anchors_to_final_adds_exact_count(self) -> None:
        base_train = [
            {
                "system": "System",
                "question": "Q1",
                "response": "Existing answer one.",
                "category": "hpn_fact_qa",
                "source_split": "hpn",
                "task_type": "fact_qa",
            }
        ]
        base_validation = [
            {
                "system": "System",
                "question": "Q2",
                "response": "Existing answer two.",
                "category": "generic_orca",
                "source_split": "generic",
                "task_type": "generic_anchor",
            }
        ]
        json_rows = normalize_hermes_json_rows(
            [
                _hermes_row("j1", "Return first object.", '{"id": 1}'),
                _hermes_row("j2", "Return second object.", '{"id": 2}'),
                _hermes_row("j3", "Return third object.", '{"id": 3}'),
            ],
            source_category="json_mode_singleturn",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train.jsonl"
            val_path = tmp_path / "validation.jsonl"
            anchors_path = tmp_path / "hermes_json_anchor.jsonl"
            output_dir = tmp_path / "out"
            _write_jsonl(train_path, base_train)
            _write_jsonl(val_path, base_validation)
            _write_jsonl(anchors_path, json_rows)

            report = append_json_anchors_to_final(
                input_train=train_path,
                input_validation=val_path,
                json_anchors=anchors_path,
                output_dir=output_dir,
                count=3,
                val_ratio=0.34,
                seed=7,
            )

            output_train = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
            output_val = [json.loads(line) for line in (output_dir / "validation.jsonl").read_text().splitlines()]

        self.assertEqual(report["selected_json_count"], 3)
        self.assertEqual(report["train_json_count"], 2)
        self.assertEqual(report["validation_json_count"], 1)
        self.assertEqual(len(output_train), 3)
        self.assertEqual(len(output_val), 2)
        for row in [*output_train, *output_val]:
            if row["category"] == "generic_json":
                json.loads(row["response"])


if __name__ == "__main__":
    unittest.main()
