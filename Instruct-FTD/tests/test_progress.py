from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.progress import append_jsonl, ensure_resume_safe, load_checkpoint_map, read_jsonl


class ProgressTests(unittest.TestCase):
    def test_append_and_load_checkpoint_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.jsonl"
            append_jsonl(checkpoint_path, {"request_id": "req_1", "status": "success"})
            append_jsonl(checkpoint_path, {"request_id": "req_2", "status": "failure"})

            rows = read_jsonl(checkpoint_path)
            checkpoint = load_checkpoint_map(checkpoint_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(checkpoint["req_1"]["status"], "success")
        self.assertEqual(checkpoint["req_2"]["status"], "failure")

    def test_ensure_resume_safe_blocks_accidental_reuse_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "pilot_manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

            with self.assertRaises(ValueError):
                ensure_resume_safe(output_dir, ["pilot_manifest.json"], resume=False)

            ensure_resume_safe(output_dir, ["pilot_manifest.json"], resume=True)


if __name__ == "__main__":
    unittest.main()
