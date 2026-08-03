from __future__ import annotations

import unittest

from _bootstrap import ROOT  # noqa: F401
from instruct_ftd.filtering import (
    ReviewRecord,
    _parse_judge_json,
    deduplicate,
    judge_records,
    heuristic_reject_reason,
)


def build_valid_example() -> dict:
    return {
        "system": "You are an HPN assistant.",
        "question": "What does the paper say about ECN-based congestion control?",
        "response": (
            "The paper says ECN-based congestion control reacts to congestion marks before severe loss "
            "appears, which allows the sender to slow down before queues become unstable. In the reported "
            "design, ECN marks are treated as an early signal that persistent queue growth is beginning, so "
            "the controller can reduce the offered load without waiting for heavy packet loss. That behavior "
            "helps preserve throughput while also reducing long queue residency, lower latency inflation, and "
            "lower jitter under load. The broader point is that the mechanism uses explicit network feedback "
            "to balance utilization and stability instead of relying on late loss events alone."
        ),
        "category": "hpn_fact_qa",
        "source_split": "hpn",
        "task_type": "fact_qa",
        "paper_id": "paper_0001",
        "paper_title": "Sample Paper",
        "word_count": 90,
        "evidence": [
            {
                "text": (
                    "ECN-based congestion control reacts to congestion marks before severe loss appears and "
                    "helps reduce persistent queue growth while preserving throughput. The mechanism uses "
                    "explicit congestion feedback to stabilize queues, limit latency inflation, and maintain "
                    "high utilization under load."
                )
            }
        ],
    }


class FilteringTests(unittest.TestCase):
    def test_valid_example_passes_heuristics(self) -> None:
        example = build_valid_example()
        self.assertEqual(heuristic_reject_reason(example), "")

    def test_bad_rag_question_format_is_rejected(self) -> None:
        example = build_valid_example()
        example["source_split"] = "rag"
        example["category"] = "rag_grounded_qa"
        example["question"] = "What does the excerpt claim?"

        self.assertEqual(heuristic_reject_reason(example), "bad_question_format")

    def test_deduplicate_removes_exact_duplicate(self) -> None:
        example_a = build_valid_example()
        example_b = build_valid_example()
        example_a["request_id"] = "req_1"
        example_b["request_id"] = "req_2"
        records = [
            ReviewRecord(example=example_a, source_path="a.jsonl", source_index=0, heuristic_score=1.0),
            ReviewRecord(example=example_b, source_path="b.jsonl", source_index=0, heuristic_score=0.9),
        ]

        kept, rejects = deduplicate(records)

        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0]["reason"], "duplicate_question_response")

    def test_parse_judge_json_handles_fences_and_noise(self) -> None:
        fenced = """```json\n{"keep": true, "groundedness": 1, "specificity": 1, "usefulness": 1, "reason": "ok"}\n```"""
        noisy = 'prefix {"keep": false, "groundedness": 0, "specificity": 0, "usefulness": 0, "reason": "bad"} suffix'

        self.assertTrue(_parse_judge_json(fenced)["keep"])
        self.assertFalse(_parse_judge_json(noisy)["keep"])

    def test_judge_records_can_resume_from_existing_rows_without_network(self) -> None:
        example = build_valid_example()
        example["request_id"] = "req_resume"
        records = [ReviewRecord(example=example, source_path="candidates.jsonl", source_index=0, heuristic_score=1.0)]
        existing_rows = [
            {
                "request_id": "req_resume",
                "paper_id": example["paper_id"],
                "category": example["category"],
                "judge_model": "local-judge",
                "judge_result": {"keep": True, "reason": "ok"},
                "raw_text": '{"keep": true, "reason": "ok"}',
                "error": "",
            }
        ]

        kept, rejects, judge_rows = judge_records(
            records,
            model="unused",
            base_url="http://127.0.0.1:9",
            existing_judge_rows=existing_rows,
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejects), 0)
        self.assertEqual(len(judge_rows), 1)


if __name__ == "__main__":
    unittest.main()
