"""Phase 0 smoke test.

Runs:
  1. config.yaml loads and validates against pydantic Config.
  2. Schema models validate hand-written samples (Passage, PaperCard, Question).
  3. io_utils round-trip: append + read + existing_ids.
  4. ollama_client.list_models() works.
  5. ollama_client.generate_json() returns parsed JSON from card_generator.

Exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

# Allow running directly: `python scripts/smoke_test.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src import io_utils
from src.ollama_client import OllamaClient, OllamaError, _extract_json
from src.schema import Evidence, EvidenceAnchor, PaperCard, Passage, Question


CHECK = "[ok] "
FAIL = "[FAIL] "


def check(label: str, fn) -> bool:
    try:
        fn()
        print(CHECK + label)
        return True
    except Exception as e:
        print(FAIL + label + f": {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return False


def test_config():
    cfg = load_config()
    assert cfg.splits.total == 300, f"expected total=300, got {cfg.splits.total}"
    assert cfg.filter.top_k_papers == 300
    assert cfg.models.card_generator
    # Resolve a relative path
    p = cfg.resolve(cfg.paths.corpus)
    assert p.is_absolute()


def test_schema_passage():
    p = Passage(paper_id="paper_0001", passage_id="paper_0001__p0", text="hello")
    assert p.passage_id.startswith("paper_0001")


def test_schema_card():
    c = PaperCard(
        paper_id="paper_0001",
        main_problem="online tuning of transfer concurrency",
        key_concepts=["concurrency", "BDP"],
        evidence_anchors=[EvidenceAnchor(passage_id="paper_0001__p3", claim="concurrency dominates throughput")],
    )
    assert c.evidence_anchors[0].passage_id == "paper_0001__p3"


def test_schema_question():
    q = Question(
        id="NB-HPN-000001",
        category="Concurrency Tuning and Scaling",
        difficulty="medium",
        question_type="reasoning",
        question="Why does adding concurrency tend to outpace adding parallelism for HPN throughput?",
        reference_answer="Concurrency adds independent I/O *and* network parallelism per file ...",
        source_papers=["paper_0182"],
        evidence=[Evidence(paper_id="paper_0182", passage_id="paper_0182__p4", quote="concurrency dominates")],
    )
    assert q.benchmark_version == "v5.0"


def test_schema_question_rejects_no_papers():
    try:
        Question(
            id="bad",
            category="x",
            difficulty="easy",
            question_type="concept",
            question="?",
            reference_answer="?",
            source_papers=[],
            evidence=[],
        )
    except Exception:
        return
    raise AssertionError("Question with empty source_papers should have failed validation")


def test_io_utils_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rt.jsonl"
        io_utils.append_jsonl(path, {"id": "a", "v": 1})
        io_utils.append_jsonl(path, {"id": "b", "v": 2})
        got = list(io_utils.read_jsonl(path))
        assert got == [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
        ids = io_utils.existing_ids(path)
        assert ids == {"a", "b"}
        assert io_utils.count_lines(path) == 2


def test_io_utils_missing_file_is_empty():
    p = Path("/tmp/__bg_nonexistent_xyz.jsonl")
    if p.exists():
        p.unlink()
    assert list(io_utils.read_jsonl(p)) == []
    assert io_utils.existing_ids(p) == set()


def test_extract_json_fence():
    out = _extract_json('```json\n{"a": 1}\n```')
    assert out == {"a": 1}


def test_extract_json_with_preamble():
    out = _extract_json('Here you go:\n{"a": 1, "b": [2, 3]}\nThanks!')
    assert out == {"a": 1, "b": [2, 3]}


def test_extract_json_array():
    out = _extract_json('[{"a": 1}, {"a": 2}]')
    assert out == [{"a": 1}, {"a": 2}]


def test_extract_json_nested_braces_in_string():
    out = _extract_json('{"k": "value with } brace"}')
    assert out == {"k": "value with } brace"}


def test_ollama_list_models():
    cfg = load_config()
    client = OllamaClient(host=cfg.models.ollama_host)
    models = client.list_models()
    assert cfg.models.card_generator in models, (
        f"configured card_generator {cfg.models.card_generator} not in installed models {models}"
    )


def test_ollama_chat_json():
    cfg = load_config()
    client = OllamaClient(host=cfg.models.ollama_host, timeout=cfg.ollama.request_timeout_seconds)
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "year": {"type": "integer"},
        },
        "required": ["name", "year"],
    }
    out = client.chat_json(
        model=cfg.models.card_generator,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Return a JSON object: {"name": "ada", "year": 1815}.'},
        ],
        temperature=0.0,
        max_retries=cfg.ollama.max_retries,
        retry_temperature=cfg.ollama.json_retry_temperature,
        schema=schema,
        think=False,
    )
    assert isinstance(out, dict), f"expected dict, got {type(out)}: {out!r}"
    assert "name" in out and "year" in out, f"missing keys: {out!r}"


def main() -> int:
    results = []
    results.append(check("config.yaml loads", test_config))
    results.append(check("schema Passage", test_schema_passage))
    results.append(check("schema PaperCard", test_schema_card))
    results.append(check("schema Question", test_schema_question))
    results.append(check("schema Question rejects empty source_papers", test_schema_question_rejects_no_papers))
    results.append(check("io_utils round-trip", test_io_utils_roundtrip))
    results.append(check("io_utils missing file is empty", test_io_utils_missing_file_is_empty))
    results.append(check("_extract_json fence", test_extract_json_fence))
    results.append(check("_extract_json with preamble", test_extract_json_with_preamble))
    results.append(check("_extract_json array", test_extract_json_array))
    results.append(check("_extract_json nested string brace", test_extract_json_nested_braces_in_string))
    results.append(check("ollama list_models", test_ollama_list_models))
    results.append(check("ollama chat_json (real call, schema-enforced)", test_ollama_chat_json))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
