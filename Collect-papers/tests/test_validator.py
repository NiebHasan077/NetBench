"""
Tests for llm_validator.LLMValidator.

All tests mock requests.post so no real LLM server is needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profile import LLMConfig
from llm_validator import (
    CACHE_FLUSH_INTERVAL,
    LLMConnectionError,
    LLMValidator,
    ValidationResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def make_config(
    backend="ollama",
    model="llama3",
    base_url="http://localhost:11434",
    api_key="",
    temperature=0.0,
    timeout=30,
    fallback_on_no_abstract="download",
    skip_journal_papers=True,
) -> LLMConfig:
    return LLMConfig(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        fallback_on_no_abstract=fallback_on_no_abstract,
        skip_journal_papers=skip_journal_papers,
    )


def make_validator(tmp_path: Path, config: LLMConfig | None = None,
                   profile_name: str = "test-profile") -> LLMValidator:
    return LLMValidator(
        config=config or make_config(),
        profile_name=profile_name,
        domain_description="Test domain: high-performance networking.",
        cache_path=tmp_path / "validation_cache.json",
    )


def _ollama_resp(text: str) -> MagicMock:
    m = MagicMock()
    m.json.return_value = {"message": {"content": text}}
    m.raise_for_status.return_value = None
    return m


def _openai_resp(text: str) -> MagicMock:
    m = MagicMock()
    m.json.return_value = {"choices": [{"message": {"content": text}}]}
    m.raise_for_status.return_value = None
    return m


# ── Tests ─────────────────────────────────────────────────────────────────

class TestResponseParsing:
    def test_relevant_response(self, tmp_path):
        v = make_validator(tmp_path)
        with patch("requests.post", return_value=_ollama_resp(
            "RELEVANT: discusses RDMA-based datacenter fabric"
        )):
            result = v.validate("W001", "RDMA Paper", "Discusses RDMA over RoCE.")
        assert result.relevant is True
        assert result.source == "llm"
        assert result.cached is False
        assert "RDMA" in result.reason or result.reason != ""

    def test_not_relevant_response(self, tmp_path):
        v = make_validator(tmp_path)
        with patch("requests.post", return_value=_ollama_resp(
            "NOT_RELEVANT: this paper is about cell biology"
        )):
            result = v.validate("W002", "Bio Paper", "A study of cell mitosis.")
        assert result.relevant is False
        assert result.source == "llm"
        assert result.cached is False

    def test_not_relevant_takes_precedence_over_relevant(self, tmp_path):
        """NOT_RELEVANT prefix must be checked before RELEVANT."""
        v = make_validator(tmp_path)
        with patch("requests.post", return_value=_ollama_resp(
            "NOT_RELEVANT: not relevant to networking"
        )):
            result = v.validate("W003", "Some Paper", "Some abstract.")
        assert result.relevant is False

    def test_malformed_response_allows_through(self, tmp_path):
        """Unparseable LLM output must never silently discard a paper."""
        v = make_validator(tmp_path)
        with patch("requests.post", return_value=_ollama_resp(
            "I think this might be interesting to you."
        )):
            result = v.validate("W004", "Paper", "Some abstract.")
        assert result.relevant is True
        assert result.source == "parse_error_allow"

    def test_case_insensitive_prefix(self, tmp_path):
        v = make_validator(tmp_path)
        with patch("requests.post", return_value=_ollama_resp(
            "relevant: matches the domain"
        )):
            result = v.validate("W005", "Paper", "Abstract.")
        assert result.relevant is True
        assert result.source == "llm"


class TestEdgeCaseFallbacks:
    def test_no_abstract_fallback_download(self, tmp_path):
        """When abstract is absent and fallback=download, allow through without LLM."""
        v = make_validator(tmp_path, make_config(fallback_on_no_abstract="download"))
        with patch("requests.post") as mock_post:
            result = v.validate("W001", "Paper", None)
        mock_post.assert_not_called()
        assert result.relevant is True
        assert result.source == "no_abstract_skipped"

    def test_no_abstract_fallback_skip(self, tmp_path):
        """When abstract is absent and fallback=skip, reject without LLM."""
        v = make_validator(tmp_path, make_config(fallback_on_no_abstract="skip"))
        with patch("requests.post") as mock_post:
            result = v.validate("W001", "Paper", None)
        mock_post.assert_not_called()
        assert result.relevant is False
        assert result.source == "no_abstract_skipped"

    def test_empty_string_abstract_treated_as_no_abstract(self, tmp_path):
        v = make_validator(tmp_path, make_config(fallback_on_no_abstract="skip"))
        with patch("requests.post") as mock_post:
            result = v.validate("W001", "Paper", "   ")
        mock_post.assert_not_called()
        assert result.relevant is False

    def test_skip_journal_papers_bypasses_llm(self, tmp_path):
        v = make_validator(tmp_path, make_config(skip_journal_papers=True))
        with patch("requests.post") as mock_post:
            result = v.validate("W001", "Journal Paper", "Abstract.", is_journal_paper=True)
        mock_post.assert_not_called()
        assert result.relevant is True
        assert result.source == "skip_journal"

    def test_skip_journal_papers_false_calls_llm(self, tmp_path):
        """When skip_journal_papers=False, journal papers still go through LLM."""
        v = make_validator(tmp_path, make_config(skip_journal_papers=False))
        with patch("requests.post", return_value=_ollama_resp(
            "RELEVANT: network paper"
        )):
            result = v.validate("W001", "Journal Paper", "Abstract.", is_journal_paper=True)
        assert result.source == "llm"


class TestCacheBehavior:
    def test_cache_hit_skips_http(self, tmp_path):
        v = make_validator(tmp_path)
        key = "W001::llama3::test-profile"
        v._cache[key] = {"relevant": True, "reason": "previously cached"}
        with patch("requests.post") as mock_post:
            result = v.validate("W001", "Paper", "Abstract.")
        mock_post.assert_not_called()
        assert result.cached is True
        assert result.reason == "previously cached"

    def test_cache_written_and_flushed_on_close(self, tmp_path):
        v = make_validator(tmp_path)
        with patch("requests.post", return_value=_ollama_resp("RELEVANT: ok")):
            v.validate("W001", "Paper", "Abstract.")
        v.close()
        cache_file = tmp_path / "validation_cache.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert "W001::llama3::test-profile" in data
        assert data["W001::llama3::test-profile"]["relevant"] is True

    def test_different_model_is_cache_miss(self, tmp_path):
        """A different model name produces a different cache key → LLM is called."""
        # Pre-populate cache with llama3 key
        cache_file = tmp_path / "validation_cache.json"
        cache_file.write_text(json.dumps({
            "W001::llama3::test-profile": {"relevant": True, "reason": "old"}
        }))
        # Create validator with different model
        v = make_validator(tmp_path, make_config(model="mistral"))
        with patch("requests.post", return_value=_ollama_resp("RELEVANT: good")) as mock_post:
            result = v.validate("W001", "Paper", "Abstract.")
        mock_post.assert_called_once()
        assert result.cached is False

    def test_periodic_flush_after_interval(self, tmp_path):
        """Cache is flushed to disk every CACHE_FLUSH_INTERVAL validations."""
        v = make_validator(tmp_path)
        cache_file = tmp_path / "validation_cache.json"
        with patch("requests.post", return_value=_ollama_resp("RELEVANT: ok")):
            for i in range(CACHE_FLUSH_INTERVAL):
                v.validate(f"W{i:03d}", f"Paper {i}", f"Abstract {i}.")
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert len(data) == CACHE_FLUSH_INTERVAL

    def test_existing_cache_loaded_on_init(self, tmp_path):
        """Validator loads an existing cache file at init time."""
        cache_file = tmp_path / "validation_cache.json"
        cache_file.write_text(json.dumps({
            "W999::llama3::test-profile": {"relevant": False, "reason": "off-topic"}
        }))
        v = make_validator(tmp_path)
        with patch("requests.post") as mock_post:
            result = v.validate("W999", "Paper", "Abstract.")
        mock_post.assert_not_called()
        assert result.relevant is False
        assert result.cached is True

    def test_corrupted_cache_file_is_ignored(self, tmp_path):
        """A corrupt cache file is silently ignored; starts with empty cache."""
        cache_file = tmp_path / "validation_cache.json"
        cache_file.write_text("{invalid json")
        v = make_validator(tmp_path)
        assert v._cache == {}


class TestBackends:
    def test_ollama_calls_correct_endpoint(self, tmp_path):
        v = make_validator(tmp_path)
        with patch("requests.post", return_value=_ollama_resp("RELEVANT: ok")) as mock_post:
            v.validate("W001", "Paper", "Abstract.")
        url = mock_post.call_args[0][0]
        assert url == "http://localhost:11434/api/chat"

    def test_openai_backend_correct_url_and_auth(self, tmp_path):
        v = make_validator(tmp_path, make_config(
            backend="openai",
            base_url="https://api.openai.com",
            api_key="sk-test123",
        ))
        with patch("requests.post", return_value=_openai_resp("RELEVANT: ok")) as mock_post:
            result = v.validate("W001", "Paper", "Abstract.")
        assert result.relevant is True
        call_args = mock_post.call_args
        assert "/v1/chat/completions" in call_args[0][0]
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test123"

    def test_openai_no_auth_header_when_key_empty(self, tmp_path):
        v = make_validator(tmp_path, make_config(
            backend="openai",
            base_url="https://api.openai.com",
            api_key="",
        ))
        with patch("requests.post", return_value=_openai_resp("RELEVANT: ok")) as mock_post:
            v.validate("W001", "Paper", "Abstract.")
        headers = mock_post.call_args[1]["headers"]
        assert "Authorization" not in headers


class TestErrorHandling:
    def test_ollama_connection_refused_raises_llm_error(self, tmp_path):
        v = make_validator(tmp_path)
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(LLMConnectionError, match="Ollama"):
                v.validate("W001", "Paper", "Abstract.")

    def test_ollama_timeout_raises_llm_error(self, tmp_path):
        v = make_validator(tmp_path)
        with patch(
            "requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            with pytest.raises(LLMConnectionError, match="timed out"):
                v.validate("W001", "Paper", "Abstract.")

    def test_openai_connection_refused_raises_llm_error(self, tmp_path):
        v = make_validator(tmp_path, make_config(backend="openai"))
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(LLMConnectionError):
                v.validate("W001", "Paper", "Abstract.")
