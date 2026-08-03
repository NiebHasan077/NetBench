"""
test_helpers.py — Unit tests for pure helper functions in collect_papers.py.

Tests normalize_title, safe_filename, and reconstruct_abstract.
Also verifies that normalize_title output is consistent with
Check-Exists/build_index.py normalization (cross-project regression guard).
Tests OpenAlexClient._get() and download_pdf() 429 retry with exponential backoff.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow importing collect_papers from parent directory without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collect_papers import normalize_title, safe_filename, reconstruct_abstract


# ── normalize_title ──────────────────────────────────────────────────────

class TestNormalizeTitle:
    def test_lowercases(self):
        assert normalize_title("RDMA Performance") == "rdma performance"

    def test_strips_punctuation(self):
        assert normalize_title("P4: A Language") == "p4 a language"

    def test_collapses_whitespace(self):
        assert normalize_title("  TCP   BBR  ") == "tcp bbr"

    def test_strips_hyphens(self):
        assert normalize_title("High-Performance Network") == "high performance network"

    def test_empty_string(self):
        assert normalize_title("") == ""

    def test_numbers_preserved(self):
        assert normalize_title("100 Gbps Links") == "100 gbps links"

    def test_special_chars_removed(self):
        # Parentheses, colons, slashes all become spaces
        result = normalize_title("NVMe-oF: (Fast Storage)")
        assert ":" not in result
        assert "(" not in result
        assert ")" not in result

    def test_consistent_with_build_index(self):
        r"""
        Regression guard: normalize_title must produce the same output as
        Check-Exists/build_index.py's normalization function.

        build_index.py uses:
            t = re.sub(r'[^a-z0-9\s]', ' ', title.lower().strip())
            t = re.sub(r'\s+', ' ', t).strip()

        Both functions must agree on the same input.
        """
        import re

        def build_index_normalize(title: str) -> str:
            t = re.sub(r"[^a-z0-9\s]", " ", title.lower().strip())
            t = re.sub(r"\s+", " ", t).strip()
            return t

        samples: list = [
            "Scalable RDMA Congestion Control in Large Datacenter Networks",
            "P4: Programming Protocol-Independent Packet Processors",
            "TCP BBR: Congestion-Based Congestion Control",
            "Swift: Delay is Simple and Effective for Congestion Control in the Datacenter",
            "DCQCN: Congestion Control for Large-Scale RDMA Deployments",
            "",
            "100 Gbps+ Network I/O at the Speed of Memory",
        ]
        for s in samples:
            assert normalize_title(s) == build_index_normalize(s), (
                f"normalize_title differs from build_index normalization for: {s!r}"
            )


# ── safe_filename ────────────────────────────────────────────────────────

class TestSafeFilename:
    def test_removes_forbidden_chars(self):
        result = safe_filename('My Paper: "Analysis" <2024>')
        for ch in [':', '"', '<', '>']:
            assert ch not in result

    def test_truncates_at_max_len(self):
        long_title = "word " * 50  # 250 chars
        result = safe_filename(long_title, max_len=150)
        assert len(result) <= 150

    def test_truncates_at_word_boundary(self):
        # Should not cut mid-word
        result = safe_filename("alpha beta gamma delta epsilon", max_len=20)
        assert not result.endswith(" ")
        # Each token in result should be a complete word
        for word in result.split():
            assert word in ["alpha", "beta", "gamma", "delta", "epsilon"]

    def test_empty_string_returns_untitled(self):
        assert safe_filename("") == "untitled"

    def test_strips_leading_trailing_dots(self):
        result = safe_filename("...Paper Title...")
        assert not result.startswith(".")
        assert not result.endswith(".")

    def test_normal_title_unchanged(self):
        result = safe_filename("RDMA Performance Analysis")
        assert result == "RDMA Performance Analysis"


# ── reconstruct_abstract ─────────────────────────────────────────────────

class TestReconstructAbstract:
    def test_correct_word_order(self):
        # OpenAlex inverted index: word → list of positions
        inverted = {
            "Networks": [2],
            "Datacenter": [1],
            "Fast": [0],
        }
        result = reconstruct_abstract(inverted)
        assert result == "Fast Datacenter Networks"

    def test_none_input_returns_none(self):
        assert reconstruct_abstract(None) is None


# ── OpenAlexClient._get() 429 retry ─────────────────────────────────────

class TestOpenAlexRateLimit:
    def _make_response(self, status_code, json_data=None, headers=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = headers or {}
        resp.json.return_value = json_data or {}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import requests as req
            resp.raise_for_status.side_effect = req.HTTPError(response=resp)
        return resp

    def test_429_retries_with_backoff(self):
        from collect_papers import OpenAlexClient

        resp_429 = self._make_response(429, headers={"Retry-After": "1"})
        # Override raise_for_status so it doesn't raise on 429 (handled before)
        resp_429.raise_for_status = MagicMock()
        resp_ok = self._make_response(200, {"results": []})

        client = OpenAlexClient("test@example.com", Path("/tmp/test_cache.json"))
        with (
            patch.object(
                client.session, "get",
                side_effect=[resp_429, resp_429, resp_ok],
            ),
            patch("collect_papers.time.sleep") as mock_sleep,
        ):
            result = client._get("works", {})

        assert result == {"results": []}
        # 2 rate-limit sleeps + 1 API_DELAY
        assert mock_sleep.call_count == 3

    def test_429_all_retries_exhausted_returns_empty(self):
        from collect_papers import OpenAlexClient, MAX_RETRIES

        resp_429 = self._make_response(429, headers={"Retry-After": "1"})
        resp_429.raise_for_status = MagicMock()

        client = OpenAlexClient("test@example.com", Path("/tmp/test_cache.json"))
        with (
            patch.object(
                client.session, "get",
                return_value=resp_429,
            ),
            patch("collect_papers.time.sleep"),
        ):
            result = client._get("works", {})

        assert result == {}


# ── download_pdf() 429 retry ────────────────────────────────────────────

class TestDownloadPdf429:
    def test_429_retries_then_succeeds(self, tmp_path):
        from collect_papers import download_pdf

        resp_429 = MagicMock()
        resp_429.__enter__ = MagicMock(return_value=resp_429)
        resp_429.__exit__ = MagicMock(return_value=False)
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "1"}

        resp_ok = MagicMock()
        resp_ok.__enter__ = MagicMock(return_value=resp_ok)
        resp_ok.__exit__ = MagicMock(return_value=False)
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        resp_ok.iter_content = MagicMock(return_value=[b"%PDF-1.4 test"])

        out = tmp_path / "paper.pdf"
        with (
            patch("collect_papers.requests.get", side_effect=[resp_429, resp_ok]),
            patch("collect_papers.time.sleep"),
        ):
            ok, reason = download_pdf("http://example.com/paper.pdf", out, "t@t.com")

        assert ok is True
        assert reason == ""
        assert out.read_bytes() == b"%PDF-1.4 test"

    def test_429_all_retries_exhausted_fails(self, tmp_path):
        from collect_papers import download_pdf

        resp_429 = MagicMock()
        resp_429.__enter__ = MagicMock(return_value=resp_429)
        resp_429.__exit__ = MagicMock(return_value=False)
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "1"}

        out = tmp_path / "paper.pdf"
        with (
            patch("collect_papers.requests.get", return_value=resp_429),
            patch("collect_papers.time.sleep"),
        ):
            ok, reason = download_pdf("http://example.com/paper.pdf", out, "t@t.com")

        assert ok is False
        assert not out.exists()

    def test_empty_dict_returns_empty_string(self):
        result = reconstruct_abstract({})
        assert result == ""

    def test_word_appearing_multiple_positions(self):
        inverted = {
            "the": [0, 3],
            "cat": [1],
            "sat": [2],
            "mat": [4],
        }
        result = reconstruct_abstract(inverted)
        assert result == "the cat sat the mat"

    def test_single_word(self):
        assert reconstruct_abstract({"hello": [0]}) == "hello"
