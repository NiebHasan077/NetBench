"""
test_s2_client.py — Unit tests for the Semantic Scholar API client.

All HTTP calls are mocked — tests run offline and deterministically.

Covers:
  - Single-page search returns papers
  - Pagination across multiple pages
  - Empty result set
  - Rate-limit 429 retry
  - API error returns empty list
  - open_access_only flag sent in params
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s2_client import S2Client, _S2_PAGE_LIMIT


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_paper(paper_id: str, title: str = "Test Paper") -> dict:
    """Build a minimal S2 paper dict."""
    return {
        "paperId": paper_id,
        "corpusId": 12345,
        "title": title,
        "abstract": "This is a test abstract.",
        "year": 2024,
        "externalIds": {"DOI": f"10.1234/{paper_id}"},
        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{paper_id}.pdf"},
        "authors": [{"authorId": "1", "name": "Alice"}],
        "venue": "SIGCOMM",
        "fieldsOfStudy": ["Computer Science"],
        "isOpenAccess": True,
    }


def _make_response(papers: list, total: int | None = None) -> dict:
    """Build a minimal S2 API response dict."""
    return {
        "total": total if total is not None else len(papers),
        "data": papers,
    }


# ── Basic search ──────────────────────────────────────────────────────────

class TestBasicSearch:
    def test_single_page_returns_papers(self):
        papers = [_make_paper("p1"), _make_paper("p2")]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response(papers)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp):
            result = client.search("RDMA", max_results=10)

        assert len(result) == 2
        assert result[0]["paperId"] == "p1"
        assert result[1]["paperId"] == "p2"

    def test_empty_results(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response([], total=0)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp):
            result = client.search("nonexistent topic", max_results=10)

        assert result == []

    def test_max_results_limits_output(self):
        papers = [_make_paper(f"p{i}") for i in range(5)]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response(papers, total=100)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp):
            result = client.search("RDMA", max_results=3)

        assert len(result) == 3


# ── Pagination ────────────────────────────────────────────────────────────

class TestPagination:
    def test_two_pages(self):
        page1 = [_make_paper(f"p{i}") for i in range(3)]
        page2 = [_make_paper(f"p{i}") for i in range(3, 5)]

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = _make_response(page1, total=5)
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = _make_response(page2, total=5)
        resp2.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", side_effect=[resp1, resp2]):
            result = client.search("RDMA", max_results=10)

        assert len(result) == 5

    def test_stops_when_total_reached(self):
        """Even if max_results is large, stops when total is exhausted."""
        papers = [_make_paper("p1")]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response(papers, total=1)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            result = client.search("RDMA", max_results=500)

        assert len(result) == 1
        assert mock_get.call_count == 1


# ── Rate limiting ─────────────────────────────────────────────────────────

class TestRateLimit:
    def test_429_retry(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "1"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = _make_response([_make_paper("p1")], total=1)
        resp_ok.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with (
            patch.object(client.session, "get", side_effect=[resp_429, resp_ok]),
            patch("s2_client.time.sleep"),
        ):
            result = client.search("RDMA", max_results=10)

        assert len(result) == 1

    def test_429_multiple_retries(self):
        """Multiple consecutive 429s should be retried with back-off."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "1"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = _make_response([_make_paper("p1")], total=1)
        resp_ok.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with (
            patch.object(
                client.session, "get",
                side_effect=[resp_429, resp_429, resp_429, resp_ok],
            ),
            patch("s2_client.time.sleep") as mock_sleep,
        ):
            result = client.search("RDMA", max_results=10)

        assert len(result) == 1
        # 3 rate-limit sleeps + 1 normal delay after success
        assert mock_sleep.call_count == 4


# ── Error handling ────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_request_exception_returns_empty(self):
        import requests as req

        client = S2Client(api_key="test-key")
        with patch.object(
            client.session, "get",
            side_effect=req.ConnectionError("connection refused"),
        ):
            result = client.search("RDMA", max_results=10)

        assert result == []


# ── Request parameters ────────────────────────────────────────────────────

class TestRequestParams:
    def test_open_access_filter_sent(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response([], total=0)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            client.search("RDMA", open_access_only=True, max_results=10)

        _, kwargs = mock_get.call_args
        assert "openAccessPdf" in kwargs["params"]

    def test_no_open_access_filter_when_disabled(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response([], total=0)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            client.search("RDMA", open_access_only=False, max_results=10)

        _, kwargs = mock_get.call_args
        assert "openAccessPdf" not in kwargs["params"]

    def test_year_range_sent(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response([], total=0)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            client.search("RDMA", year_range="2022-2026", max_results=10)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["year"] == "2022-2026"

    def test_fields_of_study_sent(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_response([], total=0)
        mock_resp.raise_for_status = MagicMock()

        client = S2Client(api_key="test-key")
        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            client.search(
                "RDMA",
                fields_of_study=["Computer Science"],
                max_results=10,
            )

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["fieldsOfStudy"] == "Computer Science"

    def test_api_key_set_in_headers(self):
        client = S2Client(api_key="my-secret-key")
        assert client.session.headers["x-api-key"] == "my-secret-key"

    def test_no_api_key_no_header(self):
        client = S2Client(api_key="")
        assert "x-api-key" not in client.session.headers
