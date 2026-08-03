"""
s2_client.py — Semantic Scholar API client for collect_papers.

Queries the Semantic Scholar Academic Graph API to discover open-access
research papers.  Returns raw API response dicts; normalization into the
pipeline's internal record format happens in collect_papers.py.

API reference: https://api.semanticscholar.org/api-docs/graph

Rate limits
-----------
  Without API key : 100 requests per 5 minutes
  With API key    : 1 request per second (free registration)

Register for a free API key at https://www.semanticscholar.org/product/api
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

# Fields to request from the S2 paper search endpoint.
_S2_FIELDS = ",".join([
    "paperId",
    "corpusId",
    "title",
    "abstract",
    "year",
    "externalIds",
    "openAccessPdf",
    "authors",
    "venue",
    "fieldsOfStudy",
    "isOpenAccess",
])

_S2_PAGE_LIMIT = 100          # S2 API max results per page
_DELAY_WITH_KEY = 1.05        # seconds between requests (with API key)
_DELAY_WITHOUT_KEY = 3.5      # conservative for unauthenticated access
_MAX_429_RETRIES = 5          # max retries on rate-limit (429) responses


class S2Client:
    """Thin wrapper around the Semantic Scholar Academic Graph API."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key.strip()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "collect-papers/1.0"
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
        self._delay = _DELAY_WITH_KEY if self.api_key else _DELAY_WITHOUT_KEY

    def search(
        self,
        query: str,
        *,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = True,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for papers matching a query string.

        Parameters
        ----------
        query:
            Free-text search query (supports S2 search syntax).
        year_range:
            Year filter, e.g. "2022-2026".
        fields_of_study:
            Restrict to specific fields, e.g. ["Computer Science"].
        open_access_only:
            If True, only return papers with an open-access PDF.
        max_results:
            Maximum number of papers to return.

        Returns
        -------
        list[dict]
            Raw S2 paper dicts with the fields listed in ``_S2_FIELDS``.
        """
        papers: List[Dict[str, Any]] = []
        offset = 0

        while len(papers) < max_results:
            limit = min(_S2_PAGE_LIMIT, max_results - len(papers))
            params: Dict[str, Any] = {
                "query": query,
                "fields": _S2_FIELDS,
                "offset": offset,
                "limit": limit,
            }
            if year_range:
                params["year"] = year_range
            if fields_of_study:
                params["fieldsOfStudy"] = ",".join(fields_of_study)
            if open_access_only:
                params["openAccessPdf"] = ""

            data = self._get("paper/search", params)
            results = data.get("data") or []
            if not results:
                break

            papers.extend(results)

            total = data.get("total", 0)
            offset += len(results)
            if offset >= total:
                break

        return papers[:max_results]

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GET request with rate limiting and retry for 429."""
        url = f"{S2_API_BASE}/{endpoint}"
        try:
            for attempt in range(_MAX_429_RETRIES + 1):
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code != 429:
                    break
                wait = float(r.headers.get("Retry-After", 10))
                # Exponential back-off: 10s, 20s, 40s, …
                wait = max(wait, 10 * (2 ** attempt))
                print(f"  \u23f3 S2 rate-limited \u2014 waiting {wait:.0f}s")
                time.sleep(wait)
            r.raise_for_status()
            time.sleep(self._delay)
            return r.json()
        except requests.RequestException as exc:
            print(f"  S2 API error: {exc}")
            return {}
