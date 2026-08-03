"""
test_pipeline.py — Integration smoke tests for the collect() pipeline.

All network calls (OpenAlex API + PDF downloads) are mocked so the tests
run offline and deterministically.

Scenarios covered:
  - All 3 candidate works download successfully (LLM disabled)
  - dry_run=True: stats updated but no files written
  - paperbase_dir dedup: 1 of 3 titles is in the existing index → skipped
  - DOI dedup: two works share a DOI → second skipped
  - Resume support: work whose OpenAlex ID is already in metadata.jsonl → skipped
  - Missing PDF URL: work with no pdf_url → skip_no_pdf incremented
  - Download failure: download_pdf returns False → download_failed logged
  - Invalid PDF: validate_pdf returns False → invalid_pdf logged, file removed
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import textwrap

import pytest

# Allow importing from parent directory without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collect_papers import (
    collect,
    normalize_title,
)
from profile import Profile


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_profile(tmp_path: Path, paperbase_index: dict | None = None) -> Profile:
    """Build a minimal Profile pointing at tmp_path directories."""
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()

    metadata_path = downloads_dir / "metadata.jsonl"
    failed_path = downloads_dir / "failed_downloads.jsonl"
    venue_cache_path = tmp_path / "test_venue_cache.json"

    paperbase_dir: Path | None = None
    if paperbase_index is not None:
        paperbase_dir = tmp_path / "paperbase"
        paperbase_dir.mkdir()
        index_path = paperbase_dir / "title_index.json"
        index_path.write_text(
            json.dumps(paperbase_index, ensure_ascii=False), encoding="utf-8"
        )

    return Profile(
        name="Test Domain",
        journals=[],
        keywords=["test keyword"],
        subfield_filter=None,
        year_min=2020,
        year_max=2025,
        max_per_query=10,
        mailto="test@example.com",
        domain_description="Test domain for unit tests.",
        paperbase_dir=paperbase_dir,
        downloads_dir=downloads_dir,
        papers_dir=downloads_dir / "papers",
        metadata_path=metadata_path,
        failed_path=failed_path,
        venue_cache_path=venue_cache_path,
        llm_validation=None,
        validation_cache_path=downloads_dir / "validation_cache.json",
        s2_config=None,
    )


def make_work(
    openalex_id: str,
    title: str,
    pdf_url: str = "https://example.com/paper.pdf",
    doi: str | None = None,
    year: int = 2023,
) -> dict:
    """Build a minimal OpenAlex work record."""
    return {
        "id": openalex_id,
        "display_name": title,
        "doi": doi,
        "publication_year": year,
        "authorships": [],
        "primary_location": {"source": {"display_name": "Test Venue"}},
        "best_oa_location": {"pdf_url": pdf_url, "license": "cc-by"},
        "locations": [],
        "abstract_inverted_index": {"Test": [0], "abstract": [1]},
    }


WORK_A = make_work("W001", "RDMA Congestion Control", doi="10.1145/001")
WORK_B = make_work("W002", "P4 Programmable Switch",  doi="10.1145/002")
WORK_C = make_work("W003", "eBPF Network Functions",  doi="10.1145/003")

THREE_WORKS = [WORK_A, WORK_B, WORK_C]


# ── Helpers ───────────────────────────────────────────────────────────────

def _mock_client_class(works: list):
    """Return a patched OpenAlexClient class whose fetch_works returns `works`."""
    mock_client = MagicMock()
    mock_client.fetch_works.return_value = works
    mock_client.resolve_all_venues.return_value = {}
    return mock_client


# ── Happy-path: 3 works, all download and validate ────────────────────────

class TestHappyPath:
    def test_all_three_downloaded(self, tmp_path):
        profile = make_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        # All three should appear in metadata.jsonl
        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 3
        titles = {r["title"] for r in records}
        assert "RDMA Congestion Control" in titles
        assert "P4 Programmable Switch" in titles
        assert "eBPF Network Functions" in titles

    def test_metadata_record_fields(self, tmp_path):
        profile = make_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        record = json.loads(profile.metadata_path.read_text().splitlines()[0])
        assert record["openalex_id"] == "W001"
        assert record["source_id"] == "W001"
        assert record["title"] == "RDMA Congestion Control"
        assert record["doi"] == "10.1145/001"
        assert record["year"] == 2023
        assert record["venue"] == "Test Venue"
        assert record["abstract"] == "Test abstract"
        assert record["local_filename"] is not None
        assert record["local_filename"].endswith(".pdf")


# ── dry_run mode ─────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_no_files_written(self, tmp_path):
        profile = make_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf") as mock_dl,
            patch("collect_papers.validate_pdf") as mock_val,
        ):
            collect(profile, mode="keyword", dry_run=True)

        # download_pdf and validate_pdf must never be called in dry_run
        mock_dl.assert_not_called()
        mock_val.assert_not_called()

    def test_dry_run_no_metadata_written(self, tmp_path):
        profile = make_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=True)

        assert not profile.metadata_path.exists()

    def test_dry_run_no_pdfs_written(self, tmp_path):
        profile = make_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=True)

        pdf_files = list(profile.papers_dir.glob("*.pdf"))
        assert pdf_files == []


# ── Paperbase dedup ───────────────────────────────────────────────────────

class TestPaperbaseDedup:
    def test_paperbase_title_skipped(self, tmp_path):
        """A work whose title is already in the paperbase index is skipped."""
        existing_title = normalize_title("RDMA Congestion Control")
        profile = make_profile(tmp_path, paperbase_index={existing_title: "some/path.pdf"})

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        # Only 2 should be downloaded (WORK_A is in paperbase)
        assert len(records) == 2
        downloaded_titles = {r["title"] for r in records}
        assert "RDMA Congestion Control" not in downloaded_titles

    def test_no_paperbase_dir_downloads_all(self, tmp_path):
        """With paperbase_dir=None, all 3 works download normally."""
        profile = make_profile(tmp_path, paperbase_index=None)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 3


# ── DOI dedup ─────────────────────────────────────────────────────────────

class TestDOIDedup:
    def test_duplicate_doi_skipped(self, tmp_path):
        """Two works sharing a DOI: only the first is downloaded."""
        profile = make_profile(tmp_path)

        # WORK_A and WORK_B given the same DOI
        work_a = make_work("W001", "Paper One", doi="10.1145/shared")
        work_b = make_work("W002", "Paper Two", doi="10.1145/shared")

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([work_a, work_b])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 1
        assert records[0]["title"] == "Paper One"


# ── Resume support ────────────────────────────────────────────────────────

class TestResumeSupport:
    def test_already_downloaded_id_skipped(self, tmp_path):
        """A work whose OpenAlex ID is already in metadata.jsonl is skipped."""
        profile = make_profile(tmp_path)

        # Pre-populate metadata.jsonl with WORK_A's ID
        existing = {"openalex_id": "W001", "title": "RDMA Congestion Control"}
        profile.metadata_path.write_text(
            json.dumps(existing) + "\n", encoding="utf-8"
        )

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        # metadata.jsonl now has 3 lines: the 1 pre-existing + 2 new
        assert len(records) == 3
        # But only 2 were actually downloaded (W001 was skipped)
        new_records = [r for r in records if r["openalex_id"] != "W001" or "local_filename" not in r]
        downloaded = [r for r in records if r.get("local_filename") is not None]
        assert len(downloaded) == 2


# ── No PDF URL ────────────────────────────────────────────────────────────

class TestNoPDFUrl:
    def test_work_without_pdf_url_skipped(self, tmp_path):
        """A work with no pdf_url in any location is skipped."""
        profile = make_profile(tmp_path)

        no_pdf_work = {
            "id": "W999",
            "display_name": "No PDF Paper",
            "doi": "10.1145/999",
            "publication_year": 2023,
            "authorships": [],
            "primary_location": {},
            "best_oa_location": {},   # no pdf_url
            "locations": [],
            "abstract_inverted_index": None,
        }

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([no_pdf_work])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        assert not profile.metadata_path.exists()


# ── Download failures ─────────────────────────────────────────────────────

class TestDownloadFailures:
    def test_download_failure_logged_to_failed(self, tmp_path):
        """A work whose PDF download fails is logged to failed_downloads.jsonl."""
        profile = make_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", return_value=(False, "HTTP 403 Forbidden (publisher blocks automated access)")),
        ):
            collect(profile, mode="keyword", dry_run=False)

        assert not profile.metadata_path.exists()
        assert profile.failed_path.exists()
        record = json.loads(profile.failed_path.read_text().splitlines()[0])
        assert record["error"] == "download_failed"
        assert record["title"] == "RDMA Congestion Control"

    def test_invalid_pdf_logged_to_failed(self, tmp_path):
        """A downloaded file that fails PDF magic-byte check is removed and logged."""
        profile = make_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=False),
        ):
            collect(profile, mode="keyword", dry_run=False)

        assert not profile.metadata_path.exists()
        assert profile.failed_path.exists()
        record = json.loads(profile.failed_path.read_text().splitlines()[0])
        assert record["error"] == "invalid_pdf"

    def test_invalid_pdf_file_removed(self, tmp_path):
        """The PDF file on disk is deleted when it fails validation."""
        profile = make_profile(tmp_path)

        # download_pdf actually creates a fake file so unlink() can be called
        def fake_download(url, out_path, mailto):
            out_path.write_bytes(b"<html>Not a PDF</html>")
            return (True, "")

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", side_effect=fake_download),
            patch("collect_papers.validate_pdf", return_value=False),
        ):
            collect(profile, mode="keyword", dry_run=False)

        pdf_files = list(profile.papers_dir.glob("*.pdf"))
        assert pdf_files == []


# ── mailto_override ───────────────────────────────────────────────────────

class TestMailtoOverride:
    def test_mailto_override_used_in_client(self, tmp_path):
        """mailto_override takes precedence over profile.mailto."""
        profile = make_profile(tmp_path)

        captured_mailto = []

        def capture_client(mailto, venue_cache_path):
            captured_mailto.append(mailto)
            return _mock_client_class([])

        with patch("collect_papers.OpenAlexClient", side_effect=capture_client):
            collect(
                profile,
                mode="keyword",
                dry_run=True,
                mailto_override="override@example.com",
            )

        assert captured_mailto[0] == "override@example.com"

    def test_profile_mailto_used_when_no_override(self, tmp_path):
        profile = make_profile(tmp_path)

        captured_mailto = []

        def capture_client(mailto, venue_cache_path):
            captured_mailto.append(mailto)
            return _mock_client_class([])

        with patch("collect_papers.OpenAlexClient", side_effect=capture_client):
            collect(profile, mode="keyword", dry_run=True)

        assert captured_mailto[0] == "test@example.com"


# ── index_override ────────────────────────────────────────────────────────

class TestIndexOverride:
    def test_index_override_takes_precedence(self, tmp_path):
        """--index CLI override uses a different index file than profile.paperbase_dir."""
        profile = make_profile(tmp_path)   # paperbase_dir=None

        # Create a separate override index containing WORK_A's title
        override_dir = tmp_path / "override_index"
        override_dir.mkdir()
        override_index = override_dir / "title_index.json"
        override_index.write_text(
            json.dumps({normalize_title("RDMA Congestion Control"): "x.pdf"}),
            encoding="utf-8",
        )

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(
                profile,
                mode="keyword",
                dry_run=False,
                index_override=override_index,
            )

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        # WORK_A should be skipped by the override index
        assert len(records) == 2
        downloaded_titles = {r["title"] for r in records}
        assert "RDMA Congestion Control" not in downloaded_titles


# ── Journal mode ──────────────────────────────────────────────────────────

class TestModes:
    def test_journal_mode_calls_resolve_all_venues(self, tmp_path):
        """In journal mode, resolve_all_venues must be called."""
        profile = make_profile(tmp_path)
        # Add a journal to the profile
        object.__setattr__(profile, "journals", [("Test Journal", "Test J")])

        mock_client = _mock_client_class([])
        mock_client.resolve_all_venues.return_value = {"Test J": "S999"}
        mock_client.fetch_works.return_value = []

        with patch("collect_papers.OpenAlexClient", return_value=mock_client):
            collect(profile, mode="journal", dry_run=True)

        mock_client.resolve_all_venues.assert_called_once()

    def test_keyword_mode_skips_resolve_all_venues(self, tmp_path):
        """In keyword-only mode, resolve_all_venues is never called."""
        profile = make_profile(tmp_path)

        mock_client = _mock_client_class([])

        with patch("collect_papers.OpenAlexClient", return_value=mock_client):
            collect(profile, mode="keyword", dry_run=True)

        mock_client.resolve_all_venues.assert_not_called()

    def test_profile_with_no_keywords_in_keyword_mode(self, tmp_path):
        """Profile with empty keywords in keyword mode: zero works fetched."""
        profile = make_profile(tmp_path)
        object.__setattr__(profile, "keywords", [])

        mock_client = _mock_client_class(THREE_WORKS)

        with patch("collect_papers.OpenAlexClient", return_value=mock_client):
            collect(profile, mode="keyword", dry_run=True)

        mock_client.fetch_works.assert_not_called()


# ── LLM validation integration ────────────────────────────────────────────

def _make_llm_config():
    from profile import LLMConfig
    return LLMConfig(
        backend="ollama",
        model="llama3",
        base_url="http://localhost:11434",
        api_key="",
        temperature=0,
        timeout=30,
        fallback_on_no_abstract="download",
        skip_journal_papers=True,
    )


def _make_profile_with_llm(tmp_path: Path) -> Profile:
    """make_profile + llm_validation configured (but LLMValidator will be mocked)."""
    profile = make_profile(tmp_path)
    object.__setattr__(profile, "llm_validation", _make_llm_config())
    return profile


class TestLLMValidation:
    def test_rejected_work_not_downloaded(self, tmp_path):
        """When LLM rejects one of three works, only two are downloaded."""
        from llm_validator import ValidationResult as VR

        profile = _make_profile_with_llm(tmp_path)

        def fake_validate(openalex_id, title, abstract, is_journal_paper=False):
            # Reject the second work by OpenAlex ID (matches make_work("W002", ...))
            if openalex_id == "W002":
                return VR(relevant=False, reason="off-topic", source="llm", cached=False)
            return VR(relevant=True, reason="relevant", source="llm", cached=False)

        mock_validator = MagicMock()
        mock_validator.validate.side_effect = fake_validate
        mock_validator.close = MagicMock()

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.LLMValidator", return_value=mock_validator),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        records = [
            json.loads(l)
            for l in profile.metadata_path.read_text().splitlines()
            if l.strip()
        ]
        assert len(records) == 2
        rejected_titles = {r["title"] for r in records}
        assert "P4 Programmable Switch" not in rejected_titles

    def test_all_accepted_all_downloaded(self, tmp_path):
        """When LLM accepts all works, all three are downloaded."""
        from llm_validator import ValidationResult as VR

        profile = _make_profile_with_llm(tmp_path)
        mock_validator = MagicMock()
        mock_validator.validate.return_value = VR(
            relevant=True, reason="relevant", source="llm", cached=False
        )
        mock_validator.close = MagicMock()

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.LLMValidator", return_value=mock_validator),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        records = [
            json.loads(l)
            for l in profile.metadata_path.read_text().splitlines()
            if l.strip()
        ]
        assert len(records) == 3

    def test_llm_connection_error_propagates(self, tmp_path):
        """LLMConnectionError raised by validate() is not swallowed by collect()."""
        from llm_validator import LLMConnectionError

        profile = _make_profile_with_llm(tmp_path)
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = LLMConnectionError("Ollama unreachable")
        mock_validator.close = MagicMock()

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.LLMValidator", return_value=mock_validator),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            with pytest.raises(LLMConnectionError):
                collect(profile, mode="keyword", dry_run=False)

    def test_validator_close_called_after_run(self, tmp_path):
        """validator.close() must be called after the processing loop."""
        from llm_validator import ValidationResult as VR

        profile = _make_profile_with_llm(tmp_path)
        mock_close = MagicMock()
        mock_validator = MagicMock()
        mock_validator.validate.return_value = VR(
            relevant=True, reason="ok", source="llm", cached=False
        )
        mock_validator.close = mock_close

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.LLMValidator", return_value=mock_validator),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        mock_close.assert_called_once()

    def test_no_llm_when_config_is_none(self, tmp_path):
        """With llm_validation=None, LLMValidator is never instantiated."""
        profile = make_profile(tmp_path)  # llm_validation=None

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.LLMValidator") as mock_llm_cls,
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        mock_llm_cls.assert_not_called()


# ── Incremental index update ──────────────────────────────────────────────

class TestIncrementalIndex:
    """After each successful download, title_index.json must gain the new title.
    Failed downloads and invalid PDFs must NOT be added to the index."""

    def _make_profile_with_index(self, tmp_path: Path) -> tuple:
        """Return (profile, index_path) with an empty pre-existing index."""
        paperbase_dir = tmp_path / "paperbase"
        paperbase_dir.mkdir()
        index_path = paperbase_dir / "title_index.json"
        index_path.write_text("{}", encoding="utf-8")

        downloads_dir = tmp_path / "downloads"
        downloads_dir.mkdir()

        profile = Profile(
            name="Test Domain",
            journals=[],
            keywords=["test keyword"],
            subfield_filter=None,
            year_min=2020,
            year_max=2025,
            max_per_query=10,
            mailto="test@example.com",
            domain_description="Test domain.",
            paperbase_dir=paperbase_dir,
            downloads_dir=downloads_dir,
            papers_dir=downloads_dir / "papers",
            metadata_path=downloads_dir / "metadata.jsonl",
            failed_path=downloads_dir / "failed_downloads.jsonl",
            venue_cache_path=tmp_path / "venue_cache.json",
            llm_validation=None,
            validation_cache_path=downloads_dir / "validation_cache.json",
            s2_config=None,
        )
        return profile, index_path

    def test_successful_download_updates_index(self, tmp_path):
        """After downloading WORK_A, its normalized title appears in title_index.json."""
        from paperbase.index_manager import IndexManager, normalize

        profile, index_path = self._make_profile_with_index(tmp_path)

        def fake_download(url, out_path, mailto):
            out_path.write_bytes(b"%PDF-1.4 fake")
            return (True, "")

        with (
            patch("collect_papers.OpenAlexClient",
                  return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", side_effect=fake_download),
            patch("collect_papers.validate_pdf", return_value=True),
            # Prevent add_pdf from trying to open the fake PDF with fitz
            patch("paperbase.index_manager.extract_title",
                  return_value=("RDMA Congestion Control", "metadata")),
        ):
            collect(profile, mode="keyword", dry_run=False)

        im = IndexManager.load(index_path)
        assert im.contains(normalize("RDMA Congestion Control"))

    def test_multiple_downloads_all_indexed(self, tmp_path):
        """All three works downloaded successfully → all three in index."""
        from paperbase.index_manager import IndexManager, normalize

        profile, index_path = self._make_profile_with_index(tmp_path)

        def fake_download(url, out_path, mailto):
            out_path.write_bytes(b"%PDF-1.4 fake")
            return (True, "")

        # Return different titles per call using side_effect cycle
        titles = [
            ("RDMA Congestion Control", "metadata"),
            ("P4 Programmable Switch", "metadata"),
            ("eBPF Network Functions", "metadata"),
        ]
        with (
            patch("collect_papers.OpenAlexClient",
                  return_value=_mock_client_class(THREE_WORKS)),
            patch("collect_papers.download_pdf", side_effect=fake_download),
            patch("collect_papers.validate_pdf", return_value=True),
            patch("paperbase.index_manager.extract_title", side_effect=titles),
        ):
            collect(profile, mode="keyword", dry_run=False)

        im = IndexManager.load(index_path)
        assert im.contains(normalize("RDMA Congestion Control"))
        assert im.contains(normalize("P4 Programmable Switch"))
        assert im.contains(normalize("eBPF Network Functions"))

    def test_download_failure_not_indexed(self, tmp_path):
        """A paper whose download fails must not appear in the index."""
        from paperbase.index_manager import IndexManager, normalize

        profile, index_path = self._make_profile_with_index(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient",
                  return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", return_value=(False, "HTTP 403 Forbidden (publisher blocks automated access)")),
        ):
            collect(profile, mode="keyword", dry_run=False)

        im = IndexManager.load(index_path)
        assert not im.contains(normalize("RDMA Congestion Control"))

    def test_invalid_pdf_not_indexed(self, tmp_path):
        """A paper that fails PDF validation must not be in the index."""
        from paperbase.index_manager import IndexManager, normalize

        profile, index_path = self._make_profile_with_index(tmp_path)

        def fake_download(url, out_path, mailto):
            out_path.write_bytes(b"<html>Not a PDF</html>")
            return (True, "")

        with (
            patch("collect_papers.OpenAlexClient",
                  return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", side_effect=fake_download),
            patch("collect_papers.validate_pdf", return_value=False),
        ):
            collect(profile, mode="keyword", dry_run=False)

        im = IndexManager.load(index_path)
        assert not im.contains(normalize("RDMA Congestion Control"))

    def test_dry_run_does_not_update_index(self, tmp_path):
        """In dry_run mode the index file must remain unchanged."""
        from paperbase.index_manager import IndexManager

        profile, index_path = self._make_profile_with_index(tmp_path)
        mtime_before = index_path.stat().st_mtime

        with patch("collect_papers.OpenAlexClient",
                   return_value=_mock_client_class(THREE_WORKS)):
            collect(profile, mode="keyword", dry_run=True)

        mtime_after = index_path.stat().st_mtime
        assert mtime_before == mtime_after

    def test_no_paperbase_dir_no_index_created(self, tmp_path):
        """With paperbase_dir=None, no title_index.json is created anywhere."""
        profile = make_profile(tmp_path)   # paperbase_dir=None

        def fake_download(url, out_path, mailto):
            out_path.write_bytes(b"%PDF-1.4 fake")
            return (True, "")

        with (
            patch("collect_papers.OpenAlexClient",
                  return_value=_mock_client_class([WORK_A])),
            patch("collect_papers.download_pdf", side_effect=fake_download),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", dry_run=False)

        index_files = list(tmp_path.rglob("title_index.json"))
        assert index_files == []


# ── Normalization unit tests ──────────────────────────────────────────────

from collect_papers import normalize_openalex_work, normalize_s2_paper


class TestNormalizeOpenAlexWork:
    def test_all_fields_extracted(self):
        paper = normalize_openalex_work(WORK_A, "keyword")
        assert paper["source_id"] == "W001"
        assert paper["title"] == "RDMA Congestion Control"
        assert paper["doi"] == "10.1145/001"
        assert paper["year"] == 2023
        assert paper["venue"] == "Test Venue"
        assert paper["abstract"] == "Test abstract"
        assert paper["source_tag"] == "keyword"
        assert paper["pdf_url"] == "https://example.com/paper.pdf"
        assert paper["license"] == "cc-by"

    def test_journal_source_tag(self):
        paper = normalize_openalex_work(WORK_A, "journal")
        assert paper["source_tag"] == "journal"


class TestNormalizeS2Paper:
    def _make_s2_paper(
        self,
        paper_id="abc123",
        title="Test S2 Paper",
        abstract="S2 abstract text.",
        doi="10.9999/s2test",
        pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
        year=2024,
    ):
        result = {
            "paperId": paper_id,
            "corpusId": 99999,
            "title": title,
            "abstract": abstract,
            "year": year,
            "externalIds": {"DOI": doi} if doi else {},
            "openAccessPdf": {"url": pdf_url} if pdf_url else None,
            "authors": [{"authorId": "1", "name": "Bob"}],
            "venue": "NSDI",
            "fieldsOfStudy": ["Computer Science"],
            "isOpenAccess": True,
        }
        return result

    def test_all_fields_extracted(self):
        paper = normalize_s2_paper(self._make_s2_paper())
        assert paper["source_id"] == "s2:abc123"
        assert paper["title"] == "Test S2 Paper"
        assert paper["doi"] == "10.9999/s2test"
        assert paper["year"] == 2024
        assert paper["venue"] == "NSDI"
        assert paper["abstract"] == "S2 abstract text."
        assert paper["source_tag"] == "s2"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/2401.00001.pdf"

    def test_no_open_access_pdf_gives_none(self):
        paper = normalize_s2_paper(self._make_s2_paper(pdf_url=None))
        assert paper["pdf_url"] is None

    def test_no_doi(self):
        paper = normalize_s2_paper(self._make_s2_paper(doi=None))
        assert paper["doi"] is None

    def test_no_abstract(self):
        paper = normalize_s2_paper(self._make_s2_paper(abstract=None))
        assert paper["abstract"] is None

    def test_authors_extracted(self):
        paper = normalize_s2_paper(self._make_s2_paper())
        assert paper["authors"] == ["Bob"]

    def test_license_is_none(self):
        """S2 API doesn't provide license info."""
        paper = normalize_s2_paper(self._make_s2_paper())
        assert paper["license"] is None


# ── Semantic Scholar pipeline integration ─────────────────────────────────


def _make_s2_config():
    from profile import S2Config
    return S2Config(
        api_key="test-key",
        fields_of_study=["Computer Science"],
        keywords=["RDMA datacenter"],
        max_per_query=10,
    )


def _make_s2_api_paper(paper_id, title, doi=None, pdf_url=None):
    """Build a raw S2 API response paper dict."""
    return {
        "paperId": paper_id,
        "corpusId": 12345,
        "title": title,
        "abstract": f"Abstract of {title}",
        "year": 2024,
        "externalIds": {"DOI": doi} if doi else {},
        "openAccessPdf": {"url": pdf_url} if pdf_url else None,
        "authors": [{"authorId": "1", "name": "Alice"}],
        "venue": "SIGCOMM",
        "fieldsOfStudy": ["Computer Science"],
        "isOpenAccess": pdf_url is not None,
    }


S2_PAPER_A = _make_s2_api_paper(
    "s2id001", "RDMA Scale-Out Networking",
    doi="10.1145/s2001", pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
)
S2_PAPER_B = _make_s2_api_paper(
    "s2id002", "SmartNIC Offload Engine",
    doi="10.1145/s2002", pdf_url="https://arxiv.org/pdf/2401.00002.pdf",
)


def _mock_s2_client_class(papers: list):
    """Return a patched S2Client instance whose search returns `papers`."""
    mock_client = MagicMock()
    mock_client.search.return_value = papers
    return mock_client


class TestS2Integration:
    def _make_s2_profile(self, tmp_path):
        """make_profile with S2 config enabled, no OpenAlex keywords."""
        profile = make_profile(tmp_path)
        object.__setattr__(profile, "s2_config", _make_s2_config())
        return profile

    def test_s2_papers_downloaded(self, tmp_path):
        """S2 papers flow through the pipeline and get downloaded."""
        profile = self._make_s2_profile(tmp_path)

        with (
            patch("collect_papers.S2Client", return_value=_mock_s2_client_class([S2_PAPER_A, S2_PAPER_B])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", sources="s2", dry_run=False)

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 2
        titles = {r["title"] for r in records}
        assert "RDMA Scale-Out Networking" in titles
        assert "SmartNIC Offload Engine" in titles

    def test_s2_records_have_source_id(self, tmp_path):
        """S2 metadata records must have source_id starting with 's2:'."""
        profile = self._make_s2_profile(tmp_path)

        with (
            patch("collect_papers.S2Client", return_value=_mock_s2_client_class([S2_PAPER_A])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", sources="s2", dry_run=False)

        record = json.loads(profile.metadata_path.read_text().splitlines()[0])
        assert record["source_id"].startswith("s2:")
        assert "openalex_id" not in record   # S2 papers have no openalex_id

    def test_s2_no_pdf_skipped(self, tmp_path):
        """S2 paper without openAccessPdf gets skipped (open-access only)."""
        profile = self._make_s2_profile(tmp_path)
        no_pdf_paper = _make_s2_api_paper(
            "s2id003", "Closed Access Paper", doi="10.1145/closed", pdf_url=None,
        )

        with (
            patch("collect_papers.S2Client", return_value=_mock_s2_client_class([no_pdf_paper])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", sources="s2", dry_run=False)

        assert not profile.metadata_path.exists()

    def test_cross_source_doi_dedup(self, tmp_path):
        """Paper with same DOI from OpenAlex and S2: only one downloaded."""
        profile = self._make_s2_profile(tmp_path)
        shared_doi = "10.1145/shared"

        # OpenAlex work with the shared DOI
        oa_work = make_work("W099", "Shared Paper OA", doi=shared_doi)
        # S2 paper with the same DOI
        s2_paper = _make_s2_api_paper(
            "s2id099", "Shared Paper S2",
            doi=shared_doi, pdf_url="https://arxiv.org/pdf/shared.pdf",
        )

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([oa_work])),
            patch("collect_papers.S2Client", return_value=_mock_s2_client_class([s2_paper])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", sources="all", dry_run=False)

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 1   # DOI dedup catches the second

    def test_s2_not_fetched_when_sources_openalex(self, tmp_path):
        """With --sources=openalex, S2Client is never instantiated."""
        profile = self._make_s2_profile(tmp_path)

        with (
            patch("collect_papers.OpenAlexClient", return_value=_mock_client_class([])),
            patch("collect_papers.S2Client") as mock_s2_cls,
        ):
            collect(profile, mode="keyword", sources="openalex", dry_run=True)

        mock_s2_cls.assert_not_called()

    def test_s2_dry_run(self, tmp_path):
        """Dry-run with S2 source — no files written."""
        profile = self._make_s2_profile(tmp_path)

        with (
            patch("collect_papers.S2Client", return_value=_mock_s2_client_class([S2_PAPER_A])),
            patch("collect_papers.download_pdf") as mock_dl,
        ):
            collect(profile, mode="keyword", sources="s2", dry_run=True)

        mock_dl.assert_not_called()
        assert not profile.metadata_path.exists()

    def test_s2_resume_skips_already_downloaded(self, tmp_path):
        """S2 paper whose source_id is in metadata.jsonl gets skipped."""
        profile = self._make_s2_profile(tmp_path)

        # Pre-populate with a source_id
        existing = {"source_id": "s2:s2id001", "title": "Already downloaded"}
        profile.metadata_path.write_text(
            json.dumps(existing) + "\n", encoding="utf-8",
        )

        with (
            patch("collect_papers.S2Client", return_value=_mock_s2_client_class([S2_PAPER_A])),
            patch("collect_papers.download_pdf", return_value=(True, "")),
            patch("collect_papers.validate_pdf", return_value=True),
        ):
            collect(profile, mode="keyword", sources="s2", dry_run=False)

        records = [
            json.loads(line)
            for line in profile.metadata_path.read_text().splitlines()
            if line.strip()
        ]
        # Only the pre-existing record — no new downloads
        new_records = [r for r in records if r.get("local_filename") is not None]
        assert len(new_records) == 0

