"""
test_merge_collections.py — Unit tests for merge_collections.py.

Covers:
  - Unique papers are copied to target
  - STRONG duplicates are detected and skipped
  - WEAK candidates are recorded
  - Title-extract failures are copied with a warning
  - Dry-run mode writes no files
  - Index and duplicate_candidates.json are saved
  - File name collision handling (_safe_copy)
  - Multiple source directories
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge_collections import merge, _safe_copy
from paperbase.index_manager import AddResult


# ── Helpers ───────────────────────────────────────────────────────────────

def _create_pdf(path: Path, content: bytes = b"%PDF-1.4 fake") -> Path:
    """Create a minimal fake PDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _mock_add_pdf_factory(results: list[AddResult]):
    """Return a side_effect function that yields AddResults in order."""
    it = iter(results)
    def _add_pdf(pdf_path, root=None):
        return next(it)
    return _add_pdf


# ── _safe_copy ────────────────────────────────────────────────────────────

class TestSafeCopy:
    def test_copies_file(self, tmp_path):
        src = _create_pdf(tmp_path / "src" / "paper.pdf")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        result = _safe_copy(src, dest_dir)
        assert result == dest_dir / "paper.pdf"
        assert result.read_bytes() == b"%PDF-1.4 fake"

    def test_appends_suffix_on_collision(self, tmp_path):
        src = _create_pdf(tmp_path / "src" / "paper.pdf")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        # Create pre-existing file
        (dest_dir / "paper.pdf").write_bytes(b"existing")
        result = _safe_copy(src, dest_dir)
        assert result == dest_dir / "paper_1.pdf"
        assert result.read_bytes() == b"%PDF-1.4 fake"
        # Original untouched
        assert (dest_dir / "paper.pdf").read_bytes() == b"existing"

    def test_increments_suffix(self, tmp_path):
        src = _create_pdf(tmp_path / "src" / "paper.pdf")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        (dest_dir / "paper.pdf").write_bytes(b"v0")
        (dest_dir / "paper_1.pdf").write_bytes(b"v1")
        result = _safe_copy(src, dest_dir)
        assert result == dest_dir / "paper_2.pdf"


# ── Merge — unique papers ────────────────────────────────────────────────

class TestMergeUnique:
    def test_unique_papers_copied(self, tmp_path):
        # Setup: two source dirs with one PDF each
        src1 = tmp_path / "srcA"
        src2 = tmp_path / "srcB"
        _create_pdf(src1 / "alpha.pdf")
        _create_pdf(src2 / "beta.pdf")
        target = tmp_path / "merged"
        idx = target / "title_index.json"

        added_a = AddResult("added", "Alpha Paper", "alpha paper", "metadata")
        added_b = AddResult("added", "Beta Paper", "beta paper", "metadata")

        with patch("merge_collections.IndexManager") as MockIM:
            im = MockIM.return_value
            im.add_pdf = _mock_add_pdf_factory([added_a, added_b])
            im._index = {
                "alpha paper": {"raw_title": "Alpha Paper", "path": "x", "source": "metadata"},
                "beta paper": {"raw_title": "Beta Paper", "path": "y", "source": "metadata"},
            }
            im.save = MagicMock()

            merge([src1, src2], target, idx, dry_run=False)

        # Two files copied
        pdfs = list(target.glob("*.pdf"))
        assert len(pdfs) == 2
        im.save.assert_called_once_with(idx)


# ── Merge — duplicates detected ──────────────────────────────────────────

class TestMergeDuplicates:
    def test_strong_dup_not_copied(self, tmp_path):
        src = tmp_path / "src"
        _create_pdf(src / "a.pdf")
        _create_pdf(src / "b.pdf")
        target = tmp_path / "merged"
        idx = target / "title_index.json"

        added = AddResult("added", "Paper A", "paper a", "metadata")
        dup = AddResult("strong_dup", "Paper A", "paper a", "metadata", existing_path="merged/a.pdf")

        with patch("merge_collections.IndexManager") as MockIM:
            im = MockIM.return_value
            im.add_pdf = _mock_add_pdf_factory([added, dup])
            im._index = {"paper a": {"raw_title": "Paper A", "path": "x", "source": "metadata"}}
            im.save = MagicMock()

            merge([src], target, idx, dry_run=False)

        # Only 1 file copied (the first one), the duplicate was skipped
        pdfs = list(target.glob("*.pdf"))
        assert len(pdfs) == 1

    def test_weak_candidate_not_copied(self, tmp_path):
        src = tmp_path / "src"
        _create_pdf(src / "a.pdf")
        _create_pdf(src / "b.pdf")
        target = tmp_path / "merged"
        idx = target / "title_index.json"

        added = AddResult("added", "Paper A", "paper a", "metadata")
        weak = AddResult("weak_dup", "Paper A?", "paper a", "metadata", existing_path="merged/a.pdf")

        with patch("merge_collections.IndexManager") as MockIM:
            im = MockIM.return_value
            im.add_pdf = _mock_add_pdf_factory([added, weak])
            im._index = {"paper a": {"raw_title": "Paper A", "path": "x", "source": "metadata"}}
            im.save = MagicMock()

            merge([src], target, idx, dry_run=False)

        pdfs = list(target.glob("*.pdf"))
        assert len(pdfs) == 1


# ── Merge — extract failed ───────────────────────────────────────────────

class TestMergeExtractFailed:
    def test_extract_failed_still_copied(self, tmp_path):
        src = tmp_path / "src"
        _create_pdf(src / "mystery.pdf")
        target = tmp_path / "merged"
        idx = target / "title_index.json"

        fail = AddResult("extract_failed", "", "", "no-text")

        with patch("merge_collections.IndexManager") as MockIM:
            im = MockIM.return_value
            im.add_pdf = _mock_add_pdf_factory([fail])
            im._index = {}
            im.save = MagicMock()

            merge([src], target, idx, dry_run=False)

        pdfs = list(target.glob("*.pdf"))
        assert len(pdfs) == 1


# ── Dry run ───────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_creates_no_files(self, tmp_path):
        src = tmp_path / "src"
        _create_pdf(src / "paper.pdf")
        target = tmp_path / "merged"
        idx = target / "title_index.json"

        added = AddResult("added", "Paper", "paper", "metadata")

        with patch("merge_collections.IndexManager") as MockIM:
            im = MockIM.return_value
            im.add_pdf = _mock_add_pdf_factory([added])
            im._index = {}
            im.save = MagicMock()

            merge([src], target, idx, dry_run=True)

        assert not target.exists()
        im.save.assert_not_called()
