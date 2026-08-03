"""
test_index_manager.py — Unit tests for paperbase/index_manager.py.

Covers:
  normalize()        — canonical normalization behaviour
  IndexManager.load()   — missing file, corrupt file, valid file
  IndexManager.contains()
  IndexManager.all_normalized_titles()
  IndexManager.add_pdf()  — extract_failed, added, strong_dup, weak_dup
  IndexManager.save()     — index written; duplicate_candidates merged
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Allow importing from parent without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperbase.index_manager import (
    normalize,
    IndexManager,
    AddResult,
    extract_title,
)


# ─────────────────────────────────────────────────────────────────────────
# normalize()
# ─────────────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_lowercases(self):
        assert normalize("RDMA Performance") == "rdma performance"

    def test_replaces_punctuation_with_space(self):
        # Colon becomes a space; then whitespace is collapsed
        assert normalize("P4: A Language") == "p4 a language"

    def test_collapses_whitespace(self):
        assert normalize("  TCP   BBR  ") == "tcp bbr"

    def test_hyphen_becomes_space(self):
        assert normalize("High-Performance Network") == "high performance network"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_numbers_preserved(self):
        assert normalize("100 Gbps Links") == "100 gbps links"

    def test_unicode_letters_preserved(self):
        # Unicode word characters (e.g. accented letters) must not be stripped
        result = normalize("Réseaux à haute performance")
        assert "r" in result and "seaux" in result

    def test_idempotent(self):
        s = "Deep Learning: Systems & Compilers"
        assert normalize(normalize(s)) == normalize(s)


# ─────────────────────────────────────────────────────────────────────────
# IndexManager.load()
# ─────────────────────────────────────────────────────────────────────────

class TestIndexManagerLoad:
    def test_load_missing_file_returns_empty(self, tmp_path):
        im = IndexManager.load(tmp_path / "nonexistent.json")
        assert len(im) == 0

    def test_load_none_returns_empty(self):
        im = IndexManager.load(None)
        assert len(im) == 0

    def test_load_valid_file(self, tmp_path):
        index_path = tmp_path / "title_index.json"
        index_path.write_text(
            json.dumps({"rdma congestion control": {"raw_title": "RDMA Congestion Control",
                                                     "path": "hpn/rdma.pdf",
                                                     "source": "metadata"}}),
            encoding="utf-8",
        )
        im = IndexManager.load(index_path)
        assert len(im) == 1
        assert im.contains("rdma congestion control")

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        index_path = tmp_path / "title_index.json"
        index_path.write_bytes(b"NOT JSON {{{")
        im = IndexManager.load(index_path)
        assert len(im) == 0


# ─────────────────────────────────────────────────────────────────────────
# IndexManager.contains() / all_normalized_titles()
# ─────────────────────────────────────────────────────────────────────────

class TestIndexManagerQueries:
    def test_contains_true(self, tmp_path):
        index_path = tmp_path / "title_index.json"
        index_path.write_text(json.dumps({"my title": {}}), encoding="utf-8")
        im = IndexManager.load(index_path)
        assert im.contains("my title") is True

    def test_contains_false(self, tmp_path):
        im = IndexManager.load(None)
        assert im.contains("anything") is False

    def test_all_normalized_titles_returns_frozenset(self, tmp_path):
        index_path = tmp_path / "title_index.json"
        index_path.write_text(
            json.dumps({"title a": {}, "title b": {}}), encoding="utf-8"
        )
        im = IndexManager.load(index_path)
        titles = im.all_normalized_titles()
        assert isinstance(titles, frozenset)
        assert titles == frozenset({"title a", "title b"})


# ─────────────────────────────────────────────────────────────────────────
# IndexManager.add_pdf()
# ─────────────────────────────────────────────────────────────────────────

# Helpers to build fake PDF paths without real files
def _fake_pdf(tmp_path: Path, name: str = "paper.pdf") -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake content for testing " + name.encode())
    return p


class TestAddPdfExtractFailed:
    def test_extract_failed_returns_correct_status(self, tmp_path):
        im = IndexManager()
        pdf = _fake_pdf(tmp_path)
        with patch("paperbase.index_manager.extract_title", return_value=("", "error: bad")):
            result = im.add_pdf(pdf)
        assert result.status == "extract_failed"
        assert result.raw_title == ""
        assert len(im) == 0

    def test_extract_failed_does_not_modify_index(self, tmp_path):
        im = IndexManager()
        pdf = _fake_pdf(tmp_path)
        with patch("paperbase.index_manager.extract_title", return_value=("", "no-text")):
            im.add_pdf(pdf)
        assert len(im) == 0


class TestAddPdfAdded:
    def test_added_returns_correct_status(self, tmp_path):
        im = IndexManager()
        pdf = _fake_pdf(tmp_path)
        with patch("paperbase.index_manager.extract_title",
                   return_value=("RDMA Congestion Control", "metadata")):
            result = im.add_pdf(pdf)
        assert result.status == "added"
        assert result.raw_title == "RDMA Congestion Control"
        assert result.norm_title == normalize("RDMA Congestion Control")

    def test_added_increments_length(self, tmp_path):
        im = IndexManager()
        pdf = _fake_pdf(tmp_path)
        with patch("paperbase.index_manager.extract_title",
                   return_value=("Some New Paper", "font-heuristic")):
            im.add_pdf(pdf)
        assert len(im) == 1

    def test_added_makes_contains_true(self, tmp_path):
        im = IndexManager()
        pdf = _fake_pdf(tmp_path)
        with patch("paperbase.index_manager.extract_title",
                   return_value=("Some New Paper", "font-heuristic")):
            im.add_pdf(pdf)
        assert im.contains(normalize("Some New Paper"))

    def test_added_stores_relative_path_when_root_given(self, tmp_path):
        root = tmp_path / "papers"
        root.mkdir()
        pdf = _fake_pdf(root, "paper.pdf")
        im = IndexManager()
        with patch("paperbase.index_manager.extract_title",
                   return_value=("Paper Title", "metadata")):
            im.add_pdf(pdf, root=root)
        # The stored path must be relative to root.parent
        titles = list(im._index.values())
        stored_path = titles[0]["path"]
        assert not Path(stored_path).is_absolute()


class TestAddPdfDuplicate:
    def _make_im_with_entry(self, tmp_path, title: str, path: str) -> IndexManager:
        index_path = tmp_path / "title_index.json"
        norm = normalize(title)
        index_path.write_text(
            json.dumps({norm: {"raw_title": title, "path": path, "source": "metadata"}}),
            encoding="utf-8",
        )
        return IndexManager.load(index_path)

    def test_strong_dup_status(self, tmp_path):
        im = self._make_im_with_entry(tmp_path, "RDMA Congestion Control", "hpn/rdma.pdf")
        pdf = _fake_pdf(tmp_path, "rdma2.pdf")

        with (
            patch("paperbase.index_manager.extract_title",
                  return_value=("RDMA Congestion Control", "metadata")),
            patch("paperbase.index_manager._is_strong_duplicate",
                  return_value=(True, {"page_count": True, "file_size": True, "text_sim": True})),
        ):
            result = im.add_pdf(pdf)

        assert result.status == "strong_dup"
        assert result.existing_path == "hpn/rdma.pdf"

    def test_strong_dup_does_not_add_to_index(self, tmp_path):
        im = self._make_im_with_entry(tmp_path, "RDMA Congestion Control", "hpn/rdma.pdf")
        pdf = _fake_pdf(tmp_path, "rdma2.pdf")

        with (
            patch("paperbase.index_manager.extract_title",
                  return_value=("RDMA Congestion Control", "metadata")),
            patch("paperbase.index_manager._is_strong_duplicate",
                  return_value=(True, {})),
        ):
            im.add_pdf(pdf)

        assert len(im) == 1  # still just the one original entry

    def test_weak_dup_status(self, tmp_path):
        im = self._make_im_with_entry(tmp_path, "RDMA Congestion Control", "hpn/rdma.pdf")
        pdf = _fake_pdf(tmp_path, "rdma2.pdf")

        with (
            patch("paperbase.index_manager.extract_title",
                  return_value=("RDMA Congestion Control", "metadata")),
            patch("paperbase.index_manager._is_strong_duplicate",
                  return_value=(False, {"page_count": False, "file_size": None, "text_sim": None})),
        ):
            result = im.add_pdf(pdf)

        assert result.status == "weak_dup"

    def test_duplicate_recorded_in_candidates(self, tmp_path):
        im = self._make_im_with_entry(tmp_path, "Title X", "dir/x.pdf")
        pdf = _fake_pdf(tmp_path, "x2.pdf")

        with (
            patch("paperbase.index_manager.extract_title",
                  return_value=("Title X", "metadata")),
            patch("paperbase.index_manager._is_strong_duplicate",
                  return_value=(True, {})),
        ):
            im.add_pdf(pdf)

        assert len(im._candidates) == 1
        cand = im._candidates[0]
        assert cand["existing"] == "dir/x.pdf"
        assert cand["confidence"] == "STRONG"


# ─────────────────────────────────────────────────────────────────────────
# IndexManager.save()
# ─────────────────────────────────────────────────────────────────────────

class TestIndexManagerSave:
    def test_save_writes_index_json(self, tmp_path):
        im = IndexManager()
        pdf = _fake_pdf(tmp_path)
        with patch("paperbase.index_manager.extract_title",
                   return_value=("My Paper", "metadata")):
            im.add_pdf(pdf)

        out = tmp_path / "title_index.json"
        im.save(out)
        assert out.exists()
        data = json.loads(out.read_text("utf-8"))
        assert normalize("My Paper") in data

    def test_save_writes_candidates_when_present(self, tmp_path):
        index_path = tmp_path / "title_index.json"
        index_path.write_text(
            json.dumps({normalize("Paper X"): {"raw_title": "Paper X",
                                                "path": "dir/x.pdf",
                                                "source": "metadata"}}),
            encoding="utf-8",
        )
        im = IndexManager.load(index_path)
        pdf = _fake_pdf(tmp_path, "x2.pdf")

        with (
            patch("paperbase.index_manager.extract_title",
                  return_value=("Paper X", "metadata")),
            patch("paperbase.index_manager._is_strong_duplicate",
                  return_value=(True, {})),
        ):
            im.add_pdf(pdf)

        im.save(index_path)
        cand_path = tmp_path / "duplicate_candidates.json"
        assert cand_path.exists()
        candidates = json.loads(cand_path.read_text("utf-8"))
        assert len(candidates) == 1

    def test_save_merges_candidates_with_existing(self, tmp_path):
        """save() must append to, not overwrite, duplicate_candidates.json."""
        cand_path = tmp_path / "duplicate_candidates.json"
        cand_path.write_text(json.dumps([{"existing": "old.pdf"}]), encoding="utf-8")

        index_path = tmp_path / "title_index.json"
        index_path.write_text(
            json.dumps({normalize("Title Z"): {"raw_title": "Title Z",
                                                "path": "z.pdf",
                                                "source": "metadata"}}),
            encoding="utf-8",
        )
        im = IndexManager.load(index_path)
        pdf = _fake_pdf(tmp_path, "z2.pdf")

        with (
            patch("paperbase.index_manager.extract_title",
                  return_value=("Title Z", "metadata")),
            patch("paperbase.index_manager._is_strong_duplicate",
                  return_value=(True, {})),
        ):
            im.add_pdf(pdf)

        im.save(index_path)
        candidates = json.loads(cand_path.read_text("utf-8"))
        assert len(candidates) == 2   # old entry preserved + new
        assert candidates[0]["existing"] == "old.pdf"

    def test_save_clears_candidates_after_flush(self, tmp_path):
        index_path = tmp_path / "title_index.json"
        index_path.write_text(
            json.dumps({normalize("Title A"): {"raw_title": "Title A",
                                                "path": "a.pdf",
                                                "source": "metadata"}}),
            encoding="utf-8",
        )
        im = IndexManager.load(index_path)
        pdf = _fake_pdf(tmp_path, "a2.pdf")

        with (
            patch("paperbase.index_manager.extract_title",
                  return_value=("Title A", "metadata")),
            patch("paperbase.index_manager._is_strong_duplicate",
                  return_value=(True, {})),
        ):
            im.add_pdf(pdf)

        assert len(im._candidates) == 1
        im.save(index_path)
        assert len(im._candidates) == 0   # reset after flush

    def test_save_no_candidates_file_when_empty(self, tmp_path):
        im = IndexManager()
        out = tmp_path / "title_index.json"
        im.save(out)
        assert not (tmp_path / "duplicate_candidates.json").exists()
