"""
test_profile.py — Unit tests for profile.py (ProfileLoader + Profile dataclass).

Tests cover:
  - Valid profiles load correctly
  - Missing required fields raise ValueError with the field name
  - Invalid year range raises ValueError
  - Optional fields produce correct defaults
  - paperbase_dir resolves relative to profile file location
  - paperbase_dir: null loads as None
  - profiles/hpn.yaml and profiles/template.yaml parse without errors
  - downloads_dir and derived paths are set correctly
"""

import sys
import textwrap
from pathlib import Path

import pytest

# Allow importing profile from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profile import load_profile, Profile

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


# ── Helpers ──────────────────────────────────────────────────────────────

def write_profile(tmp_path: Path, content: str) -> Path:
    """Write a YAML string to a temp profile file and return its path."""
    p = tmp_path / "test_profile.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


MINIMAL_VALID = """
    name: "Test Domain"
    mailto: "test@example.com"
    keywords:
      - '"test" keyword'
"""

FULL_VALID = """
    name: "Full Domain"
    mailto: "user@example.com"
    year_min: 2020
    year_max: 2024
    max_per_query: 200
    domain_description: "Papers about testing."
    subfield_filter: "primary_topic.subfield.id:subfields/1705"
    journals:
      - search: "Test Journal Full Name"
        match: "Test Journal"
    keywords:
      - '"test" keyword'
      - '"another" query'
"""


# ── Required field validation ────────────────────────────────────────────

class TestRequiredFields:
    def test_missing_name_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            mailto: "test@example.com"
            keywords:
              - "test"
        """)
        with pytest.raises(ValueError, match="name"):
            load_profile(p)

    def test_missing_mailto_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            keywords:
              - "test"
        """)
        with pytest.raises(ValueError, match="mailto"):
            load_profile(p)

    def test_empty_name_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            name: ""
            mailto: "test@example.com"
            keywords:
              - "test"
        """)
        with pytest.raises(ValueError, match="name"):
            load_profile(p)

    def test_no_journals_no_keywords_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
        """)
        with pytest.raises(ValueError, match="journals.*keywords|keywords.*journals"):
            load_profile(p)

    def test_empty_journals_and_keywords_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
            journals: []
            keywords: []
        """)
        with pytest.raises(ValueError):
            load_profile(p)


# ── Year range validation ────────────────────────────────────────────────

class TestYearRange:
    def test_year_min_greater_than_year_max_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            year_min: 2025
            year_max: 2020
        """)
        with pytest.raises(ValueError, match="year_min"):
            load_profile(p)

    def test_equal_years_valid(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            year_min: 2023
            year_max: 2023
        """)
        profile = load_profile(p)
        assert profile.year_min == 2023
        assert profile.year_max == 2023


# ── Defaults ─────────────────────────────────────────────────────────────

class TestDefaults:
    def test_year_min_default(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.year_min == 2018

    def test_year_max_default(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.year_max == 2026

    def test_max_per_query_default(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.max_per_query == 150

    def test_subfield_filter_default_is_none(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.subfield_filter is None

    def test_journals_default_is_empty(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.journals == []

    def test_paperbase_dir_default_is_none(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.paperbase_dir is None

    def test_downloads_dir_default_derived_from_name(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        # Default: <profile_dir>/../downloads/<slug>/
        # profile_dir is tmp_path, so parent of tmp_path / downloads / test_domain
        assert profile.downloads_dir == (tmp_path.parent / "downloads" / "test_domain").resolve()

    def test_metadata_path_inside_downloads_dir(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.metadata_path == profile.downloads_dir / "metadata.jsonl"

    def test_papers_dir_inside_downloads_dir(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.papers_dir == profile.downloads_dir / "papers"

    def test_failed_path_inside_downloads_dir(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.failed_path == profile.downloads_dir / "failed_downloads.jsonl"

    def test_venue_cache_in_profile_dir(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.venue_cache_path.parent == tmp_path
        assert profile.venue_cache_path.name.endswith("_venue_cache.json")


# ── Path resolution ───────────────────────────────────────────────────────

class TestPathResolution:
    def test_paperbase_dir_null_loads_as_none(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            paperbase_dir: null
        """)
        profile = load_profile(p)
        assert profile.paperbase_dir is None

    def test_paperbase_dir_relative_resolved_from_profile_dir(self, tmp_path):
        # Create a fake sibling directory
        sibling = tmp_path / "My-Check-Exists"
        sibling.mkdir()
        p = write_profile(tmp_path, f"""
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            paperbase_dir: "My-Check-Exists"
        """)
        profile = load_profile(p)
        assert profile.paperbase_dir == sibling.resolve()

    def test_downloads_dir_relative_resolved_from_profile_dir(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            downloads_dir: "my-downloads"
        """)
        profile = load_profile(p)
        assert profile.downloads_dir == (tmp_path / "my-downloads").resolve()

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_profile(tmp_path / "nonexistent.yaml")


# ── Journal parsing ───────────────────────────────────────────────────────

class TestJournalParsing:
    def test_journals_parsed_as_tuples(self, tmp_path):
        p = write_profile(tmp_path, FULL_VALID)
        profile = load_profile(p)
        assert len(profile.journals) == 1
        search, match = profile.journals[0]
        assert search == "Test Journal Full Name"
        assert match == "Test Journal"

    def test_journal_missing_match_key_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
            journals:
              - search: "Some Journal"
            keywords: ["test"]
        """)
        with pytest.raises(ValueError, match="match"):
            load_profile(p)

    def test_journal_missing_search_key_raises(self, tmp_path):
        p = write_profile(tmp_path, """
            name: "Domain"
            mailto: "test@example.com"
            journals:
              - match: "Some Journal"
            keywords: ["test"]
        """)
        with pytest.raises(ValueError, match="search"):
            load_profile(p)


# ── Full load ─────────────────────────────────────────────────────────────

class TestFullLoad:
    def test_full_profile_loads_all_fields(self, tmp_path):
        p = write_profile(tmp_path, FULL_VALID)
        profile = load_profile(p)
        assert profile.name == "Full Domain"
        assert profile.mailto == "user@example.com"
        assert profile.year_min == 2020
        assert profile.year_max == 2024
        assert profile.max_per_query == 200
        assert profile.domain_description == "Papers about testing."
        assert profile.subfield_filter == "primary_topic.subfield.id:subfields/1705"
        assert len(profile.keywords) == 2

    def test_returns_profile_instance(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert isinstance(profile, Profile)


# ── Shipped profiles parse cleanly ───────────────────────────────────────

class TestShippedProfiles:
    def test_hpn_yaml_parses(self):
        """profiles/hpn.yaml must load without errors (regression guard)."""
        profile = load_profile(PROFILES_DIR / "hpn.yaml")
        assert profile.name == "High Performance Networking"
        assert len(profile.journals) >= 1
        assert len(profile.keywords) >= 10

    def test_hpn_yaml_subfield_filter_is_none(self):
        """subfield_filter is currently commented out in hpn.yaml."""
        profile = load_profile(PROFILES_DIR / "hpn.yaml")
        assert profile.subfield_filter is None

    def test_hpn_yaml_year_range_valid(self):
        profile = load_profile(PROFILES_DIR / "hpn.yaml")
        assert profile.year_min <= profile.year_max

    def test_template_yaml_parses(self):
        """profiles/template.yaml must parse without errors (regression guard).
        It has no keywords/journals, so we expect a ValueError — that's fine.
        The point is it must not fail on YAML syntax."""
        try:
            load_profile(PROFILES_DIR / "template.yaml")
        except ValueError:
            pass  # Expected: template has no keywords/journals filled in
        except Exception as exc:
            pytest.fail(f"template.yaml raised unexpected error: {exc}")

    def test_hpn_yaml_llm_validation_is_configured(self):
        """hpn.yaml ships with llm_validation enabled → Profile.llm_validation is set."""
        profile = load_profile(PROFILES_DIR / "hpn.yaml")
        assert profile.llm_validation is not None
        assert profile.llm_validation.backend == "ollama"
        assert profile.llm_validation.model == "qwen3.5:9b"

    def test_hpn_yaml_validation_cache_path_set(self):
        profile = load_profile(PROFILES_DIR / "hpn.yaml")
        assert profile.validation_cache_path == profile.downloads_dir / "validation_cache.json"


# ── LLM config loading ────────────────────────────────────────────────────

class TestLLMConfig:
    def test_absent_llm_section_gives_none(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.llm_validation is None

    def test_enabled_false_gives_none(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            llm_validation:
              enabled: false
              backend: ollama
              model: llama3
              base_url: "http://localhost:11434"
        """))
        profile = load_profile(p)
        assert profile.llm_validation is None

    def test_valid_ollama_config_loaded(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            llm_validation:
              enabled: true
              backend: ollama
              model: "llama3"
              base_url: "http://localhost:11434"
              api_key: ""
              temperature: 0
              timeout: 45
              fallback_on_no_abstract: "download"
              skip_journal_papers: true
        """))
        profile = load_profile(p)
        cfg = profile.llm_validation
        assert cfg is not None
        assert cfg.backend == "ollama"
        assert cfg.model == "llama3"
        assert cfg.timeout == 45
        assert cfg.fallback_on_no_abstract == "download"
        assert cfg.skip_journal_papers is True

    def test_valid_openai_config_loaded(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            llm_validation:
              enabled: true
              backend: openai
              model: "gpt-4o-mini"
              base_url: "https://api.openai.com"
              api_key: "sk-test"
              temperature: 0.0
              timeout: 60
              fallback_on_no_abstract: "skip"
              skip_journal_papers: false
        """))
        profile = load_profile(p)
        cfg = profile.llm_validation
        assert cfg is not None
        assert cfg.backend == "openai"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.api_key == "sk-test"
        assert cfg.fallback_on_no_abstract == "skip"
        assert cfg.skip_journal_papers is False

    def test_invalid_backend_raises(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            llm_validation:
              enabled: true
              backend: "llamacpp"
              model: llama3
        """))
        with pytest.raises(ValueError, match="backend"):
            load_profile(p)

    def test_invalid_fallback_raises(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            llm_validation:
              enabled: true
              backend: ollama
              model: llama3
              fallback_on_no_abstract: "ignore"
        """))
        with pytest.raises(ValueError, match="fallback_on_no_abstract"):
            load_profile(p)

    def test_validation_cache_path_inside_downloads(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.validation_cache_path == profile.downloads_dir / "validation_cache.json"

    def test_llm_defaults_applied(self, tmp_path):
        """Fields omitted from yaml get sensible defaults."""
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            llm_validation:
              enabled: true
              backend: ollama
        """))
        profile = load_profile(p)
        cfg = profile.llm_validation
        assert cfg.model == "llama3"
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.api_key == ""
        assert cfg.temperature == 0.0
        assert cfg.timeout == 30
        assert cfg.fallback_on_no_abstract == "download"
        assert cfg.skip_journal_papers is True


# ── S2 config loading ─────────────────────────────────────────────────────

class TestS2Config:
    def test_absent_s2_section_gives_none(self, tmp_path):
        p = write_profile(tmp_path, MINIMAL_VALID)
        profile = load_profile(p)
        assert profile.s2_config is None

    def test_enabled_false_gives_none(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            semantic_scholar:
              enabled: false
              keywords: ["test"]
        """))
        profile = load_profile(p)
        assert profile.s2_config is None

    def test_enabled_true_loaded(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            semantic_scholar:
              enabled: true
              api_key: "my-key"
              fields_of_study: ["Computer Science"]
              max_per_query: 200
              keywords:
                - "RDMA datacenter"
                - "SmartNIC offload"
        """))
        profile = load_profile(p)
        cfg = profile.s2_config
        assert cfg is not None
        assert cfg.api_key == "my-key"
        assert cfg.fields_of_study == ["Computer Science"]
        assert cfg.max_per_query == 200
        assert len(cfg.keywords) == 2

    def test_defaults_applied(self, tmp_path):
        """Omitted fields get sensible defaults."""
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            semantic_scholar:
              enabled: true
              keywords: ["test query"]
        """))
        profile = load_profile(p)
        cfg = profile.s2_config
        assert cfg is not None
        assert cfg.api_key == ""
        assert cfg.fields_of_study == ["Computer Science"]
        assert cfg.max_per_query == 100

    def test_empty_keywords_raises(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            semantic_scholar:
              enabled: true
              keywords: []
        """))
        with pytest.raises(ValueError, match="keywords"):
            load_profile(p)

    def test_missing_keywords_raises(self, tmp_path):
        p = write_profile(tmp_path, textwrap.dedent("""\
            name: "Domain"
            mailto: "test@example.com"
            keywords: ["test"]
            semantic_scholar:
              enabled: true
        """))
        with pytest.raises(ValueError, match="keywords"):
            load_profile(p)

    def test_hpn_yaml_s2_config_loaded(self):
        """hpn.yaml ships with semantic_scholar enabled."""
        profile = load_profile(PROFILES_DIR / "hpn.yaml")
        assert profile.s2_config is not None
        assert len(profile.s2_config.keywords) >= 5
        assert "Computer Science" in profile.s2_config.fields_of_study
