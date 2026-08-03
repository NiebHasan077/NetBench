"""
llm_validator.py — LLM-based abstract relevance validator for collect_papers.

Validates whether a paper's title + abstract are relevant to the configured
domain by calling an LLM before the PDF is downloaded.

Supports two backends:
  ollama — local Ollama server (free, private, no rate limits)
  openai — any OpenAI-compatible API endpoint (OpenAI, Together, LM Studio…)

Both backends share the same prompt template and deterministic response
parsing.  The prompt enforces a strict single-line format so parsing never
relies on fragile JSON extraction from free-form LLM output.

Validation cache
----------------
Decisions are cached in {downloads_dir}/validation_cache.json, keyed by
  "{openalex_id}::{model}::{profile_name}"
The composite key prevents stale hits if the model or profile changes.
The cache is loaded once at init and flushed every CACHE_FLUSH_INTERVAL
validations plus on close(), so an interrupted run does not lose results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import requests

if TYPE_CHECKING:
    from profile import LLMConfig

# Flush the cache to disk after this many new validations
CACHE_FLUSH_INTERVAL = 10

# ── Prompt template ───────────────────────────────────────────────────────
# The strict RELEVANT / NOT_RELEVANT prefix format is enforced so response
# parsing uses only a simple startswith() check — no regex, no JSON parsing.

_SYSTEM_TEMPLATE = """\
You are a research paper relevance classifier.
Domain: {domain_description}"""

_USER_TEMPLATE = """\
Title: {title}
Abstract: {abstract}

Is this paper relevant to the domain above?
Reply with exactly one line in one of these two formats:
  RELEVANT: <one sentence reason>
  NOT_RELEVANT: <one sentence reason>"""


# ── Exceptions ────────────────────────────────────────────────────────────

class LLMConnectionError(RuntimeError):
    """Raised when the LLM backend cannot be reached or times out."""


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    relevant: bool
    reason: str
    # source values:
    #   "llm"               — LLM made the decision
    #   "no_abstract_skipped" — abstract was absent; fallback policy applied
    #   "skip_journal"      — journal paper, validation bypassed per config
    #   "parse_error_allow" — LLM response unparseable; allowed through safely
    source: str
    cached: bool


# ── Validator ─────────────────────────────────────────────────────────────

class LLMValidator:
    """Validates paper relevance against a domain using an LLM.

    Parameters
    ----------
    config:
        LLMConfig from the loaded profile.
    profile_name:
        Used as part of the cache key to prevent cross-profile stale hits.
    domain_description:
        Plain-English description of the research domain.  Passed verbatim
        into the system prompt.
    cache_path:
        Path to the validation_cache.json file.

    Contract for edge cases
    -----------------------
    - No abstract: applies config.fallback_on_no_abstract ("download"/"skip")
      without calling the LLM.
    - LLM response unparseable: returns relevant=True with
      source="parse_error_allow".  Never discards a paper silently due to an
      LLM formatting failure.
    - LLM unreachable: raises LLMConnectionError immediately.  The pipeline
      does NOT fall back to downloading without validation, as that would
      defeat the purpose of the feature.
    """

    def __init__(
        self,
        config: "LLMConfig",
        profile_name: str,
        domain_description: str,
        cache_path: Path,
    ) -> None:
        self.config = config
        self.profile_name = profile_name
        self.domain_description = domain_description
        self.cache_path = cache_path
        self._cache: dict = {}
        self._validations_since_flush: int = 0

        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    # ── Public API ────────────────────────────────────────────────────

    def validate(
        self,
        openalex_id: str,
        title: str,
        abstract: Optional[str],
        is_journal_paper: bool = False,
    ) -> ValidationResult:
        """Classify a paper as relevant or not relevant to the domain.

        Parameters
        ----------
        openalex_id:
            OpenAlex work ID used as cache key component.
        title:
            Paper title.
        abstract:
            Reconstructed abstract text, or None if unavailable.
        is_journal_paper:
            If True and config.skip_journal_papers is True, returns
            relevant=True immediately without calling the LLM.

        Returns
        -------
        ValidationResult
            Always returns a result; never raises for LLM parse failures.

        Raises
        ------
        LLMConnectionError
            If the backend cannot be reached (connection refused or timeout).
        """
        # Fast path: journal papers bypass validation when configured
        if is_journal_paper and self.config.skip_journal_papers:
            return ValidationResult(
                relevant=True,
                reason="journal paper — validation skipped per config",
                source="skip_journal",
                cached=False,
            )

        # Fast path: no abstract → apply fallback policy without LLM call
        if abstract is None or abstract.strip() == "":
            relevant = self.config.fallback_on_no_abstract == "download"
            return ValidationResult(
                relevant=relevant,
                reason="no abstract available — fallback policy applied",
                source="no_abstract_skipped",
                cached=False,
            )

        # Cache lookup
        key = self._cache_key(openalex_id)
        if key in self._cache:
            entry = self._cache[key]
            return ValidationResult(
                relevant=entry["relevant"],
                reason=entry["reason"],
                source="llm",
                cached=True,
            )

        # Call LLM and parse response
        raw_response = self._call_llm(title, abstract)
        result = self._parse_response(raw_response)

        # Store in cache and periodically flush to disk
        self._cache[key] = {"relevant": result.relevant, "reason": result.reason}
        self._validations_since_flush += 1
        if self._validations_since_flush >= CACHE_FLUSH_INTERVAL:
            self._flush_cache()
            self._validations_since_flush = 0

        return result

    def close(self) -> None:
        """Flush the validation cache to disk. Call after a collection run."""
        if self._cache:
            self._flush_cache()

    # ── Cache helpers ─────────────────────────────────────────────────

    def _cache_key(self, openalex_id: str) -> str:
        return f"{openalex_id}::{self.config.model}::{self.profile_name}"

    def _flush_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── LLM backends ──────────────────────────────────────────────────

    def _call_llm(self, title: str, abstract: str) -> str:
        """Call the configured backend and return the raw text response."""
        system = _SYSTEM_TEMPLATE.format(domain_description=self.domain_description)
        user = _USER_TEMPLATE.format(title=title, abstract=abstract)

        if self.config.backend == "ollama":
            return self._call_ollama(system, user)
        elif self.config.backend == "openai":
            return self._call_openai(system, user)
        else:
            raise ValueError(f"Unknown LLM backend: {self.config.backend!r}")

    def _call_ollama(self, system: str, user: str) -> str:
        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": self.config.temperature},
        }
        try:
            r = requests.post(url, json=payload, timeout=self.config.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Ollama not reachable at {self.config.base_url} — "
                f"is Ollama running?  (run: ollama serve)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"Ollama timed out after {self.config.timeout}s "
                f"at {self.config.base_url}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(
                f"Ollama request failed: {exc}"
            ) from exc

    def _call_openai(self, system: str, user: str) -> str:
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        headers: dict = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
        }
        try:
            r = requests.post(
                url, json=payload, headers=headers, timeout=self.config.timeout
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"OpenAI-compatible endpoint not reachable at {self.config.base_url}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"OpenAI endpoint timed out after {self.config.timeout}s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(
                f"OpenAI request failed: {exc}"
            ) from exc

    # ── Response parsing ──────────────────────────────────────────────

    def _parse_response(self, text: str) -> ValidationResult:
        """Parse the LLM's single-line response into a ValidationResult.

        Expected formats (case-insensitive prefix match):
          RELEVANT: <one-sentence reason>
          NOT_RELEVANT: <one-sentence reason>

        If neither prefix is found, returns relevant=True with
        source="parse_error_allow".  This ensures an LLM formatting glitch
        never silently causes a paper to be discarded.
        """
        stripped = text.strip()
        upper = stripped.upper()

        if upper.startswith("NOT_RELEVANT"):
            reason = stripped[len("NOT_RELEVANT"):].lstrip(": ").strip()
            return ValidationResult(
                relevant=False,
                reason=reason or "not relevant to domain",
                source="llm",
                cached=False,
            )

        if upper.startswith("RELEVANT"):
            reason = stripped[len("RELEVANT"):].lstrip(": ").strip()
            return ValidationResult(
                relevant=True,
                reason=reason or "relevant to domain",
                source="llm",
                cached=False,
            )

        # Unparseable response — allow through safely
        return ValidationResult(
            relevant=True,
            reason=f"parse error — LLM response: {stripped[:100]!r}",
            source="parse_error_allow",
            cached=False,
        )
