"""Sync Ollama client with retry and tolerant JSON extraction.

Calls Ollama's HTTP API directly via `requests` so we don't depend on the
ollama Python package. JSON-mode responses from local models often arrive
wrapped in markdown fences or with trailing commentary; `_extract_json`
peels those layers off before parsing.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional, Union

import requests

JSONValue = Union[dict[str, Any], list[Any]]


class OllamaError(RuntimeError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> JSONValue:
    """Tolerantly extract a JSON object/array from a model response."""
    if not text or not text.strip():
        raise ValueError("empty response")
    s = text.strip()

    # Strip markdown fence if present.
    fence = _FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()

    # Try direct parse first.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fall back: find the outermost {...} or [...].
    obj_start = _find_outermost(s, "{", "}")
    arr_start = _find_outermost(s, "[", "]")
    candidates: list[tuple[int, int]] = []
    if obj_start is not None:
        candidates.append(obj_start)
    if arr_start is not None:
        candidates.append(arr_start)
    if not candidates:
        raise ValueError(f"no JSON object found in response: {s[:200]!r}")
    # Take whichever brace appears first.
    start, end = min(candidates, key=lambda se: se[0])
    return json.loads(s[start:end + 1])


def _find_outermost(s: str, open_ch: str, close_ch: str) -> Optional[tuple[int, int]]:
    """Return (start, end) indices of the outermost balanced pair, or None."""
    start = s.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return start, i
    return None


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 600,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"could not list models: {e}") from e
        return [m["name"] for m in r.json().get("models", [])]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        num_predict: int = 2048,
        format: Optional[Union[str, dict[str, Any]]] = None,
        repeat_penalty: Optional[float] = None,
        think: Optional[bool] = None,
    ) -> str:
        """Call Ollama's /api/chat with a list of role/content messages.

        `think=False` disables hidden chain-of-thought for thinking models like
        qwen3 family — required to get cooperative structured-JSON output via /api/chat.
        Non-thinking models silently ignore the flag.
        """
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict,
        }
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think

        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"chat failed for {model}: {e}") from e

        data = r.json()
        return data.get("message", {}).get("content", "")

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        retry_temperature: float = 0.2,
        max_retries: int = 1,
        num_predict: int = 4096,
        schema: Optional[dict[str, Any]] = None,
        repeat_penalty: Optional[float] = 1.15,
        think: Optional[bool] = False,
    ) -> JSONValue:
        """Chat-mode equivalent of generate_json. Schema-enforced parsing skips the
        tolerant fallback (truncated JSON should fail loudly and retry, not silently
        return an inner array).
        """
        last_err: Optional[Exception] = None
        fmt: Union[str, dict[str, Any]] = schema if schema is not None else "json"
        for attempt in range(max_retries + 1):
            t = temperature if attempt == 0 else retry_temperature
            raw = self.chat(
                model=model,
                messages=messages,
                temperature=t,
                num_predict=num_predict,
                format=fmt,
                repeat_penalty=repeat_penalty,
                think=think,
            )
            try:
                if schema is not None:
                    return json.loads(raw)
                return _extract_json(raw)
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
        raise OllamaError(
            f"could not parse chat JSON after {max_retries + 1} attempt(s): {last_err}"
        )

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        num_predict: int = 2048,
        format: Optional[Union[str, dict[str, Any]]] = None,
        repeat_penalty: Optional[float] = None,
    ) -> str:
        """Call Ollama's /api/generate. `format` may be the literal "json" or a JSON schema dict."""
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict,
        }
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if format is not None:
            payload["format"] = format

        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"generate failed for {model}: {e}") from e

        data = r.json()
        return data.get("response", "")

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        retry_temperature: float = 0.2,
        max_retries: int = 1,
        num_predict: int = 4096,
        schema: Optional[dict[str, Any]] = None,
        repeat_penalty: Optional[float] = 1.15,
    ) -> JSONValue:
        """Generate, then extract+parse JSON tolerantly. One retry on parse failure.

        If `schema` is provided, Ollama enforces it server-side (structured output) and
        the response is guaranteed valid JSON matching the schema. Otherwise we ask for
        plain JSON and tolerate light wrapping (markdown fences, preamble text).
        """
        last_err: Optional[Exception] = None
        fmt: Union[str, dict[str, Any]] = schema if schema is not None else "json"
        for attempt in range(max_retries + 1):
            t = temperature if attempt == 0 else retry_temperature
            raw = self.generate(
                model=model,
                prompt=prompt,
                system=system,
                temperature=t,
                num_predict=num_predict,
                format=fmt,
                repeat_penalty=repeat_penalty,
            )
            try:
                if schema is not None:
                    # With schema enforcement, the response must be a complete JSON
                    # value matching the schema. Don't use the tolerant fallback —
                    # it can return an inner array if the outermost object is
                    # truncated mid-generation.
                    return json.loads(raw)
                return _extract_json(raw)
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
        raise OllamaError(
            f"could not parse JSON after {max_retries + 1} attempt(s): {last_err}"
        )
