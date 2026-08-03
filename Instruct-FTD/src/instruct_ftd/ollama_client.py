"""Minimal local Ollama client for Phase 5 pilot generation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaClient:
    """Tiny client for the local Ollama `/api/chat` endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_sec: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_predict: int = 500,
        think: bool = False,
    ) -> dict:
        """Call the local Ollama chat API and return the decoded JSON response."""

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "think": think,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach Ollama at {self.base_url}: {exc}") from exc
