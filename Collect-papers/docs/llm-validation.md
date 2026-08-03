# LLM Abstract Validation

When enabled, each candidate paper's **title + abstract** is sent to an LLM before the PDF is downloaded.  
Only papers the LLM classifies as relevant to the configured `domain_description` are downloaded.

---

## Enable in your profile

```yaml
llm_validation:
  enabled: true
  backend: ollama            # "ollama" (local) | "openai" (OpenAI-compatible API)
  model: "llama3"
  base_url: "http://localhost:11434"
  api_key: ""                # leave empty for Ollama
  temperature: 0             # 0 = fully deterministic
  timeout: 30                # seconds per LLM call
  fallback_on_no_abstract: "download"   # "download" (safe) | "skip"
  skip_journal_papers: true             # bypass LLM for journal-sourced papers
```

---

## Backends

### Ollama (local, free, private)

```bash
ollama serve               # start the server (keep running)
ollama pull llama3         # download model once (~4 GB)
```

```yaml
llm_validation:
  enabled: true
  backend: ollama
  model: "llama3"
  base_url: "http://localhost:11434"
  api_key: ""
```

### OpenAI-compatible APIs

Works with OpenAI, Together AI, LM Studio, and any OpenAI-compatible endpoint:

```yaml
llm_validation:
  enabled: true
  backend: openai
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com"
  api_key: "sk-..."
```

---

## How it works

1. After all dedup checks pass, the abstract is reconstructed from OpenAlex's `abstract_inverted_index`.
2. A structured prompt is sent to the LLM:  
   *"Is this paper relevant to `<domain_description>`?"*
3. The LLM must respond with `RELEVANT: <reason>` or `NOT_RELEVANT: <reason>`.
4. The decision is cached in `downloads/<slug>/validation_cache.json` — keyed by  
   `{openalex_id}::{model}::{profile_name}` so cache entries are automatically invalidated when the model or profile changes.
5. If the response cannot be parsed, the paper is **allowed through** — the system never silently discards papers on a parsing failure.

---

## Edge case behaviour

| Situation | Behaviour |
|---|---|
| No abstract available | Apply `fallback_on_no_abstract` policy — no LLM call made |
| `skip_journal_papers: true` + journal paper | Allowed through without LLM call |
| LLM response unparseable | Allowed through (`source: parse_error_allow`) |
| LLM unreachable | Raises `LLMConnectionError` and exits — does not fall back to downloading unvalidated papers |

---

## Caching

Responses are stored in `downloads/<slug>/validation_cache.json` and reused on subsequent runs.  
A cache hit avoids an LLM call entirely — useful when re-running after partial failures.

To force a fresh LLM evaluation, delete `validation_cache.json` or change the model name in the profile.
