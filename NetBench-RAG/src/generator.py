"""
Generation module for the NetBench-RAG pipeline.

Two backends, selected via config.yaml ``generation.backend``:

  openai  — GPT-4o via the OpenAI API (default)
  local   — Any HuggingFace instruction model (Llama or Qwen family)

Both receive the same system prompt and numbered-excerpt user message.
The Generator class is backend-agnostic; callers pass retrieved Chunk objects
and get back a GenerationResult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .chunker import Chunk


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationResult:
    """Output from one generation call."""
    answer: str                        # LLM response text
    backend: str                       # "openai" | "local"
    model: str                         # e.g. "gpt-4o" or "Llama-3.1-8B-trained-new-instruct"
    context_chunks: list[Chunk]        # chunks that were passed as context
    prompt_tokens: int = 0             # input token count (diagnostic)


# ═══════════════════════════════════════════════════════════════════════════════
# Context + prompt formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_context(chunks: list[Chunk]) -> str:
    """
    Build the numbered-excerpt context block passed to the LLM.

    Format per excerpt:
        [N] Paper: '<title>' | Section: <num> <heading>
        <chunk body text>
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.paper_title
        if len(title) > 80:
            title = title[:77] + "…"

        if chunk.section_number:
            section = f"{chunk.section_number} {chunk.section_heading}".strip()
        else:
            section = chunk.section_heading or "Body"

        header = f"[{i}] Paper: '{title}' | Section: {section}"
        parts.append(f"{header}\n{chunk.text}")

    return "\n\n".join(parts)


def format_user_message(query: str, context: str) -> str:
    """Combine the numbered excerpts and the question into a single user turn."""
    return f"Excerpts:\n\n{context}\n\nQuestion: {query}"


# ═══════════════════════════════════════════════════════════════════════════════
# Backend protocol
# ═══════════════════════════════════════════════════════════════════════════════

class _Backend(Protocol):
    """Minimal interface both backends must satisfy."""
    model: str

    def generate(self, system_prompt: str, user_message: str) -> tuple[str, int]:
        """Return (answer_text, prompt_token_count)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI backend
# ═══════════════════════════════════════════════════════════════════════════════

class _OpenAIBackend:
    """
    Calls GPT-4o (or any OpenAI chat model) via the openai SDK.

    The API key is read from a plain-text file (path from config).
    """

    def __init__(
        self,
        api_key_path: str,
        model: str = "gpt-4o",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        from openai import OpenAI

        key_file = Path(api_key_path)
        if not key_file.exists():
            raise FileNotFoundError(
                f"OpenAI API key file not found: {key_file}\n"
                "Create it with: echo 'sk-...' > openai_api_key.txt"
            )
        api_key = key_file.read_text().strip()
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, system_prompt: str, user_message: str) -> tuple[str, int]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        answer = (response.choices[0].message.content or "").strip()
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        return answer, prompt_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Local HuggingFace backend
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_model_family(model_path: str) -> str:
    """Return 'qwen', 'gemma', or 'llama' based on the model path."""
    path_lower = model_path.lower()
    if "qwen" in path_lower:
        return "qwen"
    if "gemma" in path_lower:
        return "gemma"
    return "llama"


# ── Per-family prompt templates ───────────────────────────────────────────

# Open-Orca template — used for Llama models fine-tuned on Orca format
_ORCA_TEMPLATE = (
    "### System:\n{system}\n\n"
    "### User:\n{instruction}\n\n"
    "### Assistant:\n"
)

# ChatML template — used for Qwen models
_CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Gemma template — fallback for older Gemma checkpoints
_GEMMA_TEMPLATE = (
    "<bos><start_of_turn>system\n{system}<end_of_turn>\n"
    "<start_of_turn>user\n{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# Stop strings per model family
_STOP_STRINGS: dict[str, list[str]] = {
    "llama": ["### User:", "### System:", "\n### "],
    "qwen":  ["<|im_start|>", "<|im_end|>"],
    "gemma": ["<end_of_turn>", "<start_of_turn>"],
}

_GEMMA4_STOP_STRINGS = ["<turn|>", "<|tool_response>", "<|tool_call>"]


def _extract_orca_answer(raw: str) -> str:
    """
    Extract the actual answer from raw Orca-format generation output.

    Some fine-tuned models echo a partial conversation (system prompt +
    user turn) before producing the real answer after a fresh
    '### Assistant:' marker.  We therefore look for the LAST occurrence
    of '### Assistant:' in the generated text and return what follows it.

    Fallback patterns handle simpler single-turn outputs:

    Pattern B (rare):
        "{echo}\n\n### System:\n{answer}\n\n### User: ..."
        → extract the content of the first ### System: block after the echo

    Pattern A (normal single-turn):
        "{answer text}\n\n### User: ..."
        → take everything before the first turn marker

    In all cases, trailing turn-marker noise is stripped.
    """
    import re

    # Primary: take text after the LAST ### Assistant: marker.
    # This handles models that echo the conversation before answering.
    last_assistant = raw.rfind("### Assistant:")
    if last_assistant != -1:
        candidate = raw[last_assistant + len("### Assistant:"):].strip()
        if candidate:
            for marker in ("\n### User:", "\n### System:", "\n### Assistant:"):
                idx = candidate.find(marker)
                if idx != -1:
                    candidate = candidate[:idx].strip()
            return candidate

    # Pattern B: model wraps the real answer inside a ### System: block
    sys_m = re.search(
        r"\n### System:\n(.*?)(?=\n### (?:User|Assistant|System):|\Z)",
        raw,
        re.DOTALL,
    )
    if sys_m:
        answer = sys_m.group(1).strip()
        # Strip any trailing "Question: ..." the model appended inside the block
        q_idx = re.search(r"\n\nQuestion:", answer)
        if q_idx:
            answer = answer[:q_idx.start()].strip()
        return answer

    # Pattern A: direct answer, possibly followed by a new turn
    for marker in ("\n### User:", "\n### System:", "\n### Assistant:"):
        idx = raw.find(marker)
        if idx != -1:
            return raw[:idx].strip()

    return raw.strip()


def _extract_chatml_answer(raw: str) -> str:
    """Extract the answer from ChatML-format generation output.

    Looks for the last ``<|im_start|>assistant\\n`` marker and returns
    everything after it, stripping trailing ChatML tokens.
    """
    marker = "<|im_start|>assistant\n"
    if marker in raw:
        raw = raw.split(marker)[-1]
    for tag in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        raw = raw.split(tag)[0]
    # Strip Qwen3.5 thinking blocks if present
    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>")[-1]
    return raw.strip()


def _extract_gemma_answer(raw: str) -> str:
    """Extract the answer from Gemma-format generation output.

    Supports both:
      - Gemma 3 style ``<start_of_turn>model``
      - Gemma 4 style ``<|turn>model`` with optional channel/thinking blocks
    """
    gemma4_marker = "<|turn>model\n"
    if gemma4_marker in raw:
        raw = raw.split(gemma4_marker)[-1]
        raw = re.sub(r"<\|channel\>.*?<channel\|>", "", raw, flags=re.DOTALL)
        raw = raw.replace("<|think|>", "")
        for tag in ("<turn|>", "<|tool_response>", "<|tool_call>", "<eos>", "<bos>"):
            raw = raw.split(tag)[0]
        return raw.strip()

    marker = "<start_of_turn>model\n"
    if marker in raw:
        raw = raw.split(marker)[-1]
    for tag in ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>"):
        raw = raw.split(tag)[0]
    return raw.strip()


class _LocalBackend:
    """
    Runs inference against a local instruction-tuned model.

    Supports two model families:
      - **Llama** — Open-Orca template (``### System: / ### User: / ### Assistant:``)
      - **Qwen**  — ChatML template (``<|im_start|>system / user / assistant``)

    The family is auto-detected from the model path.
    Model is loaded once and kept in memory; generation is in inference_mode.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = Path(model_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Local model not found: {resolved}\n"
                "Check paths.local_model in config.yaml."
            )

        self._model_family = _detect_model_family(str(resolved))
        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(str(resolved))
        # Qwen tokenizers ship without a dedicated pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            str(resolved),
            dtype=torch_dtype,
            device_map=device_map,
        )
        self._model.eval()

        self._model_type = getattr(self._model.config, "model_type", "")
        self._uses_native_chat_template = (
            self._model_type.startswith("gemma4")
            and bool(getattr(self._tokenizer, "chat_template", None))
        )
        self.model = resolved.name   # e.g. "Qwen3.5-2B-Base-instruct"
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample
        self._repetition_penalty = repetition_penalty
        self._normalise_generation_config()

        print(f"   Model family : {self._model_family}")
        print(f"   Model type   : {self._model_type or 'unknown'}")
        print(f"   Pad token    : {self._tokenizer.pad_token!r}")

    def _normalise_generation_config(self) -> None:
        """
        Align the model's default generation_config with this runtime.

        Some local checkpoints ship sampling defaults (e.g. top_p/top_k) in
        generation_config.json. When NetBench-RAG forces greedy decoding with
        ``do_sample=False``, Transformers warns on every call unless those
        sampling-only fields are cleared from the model defaults first.
        """
        gc = self._model.generation_config
        gc.do_sample = self._do_sample
        gc.max_new_tokens = self._max_new_tokens
        gc.repetition_penalty = self._repetition_penalty

        if self._do_sample:
            gc.temperature = self._temperature
            return

        for attr in (
            "top_k",
            "top_p",
            "min_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "penalty_alpha",
        ):
            if hasattr(gc, attr):
                setattr(gc, attr, None)
        if hasattr(gc, "temperature"):
            gc.temperature = None

    def _build_prompt(self, system_prompt: str, user_message: str) -> str:
        """Build the prompt string using the family-appropriate template."""
        if self._model_family == "qwen":
            return _CHATML_TEMPLATE.format(system=system_prompt, instruction=user_message)
        if self._model_family == "gemma":
            if self._uses_native_chat_template:
                return self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return _GEMMA_TEMPLATE.format(system=system_prompt, instruction=user_message)
        return _ORCA_TEMPLATE.format(system=system_prompt, instruction=user_message)

    def _resolve_eos_ids(self) -> list[int]:
        """Return all EOS / end-of-turn token IDs for the model family."""
        eos_ids: list[int] = []
        vocab = self._tokenizer.get_vocab()

        def _add_eos_id(token_id: int | None) -> None:
            if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
                eos_ids.append(token_id)

        config_eos = getattr(self._model.config, "eos_token_id", None)
        if isinstance(config_eos, (list, tuple)):
            for token_id in config_eos:
                _add_eos_id(token_id)
        else:
            _add_eos_id(config_eos)

        _add_eos_id(self._tokenizer.eos_token_id)

        if self._model_family == "qwen":
            # ChatML end-of-turn marker
            for candidate in ("<|im_end|>", "<|endoftext|>"):
                if candidate in vocab:
                    _add_eos_id(vocab[candidate])
        elif self._model_family == "gemma":
            candidates = ("<turn|>", "<end_of_turn>", "<eos>") if self._uses_native_chat_template else ("<end_of_turn>", "<eos>")
            for candidate in candidates:
                if candidate in vocab:
                    _add_eos_id(vocab[candidate])
        else:
            # Llama end-of-turn marker
            eot_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
            _add_eos_id(eot_id)

        return eos_ids

    def _extract_answer(self, decoded: str) -> str:
        """Extract the assistant answer from raw decoded output."""
        if self._model_family == "qwen":
            return _extract_chatml_answer(decoded)
        if self._model_family == "gemma":
            return _extract_gemma_answer(decoded)

        # Llama / Orca extraction
        if "### Assistant:" in decoded:
            decoded = decoded.split("### Assistant:")[-1].strip()
        stop_strings = _STOP_STRINGS["llama"]
        for ss in stop_strings:
            if ss in decoded:
                decoded = decoded.split(ss)[0].strip()
        return _extract_orca_answer(decoded) if "### " in decoded else decoded

    def generate(self, system_prompt: str, user_message: str) -> tuple[str, int]:
        import torch

        prompt_str = self._build_prompt(system_prompt, user_message)
        if self._uses_native_chat_template:
            stop_strings = _GEMMA4_STOP_STRINGS
        else:
            stop_strings = _STOP_STRINGS.get(self._model_family, _STOP_STRINGS["llama"])

        # Qwen supports 262144 context; Gemma-3 supports 131072; Llama 3.x supports 8192
        max_length = 131072 if self._model_family in ("qwen", "gemma") else 8192

        inputs = self._tokenizer(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids      = inputs.input_ids.to(self._model.device)
        attention_mask = inputs.attention_mask.to(self._model.device)

        eos_ids = self._resolve_eos_ids()

        gen_kwargs: dict = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample":      self._do_sample,
            "pad_token_id":   self._tokenizer.pad_token_id,
            "eos_token_id":   eos_ids,
            "attention_mask": attention_mask,
            "repetition_penalty": self._repetition_penalty,
            "stop_strings":   stop_strings,
            "tokenizer":      self._tokenizer,
        }
        if self._do_sample:
            gen_kwargs["temperature"] = self._temperature

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        # Decode and extract the assistant answer using family-specific logic
        # skip_special_tokens=False so we can reliably find ChatML markers
        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=False)
        answer = self._extract_answer(decoded)
        return answer, int(input_ids.shape[1])


# ═══════════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════════

class Generator:
    """
    Backend-agnostic generation wrapper.

    Handles context formatting and delegates actual LLM calls to the
    configured backend.

    Parameters
    ----------
    backend       : _OpenAIBackend | _LocalBackend
    system_prompt : shared system prompt (from config.yaml)
    backend_name  : "openai" | "local" (for result metadata)
    """

    def __init__(
        self,
        backend: _OpenAIBackend | _LocalBackend,
        system_prompt: str,
        backend_name: str,
    ) -> None:
        self._backend = backend
        self._system_prompt = system_prompt.strip()
        self.backend_name = backend_name

    def generate(self, query: str, chunks: list[Chunk]) -> GenerationResult:
        """
        Generate an answer for *query* using *chunks* as context.

        Parameters
        ----------
        query  : the user's question
        chunks : retrieved and reranked Chunk objects (ordered best-first)

        Returns
        -------
        GenerationResult with the answer and metadata
        """
        context = format_context(chunks)
        user_message = format_user_message(query, context)
        answer, prompt_tokens = self._backend.generate(self._system_prompt, user_message)

        return GenerationResult(
            answer=answer,
            backend=self.backend_name,
            model=self._backend.model,
            context_chunks=chunks,
            prompt_tokens=prompt_tokens,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def generator_from_config(cfg: dict, backend_override: str | None = None) -> Generator:
    """
    Build a Generator from config.yaml.

    Parameters
    ----------
    cfg              : full parsed config dict
    backend_override : "openai" | "local" — overrides config if supplied
                       (used by query.py --backend flag)
    """
    gen_cfg = cfg.get("generation", {})
    paths = cfg.get("paths", {})

    backend_name = backend_override or gen_cfg.get("backend", "openai")
    system_prompt = gen_cfg.get("system_prompt", "You are a helpful assistant.")

    if backend_name == "openai":
        oa = gen_cfg.get("openai", {})
        backend: _OpenAIBackend | _LocalBackend = _OpenAIBackend(
            api_key_path=paths.get("openai_api_key", "openai_api_key.txt"),
            model=oa.get("model", "gpt-4o"),
            max_tokens=oa.get("max_tokens", 512),
            temperature=oa.get("temperature", 0.3),
        )
    elif backend_name == "local":
        loc = gen_cfg.get("local", {})
        backend = _LocalBackend(
            model_path=paths.get("local_model", "../LLM-Training/models/instruction/Llama-3.1-8B-trained-new-instruct"),
            max_new_tokens=loc.get("max_new_tokens", 512),
            temperature=loc.get("temperature", 0.3),
            do_sample=loc.get("do_sample", False),
            repetition_penalty=loc.get("repetition_penalty", 1.1),
            dtype=loc.get("dtype", "bfloat16"),
            device_map=loc.get("device_map", "auto"),
        )
    else:
        raise ValueError(
            f"Unknown backend {backend_name!r}. Choose 'openai' or 'local'."
        )

    return Generator(
        backend=backend,
        system_prompt=system_prompt,
        backend_name=backend_name,
    )
