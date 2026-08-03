#!/usr/bin/env python3
"""
Model-family utilities shared across training and data-preparation scripts.

Provides:
  - Model-family detection (Qwen vs Llama vs Gemma) from a path or HF model ID
  - Stop-token / EOS-token resolution per family
  - LoRA target-module lists per family
  - Tokenizer padding fixes
"""

from __future__ import annotations
from typing import List


# ── LoRA target modules ────────────────────────────────────────────────────────
# Qwen3.5 and LLaMA 3.x share identical projection layer names, so a single list
# covers both.  Defined separately in case they diverge in future model releases.
_LLAMA_TARGET_MODULES: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

_QWEN_TARGET_MODULES: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

_GEMMA_TARGET_MODULES: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]


def detect_model_family(model_path_or_name: str) -> str:
    """
    Return 'qwen', 'gemma', or 'llama' based on the model path or HuggingFace model ID.

    >>> detect_model_family("Qwen/Qwen3.5-4B-Base")
    'qwen'
    >>> detect_model_family("models/base/Llama-3.1-8B-base")
    'llama'
    >>> detect_model_family("models/base/gemma-3-4b")
    'gemma'
    """
    lower = model_path_or_name.lower()
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    return "llama"


def get_eos_token_str(tokenizer, model_family: str) -> str:
    """
    Return the EOS / stop-token string for the given model family.

    Qwen3.5 (ChatML format)  →  ``<|im_end|>`` (preferred), else ``<|endoftext|>``
    Gemma 3                   →  ``<end_of_turn>`` (preferred), else ``<eos>``
    LLaMA 3.x                →  ``<|end_of_text|>`` (preferred), else ``<|eot_id|>``

    Falls back to ``tokenizer.eos_token`` when the preferred token is absent
    from the vocabulary (e.g. older tokenizer snapshots).
    """
    vocab = tokenizer.get_vocab()

    if model_family == "qwen":
        for candidate in ("<|im_end|>", "<|endoftext|>"):
            if candidate in vocab:
                return candidate
        return tokenizer.eos_token or "<|endoftext|>"
    elif model_family == "gemma":
        for candidate in ("<end_of_turn>", "<eos>"):
            if candidate in vocab:
                return candidate
        return tokenizer.eos_token or "<eos>"
    else:
        for candidate in ("<|end_of_text|>", "<|eot_id|>"):
            if candidate in vocab:
                return candidate
        return tokenizer.eos_token or "<|end_of_text|>"


def get_eos_token_id(tokenizer, model_family: str) -> int:
    """Return the token ID for the EOS/stop token of the given model family."""
    eos_str = get_eos_token_str(tokenizer, model_family)
    return tokenizer.convert_tokens_to_ids(eos_str)


# Gemma 4 is a multimodal architecture (vision/audio towers + text decoder).
# Its vision/audio towers contain Gemma4ClippableLinear layers that PEFT cannot
# wrap.  We use a regex pattern that targets ONLY the text-decoder's standard
# Linear projections.
_GEMMA4_TARGET_MODULES_RE: str = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


def is_gemma4(model_path_or_name: str) -> bool:
    """Return True when the model directory looks like a Gemma 4 variant.

    Checks config.json for ``model_type == 'gemma4'`` if the path exists on
    disk.  Falls back to simple name heuristics (``gemma-4`` or ``gemma4``).
    """
    import json
    from pathlib import Path

    cfg_path = Path(model_path_or_name) / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                data = json.load(f)
            return data.get("model_type", "").startswith("gemma4")
        except Exception:
            pass
    lower = model_path_or_name.lower()
    return "gemma-4" in lower or "gemma4" in lower


def get_lora_target_modules(model_family: str, model_path: str | None = None):
    """Return the LoRA target-module names for the given model family.

    For Gemma 4 models this returns a regex *string* (not a list) that scopes
    LoRA to the text-decoder layers only, avoiding unsupported
    Gemma4ClippableLinear modules in the vision/audio towers.

    Parameters
    ----------
    model_family : str
        One of 'qwen', 'gemma', 'llama'.
    model_path : str, optional
        Path to the local model directory.  Used to distinguish Gemma 3
        (simple list) from Gemma 4 (regex pattern).
    """
    if model_family == "qwen":
        return list(_QWEN_TARGET_MODULES)
    if model_family == "gemma":
        if model_path and is_gemma4(model_path):
            return _GEMMA4_TARGET_MODULES_RE
        return list(_GEMMA_TARGET_MODULES)
    return list(_LLAMA_TARGET_MODULES)


def get_gemma_token_type_key(model) -> str:
    """Return the correct token-type-ids field name for a Gemma model.

    Gemma 3 (``Gemma3ForConditionalGeneration``) expects ``token_type_ids``.
    Gemma 4 (``Gemma4ForConditionalGeneration``) renamed the field to
    ``mm_token_type_ids``.  Inspecting ``model.config.model_type`` is the
    most reliable way to distinguish the two at runtime.
    """
    model_type = getattr(model.config, "model_type", "")
    if model_type.startswith("gemma4"):
        return "mm_token_type_ids"
    return "token_type_ids"


def fix_tokenizer_padding(tokenizer, model_family: str) -> None:
    """
    Ensure the tokenizer has a pad token and correct padding side.

    Qwen3.5 tokenizers ship without a dedicated pad token.  We assign the EOS
    token as the pad token so HuggingFace Trainer padding works correctly.
    Padding side is set to 'right' for instruction fine-tuning (required when
    labels are provided with the Trainer's DataCollator).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
