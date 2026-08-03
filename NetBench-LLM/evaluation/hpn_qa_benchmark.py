#!/usr/bin/env python3
"""
HPN Q&A Benchmark — Answer Generation
======================================

Prompts instruction-tuned models with HPN domain questions from the
benchmark dataset and saves model answers to Excel.

Supports two backends:
  • Local HuggingFace models (default) — loaded with device_map="auto"
  • API models — GPT-4o / OpenAI (gpt-*, o1-*, o3-*, o4-*) and
                  Gemini (gemini-*) detected automatically from the model name

Each model is loaded once, prompted with all questions, then unloaded
before the next model is loaded — avoids repeated load/unload overhead.

Output:
    outputs/evaluations/answers/hpn_answers_{model_name}.xlsx

The Excel workbook has two sheets:
    • **Metadata** — model path, generation config, timing
    • **Answers**  — one row per question with model responses

Usage:
    # Single local model
    python evaluation/hpn_qa_benchmark.py \\
        --model_path models/instruction/Llama-3.1-8B-base-instruct

    # Two local models (processed sequentially)
    python evaluation/hpn_qa_benchmark.py \\
        --model_path models/instruction/Llama-3.1-8B-base-instruct \\
                     models/instruction/Llama-3.1-8B-trained-new-instruct

    # API model — GPT-4o
    python evaluation/hpn_qa_benchmark.py \\
        --model_path gpt-4o \\
        --openai_api_key_file openai_api_key.txt

    # API model — Gemini
    python evaluation/hpn_qa_benchmark.py \\
        --model_path gemini-2.5-pro \\
        --gemini_api_key_file gemini_api_key.txt

    # Mix of local + API in one run
    python evaluation/hpn_qa_benchmark.py \\
        --model_path models/instruction/Llama-3.1-8B-base-instruct gpt-4o
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.model_utils import detect_model_family, get_eos_token_str, fix_tokenizer_padding
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# Optional API imports — only needed when using API backends
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_BENCHMARK = "data/prompts/hpn_benchmark_v5.0.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/evaluations/answers"
DEFAULT_RPM = 15          # requests per minute for API models
MAX_RETRIES = 5           # for 429 / rate-limit errors
MAX_BACKOFF_SEC = 60      # cap for exponential backoff

# Prefixes used to detect API model types
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")
_GEMINI_PREFIX = "gemini-"


def _is_openai_model(model_name: str) -> bool:
    return model_name.lower().startswith(_OPENAI_PREFIXES)


def _is_gemini_model(model_name: str) -> bool:
    return model_name.lower().startswith(_GEMINI_PREFIX)


def _is_api_model(model_name: str) -> bool:
    return _is_openai_model(model_name) or _is_gemini_model(model_name)


def _benchmark_tag(benchmark_path: str) -> str:
    """Extract a concise tag from the benchmark filename.

    Examples:
        'hpn_qa_benchmark_v4_general_skills.json' → 'v4_general_skills'
        'hpn_benchmark_v5.0_all.jsonl'            → 'v5.0_all'
        'hpn_qa_benchmark.json'                   → ''  (no tag)
    """
    stem = Path(benchmark_path).stem
    for prefix in ("hpn_qa_benchmark", "hpn_benchmark"):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return stem[len(prefix) + 1:]   # strip prefix + underscore
    return ""

# HPN-expert system prompt — used for both local (Orca template) and API models
HPN_SYSTEM_PROMPT = (
    "You are an expert in high-performance networking (HPN), "
    "HPC data transfer systems, and computer networking.\n\n"
    "Answer the question below in a focused response (100–350 words). "
    "Be technically precise: include specific numbers, thresholds, or "
    "formulas where relevant. Explain the underlying reasoning, not just "
    "the conclusion."
)

# Open-Orca template — must match the training template (Llama local models)
ORCA_TEMPLATE = (
    "### System:\n{system}\n\n"
    "### User:\n{instruction}\n\n"
    "### Assistant:\n"
)

# ChatML template — must match the training template (Qwen local models)
CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Gemma template — Gemma 3 folds the system prompt into the first user turn
GEMMA_TEMPLATE = (
    "<start_of_turn>user\n{system}\n\n{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# Stop strings per model family — prevent the model from continuing past
# the first answer (e.g. self-asking follow-up questions or repeating content)
_STOP_STRINGS = {
    "llama": ["### User:", "### System:", "\n### "],
    "qwen":  ["<|im_start|>", "<|im_end|>"],
    "gemma": ["<start_of_turn>", "<end_of_turn>"],
}


# ═══════════════════════════════════════════════════════════════════════
# API key loading
# ═══════════════════════════════════════════════════════════════════════

def _load_key_from_file(key_file: str | None) -> str | None:
    """Read and return the first non-empty line from *key_file*, or None."""
    if not key_file:
        return None
    path = Path(key_file)
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


def load_gemini_api_key(key_file: str | None) -> str:
    """Load the Gemini API key from file or env var.

    Priority:
        1. --gemini_api_key_file (CLI flag pointing to a file)
        2. GEMINI_API_KEY environment variable
    """
    key = _load_key_from_file(key_file)
    if key:
        print(f"🔑 Gemini API key loaded from: {key_file}")
        return key

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        print("🔑 Gemini API key loaded from GEMINI_API_KEY environment variable")
        return key

    print("❌ No Gemini API key found.")
    print("   Provide one via:")
    print("     --gemini_api_key_file <path>")
    print("     GEMINI_API_KEY=<key>  (env var)")
    sys.exit(1)


def load_openai_api_key(key_file: str | None) -> str:
    """Load the OpenAI API key from file or env var.

    Priority:
        1. --openai_api_key_file (CLI flag pointing to a file)
        2. OPENAI_API_KEY environment variable
    """
    key = _load_key_from_file(key_file)
    if key:
        print(f"🔑 OpenAI API key loaded from: {key_file}")
        return key

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        print("🔑 OpenAI API key loaded from OPENAI_API_KEY environment variable")
        return key

    print("❌ No OpenAI API key found.")
    print("   Provide one via:")
    print("     --openai_api_key_file <path>")
    print("     OPENAI_API_KEY=<key>  (env var)")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark loader
# ═══════════════════════════════════════════════════════════════════════

def _normalize_question(raw: dict) -> dict:
    """Normalise a v5 JSONL question record to the pipeline's expected schema.

    Drops generation metadata that is not needed during benchmarking.
    """
    return {
        "id":               raw["id"],
        "category":         raw.get("category", ""),
        "difficulty":       raw.get("difficulty", ""),
        "question_type":    raw.get("question_type", ""),
        "question":         raw.get("question", ""),
        "reference_answer": raw.get("reference_answer", ""),
        "keywords":         raw.get("keywords", []),
    }


def load_benchmark(benchmark_path: str) -> dict:
    """Load the HPN Q&A benchmark file (JSON or JSONL).

    Accepts:
        * ``.json``  — legacy format with top-level ``metadata`` / ``questions``
        * ``.jsonl`` — flat format (one question per line); fields are normalised
                       to match the pipeline schema via :func:`_normalize_question`.

    Returns:
        dict with keys 'metadata' and 'questions'.
    """
    path = Path(benchmark_path)
    if not path.exists():
        print(f"❌ Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    if path.suffix == ".jsonl":
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                questions.append(_normalize_question(json.loads(line)))
        metadata = {
            "title":            f"HPN Benchmark — {path.stem}",
            "categories":       sorted({q["category"] for q in questions}),
            "difficulty_levels": sorted({q["difficulty"] for q in questions}),
        }
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        metadata = data.get("metadata", {})

    print(f"📋 Loaded benchmark: {metadata.get('title', benchmark_path)}")
    print(f"   Questions : {len(questions)}")
    print(f"   Categories: {len(metadata.get('categories', []))}")
    print(f"   Difficulty: {metadata.get('difficulty_levels', [])}")
    return {"metadata": metadata, "questions": questions}


# ═══════════════════════════════════════════════════════════════════════
# Resume support — load already-generated answers
# ═══════════════════════════════════════════════════════════════════════

def load_existing_answers(output_path: str) -> dict[str, dict]:
    """Read a previously saved answer Excel file and return valid answers.

    A row is considered valid (resumable) when its model_answer is non-empty
    and does not start with '[ERROR'.  Rows with empty or error answers are
    treated as not-yet-generated and will be regenerated on the next run.

    Returns:
        dict mapping str(question_id) → answer_dict
    """
    if not Path(output_path).exists():
        return {}

    try:
        wb = load_workbook(output_path, read_only=True, data_only=True)
        if "Answers" not in wb.sheetnames:
            wb.close()
            return {}

        ws = wb["Answers"]
        headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]

        existing = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            record = {}
            for col_idx, header in enumerate(headers):
                record[header] = row[col_idx] if col_idx < len(row) else None

            q_id = str(record.get("id", "")).strip()
            answer = str(record.get("model_answer") or "").strip()

            if not q_id or not answer or answer.startswith("[ERROR"):
                continue  # will be regenerated

            # Normalise keywords back to list
            kw_raw = record.get("keywords") or ""
            record["keywords"] = (
                [k.strip() for k in str(kw_raw).split(",") if k.strip()]
                if kw_raw else []
            )
            record["generation_time_sec"] = float(
                record.get("generation_time_sec") or 0.0
            )
            existing[q_id] = record

        wb.close()
        if existing:
            print(f"♻️  Resuming: {len(existing)} question(s) already answered "
                  f"in {output_path}")
        return existing

    except Exception as e:
        print(f"⚠️  Could not load existing answers: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════
# Local model loading / unloading
# ═══════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """Load model and tokenizer from a local directory.

    Returns:
        (model, tokenizer, model_family)
    """
    print(f"\n{'=' * 70}")
    print(f"🔄 Loading model: {model_path}")
    print(f"{'=' * 70}")

    model_family = detect_model_family(model_path)
    print(f"   Model family: {model_family}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    fix_tokenizer_padding(tokenizer, model_family)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("✅ Model loaded\n")
    return model, tokenizer, model_family


def unload_model(model, tokenizer):
    """Free GPU memory after finishing with a model."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🗑️  Model unloaded, GPU memory freed\n")


# ═══════════════════════════════════════════════════════════════════════
# Local model: prompt building & generation
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(question: str, model_family: str = "llama", tokenizer=None) -> str:
    """Build a prompt with the HPN system prompt using the model's native template."""
    if model_family == "qwen":
        # Qwen3.5 is a hybrid-reasoning model; use its native chat template with
        # thinking disabled so it answers directly (no <think> blocks), keeping it
        # consistent with every other model in the benchmark. Fall back to the
        # hand-written ChatML template if the tokenizer ships no chat template.
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": HPN_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return CHATML_TEMPLATE.format(
            system=HPN_SYSTEM_PROMPT,
            instruction=question,
        )
    if model_family == "gemma":
        return GEMMA_TEMPLATE.format(
            system=HPN_SYSTEM_PROMPT,
            instruction=question,
        )
    return ORCA_TEMPLATE.format(
        system=HPN_SYSTEM_PROMPT,
        instruction=question,
    )


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    model_family: str = "llama",
) -> tuple[str, float]:
    """Generate an answer for a single question using a local model.

    Returns:
        (answer_text, generation_time_seconds)
    """
    prompt = build_prompt(question, model_family=model_family, tokenizer=tokenizer)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    # Stop strings prevent the model from continuing past the first answer
    # (e.g. self-asking follow-up questions in base-model style)
    stop_strings = _STOP_STRINGS.get(model_family, _STOP_STRINGS["llama"])

    do_sample = temperature > 0.0
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        stop_strings=stop_strings,
        tokenizer=tokenizer,
    )
    elapsed = time.perf_counter() - start

    # Decode only the newly generated tokens
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Strip everything before the assistant marker to get the response only
    if model_family == "qwen":
        marker = "<|im_start|>assistant\n"
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
        # Remove trailing <|im_end|> and any other special tokens
        for tag in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            decoded = decoded.split(tag)[0]
        # Drop the Qwen3.5 thinking block (empty when thinking is disabled)
        if "<think>" in decoded and "</think>" in decoded:
            decoded = decoded.split("</think>")[-1]
        decoded = decoded.strip()
    elif model_family == "gemma":
        marker = "<start_of_turn>model\n"
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
        for tag in ("<end_of_turn>", "<start_of_turn>", "<eos>"):
            decoded = decoded.split(tag)[0]
        decoded = decoded.strip()
    else:
        if "### Assistant:" in decoded:
            decoded = decoded.split("### Assistant:")[-1].strip()
        # Clean up any trailing stop-string fragments
        for ss in stop_strings:
            if ss in decoded:
                decoded = decoded.split(ss)[0].strip()

    return decoded, elapsed


# ═══════════════════════════════════════════════════════════════════════
# API backend: answer generation (OpenAI + Gemini)
# ═══════════════════════════════════════════════════════════════════════

def _generate_answer_openai(
    client,
    model_name: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    rpm_delay: float,
) -> tuple[str, float]:
    """Generate an answer using the OpenAI Chat Completions API.

    Returns:
        (answer_text, generation_time_seconds)
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            time.sleep(rpm_delay)
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": HPN_SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            elapsed = time.perf_counter() - start
            answer = (response.choices[0].message.content or "").strip()
            return answer, elapsed

        except Exception as e:
            err_str = str(e).lower()
            is_retryable = (
                "429" in err_str
                or "503" in err_str
                or "rate_limit" in err_str
                or "rate limit" in err_str
                or "overloaded" in err_str
                or "server_error" in err_str
            )
            if is_retryable and attempt < MAX_RETRIES:
                backoff = min(2 ** (attempt + 1), MAX_BACKOFF_SEC)
                print(f"    ⏳ Rate-limited — backing off {backoff}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
                continue
            else:
                print(f"    ❌ OpenAI error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                return f"[ERROR: {e}]", 0.0

    return "[ERROR: max retries exceeded]", 0.0


def _generate_answer_gemini(
    client,
    model_name: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    rpm_delay: float,
) -> tuple[str, float]:
    """Generate an answer using the Gemini API.

    Returns:
        (answer_text, generation_time_seconds)
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            time.sleep(rpm_delay)
            start = time.perf_counter()
            # Gemini 2.5-Pro is a thinking model: its internal reasoning tokens
            # count against max_output_tokens. With 512, thinking exhausts the
            # entire budget before producing any text (FinishReason.MAX_TOKENS).
            # We add 8192 tokens of headroom for thinking so the model can still
            # produce ~512 tokens of actual response after reasoning completes.
            # temperature=0.0 is also omitted — it causes silent empty output
            # on thinking models; the API default is used instead.
            gemini_config_kwargs = {
                "system_instruction": HPN_SYSTEM_PROMPT,
                "max_output_tokens": max_new_tokens + 8192,
            }
            if temperature > 0.0:
                gemini_config_kwargs["temperature"] = temperature

            response = client.models.generate_content(
                model=model_name,
                contents=question,
                config=genai_types.GenerateContentConfig(**gemini_config_kwargs),
            )
            elapsed = time.perf_counter() - start

            # Try response.text first (works for non-thinking models)
            answer = ""
            try:
                if response.text:
                    answer = response.text.strip()
            except Exception:
                pass

            # Fallback: iterate parts directly (handles thinking models like
            # gemini-2.5-pro where thought parts precede the text part)
            if not answer and response.candidates:
                try:
                    for part in response.candidates[0].content.parts:
                        if getattr(part, "thought", False):
                            continue  # skip internal thinking tokens
                        if hasattr(part, "text") and part.text:
                            answer += part.text
                    answer = answer.strip()
                except Exception:
                    pass

            if not answer:
                print(f"    ⚠️  Empty response — diagnosing:")
                # prompt_feedback indicates if the request itself was blocked
                try:
                    pf = response.prompt_feedback
                    print(f"       prompt_feedback     : {pf}")
                except Exception as pf_err:
                    print(f"       prompt_feedback     : (error reading) {pf_err}")
                # candidates — number and finish reasons
                try:
                    cands = response.candidates
                    print(f"       candidates count    : {len(cands)}")
                    for i, c in enumerate(cands):
                        print(f"       candidate[{i}].finish_reason : "
                              f"{getattr(c, 'finish_reason', 'N/A')}")
                        try:
                            print(f"       candidate[{i}].safety_ratings: "
                                  f"{c.safety_ratings}")
                        except Exception:
                            pass
                        try:
                            parts = c.content.parts
                            print(f"       candidate[{i}].parts count  : {len(parts)}")
                            for j, p in enumerate(parts):
                                print(f"         part[{j}] thought={getattr(p, 'thought', False)} "
                                      f"text_len={len(p.text) if getattr(p, 'text', None) else 0}")
                        except Exception as pe:
                            print(f"       candidate[{i}].parts error  : {pe}")
                except Exception as ce:
                    print(f"       candidates error    : {ce}")
                # response.text property exception (if any)
                try:
                    _ = response.text
                except Exception as te:
                    print(f"       response.text raised: {te}")

            return answer, elapsed

        except Exception as e:
            err_str = str(e).lower()
            is_retryable = (
                "429" in err_str
                or "503" in err_str
                or "resource_exhausted" in err_str
                or "resourceexhausted" in err_str
                or "unavailable" in err_str
                or "overloaded" in err_str
                or "quota" in err_str
            )
            if is_retryable and attempt < MAX_RETRIES:
                backoff = min(2 ** (attempt + 1), MAX_BACKOFF_SEC)
                print(f"    ⏳ Rate-limited — backing off {backoff}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
                continue
            else:
                print(f"    ❌ Gemini error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                return f"[ERROR: {e}]", 0.0

    return "[ERROR: max retries exceeded]", 0.0


def generate_answer_api(
    client,
    model_name: str,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    rpm_delay: float = 4.0,
) -> tuple[str, float]:
    """Dispatch to the correct API backend based on model name.

    Returns:
        (answer_text, generation_time_seconds)
    """
    if _is_openai_model(model_name):
        return _generate_answer_openai(
            client, model_name, question, max_new_tokens, temperature, rpm_delay
        )
    return _generate_answer_gemini(
        client, model_name, question, max_new_tokens, temperature, rpm_delay
    )


def _init_api_client(model_name: str, gemini_key_file: str, openai_key_file: str):
    """Initialise and return the appropriate API client for *model_name*."""
    if _is_openai_model(model_name):
        if not _OPENAI_AVAILABLE:
            print("❌ openai package not installed. Run: pip install openai")
            sys.exit(1)
        api_key = load_openai_api_key(openai_key_file)
        client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI client initialised for model: {model_name}")
        return client
    else:
        if not _GENAI_AVAILABLE:
            print("❌ google-genai package not installed. Run: pip install google-genai")
            sys.exit(1)
        api_key = load_gemini_api_key(gemini_key_file)
        client = genai.Client(api_key=api_key)
        print(f"✅ Gemini client initialised for model: {model_name}")
        return client


# ═══════════════════════════════════════════════════════════════════════
# Excel writer
# ═══════════════════════════════════════════════════════════════════════

# Style constants
_HEADER_FONT = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
_HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
_HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
_WRAP_ALIGN = Alignment(vertical="top", wrap_text=True)
_THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
_META_KEY_FONT = Font(name="Calibri", bold=True, size=11)


def _style_header_row(ws, num_cols: int):
    """Apply styling to the first (header) row."""
    for col in range(1, num_cols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _HEADER_ALIGN
        cell.border = _THIN_BORDER


def save_answers_to_excel(
    output_path: str,
    model_path: str,
    model_name: str,
    benchmark_file: str,
    answers: list[dict],
    gen_config: dict,
    total_time: float,
):
    """Save model answers and metadata to an Excel workbook.

    Args:
        output_path:    Path for the .xlsx file.
        model_path:     Full path to the model directory (or API model name).
        model_name:     Short model name (directory basename or API model name).
        benchmark_file: Path to the benchmark JSON used.
        answers:        List of dicts, one per question.
        gen_config:     Dict with temperature, max_new_tokens, top_p.
        total_time:     Total wall-clock time for generation (seconds).
    """
    wb = Workbook()

    # ── Metadata sheet ──────────────────────────────────────────────
    ws_meta = wb.active
    ws_meta.title = "Metadata"

    meta_rows = [
        ("Key", "Value"),
        ("model_path", model_path),
        ("model_name", model_name),
        ("benchmark_file", benchmark_file),
        ("timestamp", datetime.now().isoformat()),
        ("total_questions", len(answers)),
        ("temperature", gen_config["temperature"]),
        ("max_new_tokens", gen_config["max_new_tokens"]),
        ("top_p", gen_config.get("top_p", "N/A")),
        ("total_generation_time_sec", f"{total_time:.1f}"),
        ("avg_generation_time_sec", f"{total_time / max(len(answers), 1):.2f}"),
    ]

    for row_idx, (key, value) in enumerate(meta_rows, start=1):
        ws_meta.cell(row=row_idx, column=1, value=key)
        ws_meta.cell(row=row_idx, column=2, value=str(value))
        if row_idx == 1:
            ws_meta.cell(row=row_idx, column=1).font = _HEADER_FONT
            ws_meta.cell(row=row_idx, column=1).fill = _HEADER_FILL
            ws_meta.cell(row=row_idx, column=1).alignment = _HEADER_ALIGN
            ws_meta.cell(row=row_idx, column=2).font = _HEADER_FONT
            ws_meta.cell(row=row_idx, column=2).fill = _HEADER_FILL
            ws_meta.cell(row=row_idx, column=2).alignment = _HEADER_ALIGN
        else:
            ws_meta.cell(row=row_idx, column=1).font = _META_KEY_FONT
            ws_meta.cell(row=row_idx, column=1).border = _THIN_BORDER
            ws_meta.cell(row=row_idx, column=2).border = _THIN_BORDER

    ws_meta.column_dimensions["A"].width = 28
    ws_meta.column_dimensions["B"].width = 80

    # ── Answers sheet ───────────────────────────────────────────────
    ws_ans = wb.create_sheet("Answers")

    headers = [
        "id",
        "category",
        "difficulty",
        "question",
        "reference_answer",
        "model_answer",
        "keywords",
        "generation_time_sec",
    ]

    # Write header row
    for col_idx, header in enumerate(headers, start=1):
        ws_ans.cell(row=1, column=col_idx, value=header)
    _style_header_row(ws_ans, len(headers))

    # Write data rows
    for row_idx, ans in enumerate(answers, start=2):
        ws_ans.cell(row=row_idx, column=1, value=ans["id"])
        ws_ans.cell(row=row_idx, column=2, value=ans["category"])
        ws_ans.cell(row=row_idx, column=3, value=ans["difficulty"])
        ws_ans.cell(row=row_idx, column=4, value=ans["question"])
        ws_ans.cell(row=row_idx, column=5, value=ans["reference_answer"])
        ws_ans.cell(row=row_idx, column=6, value=ans["model_answer"])
        ws_ans.cell(
            row=row_idx, column=7,
            value=", ".join(ans.get("keywords", []))
        )
        ws_ans.cell(
            row=row_idx, column=8,
            value=round(ans["generation_time_sec"], 3)
        )

        # Apply wrap + border to each cell
        for col in range(1, len(headers) + 1):
            cell = ws_ans.cell(row=row_idx, column=col)
            cell.alignment = _WRAP_ALIGN
            cell.border = _THIN_BORDER

    # Column widths
    col_widths = {
        "A": 8,   # id
        "B": 30,  # category
        "C": 12,  # difficulty
        "D": 60,  # question
        "E": 80,  # reference_answer
        "F": 80,  # model_answer
        "G": 40,  # keywords
        "H": 18,  # generation_time_sec
    }
    for col_letter, width in col_widths.items():
        ws_ans.column_dimensions[col_letter].width = width

    # Freeze the header row
    ws_ans.freeze_panes = "A2"

    # Auto-filter
    ws_ans.auto_filter.ref = (
        f"A1:{get_column_letter(len(headers))}{len(answers) + 1}"
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"💾 Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline — local model
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark_for_model(
    model_path: str,
    questions: list[dict],
    gen_config: dict,
    benchmark_file: str,
    output_dir: str,
):
    """Run the full benchmark for a single local HuggingFace model.

    1. Load existing answers (resume support)
    2. Load model
    3. Generate answers for unanswered questions only
    4. Save to Excel (merging existing + new)
    5. Unload model
    """
    model_name = Path(model_path).name

    # Check model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Resolve output path early so we can check for existing answers
    tag = _benchmark_tag(benchmark_file)
    tag_suffix = f"_{tag}" if tag else ""
    output_path = os.path.join(output_dir, f"hpn_answers_{model_name}{tag_suffix}.xlsx")

    # Resume: load already-generated answers and filter questions
    existing = load_existing_answers(output_path)
    questions_to_run = [q for q in questions if str(q["id"]) not in existing]

    if not questions_to_run:
        print(f"✅ {model_name} — all {len(questions)} answers already present, skipping.")
        return

    if existing:
        print(f"   Generating {len(questions_to_run)} remaining question(s).")

    # Load
    model, tokenizer, model_family = load_model(model_path)

    # Generate answers for remaining questions
    new_answers = []
    total_start = time.perf_counter()

    print(f"📝 Generating answers for {len(questions_to_run)} question(s)...")
    for q in tqdm(questions_to_run, desc=f"  {model_name}", unit="Q"):
        answer_text, gen_time = generate_answer(
            model,
            tokenizer,
            question=q["question"],
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            model_family=model_family,
        )

        new_answers.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "model_answer": answer_text,
            "keywords": q.get("keywords", []),
            "generation_time_sec": gen_time,
        })

    total_time = time.perf_counter() - total_start

    # Unload before saving (frees GPU for the next model)
    unload_model(model, tokenizer)

    # Merge: rebuild full answer list in original question order
    new_by_id = {str(a["id"]): a for a in new_answers}
    answers = []
    for q in questions:
        q_id = str(q["id"])
        if q_id in existing:
            answers.append(existing[q_id])
        elif q_id in new_by_id:
            answers.append(new_by_id[q_id])

    save_answers_to_excel(
        output_path=output_path,
        model_path=model_path,
        model_name=model_name,
        benchmark_file=benchmark_file,
        answers=answers,
        gen_config=gen_config,
        total_time=total_time,
    )

    # Print summary
    avg_time = total_time / max(len(new_answers), 1)
    print(f"\n{'─' * 70}")
    print(f"✅ {model_name} — {len(new_answers)} new answer(s) generated "
          f"in {total_time:.1f}s (avg {avg_time:.2f}s/question)")
    print(f"   Total saved: {len(answers)}/{len(questions)} questions")
    print(f"   Output: {output_path}")
    print(f"{'─' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline — API model
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark_for_api_model(
    model_name: str,
    questions: list[dict],
    gen_config: dict,
    benchmark_file: str,
    output_dir: str,
    client,
    rpm_delay: float,
):
    """Run the full benchmark for a single API model (OpenAI or Gemini).

    No model loading/unloading — uses the pre-initialised API client.
    Rate limiting is applied between each API call via rpm_delay.
    Supports resume: already-answered questions are skipped on rerun.
    """
    provider = "OpenAI" if _is_openai_model(model_name) else "Gemini"
    print(f"\n{'=' * 70}")
    print(f"🌐 API model: {model_name}")
    print(f"   Provider  : {provider}")
    print(f"   Rate limit: {60.0 / rpm_delay:.0f} req/min  ({rpm_delay:.1f}s between calls)")
    print(f"{'=' * 70}\n")

    # Resolve output path early so we can check for existing answers
    tag = _benchmark_tag(benchmark_file)
    tag_suffix = f"_{tag}" if tag else ""
    output_path = os.path.join(output_dir, f"hpn_answers_{model_name}{tag_suffix}.xlsx")

    # Resume: load already-generated answers and filter questions
    existing = load_existing_answers(output_path)
    questions_to_run = [q for q in questions if str(q["id"]) not in existing]

    if not questions_to_run:
        print(f"✅ {model_name} — all {len(questions)} answers already present, skipping.")
        return

    if existing:
        print(f"   Generating {len(questions_to_run)} remaining question(s).")

    new_answers = []
    total_start = time.perf_counter()

    print(f"📝 Generating answers for {len(questions_to_run)} question(s)...")
    for q in tqdm(questions_to_run, desc=f"  {model_name}", unit="Q"):
        answer_text, gen_time = generate_answer_api(
            client=client,
            model_name=model_name,
            question=q["question"],
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            rpm_delay=rpm_delay,
        )

        new_answers.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "model_answer": answer_text,
            "keywords": q.get("keywords", []),
            "generation_time_sec": gen_time,
        })

    total_time = time.perf_counter() - total_start

    # Merge: rebuild full answer list in original question order
    new_by_id = {str(a["id"]): a for a in new_answers}
    answers = []
    for q in questions:
        q_id = str(q["id"])
        if q_id in existing:
            answers.append(existing[q_id])
        elif q_id in new_by_id:
            answers.append(new_by_id[q_id])

    save_answers_to_excel(
        output_path=output_path,
        model_path=model_name,   # API models have no local path
        model_name=model_name,
        benchmark_file=benchmark_file,
        answers=answers,
        gen_config={**gen_config, "top_p": "N/A"},
        total_time=total_time,
    )

    # Print summary
    avg_time = total_time / max(len(new_answers), 1)
    print(f"\n{'─' * 70}")
    print(f"✅ {model_name} — {len(new_answers)} new answer(s) generated "
          f"in {total_time:.1f}s (avg {avg_time:.2f}s/question)")
    print(f"   Total saved: {len(answers)}/{len(questions)} questions")
    print(f"   Output: {output_path}")
    print(f"{'─' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HPN Q&A Benchmark — generate answers from local or API models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single local model\n"
            "  python evaluation/hpn_qa_benchmark.py \\\n"
            "      --model_path models/instruction/Llama-3.1-8B-base-instruct\n"
            "\n"
            "  # Two local models (sequential: load → generate → unload → next)\n"
            "  python evaluation/hpn_qa_benchmark.py \\\n"
            "      --model_path models/instruction/Llama-3.1-8B-base-instruct \\\n"
            "                   models/instruction/Llama-3.1-8B-trained-new-instruct\n"
            "\n"
            "  # GPT-4o via OpenAI API\n"
            "  python evaluation/hpn_qa_benchmark.py \\\n"
            "      --model_path gpt-4o\n"
            "\n"
            "  # Gemini via Google API\n"
            "  python evaluation/hpn_qa_benchmark.py \\\n"
            "      --model_path gemini-2.5-pro\n"
        ),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Model path(s) or API model name(s). "
            "Local models: path to an instruction-tuned model directory. "
            "API models: gpt-4o, gemini-2.5-pro, etc. "
            "Mixed lists are allowed — local models are loaded/unloaded sequentially; "
            "API models use the configured API key."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=DEFAULT_BENCHMARK,
        help=f"Path to HPN Q&A benchmark JSON (default: {DEFAULT_BENCHMARK})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output Excel files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per answer (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature (default: 0.0 — deterministic). "
            "Set > 0 to enable stochastic sampling."
        ),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p for local models (default: 0.9)",
    )
    parser.add_argument(
        "--gemini_api_key_file",
        type=str,
        default="gemini_api_key.txt",
        help="Path to file containing the Gemini API key (default: gemini_api_key.txt)",
    )
    parser.add_argument(
        "--openai_api_key_file",
        type=str,
        default="openai_api_key.txt",
        help="Path to file containing the OpenAI API key (default: openai_api_key.txt)",
    )
    parser.add_argument(
        "--requests_per_minute",
        type=int,
        default=DEFAULT_RPM,
        help=(
            f"Rate limit for API model calls in requests per minute "
            f"(default: {DEFAULT_RPM}). Applies to OpenAI and Gemini backends only."
        ),
    )

    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🏁  HPN Q&A Benchmark — Answer Generation")
    print("=" * 70)

    # ── Load benchmark ──────────────────────────────────────────────
    benchmark = load_benchmark(args.benchmark)
    questions = benchmark["questions"]

    gen_config = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
    }

    rpm_delay = 60.0 / args.requests_per_minute

    deterministic = args.temperature == 0.0
    print(f"\n⚙️  Generation config:")
    print(f"   temperature   = {gen_config['temperature']} "
          f"({'greedy / deterministic' if deterministic else 'sampling'})")
    print(f"   max_new_tokens= {gen_config['max_new_tokens']}")
    if not deterministic:
        print(f"   top_p         = {gen_config['top_p']}")
    print(f"\n📂 Output dir: {args.output_dir}")
    print(f"🤖 Models to benchmark: {len(args.model_path)}")
    for i, mp in enumerate(args.model_path, 1):
        backend = "API" if _is_api_model(mp) else "local"
        print(f"   {i}. {mp}  [{backend}]")

    # ── Run benchmark for each model ────────────────────────────────
    overall_start = time.perf_counter()
    tag = _benchmark_tag(args.benchmark)
    tag_suffix = f"_{tag}" if tag else ""

    for model_path in args.model_path:
        if _is_api_model(model_path):
            client = _init_api_client(
                model_path,
                args.gemini_api_key_file,
                args.openai_api_key_file,
            )
            run_benchmark_for_api_model(
                model_name=model_path,
                questions=questions,
                gen_config=gen_config,
                benchmark_file=args.benchmark,
                output_dir=args.output_dir,
                client=client,
                rpm_delay=rpm_delay,
            )
        else:
            run_benchmark_for_model(
                model_path=model_path,
                questions=questions,
                gen_config=gen_config,
                benchmark_file=args.benchmark,
                output_dir=args.output_dir,
            )

    overall_time = time.perf_counter() - overall_start

    # ── Final summary ───────────────────────────────────────────────
    print("=" * 70)
    print(f"🏁 All done! {len(args.model_path)} model(s) benchmarked "
          f"in {overall_time:.1f}s")
    print(f"   Output files in: {args.output_dir}/")
    for mp in args.model_path:
        name = mp if _is_api_model(mp) else Path(mp).name
        print(f"     • hpn_answers_{name}{tag_suffix}.xlsx")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
