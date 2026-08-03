"""
frontend/tabs/chat_tab.py
──────────────────────────
Chat tab for the LLM Training Dashboard.

Sub-tabs:
  • Single Chat  — stream tokens from one loaded model in real time.
  • Compare      — stream two models side-by-side into separate chatbots.

Streaming mechanism:
  TextIteratorStreamer (transformers) is created, then model.generate() is
  run in a background threading.Thread. The generator yields one accumulated
  string per token so Gradio re-renders the chatbot incrementally.

Prompt templates (matching training in prepare_instruction_data.py):
  INSTRUCT mode  → Open-Orca template with ### System / User / Assistant
  TEXT COMPLETION → raw user text passed directly to the model

Multi-turn conversation history:
  INSTRUCT mode   → full history included in prompt (all prior turns)
  TEXT COMPLETION → single-turn only (history is shown but not re-fed)

Must be called inside a gr.Blocks() context.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

import gradio as gr
from transformers import TextIteratorStreamer

from frontend.core.model_manager import manager


class _StopStringHit(Exception):
    """Raised internally when a stop-string is detected in the stream."""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates per model family
# (identical to inference/test_instruction_model.py)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM = (
    "You are a helpful assistant with expertise in high-performance networking, "
    "HPC data transfer systems, and computer networking. Give concise, "
    "technically precise answers."
)

# ── Llama / Open-Orca ─────────────────────────────────────────────────────────
_ORCA_TURN = "### User:\n{user}\n\n### Assistant:\n{assistant}\n\n"
_ORCA_FINAL = "### User:\n{user}\n\n### Assistant:\n"
_ORCA_SYSTEM_HEADER = "### System:\n{system}\n\n"

# ── Qwen / ChatML ────────────────────────────────────────────────────────────
_CHATML_TURN = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n"
_CHATML_FINAL = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
_CHATML_SYSTEM_HEADER = "<|im_start|>system\n{system}<|im_end|>\n"

# ── Gemma ─────────────────────────────────────────────────────────────────────
_GEMMA_TURN = "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>\n"
_GEMMA_FINAL = "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"

# Stop strings per model family — when the model emits any of these, generation
# should halt.  Prevents continuing in "base-model" style.
_STOP_STRINGS = {
    "llama": ["### User:", "### System:", "\n### "],
    "qwen":  ["<|im_start|>", "<|im_end|>"],
    "gemma": ["<start_of_turn>", "<end_of_turn>"],
}


def _get_model_family(model_nickname: str) -> str:
    """Detect model family from the loaded model's nickname/path."""
    from utils.model_utils import detect_model_family
    loaded = manager.get_model(model_nickname)
    if loaded and loaded.info and loaded.info.path:
        return detect_model_family(str(loaded.info.path))
    return detect_model_family(model_nickname)


def _build_instruct_prompt(
    message: str,
    history: list[dict],   # [{"role": "user"|"assistant", "content": str}, ...]
    system: str,
    model_family: str = "llama",
) -> str:
    """
    Build a full multi-turn prompt from Gradio chat history using the
    appropriate template for the model family.

    history contains prior answered turns; message is the new user query.
    """
    # Extract prior turn pairs
    history_pairs: list[tuple[str, str]] = []
    user_buf: str | None = None
    for h in history:
        role, content = h.get("role", ""), h.get("content", "")
        if role == "user":
            user_buf = content
        elif role == "assistant" and user_buf is not None:
            history_pairs.append((user_buf, content))
            user_buf = None

    sys = system or _DEFAULT_SYSTEM

    if model_family == "qwen":
        parts = [_CHATML_SYSTEM_HEADER.format(system=sys)]
        for user_msg, asst_msg in history_pairs:
            parts.append(_CHATML_TURN.format(user=user_msg, assistant=asst_msg))
        parts.append(_CHATML_FINAL.format(user=message))
    elif model_family == "gemma":
        # Gemma folds system prompt into the first user turn
        parts = []
        for i, (user_msg, asst_msg) in enumerate(history_pairs):
            u = f"{sys}\n\n{user_msg}" if i == 0 else user_msg
            parts.append(_GEMMA_TURN.format(user=u, assistant=asst_msg))
        first_user = message if history_pairs else f"{sys}\n\n{message}"
        parts.append(_GEMMA_FINAL.format(user=first_user))
    else:
        parts = [_ORCA_SYSTEM_HEADER.format(system=sys)]
        for user_msg, asst_msg in history_pairs:
            parts.append(_ORCA_TURN.format(user=user_msg, assistant=asst_msg))
        parts.append(_ORCA_FINAL.format(user=message))

    return "".join(parts)


def _build_completion_prompt(message: str) -> str:
    """Text-completion mode: pass the user text directly."""
    return message


# ──────────────────────────────────────────────────────────────────────────────
# Core streaming generator
# ──────────────────────────────────────────────────────────────────────────────

def _stream_response(
    model_nickname: str,
    message: str,
    history: list[dict],
    mode: str,           # "Instruct" | "Text Completion"
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Iterator[str]:
    """
    Generator that yields the growing assistant response one token at a time.
    Designed to be used directly as a gr.ChatInterface fn or as a manual
    gr.Chatbot streaming fn.

    Yields the partial response string (accumulates — not just the new token).
    Appends a stats line (time + tokens/s) as the very last yield.
    """
    # ── Precondition ────────────────────────────────────────────────────────
    if not model_nickname:
        yield "⚠️ No model selected. Choose a loaded model from the dropdown."
        return

    loaded = manager.get_model(model_nickname)
    if loaded is None:
        yield (
            f"⚠️ **{model_nickname}** is not loaded. "
            "Go to the ⚙️ System tab and load it first."
        )
        return

    model     = loaded.model
    tokenizer = loaded.tokenizer

    # ── Detect model family for correct template ────────────────────────────
    model_family = _get_model_family(model_nickname)

    # ── Build prompt ─────────────────────────────────────────────────────────
    if mode == "Instruct":
        prompt = _build_instruct_prompt(message, history, system_prompt, model_family=model_family)
    else:
        prompt = _build_completion_prompt(message)

    # ── Tokenise ─────────────────────────────────────────────────────────────
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # ── Set up streaming ─────────────────────────────────────────────────────
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0,
    )

    gen_kwargs = {
        **inputs,
        "streamer":             streamer,
        "max_new_tokens":       max_new_tokens,
        "temperature":          max(temperature, 1e-2),  # avoid 0 with do_sample
        "top_p":                top_p,
        "do_sample":            temperature > 0.0,
        "pad_token_id":         tokenizer.eos_token_id,
        "repetition_penalty":   1.1,
    }

    # In Instruct mode, stop when the model tries to start a new turn
    stop_strings = _STOP_STRINGS.get(model_family, _STOP_STRINGS["llama"])
    if mode == "Instruct":
        gen_kwargs["stop_strings"] = stop_strings
        gen_kwargs["tokenizer"] = tokenizer

    gen_thread = threading.Thread(
        target=model.generate,
        kwargs=gen_kwargs,
        daemon=True,
    )

    # ── Stream ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    gen_thread.start()

    partial = ""
    token_count = 0

    try:
        for token in streamer:
            # Strip any stop-string fragments that leak into the last token
            if mode == "Instruct":
                for ss in stop_strings:
                    if ss in partial + token:
                        token = (partial + token).split(ss)[0][len(partial):]
                        partial += token
                        token_count += 1
                        raise _StopStringHit()
            partial += token
            token_count += 1
            yield partial
    except _StopStringHit:
        pass  # clean stop — model tried to start a new turn
    except Exception as exc:
        yield f"{partial}\n\n_[Streaming error: {exc}]_"
        return
    finally:
        gen_thread.join(timeout=5)
        if gen_thread.is_alive():
            # Thread did not finish within 5 s — log a warning but don't block.
            import warnings
            warnings.warn(
                f"[chat_tab] generate() thread for '{model_nickname}' is still "
                "alive after 5 s join — the thread may be leaked.",
                RuntimeWarning,
                stacklevel=2,
            )

    elapsed = time.perf_counter() - t0
    tps = token_count / elapsed if elapsed > 0 else 0.0

    stats = (
        f"\n\n---\n"
        f"_⏱ {elapsed:.1f} s · {token_count} tokens · {tps:.1f} tok/s_"
    )
    yield partial + stats


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _loaded_model_choices() -> list[str]:
    return list(manager.get_loaded_models().keys())


def _refresh_loaded_model_dropdown(current: str | None):
    choices = _loaded_model_choices()
    value = current if current in choices else None
    return gr.Dropdown(choices=choices, value=value, interactive=True)


def _auto_mode(nickname: str) -> str:
    """Return default mode string based on model type."""
    loaded = manager.get_model(nickname)
    if loaded and loaded.is_instruct:
        return "Instruct"
    return "Text Completion"


def _system_prompt_visibility(mode: str) -> dict:
    return gr.update(visible=(mode == "Instruct"))


# ──────────────────────────────────────────────────────────────────────────────
# Single Chat sub-tab builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_single_chat() -> None:
    """Builds the Single Chat layout inside the current gr.Tab context."""

    # ── Controls row ────────────────────────────────────────────────────────
    with gr.Row():
        model_dd = gr.Dropdown(
            choices=_loaded_model_choices(),
            value=None,
            label="Loaded Model",
            scale=4,
            interactive=True,
        )
        mode_radio = gr.Radio(
            choices=["Instruct", "Text Completion"],
            value="Instruct",
            label="Mode",
            scale=3,
        )

    system_prompt = gr.Textbox(
        value=_DEFAULT_SYSTEM,
        label="System Prompt",
        placeholder="You are a helpful assistant.",
        lines=2,
        visible=True,
    )

    gr.Markdown(
        "_**Instruct** — auto-selects template by model family "
        "(Open-Orca for Llama, ChatML for Qwen, Gemma turns for Gemma) "
        "with full conversation history. "
        "**Text Completion** — passes your text directly to the model (single-turn)._"
    )

    with gr.Row():
        max_tokens_sl = gr.Slider(
            minimum=32, maximum=2048, value=512, step=32,
            label="Max new tokens", scale=3,
        )

    with gr.Accordion("⚙️ Advanced generation settings", open=False):
        with gr.Row():
            temperature_sl = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                label="Temperature", scale=3,
            )
            top_p_sl = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                label="Top-p", scale=2,
            )

    # ── Chatbot ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    chatbot = gr.Chatbot(
        label="Chat",
        height=480,
        buttons=["copy_all"],
        avatar_images=("👤", "🤖"),
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your message and press ⏎ to send…",
            label="",
            scale=9,
            lines=1,
        )
        send_btn  = gr.Button("Send ▶",  variant="primary", scale=1)
        clear_btn = gr.Button("Clear 🗑", variant="secondary", scale=1)

    # ── Streaming submit function ─────────────────────────────────────────────
    def _submit(message, history, model_nick, mode, sys_p, max_tok, temp, tp):
        if not message.strip():
            yield history or [], ""
            return

        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})

        try:
            for chunk in _stream_response(
                model_nick, message, history[:-2],
                mode, sys_p, max_tok, temp, tp,
            ):
                history[-1]["content"] = chunk
                yield history, ""
        except Exception as exc:
            gr.Warning(f"Chat error: {exc}")
            history[-1]["content"] = f"⚠️ Error: {exc}"
            yield history, ""

    # ── Event wiring ─────────────────────────────────────────────────────────
    submit_inputs  = [msg_box, chatbot, model_dd, mode_radio,
                      system_prompt, max_tokens_sl, temperature_sl, top_p_sl]
    submit_outputs = [chatbot, msg_box]

    msg_box.submit(fn=_submit, inputs=submit_inputs, outputs=submit_outputs)
    send_btn.click(fn=_submit, inputs=submit_inputs, outputs=submit_outputs)
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_box])

    # Auto-set mode when model is selected
    model_dd.change(
        fn=_auto_mode,
        inputs=[model_dd],
        outputs=[mode_radio],
    )

    # Show/hide system prompt based on mode
    mode_radio.change(
        fn=_system_prompt_visibility,
        inputs=[mode_radio],
        outputs=[system_prompt],
    )

    # Refresh model dropdown every 5 s
    gr.Timer(value=5).tick(
        fn=_refresh_loaded_model_dropdown,
        inputs=[model_dd],
        outputs=[model_dd],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Compare sub-tab builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_compare() -> None:
    """Builds the Compare Two Models layout inside the current gr.Tab context."""

    gr.Markdown(
        "Load **two different models** in the ⚙️ System tab, then select one "
        "per column. Both will receive the same message and stream simultaneously."
    )

    # ── Shared controls ───────────────────────────────────────────────────────
    with gr.Row():
        shared_mode = gr.Radio(
            choices=["Instruct", "Text Completion"],
            value="Instruct",
            label="Mode (applies to both)",
            scale=4,
        )
        shared_max_tokens = gr.Slider(
            32, 2048, value=512, step=32, label="Max new tokens", scale=3,
        )
        shared_temp = gr.Slider(
            0.0, 2.0, value=0.7, step=0.05, label="Temperature", scale=3,
        )

    shared_system = gr.Textbox(
        value=_DEFAULT_SYSTEM,
        label="Shared System Prompt",
        lines=2,
        visible=True,
    )

    shared_mode.change(
        fn=_system_prompt_visibility,
        inputs=[shared_mode],
        outputs=[shared_system],
    )

    # ── Two model columns ─────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column():
            left_dd = gr.Dropdown(
                choices=_loaded_model_choices(),
                value=None,
                label="Left Model",
                interactive=True,
            )
            left_bot = gr.Chatbot(
                label="Left",
                height=400,
                buttons=["copy_all"],
                avatar_images=("👤", "🤖"),
            )

        with gr.Column():
            right_dd = gr.Dropdown(
                choices=_loaded_model_choices(),
                value=None,
                label="Right Model",
                interactive=True,
            )
            right_bot = gr.Chatbot(
                label="Right",
                height=400,
                buttons=["copy_all"],
                avatar_images=("👤", "🤖"),
            )

    # ── Shared input ──────────────────────────────────────────────────────────
    with gr.Row():
        cmp_msg  = gr.Textbox(
            placeholder="Type a message and press ⏎ to send to both models…",
            label="",
            scale=9,
            lines=1,
        )
        cmp_send  = gr.Button("Send to Both ▶", variant="primary", scale=2)
        cmp_clear = gr.Button("Clear Both 🗑",   variant="secondary", scale=1)

    # ── Streaming generators per column ───────────────────────────────────────
    # Each generator writes ONLY to its own chatbot to prevent concurrent
    # generators from overwriting each other's live streaming updates.

    def _stream_left(message, left_hist, left_nick,
                     mode, sys_p, max_tok, temp):
        if not message.strip():
            yield left_hist or []
            return
        left_hist = left_hist or []
        left_hist.append({"role": "user",      "content": message})
        left_hist.append({"role": "assistant", "content": "⏳ Generating…"})
        try:
            for chunk in _stream_response(
                left_nick, message, left_hist[:-2],
                mode, sys_p, max_tok, temp, 0.9,
            ):
                left_hist[-1]["content"] = chunk
                yield left_hist
        except Exception as exc:
            gr.Warning(f"Left model error: {exc}")
            left_hist[-1]["content"] = f"⚠️ Error: {exc}"
            yield left_hist

    def _stream_right(message, right_hist, right_nick,
                      mode, sys_p, max_tok, temp):
        if not message.strip():
            yield right_hist or []
            return
        right_hist = right_hist or []
        right_hist.append({"role": "user",      "content": message})
        right_hist.append({"role": "assistant", "content": "⏳ Generating…"})
        try:
            for chunk in _stream_response(
                right_nick, message, right_hist[:-2],
                mode, sys_p, max_tok, temp, 0.9,
            ):
                right_hist[-1]["content"] = chunk
                yield right_hist
        except Exception as exc:
            gr.Warning(f"Right model error: {exc}")
            right_hist[-1]["content"] = f"⚠️ Error: {exc}"
            yield right_hist

    left_inputs  = [cmp_msg, left_bot,  left_dd,  shared_mode, shared_system, shared_max_tokens, shared_temp]
    right_inputs = [cmp_msg, right_bot, right_dd, shared_mode, shared_system, shared_max_tokens, shared_temp]

    # Each event targets only its own chatbot — no cross-component interference
    cmp_send.click(fn=_stream_left,  inputs=left_inputs,  outputs=[left_bot])
    cmp_send.click(fn=_stream_right, inputs=right_inputs, outputs=[right_bot])
    cmp_send.click(fn=lambda: "", outputs=[cmp_msg])

    cmp_msg.submit(fn=_stream_left,  inputs=left_inputs,  outputs=[left_bot])
    cmp_msg.submit(fn=_stream_right, inputs=right_inputs, outputs=[right_bot])
    cmp_msg.submit(fn=lambda: "", outputs=[cmp_msg])

    cmp_clear.click(fn=lambda: ([], [], ""), outputs=[left_bot, right_bot, cmp_msg])

    # Refresh model dropdowns every 5 s
    cmp_timer = gr.Timer(value=5)
    cmp_timer.tick(
        fn=lambda left, right: (
            _refresh_loaded_model_dropdown(left),
            _refresh_loaded_model_dropdown(right),
        ),
        inputs=[left_dd, right_dd],
        outputs=[left_dd, right_dd],
    )

    # Auto-set mode when left model changes
    left_dd.change(fn=_auto_mode, inputs=[left_dd], outputs=[shared_mode])
    right_dd.change(fn=_auto_mode, inputs=[right_dd], outputs=[shared_mode])


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint — called from app.py inside gr.Tabs()
# ──────────────────────────────────────────────────────────────────────────────

def build_chat_tab() -> None:
    """
    Build the Chat tab.
    Must be called inside a gr.Blocks() / gr.Tabs() context.
    """
    with gr.Tab("💬 Chat"):
        with gr.Tabs():
            with gr.Tab("💬 Single Chat"):
                _build_single_chat()
            with gr.Tab("⚖️ Compare Two Models"):
                _build_compare()
