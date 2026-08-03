"""
frontend/tabs/system_tab.py
────────────────────────────
System tab for the LLM Training Dashboard.

Displays:
  - Live GPU VRAM + temperature + utilisation (auto-refreshes every 5 s)
  - Currently loaded models with VRAM consumed
  - Model loader: scan available models, load, unload, refresh model list

Must be called inside a gr.Blocks() context:
    with gr.Blocks() as demo:
        build_system_tab()
"""

from __future__ import annotations

import gradio as gr

from frontend.core.model_manager import manager, ModelType


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _vram_table() -> list[dict]:
    """Return VRAM data as a list-of-dicts for gr.DataFrame."""
    rows = []
    for v in manager.get_vram_usage():
        rows.append({
            "GPU":        f"{v.gpu_index} — {v.gpu_name}",
            "Used (GB)":  round(v.used_gb, 2),
            "Total (GB)": round(v.total_gb, 2),
            "Free (GB)":  round(v.free_gb, 2),
            "Used %":     round(v.used_pct, 1),
            "Temp (°C)":  v.temperature_c if v.temperature_c is not None else "N/A",
            "Util %":     v.utilization_pct if v.utilization_pct is not None else "N/A",
        })
    if not rows:
        rows.append({"GPU": "No CUDA GPU detected", "Used (GB)": 0,
                     "Total (GB)": 0, "Free (GB)": 0,
                     "Used %": 0, "Temp (°C)": "N/A", "Util %": "N/A"})
    return rows


def _loaded_models_md() -> str:
    """Return a Markdown summary of currently loaded models."""
    loaded = manager.get_loaded_models()
    if not loaded:
        return "_No models currently loaded._"
    lines = []
    for name, m in loaded.items():
        vram_gb = m.vram_bytes_on_load / (1024 ** 3)
        lines.append(
            f"- **{name}** &nbsp; `{m.model_type.label}` &nbsp; "
            f"≈ {vram_gb:.2f} GB VRAM on load"
        )
    return "\n".join(lines)


def _available_choices() -> list[tuple[str, str]]:
    """
    Build dropdown choices for available models.
    Returns list of (display_label, nickname) tuples grouped by type.
    """
    models = manager.list_available_models()
    # Group by type for visual separation
    order = [ModelType.BASE, ModelType.PRETRAINED, ModelType.INSTRUCT,
             ModelType.PROFILED_BASE, ModelType.PROFILED_INSTRUCT]
    grouped: dict[ModelType, list] = {t: [] for t in order}
    for m in models:
        grouped[m.model_type].append(m)

    choices = []
    for mtype in order:
        for m in grouped[mtype]:
            choices.append((m.display_name, m.nickname))
    return choices


def _loaded_choices() -> list[str]:
    """Return the nicknames of currently loaded models."""
    return list(manager.get_loaded_models().keys())


def _refresh_available_dropdown(current: str | None):
    choices = _available_choices()
    valid_values = [value for _, value in choices]
    value = current if current in valid_values else None
    return gr.Dropdown(choices=choices, value=value, interactive=True)


def _refresh_loaded_dropdown(current: str | None):
    choices = _loaded_choices()
    value = current if current in choices else None
    return gr.Dropdown(choices=choices, value=value, interactive=True)


# ──────────────────────────────────────────────────────────────────────────────
# Event handlers
# ──────────────────────────────────────────────────────────────────────────────

def _refresh_vram():
    return _vram_table()


def _refresh_loaded_md():
    return _loaded_models_md()


def _rescan_models(current: str | None):
    """Re-scan the models/ directory and return updated dropdown choices."""
    return _refresh_available_dropdown(current)


def _load_model(nickname: str | None):
    """
    Load the selected model.
    Returns (status_message, updated_loaded_summary, updated_unload_dropdown, vram_table)
    """
    if not nickname:
        gr.Warning("Please select a model from the dropdown first.")
        return (
            "⚠️ Please select a model from the dropdown first.",
            _loaded_models_md(),
            gr.Dropdown(choices=_loaded_choices(), value=None),
            _vram_table(),
        )

    try:
        manager.load_model(nickname, max_loaded=2)
        status = f"✅ **{nickname}** loaded successfully."
    except RuntimeError as e:
        status = f"❌ {e}"
    except Exception as e:
        status = f"❌ Unexpected error: {e}"

    return (
        status,
        _loaded_models_md(),
        gr.Dropdown(choices=_loaded_choices(), value=None),
        _vram_table(),
    )


def _unload_model(nickname: str | None):
    """
    Unload the selected model.
    Returns (status_message, updated_loaded_summary, updated_unload_dropdown, vram_table)
    """
    if not nickname:
        gr.Warning("Please select a loaded model to unload.")
        return (
            "⚠️ Please select a loaded model to unload.",
            _loaded_models_md(),
            gr.Dropdown(choices=_loaded_choices(), value=None),
            _vram_table(),
        )
    try:
        manager.unload_model(nickname)
        status = f"✅ **{nickname}** unloaded. GPU memory freed."
    except KeyError as e:
        status = f"❌ {e}"
    except Exception as e:
        status = f"❌ Unexpected error: {e}"

    return (
        status,
        _loaded_models_md(),
        gr.Dropdown(choices=_loaded_choices(), value=None),
        _vram_table(),
    )


def _unload_all():
    """Unload every loaded model."""
    try:
        manager.unload_all()
        status = "✅ All models unloaded."
    except Exception as e:
        status = f"❌ {e}"
    return (
        status,
        _loaded_models_md(),
        gr.Dropdown(choices=_loaded_choices(), value=None),
        _vram_table(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab builder — call inside gr.Blocks()
# ──────────────────────────────────────────────────────────────────────────────

def build_system_tab() -> None:
    """
    Build the System tab UI.
    Must be called inside a gr.Blocks() context.
    """
    with gr.Tab("⚙️ System"):

        # ── GPU Stats ──────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### GPU Status")
            gr.Markdown(
                "_Refreshes automatically every 5 seconds. "
                "'Used %' reflects this process's allocated VRAM._"
            )
            vram_df = gr.DataFrame(
                value=_vram_table,
                headers=["GPU", "Used (GB)", "Total (GB)", "Free (GB)",
                         "Used %", "Temp (°C)", "Util %"],
                interactive=False,
                wrap=False,
            )
            gpu_timer = gr.Timer(value=5)
            gpu_timer.tick(fn=_refresh_vram, outputs=vram_df)

        gr.Markdown("---")

        # ── Loaded Models ──────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Loaded Models")
            loaded_md = gr.Markdown(value=_loaded_models_md)

        gr.Markdown("---")

        # ── Model Loader ───────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Load a Model")
            gr.Markdown(
                "Select a model from the list below and click **Load**. "
                "Up to **2 models** can be loaded simultaneously (for side-by-side Chat Compare). "
                "Loading the 8B model takes ~30–60 s; the 1B model takes ~10–20 s."
            )

            with gr.Row():
                available_dd = gr.Dropdown(
                    choices=_available_choices(),
                    value=None,
                    label="Available Models",
                    scale=5,
                    interactive=True,
                )
                rescan_btn = gr.Button("🔄 Rescan", scale=1, variant="secondary")

            with gr.Row():
                load_btn   = gr.Button("⬆️ Load Selected",  variant="primary",   scale=2)
                unload_dd  = gr.Dropdown(
                    choices=_loaded_choices(),
                    value=None,
                    label="Loaded Model to Unload",
                    scale=3,
                    interactive=True,
                )
                unload_btn    = gr.Button("⬇️ Unload Selected", variant="secondary", scale=2)
                unload_all_btn = gr.Button("🗑️ Unload All",     variant="stop",      scale=1)

            status_md = gr.Markdown(value="", label="Status")

        # ── Wire up events ─────────────────────────────────────────────────
        shared_outputs = [status_md, loaded_md, unload_dd, vram_df]

        load_btn.click(
            fn=_load_model,
            inputs=[available_dd],
            outputs=shared_outputs,
        )

        unload_btn.click(
            fn=_unload_model,
            inputs=[unload_dd],
            outputs=shared_outputs,
        )

        unload_all_btn.click(
            fn=_unload_all,
            inputs=[],
            outputs=shared_outputs,
        )

        rescan_btn.click(
            fn=_rescan_models,
            inputs=[available_dd],
            outputs=[available_dd],
        )

        # Keep loaded_md and unload_dd fresh on every GPU timer tick too
        gpu_timer.tick(fn=_refresh_loaded_md, outputs=loaded_md)
        gpu_timer.tick(
            fn=_refresh_loaded_dropdown,
            inputs=[unload_dd],
            outputs=unload_dd,
        )
        gpu_timer.tick(
            fn=_refresh_available_dropdown,
            inputs=[available_dd],
            outputs=available_dd,
        )
