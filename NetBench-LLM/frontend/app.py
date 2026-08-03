"""
frontend/app.py
───────────────
Main entry point for the LLM Training Dashboard (Gradio).

Run from the project root:
    python frontend/app.py

The dashboard opens at http://localhost:7860
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure project root is on the Python path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from frontend.core.model_manager import manager
from frontend.tabs.system_tab    import build_system_tab
from frontend.tabs.training_tab  import build_training_tab
from frontend.tabs.chat_tab      import build_chat_tab
from frontend.tabs.benchmark_tab import build_benchmark_tab
from frontend.tabs.profiling_tab import build_profiling_tab
from frontend.tabs.lora_tab      import build_lora_tab

_APP_VERSION = "0.4.0"


def _git_hash() -> str:
    """Return the short git commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _artifact_count(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob(pattern))


def _model_count() -> int:
    models_root = Path(__file__).resolve().parent.parent / "models"
    if not models_root.exists():
        return 0
    total = 0
    for category in models_root.iterdir():
        if not category.is_dir():
            continue
        for model_dir in category.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                total += 1
    return total


def _overview_html() -> str:
    project_root = Path(__file__).resolve().parent.parent
    instruction_data = _artifact_count(project_root / "data" / "instruction", "*/train")
    reports = _artifact_count(project_root / "outputs" / "evaluations" / "reports", "*.md")
    profiling_runs = _artifact_count(project_root / "outputs" / "profiling_results", "*.json")
    loaded = manager.loaded_count()
    models = _model_count()

    cards = [
        ("Models", str(models), "Local model directories discovered"),
        ("Loaded", str(loaded), "Models currently resident in GPU memory"),
        ("Instruction Data", str(instruction_data), "Prepared instruction-training datasets"),
        ("Benchmark Reports", str(reports), "Markdown reports under outputs/evaluations/reports"),
        ("Profiling Runs", str(profiling_runs), "JSON runs under outputs/profiling_results"),
    ]

    html_cards = "".join(
        f"""
        <div class="llmtd-card">
          <div class="llmtd-card-label">{label}</div>
          <div class="llmtd-card-value">{value}</div>
          <div class="llmtd-card-sub">{sub}</div>
        </div>
        """
        for label, value, sub in cards
    )

    return f"""
    <div class="llmtd-overview">
      <div class="llmtd-overview-grid">
        {html_cards}
      </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Build the app
# ──────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    _commit = _git_hash()
    with gr.Blocks(title="LLM Training Dashboard") as demo:

        gr.Markdown(
            f"# 🧠 LLM Training Dashboard"
            f"  <span style='font-size:0.75em;color:#888;'>v{_APP_VERSION} · {_commit}</span>\n"
            "Interactive UI for training, model management, chat, benchmarking, profiling, and LoRA."
        )

        overview = gr.HTML(value=_overview_html())
        gr.Timer(value=10).tick(fn=_overview_html, outputs=[overview])

        with gr.Tabs():
            build_system_tab()

            build_training_tab()

            build_chat_tab()

            build_benchmark_tab()

            build_profiling_tab()

            build_lora_tab()

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Training Dashboard")
    parser.add_argument("--host",  default="0.0.0.0",  help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",  default=7860, type=int, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true",  help="Create a public Gradio link")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1480px !important; }
            footer { display: none !important; }
            .llmtd-overview { margin: 0 0 18px 0; }
            .llmtd-overview-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
            }
            .llmtd-card {
                border: 1px solid rgba(24, 35, 53, 0.10);
                border-radius: 16px;
                padding: 14px 16px;
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,248,252,0.96));
                box-shadow: 0 10px 24px rgba(22, 30, 43, 0.06);
            }
            .llmtd-card-label {
                font-size: 12px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #5b6471;
                margin-bottom: 8px;
            }
            .llmtd-card-value {
                font-size: 28px;
                line-height: 1;
                font-weight: 700;
                color: #132033;
                margin-bottom: 8px;
            }
            .llmtd-card-sub {
                font-size: 12px;
                color: #5f6978;
                line-height: 1.35;
            }
            .block-title, .gr-markdown h3 { scroll-margin-top: 16px; }
        """,
    )
