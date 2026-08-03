"""
frontend/tabs/lora_tab.py
──────────────────────────
LoRA Adapter Training tab for the LLM Training Dashboard.

Provides a 4-panel UI for the LoRA pipeline:

  Panel 1 — Prepare Data
    Wraps training/prepare_lora_data.py
    Tokenizes the HPN research corpus into fixed-length blocks for CLM training.

  Panel 2 — Train LoRA
    Wraps training/lora_finetune.py
    Trains a QLoRA adapter on top of a selected instruct model.
    Output: models/lora/<model-name>-lora/

  Panel 3 — Merge Adapter
    Wraps training/merge_lora_adapter.py
    Merges a trained adapter back into base model weights.
    Output: models/lora-merged/<adapter-name>-merged/

  Panel 4 — Adapters
    Shows saved adapters and merged models on disk.
    Directs to the Benchmark tab for full evaluation.

Each operation runs as a subprocess through JobRunner so stdout/stderr is
streamed live into the log textbox.  The pipeline is loosely coupled: no
existing files are modified.

Must be called inside a gr.Blocks() context.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gradio as gr

from frontend.core.job_runner import make_runner

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

_ROOT         = Path(__file__).resolve().parent.parent.parent   # project root
_PYTHON       = sys.executable

_MODELS_DIR       = _ROOT / "models"
_DATA_DIR         = _ROOT / "data" / "processed"
_LORA_ADAPTER_DIR = _MODELS_DIR / "lora"
_LORA_MERGED_DIR  = _MODELS_DIR / "lora-merged"
_DEFAULT_CORPUS   = str(_ROOT / "data" / "raw" / "research_corpus_new.json")

# ──────────────────────────────────────────────────────────────────────────────
# JobRunner instances (one per operation)
# ──────────────────────────────────────────────────────────────────────────────

_runner_prep  = make_runner("lora_prep")
_runner_train = make_runner("lora_train")
_runner_merge = make_runner("lora_merge")

# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _instruct_models() -> list[str]:
    """Scan models/instruction/ for candidate source models."""
    out = []
    d = _MODELS_DIR / "instruction"
    if d.is_dir():
        out += [
            str(p.relative_to(_ROOT))
            for p in sorted(d.iterdir())
            if p.is_dir()
        ]
    return out or ["(no instruction models found)"]


def _data_dirs() -> list[str]:
    """Scan data/processed/ for Arrow datasets (directories with train/ subdir)."""
    out = []
    if _DATA_DIR.is_dir():
        for p in sorted(_DATA_DIR.iterdir()):
            if p.is_dir() and (p / "train").is_dir():
                out.append(str(p.relative_to(_ROOT)))
    return out or ["(no processed datasets found — run Prepare Data first)"]


def _lora_adapters() -> list[str]:
    """Scan models/lora/ for trained adapters (dirs with adapter_config.json)."""
    if not _LORA_ADAPTER_DIR.is_dir():
        return []
    return [
        str(p.relative_to(_ROOT))
        for p in sorted(_LORA_ADAPTER_DIR.iterdir())
        if p.is_dir() and (p / "adapter_config.json").exists()
    ]


def _merged_models() -> list[str]:
    """Scan models/lora-merged/ for fully merged models."""
    if not _LORA_MERGED_DIR.is_dir():
        return []
    return [
        str(p.relative_to(_ROOT))
        for p in sorted(_LORA_MERGED_DIR.iterdir())
        if p.is_dir() and (p / "config.json").exists()
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helper (identical to benchmark_tab.py)
# ──────────────────────────────────────────────────────────────────────────────

def _run_subprocess(cmd: list[str], cancel_event=None) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(_ROOT),
    )
    tail: list[str] = []
    for line in proc.stdout:
        if cancel_event and cancel_event.is_set():
            proc.terminate()
            proc.wait()
            print("\n[Cancelled by user]")
            return
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 20:
            tail.pop(0)
    proc.wait()
    if proc.returncode not in (0, None):
        print(f"\n❌ Process exited with code {proc.returncode}. Last output:")
        for ln in tail:
            print(f"❌  {ln}", end="", flush=True)
        raise RuntimeError(
            f"Subprocess exited with code {proc.returncode}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Job functions (run inside JobRunner threads)
# ──────────────────────────────────────────────────────────────────────────────

def _prep_job(
    model_path: str,
    corpus_file: str,
    output_dir: str,
    max_length: int,
    reuse_cpt: str,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON, "training/prepare_lora_data.py",
        "--model_path",  model_path,
        "--corpus_file", corpus_file,
        "--output_dir",  output_dir,
        "--max_length",  str(max_length),
    ]
    if reuse_cpt.strip():
        cmd += ["--reuse_cpt_data", reuse_cpt.strip()]
    _run_subprocess(cmd, _cancel_event)


def _train_job(
    model_path: str,
    data_dir: str,
    lora_rank: int,
    lora_alpha: int,
    lr: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    quantize_4bit: bool,
    target_modules: str,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON, "training/lora_finetune.py",
        "--model_path", model_path,
        "--data_dir",   data_dir,
        "--lora_rank",  str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--lr",         lr.strip(),
        "--epochs",     str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
    ]
    if quantize_4bit:
        cmd.append("--quantize_4bit")
    if target_modules.strip():
        cmd += ["--target_modules", target_modules.strip()]
    _run_subprocess(cmd, _cancel_event)


def _merge_job(
    adapter_path: str,
    base_model_path: str,
    output_dir: str,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON, "training/merge_lora_adapter.py",
        "--adapter_path",    adapter_path,
        "--base_model_path", base_model_path,
        "--output_dir",      output_dir,
    ]
    _run_subprocess(cmd, _cancel_event)


# ──────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ──────────────────────────────────────────────────────────────────────────────

_STATUS_ICONS = {
    "IDLE":    "⬜ Idle",
    "RUNNING": "🔄 Running…",
    "DONE":    "✅ Done",
    "ERROR":   "❌ Error",
}


def _status_label(runner) -> str:
    return _STATUS_ICONS.get(runner.get_status().name, runner.get_status().name)


def _poll_logs(runner) -> tuple[str, str]:
    runner.read_logs()
    return runner.get_full_log(), _status_label(runner)


# ──────────────────────────────────────────────────────────────────────────────
# Panel 1 — Prepare Data
# ──────────────────────────────────────────────────────────────────────────────

def _build_prepare_data() -> None:
    gr.Markdown(
        "### Step 1 — Prepare LoRA Training Data\n"
        "Tokenizes `data/raw/research_corpus_new.json` into fixed-length "
        "blocks using the causal-LM packing strategy (same as CPT prep).  "
        "Output is saved to `data/processed/lora/` by default.\n\n"
        "If you have already run the CPT data preparation for a model of the "
        "same size, enter that path in **Reuse CPT data** to skip re-tokenization."
    )

    with gr.Row():
        prep_model = gr.Dropdown(
            choices=_instruct_models(),
            value=None,
            label="Source model (tokenizer)",
            scale=5,
            interactive=True,
        )
        prep_rescan = gr.Button("🔄 Rescan", scale=1)

    prep_corpus = gr.Textbox(
        value=_DEFAULT_CORPUS,
        label="Corpus file",
        interactive=True,
    )

    prep_output = gr.Textbox(
        value="data/processed/lora",
        label="Output directory",
        interactive=True,
    )

    with gr.Row():
        prep_max_length = gr.Slider(
            512, 2048, value=2048, step=512,
            label="Block size (tokens)",
            scale=4,
        )
        prep_reuse_cpt = gr.Textbox(
            value="",
            label="Reuse CPT data (path, leave blank to tokenize from scratch)",
            placeholder="e.g. data/processed/llama-3.1-8b",
            scale=6,
        )

    with gr.Row():
        prep_run    = gr.Button("▶ Prepare Data", variant="primary", scale=3)
        prep_cancel = gr.Button("⏹ Cancel",        variant="stop",    scale=1)
        prep_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2
        )

    prep_log = gr.Textbox(
        label="Log output", lines=12, max_lines=200,
        interactive=False, autoscroll=True,
    )

    def _submit_prep(model, corpus, output, max_len, reuse_cpt):
        if not model or model.startswith("("):
            gr.Warning("Select a source model first.")
            return "⬜ Idle", ""
        if _runner_prep.get_status().name == "RUNNING":
            gr.Warning("Data preparation is already running.")
            return _status_label(_runner_prep), ""
        _runner_prep.reset()
        _runner_prep.submit(
            _prep_job,
            model, corpus, output, int(max_len), reuse_cpt,
        )
        return _status_label(_runner_prep), ""

    prep_run.click(
        fn=_submit_prep,
        inputs=[prep_model, prep_corpus, prep_output,
                prep_max_length, prep_reuse_cpt],
        outputs=[prep_status, prep_log],
    )
    prep_cancel.click(
        fn=lambda: (_runner_prep.cancel(), _status_label(_runner_prep))[1],
        outputs=[prep_status],
    )
    prep_rescan.click(
        fn=lambda: gr.Dropdown(choices=_instruct_models(), interactive=True),
        outputs=[prep_model],
    )

    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_prep),
        outputs=[prep_log, prep_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Panel 2 — Train LoRA
# ──────────────────────────────────────────────────────────────────────────────

def _build_train_lora() -> None:
    gr.Markdown(
        "### Step 2 — Train LoRA Adapter\n"
        "Trains a LoRA adapter on a selected instruct model using the HPN corpus.  "
        "Only the small adapter matrices are trained; the base model stays frozen.  "
        "Adapter is saved to `models/lora/<model-name>-lora/`.\n\n"
        "**Default mode: full bfloat16** — recommended for 2×48 GB RTX A6000 "
        "(~20–25 GB used per GPU with rank=64, batch=4).  "
        "Enable **4-bit QLoRA** only if VRAM < 24 GB.\n\n"
        "**Recommended models for primary experiments:**  "
        "`Llama-3.2-1B-base-instruct`, `Llama-3.1-8B-base-instruct`  \n"
        "**Ablation only:**  `*-trained-new-instruct` variants"
    )

    with gr.Row():
        train_model = gr.Dropdown(
            choices=_instruct_models(),
            value=None,
            label="Source instruct model",
            scale=5,
            interactive=True,
        )
        train_rescan_model = gr.Button("🔄 Rescan", scale=1)

    with gr.Row():
        train_data = gr.Dropdown(
            choices=_data_dirs(),
            value=None,
            label="Processed dataset (from Step 1)",
            scale=5,
            interactive=True,
        )
        train_rescan_data = gr.Button("🔄 Rescan", scale=1)

    gr.Markdown("**LoRA hyperparameters**  *(defaults tuned for 2×48 GB)*")
    with gr.Row():
        train_rank  = gr.Slider(4, 128,  value=64, step=4,
                                label="LoRA rank (r)", scale=3)
        train_alpha = gr.Slider(4, 256,  value=128, step=4,
                                label="LoRA alpha  (scaling = alpha/r)", scale=3)
        train_lr    = gr.Textbox(value="2e-4", label="Learning rate", scale=2)

    gr.Markdown("**Training hyperparameters**  *(effective batch = batch × grad_accum)*")
    with gr.Row():
        train_epochs     = gr.Slider(1, 20, value=3, step=1,
                                     label="Epochs", scale=2)
        train_batch      = gr.Slider(1, 16, value=4, step=1,
                                     label="Per-device batch size", scale=2)
        train_grad_accum = gr.Slider(1, 32, value=4, step=1,
                                     label="Gradient accumulation steps", scale=2)

    train_target_modules = gr.Textbox(
        value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        label="LoRA target modules (comma-separated; default works for Llama & Qwen)",
        interactive=True,
    )

    train_quantize = gr.Checkbox(
        value=False,
        label="Use 4-bit QLoRA (only for GPUs < 24 GB — not needed on 2×48 GB)",
    )

    with gr.Row():
        train_run    = gr.Button("▶ Train LoRA", variant="primary", scale=3)
        train_cancel = gr.Button("⏹ Cancel",     variant="stop",    scale=1)
        train_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2
        )

    train_log = gr.Textbox(
        label="Log output", lines=14, max_lines=300,
        interactive=False, autoscroll=True,
    )

    def _submit_train(model, data, rank, alpha, lr, epochs, batch, grad, quantize, target_mods):
        if not model or model.startswith("("):
            gr.Warning("Select a source instruct model.")
            return "⬜ Idle", ""
        if not data or data.startswith("("):
            gr.Warning("Select (or prepare) a processed dataset.")
            return "⬜ Idle", ""
        if _runner_train.get_status().name == "RUNNING":
            gr.Warning("LoRA training is already running.")
            return _status_label(_runner_train), ""
        _runner_train.reset()
        _runner_train.submit(
            _train_job,
            model, data,
            int(rank), int(alpha), lr,
            int(epochs), int(batch), int(grad),
            bool(quantize),
            target_mods,
        )
        return _status_label(_runner_train), ""

    train_run.click(
        fn=_submit_train,
        inputs=[train_model, train_data, train_rank, train_alpha, train_lr,
                train_epochs, train_batch, train_grad_accum, train_quantize,
                train_target_modules],
        outputs=[train_status, train_log],
    )
    train_cancel.click(
        fn=lambda: (_runner_train.cancel(), _status_label(_runner_train))[1],
        outputs=[train_status],
    )
    train_rescan_model.click(
        fn=lambda: gr.Dropdown(choices=_instruct_models(), interactive=True),
        outputs=[train_model],
    )
    train_rescan_data.click(
        fn=lambda: gr.Dropdown(choices=_data_dirs(), interactive=True),
        outputs=[train_data],
    )

    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_train),
        outputs=[train_log, train_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Panel 3 — Merge Adapter
# ──────────────────────────────────────────────────────────────────────────────

def _build_merge_adapter() -> None:
    gr.Markdown(
        "### Step 3 — Merge Adapter into Base Model\n"
        "Merges a trained LoRA adapter back into the base model weights, "
        "producing a standard full-weight model in `models/lora-merged/`.  "
        "The merged model has **no PEFT dependency** and works directly with "
        "the existing **Benchmark** tab for evaluation."
    )

    with gr.Row():
        merge_adapter = gr.Dropdown(
            choices=_lora_adapters(),
            value=None,
            label="Trained LoRA adapter",
            scale=5,
            interactive=True,
        )
        merge_rescan_adapter = gr.Button("🔄 Rescan", scale=1)

    with gr.Row():
        merge_base = gr.Dropdown(
            choices=_instruct_models(),
            value=None,
            label="Base model (used during LoRA training)",
            scale=5,
            interactive=True,
        )
        merge_rescan_base = gr.Button("🔄 Rescan", scale=1)

    merge_output = gr.Textbox(
        value="",
        label="Output path (leave blank for auto: models/lora-merged/<adapter-name>-merged/)",
        placeholder="e.g. models/lora-merged/Llama-3.1-8B-base-instruct-lora-merged",
        interactive=True,
    )

    with gr.Row():
        merge_run    = gr.Button("▶ Merge", variant="primary", scale=3)
        merge_cancel = gr.Button("⏹ Cancel",  variant="stop",  scale=1)
        merge_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2
        )

    merge_log = gr.Textbox(
        label="Log output", lines=10, max_lines=150,
        interactive=False, autoscroll=True,
    )

    def _submit_merge(adapter, base, output):
        if not adapter:
            gr.Warning("Select a LoRA adapter to merge.")
            return "⬜ Idle", ""
        if not base or base.startswith("("):
            gr.Warning("Select the base model used during training.")
            return "⬜ Idle", ""
        if _runner_merge.get_status().name == "RUNNING":
            gr.Warning("Merge is already running.")
            return _status_label(_runner_merge), ""

        # Resolve output: empty string → let the script auto-derive
        output_arg = output.strip() if output.strip() else (
            str(_LORA_MERGED_DIR / f"{Path(adapter).name}-merged")
        )
        _runner_merge.reset()
        _runner_merge.submit(_merge_job, adapter, base, output_arg)
        return _status_label(_runner_merge), ""

    merge_run.click(
        fn=_submit_merge,
        inputs=[merge_adapter, merge_base, merge_output],
        outputs=[merge_status, merge_log],
    )
    merge_cancel.click(
        fn=lambda: (_runner_merge.cancel(), _status_label(_runner_merge))[1],
        outputs=[merge_status],
    )
    merge_rescan_adapter.click(
        fn=lambda: gr.Dropdown(choices=_lora_adapters(), interactive=True),
        outputs=[merge_adapter],
    )
    merge_rescan_base.click(
        fn=lambda: gr.Dropdown(choices=_instruct_models(), interactive=True),
        outputs=[merge_base],
    )

    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_merge),
        outputs=[merge_log, merge_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Panel 4 — Adapters & Merged Models
# ──────────────────────────────────────────────────────────────────────────────

def _build_adapters_panel() -> None:
    gr.Markdown(
        "### Saved LoRA Adapters & Merged Models\n"
        "All adapters in `models/lora/` and merged models in `models/lora-merged/`.  "
        "Click **Refresh** after training or merging.\n\n"
        "To evaluate a merged model, go to the **📊 Benchmark** tab and select "
        "the merged model from `models/lora-merged/` in Phase 1."
    )

    adapters_box = gr.Textbox(
        value=_scan_lora_outputs,
        label="LoRA outputs",
        lines=16,
        max_lines=40,
        interactive=False,
    )

    refresh_btn = gr.Button("🔄 Refresh")

    refresh_btn.click(fn=_scan_lora_outputs, outputs=[adapters_box])

    gr.Timer(value=10).tick(fn=_scan_lora_outputs, outputs=[adapters_box])


def _scan_lora_outputs() -> str:
    """Return a formatted listing of lora/ and lora-merged/ directories."""
    lines = []

    def _scan_dir(d: Path, label: str):
        if not d.is_dir():
            lines.append(f"📁 {label}/  (not created yet)")
            return
        entries = [p for p in sorted(d.iterdir()) if p.is_dir()]
        lines.append(f"📁 {label}/  ({len(entries)} item{'s' if len(entries) != 1 else ''})")
        for p in entries:
            # Check what files are present
            has_adapter = (p / "adapter_config.json").exists()
            has_model   = (p / "config.json").exists()
            tag = "🔌 adapter" if has_adapter else ("⚡ merged" if has_model else "?")
            try:
                size_mb = sum(
                    f.stat().st_size for f in p.rglob("*") if f.is_file()
                ) / 1024**2
                lines.append(f"   {tag}  {p.name}  ({size_mb:.0f} MB)")
            except Exception:
                lines.append(f"   {tag}  {p.name}")

    _scan_dir(_LORA_ADAPTER_DIR, "models/lora")
    lines.append("")
    _scan_dir(_LORA_MERGED_DIR,  "models/lora-merged")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point — called from app.py inside gr.Tabs()
# ──────────────────────────────────────────────────────────────────────────────

def build_lora_tab() -> None:
    """
    Build the LoRA Training tab.
    Must be called inside a gr.Blocks() / gr.Tabs() context.
    """
    with gr.Tab("🔌 LoRA Training"):
        with gr.Tabs():
            with gr.Tab("📦 1. Prepare Data"):
                _build_prepare_data()
            with gr.Tab("🚀 2. Train LoRA"):
                _build_train_lora()
            with gr.Tab("🔗 3. Merge Adapter"):
                _build_merge_adapter()
            with gr.Tab("📂 Adapters"):
                _build_adapters_panel()
