"""
frontend/tabs/training_tab.py
─────────────────────────────
Training tab for the LLM Training Dashboard.

Wraps the project's main full-weight training pipeline:

  1. Prepare pretraining data        → training/prepare_data.py
  2. Continual pretraining           → training/pretrain_transformers.py
  3. Prepare instruction data        → training/prepare_instruction_data.py
  4. Instruction fine-tuning         → training/instruction_finetune.py
  5. Outputs viewer                  → scans generated datasets and models

Like the Benchmark / Profiling / LoRA tabs, each long-running stage is
executed as a subprocess inside a JobRunner worker thread. Stdout/stderr is
streamed live into the UI via gr.Timer polling.
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

_ROOT = Path(__file__).resolve().parent.parent.parent
_PYTHON = sys.executable

_MODELS_DIR = _ROOT / "models"
_DATA_DIR = _ROOT / "data"

_PRETRAINED_MODELS_DIR = _MODELS_DIR / "pretrained"
_INSTRUCTION_MODELS_DIR = _MODELS_DIR / "instruction"

_PROCESSED_DATA_DIR = _DATA_DIR / "processed"
_INSTRUCTION_DATA_DIR = _DATA_DIR / "instruction"

_DEFAULT_CORPUS = str(_DATA_DIR / "raw" / "research_corpus_new.json")


# ──────────────────────────────────────────────────────────────────────────────
# JobRunner instances
# ──────────────────────────────────────────────────────────────────────────────

_runner_prepare_pretrain = make_runner("train_prepare_pretrain")
_runner_pretrain = make_runner("train_pretrain")
_runner_prepare_instruct = make_runner("train_prepare_instruct")
_runner_instruct = make_runner("train_instruct")


# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scan_models(*subdirs: str) -> list[str]:
    out: list[str] = []
    for sub in subdirs:
        d = _MODELS_DIR / sub
        if d.is_dir():
            out += [
                str(p.relative_to(_ROOT))
                for p in sorted(d.iterdir())
                if p.is_dir() and (p / "config.json").exists()
            ]
    return out


def _base_models() -> list[str]:
    return _scan_models("base") or ["(no base models found)"]


def _base_and_pretrained_models() -> list[str]:
    return _scan_models("base", "pretrained") or ["(no base/pretrained models found)"]


def _processed_pretrain_dirs() -> list[str]:
    if not _PROCESSED_DATA_DIR.is_dir():
        return ["(no processed datasets found)"]
    return [
        str(p.relative_to(_ROOT))
        for p in sorted(_PROCESSED_DATA_DIR.iterdir())
        if p.is_dir() and (p / "train").is_dir()
    ] or ["(no processed datasets found)"]


def _instruction_data_dirs() -> list[str]:
    if not _INSTRUCTION_DATA_DIR.is_dir():
        return ["(no instruction datasets found)"]
    return [
        str(p.relative_to(_ROOT))
        for p in sorted(_INSTRUCTION_DATA_DIR.iterdir())
        if p.is_dir() and (p / "train").is_dir()
    ] or ["(no instruction datasets found)"]


def _presplit_instruction_dirs() -> list[str]:
    out = []
    for p in sorted(_DATA_DIR.rglob("*")):
        if not p.is_dir():
            continue
        if (p / "train.jsonl").exists() and (p / "validation.jsonl").exists():
            out.append(str(p.relative_to(_ROOT)))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Output path helpers
# ──────────────────────────────────────────────────────────────────────────────

def _model_basename(model_path: str) -> str:
    return Path(model_path).name


def _model_slug(model_path: str) -> str:
    name = _model_basename(model_path).lower()
    for suffix in (
        "-base-instruct",
        "-trained-new-instruct",
        "-instruct",
        "-base",
        "-trained-new",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _default_pretrain_data_output(model_path: str) -> str:
    return f"data/processed/{_model_slug(model_path)}"


def _default_pretrain_model_output(model_path: str) -> str:
    name = _model_basename(model_path)
    if name.endswith("-base"):
        name = name[:-5]
    return f"models/pretrained/{name}-trained-new"


def _default_instruction_data_output(model_path: str) -> str:
    return f"data/instruction/{_model_basename(model_path).lower()}"


def _default_instruction_model_output(model_path: str) -> str:
    return f"models/instruction/{_model_basename(model_path)}-instruct"


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess runner
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
        raise RuntimeError(f"Subprocess exited with code {proc.returncode}")


# ──────────────────────────────────────────────────────────────────────────────
# Job functions
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_pretrain_data_job(
    model_name: str,
    input_file: str,
    output_dir: str,
    max_length: int,
    test_size: float,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON,
        "training/prepare_data.py",
        "--input_file",
        input_file,
        "--output_dir",
        output_dir,
        "--model_name",
        model_name,
        "--max_length",
        str(max_length),
        "--test_size",
        str(test_size),
    ]
    _run_subprocess(cmd, _cancel_event)


def _pretrain_job(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_epochs: int,
    train_batch: int,
    eval_batch: int,
    grad_accum: int,
    learning_rate: str,
    optimizer: str,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON,
        "training/pretrain_transformers.py",
        "--model_path",
        model_path,
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
        "--num_train_epochs",
        str(num_epochs),
        "--per_device_train_batch_size",
        str(train_batch),
        "--per_device_eval_batch_size",
        str(eval_batch),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--learning_rate",
        learning_rate.strip(),
        "--optim",
        optimizer,
    ]
    _run_subprocess(cmd, _cancel_event)


def _prepare_instruction_data_job(
    model_path: str,
    output_dir: str,
    input_dir: str,
    max_length: int,
    max_samples: str,
    val_ratio: float,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON,
        "training/prepare_instruction_data.py",
        "--model_path",
        model_path,
        "--output_dir",
        output_dir,
        "--max_length",
        str(max_length),
        "--val_ratio",
        str(val_ratio),
    ]
    if input_dir.strip():
        cmd += ["--input_dir", input_dir.strip()]
    if max_samples.strip():
        cmd += ["--max_samples", max_samples.strip()]
    _run_subprocess(cmd, _cancel_event)


def _instruction_finetune_job(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: str,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON,
        "training/instruction_finetune.py",
        "--model_path",
        model_path,
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--learning_rate",
        learning_rate.strip(),
    ]
    _run_subprocess(cmd, _cancel_event)


# ──────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ──────────────────────────────────────────────────────────────────────────────

_STATUS_ICONS = {
    "IDLE": "⬜ Idle",
    "RUNNING": "🔄 Running…",
    "DONE": "✅ Done",
    "ERROR": "❌ Error",
}


def _status_label(runner) -> str:
    return _STATUS_ICONS.get(runner.get_status().name, runner.get_status().name)


def _poll_logs(runner) -> tuple[str, str]:
    runner.read_logs()
    return runner.get_full_log(), _status_label(runner)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-tab 1 — Prepare pretraining data
# ──────────────────────────────────────────────────────────────────────────────

def _build_prepare_pretrain_data() -> None:
    gr.Markdown(
        "### Step 1 — Prepare Continual-Pretraining Data\n"
        "Tokenizes the research corpus into fixed-length packed causal-LM blocks "
        "for full-weight continual pretraining. Uses the selected base model's "
        "tokenizer and writes a Hugging Face Arrow dataset to `data/processed/`."
    )

    with gr.Row():
        prep_model = gr.Dropdown(
            choices=_base_models(),
            value=None,
            label="Base model / tokenizer",
            scale=5,
            interactive=True,
        )
        prep_rescan = gr.Button("🔄 Rescan", scale=1)

    prep_input = gr.Textbox(
        value=_DEFAULT_CORPUS,
        label="Input corpus JSON",
        interactive=True,
    )
    prep_output = gr.Textbox(
        value="",
        label="Output dataset directory",
        placeholder="Auto-fills from selected model",
        interactive=True,
    )

    with gr.Row():
        prep_max_length = gr.Slider(
            512, 4096, value=2048, step=512,
            label="Sequence length", scale=3,
        )
        prep_val = gr.Slider(
            0.01, 0.20, value=0.05, step=0.01,
            label="Validation split", scale=2,
        )

    with gr.Row():
        prep_run = gr.Button("▶ Prepare Data", variant="primary", scale=3)
        prep_cancel = gr.Button("⏹ Cancel", variant="stop", scale=1)
        prep_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2,
        )

    prep_log = gr.Textbox(
        label="Log output", lines=12, max_lines=200,
        interactive=False, autoscroll=True,
    )

    def _submit(model, input_file, output_dir, max_length, val_ratio):
        if not model or model.startswith("("):
            gr.Warning("Select a base model before running.")
            return "⬜ Idle", ""
        if _runner_prepare_pretrain.get_status().name == "RUNNING":
            gr.Warning("Pretraining data preparation is already running.")
            return _status_label(_runner_prepare_pretrain), ""
        final_output = output_dir.strip() or _default_pretrain_data_output(model)
        _runner_prepare_pretrain.reset()
        _runner_prepare_pretrain.submit(
            _prepare_pretrain_data_job,
            model,
            input_file,
            final_output,
            int(max_length),
            float(val_ratio),
        )
        return _status_label(_runner_prepare_pretrain), ""

    prep_model.change(
        fn=lambda model: _default_pretrain_data_output(model) if model and not model.startswith("(") else "",
        inputs=[prep_model],
        outputs=[prep_output],
    )
    prep_run.click(
        fn=_submit,
        inputs=[prep_model, prep_input, prep_output, prep_max_length, prep_val],
        outputs=[prep_status, prep_log],
    )
    prep_cancel.click(
        fn=lambda: (_runner_prepare_pretrain.cancel(), _status_label(_runner_prepare_pretrain))[1],
        outputs=[prep_status],
    )
    prep_rescan.click(
        fn=lambda: gr.Dropdown(choices=_base_models(), interactive=True),
        outputs=[prep_model],
    )
    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_prepare_pretrain),
        outputs=[prep_log, prep_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sub-tab 2 — Continual pretraining
# ──────────────────────────────────────────────────────────────────────────────

def _build_pretrain() -> None:
    gr.Markdown(
        "### Step 2 — Continual Pretraining\n"
        "Runs full-weight domain adaptation on a local base model using the "
        "processed dataset from Step 1. This wraps `training/pretrain_transformers.py` "
        "rather than the lightweight profiler."
    )

    with gr.Row():
        pre_model = gr.Dropdown(
            choices=_base_models(),
            value=None,
            label="Base model",
            scale=5,
            interactive=True,
        )
        pre_model_rescan = gr.Button("🔄 Rescan", scale=1)

    with gr.Row():
        pre_data = gr.Dropdown(
            choices=_processed_pretrain_dirs(),
            value=None,
            label="Processed dataset",
            scale=5,
            interactive=True,
        )
        pre_data_rescan = gr.Button("🔄 Rescan", scale=1)

    pre_output = gr.Textbox(
        value="",
        label="Output trained-model directory",
        placeholder="Auto-fills from selected model",
        interactive=True,
    )

    with gr.Row():
        pre_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs", scale=2)
        pre_train_batch = gr.Slider(1, 8, value=2, step=1, label="Train batch", scale=2)
        pre_eval_batch = gr.Slider(1, 8, value=2, step=1, label="Eval batch", scale=2)
        pre_grad = gr.Slider(1, 32, value=8, step=1, label="Grad accum", scale=2)

    with gr.Row():
        pre_lr = gr.Textbox(value="2e-5", label="Learning rate", scale=2)
        pre_optim = gr.Dropdown(
            choices=["adamw_torch", "adamw_torch_fused", "paged_adamw_8bit", "adamw_bnb_8bit"],
            value="adamw_torch",
            label="Optimizer",
            scale=3,
            interactive=True,
        )

    with gr.Row():
        pre_run = gr.Button("▶ Run Continual Pretraining", variant="primary", scale=3)
        pre_cancel = gr.Button("⏹ Cancel", variant="stop", scale=1)
        pre_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2,
        )

    pre_log = gr.Textbox(
        label="Log output", lines=14, max_lines=300,
        interactive=False, autoscroll=True,
    )

    def _submit(model, data_dir, output_dir, epochs, train_batch, eval_batch, grad_accum, lr, optim):
        if not model or model.startswith("("):
            gr.Warning("Select a base model before running.")
            return "⬜ Idle", ""
        if not data_dir or data_dir.startswith("("):
            gr.Warning("Select a processed dataset before running.")
            return "⬜ Idle", ""
        if _runner_pretrain.get_status().name == "RUNNING":
            gr.Warning("Continual pretraining is already running.")
            return _status_label(_runner_pretrain), ""
        final_output = output_dir.strip() or _default_pretrain_model_output(model)
        _runner_pretrain.reset()
        _runner_pretrain.submit(
            _pretrain_job,
            model,
            data_dir,
            final_output,
            int(epochs),
            int(train_batch),
            int(eval_batch),
            int(grad_accum),
            lr,
            optim,
        )
        return _status_label(_runner_pretrain), ""

    pre_model.change(
        fn=lambda model: _default_pretrain_model_output(model) if model and not model.startswith("(") else "",
        inputs=[pre_model],
        outputs=[pre_output],
    )
    pre_run.click(
        fn=_submit,
        inputs=[
            pre_model,
            pre_data,
            pre_output,
            pre_epochs,
            pre_train_batch,
            pre_eval_batch,
            pre_grad,
            pre_lr,
            pre_optim,
        ],
        outputs=[pre_status, pre_log],
    )
    pre_cancel.click(
        fn=lambda: (_runner_pretrain.cancel(), _status_label(_runner_pretrain))[1],
        outputs=[pre_status],
    )
    pre_model_rescan.click(
        fn=lambda: gr.Dropdown(choices=_base_models(), interactive=True),
        outputs=[pre_model],
    )
    pre_data_rescan.click(
        fn=lambda: gr.Dropdown(choices=_processed_pretrain_dirs(), interactive=True),
        outputs=[pre_data],
    )
    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_pretrain),
        outputs=[pre_log, pre_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sub-tab 3 — Prepare instruction data
# ──────────────────────────────────────────────────────────────────────────────

def _build_prepare_instruction_data() -> None:
    detected = _presplit_instruction_dirs()
    detected_hint = ", ".join(detected[:3]) if detected else "none detected"

    gr.Markdown(
        "### Step 3 — Prepare Instruction Data\n"
        "Builds the instruction-training dataset for a selected base or pretrained model. "
        "By default this downloads and mixes Open-Orca + Dolly. If you provide a local "
        "pre-split JSONL directory, that is used instead.\n\n"
        f"Detected local pre-split datasets: `{detected_hint}`"
    )

    with gr.Row():
        instprep_model = gr.Dropdown(
            choices=_base_and_pretrained_models(),
            value=None,
            label="Target model / tokenizer",
            scale=5,
            interactive=True,
        )
        instprep_model_rescan = gr.Button("🔄 Rescan", scale=1)

    instprep_output = gr.Textbox(
        value="",
        label="Output instruction-dataset directory",
        placeholder="Auto-fills from selected model",
        interactive=True,
    )
    instprep_input_dir = gr.Textbox(
        value="",
        label="Optional local pre-split JSONL directory",
        placeholder="Leave blank to download Orca + Dolly, or use e.g. data/Instruct-FTD/v3_run",
        interactive=True,
    )

    with gr.Row():
        instprep_max_length = gr.Slider(
            512, 4096, value=2048, step=512,
            label="Max token length", scale=2,
        )
        instprep_val = gr.Slider(
            0.01, 0.20, value=0.05, step=0.01,
            label="Validation split", scale=2,
        )
        instprep_max_samples = gr.Textbox(
            value="5000",
            label="Max samples (blank = all)",
            scale=2,
        )

    with gr.Row():
        instprep_run = gr.Button("▶ Prepare Instruction Data", variant="primary", scale=3)
        instprep_cancel = gr.Button("⏹ Cancel", variant="stop", scale=1)
        instprep_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2,
        )

    instprep_log = gr.Textbox(
        label="Log output", lines=12, max_lines=220,
        interactive=False, autoscroll=True,
    )

    def _submit(model, output_dir, input_dir, max_length, max_samples, val_ratio):
        if not model or model.startswith("("):
            gr.Warning("Select a base or pretrained model before running.")
            return "⬜ Idle", ""
        if _runner_prepare_instruct.get_status().name == "RUNNING":
            gr.Warning("Instruction-data preparation is already running.")
            return _status_label(_runner_prepare_instruct), ""
        final_output = output_dir.strip() or _default_instruction_data_output(model)
        _runner_prepare_instruct.reset()
        _runner_prepare_instruct.submit(
            _prepare_instruction_data_job,
            model,
            final_output,
            input_dir,
            int(max_length),
            max_samples,
            float(val_ratio),
        )
        return _status_label(_runner_prepare_instruct), ""

    instprep_model.change(
        fn=lambda model: _default_instruction_data_output(model) if model and not model.startswith("(") else "",
        inputs=[instprep_model],
        outputs=[instprep_output],
    )
    instprep_run.click(
        fn=_submit,
        inputs=[
            instprep_model,
            instprep_output,
            instprep_input_dir,
            instprep_max_length,
            instprep_max_samples,
            instprep_val,
        ],
        outputs=[instprep_status, instprep_log],
    )
    instprep_cancel.click(
        fn=lambda: (_runner_prepare_instruct.cancel(), _status_label(_runner_prepare_instruct))[1],
        outputs=[instprep_status],
    )
    instprep_model_rescan.click(
        fn=lambda: gr.Dropdown(choices=_base_and_pretrained_models(), interactive=True),
        outputs=[instprep_model],
    )
    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_prepare_instruct),
        outputs=[instprep_log, instprep_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sub-tab 4 — Instruction fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def _build_instruction_finetune() -> None:
    gr.Markdown(
        "### Step 4 — Instruction Fine-Tuning\n"
        "Runs the full instruction fine-tuning stage on a selected base or "
        "pretrained model using a prepared instruction dataset from Step 3."
    )

    with gr.Row():
        inst_model = gr.Dropdown(
            choices=_base_and_pretrained_models(),
            value=None,
            label="Source model",
            scale=5,
            interactive=True,
        )
        inst_model_rescan = gr.Button("🔄 Rescan", scale=1)

    with gr.Row():
        inst_data = gr.Dropdown(
            choices=_instruction_data_dirs(),
            value=None,
            label="Instruction dataset",
            scale=5,
            interactive=True,
        )
        inst_data_rescan = gr.Button("🔄 Rescan", scale=1)

    inst_output = gr.Textbox(
        value="",
        label="Output instruction-model directory",
        placeholder="Auto-fills from selected model",
        interactive=True,
    )

    with gr.Row():
        inst_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs", scale=2)
        inst_batch = gr.Slider(1, 8, value=2, step=1, label="Batch size", scale=2)
        inst_grad = gr.Slider(1, 32, value=8, step=1, label="Grad accum", scale=2)
        inst_lr = gr.Textbox(value="2e-5", label="Learning rate", scale=2)

    with gr.Row():
        inst_run = gr.Button("▶ Run Instruction FT", variant="primary", scale=3)
        inst_cancel = gr.Button("⏹ Cancel", variant="stop", scale=1)
        inst_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2,
        )

    inst_log = gr.Textbox(
        label="Log output", lines=14, max_lines=300,
        interactive=False, autoscroll=True,
    )

    def _submit(model, data_dir, output_dir, epochs, batch, grad_accum, lr):
        if not model or model.startswith("("):
            gr.Warning("Select a source model before running.")
            return "⬜ Idle", ""
        if not data_dir or data_dir.startswith("("):
            gr.Warning("Select an instruction dataset before running.")
            return "⬜ Idle", ""
        if _runner_instruct.get_status().name == "RUNNING":
            gr.Warning("Instruction fine-tuning is already running.")
            return _status_label(_runner_instruct), ""
        final_output = output_dir.strip() or _default_instruction_model_output(model)
        _runner_instruct.reset()
        _runner_instruct.submit(
            _instruction_finetune_job,
            model,
            data_dir,
            final_output,
            int(epochs),
            int(batch),
            int(grad_accum),
            lr,
        )
        return _status_label(_runner_instruct), ""

    inst_model.change(
        fn=lambda model: _default_instruction_model_output(model) if model and not model.startswith("(") else "",
        inputs=[inst_model],
        outputs=[inst_output],
    )
    inst_run.click(
        fn=_submit,
        inputs=[inst_model, inst_data, inst_output, inst_epochs, inst_batch, inst_grad, inst_lr],
        outputs=[inst_status, inst_log],
    )
    inst_cancel.click(
        fn=lambda: (_runner_instruct.cancel(), _status_label(_runner_instruct))[1],
        outputs=[inst_status],
    )
    inst_model_rescan.click(
        fn=lambda: gr.Dropdown(choices=_base_and_pretrained_models(), interactive=True),
        outputs=[inst_model],
    )
    inst_data_rescan.click(
        fn=lambda: gr.Dropdown(choices=_instruction_data_dirs(), interactive=True),
        outputs=[inst_data],
    )
    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_instruct),
        outputs=[inst_log, inst_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sub-tab 5 — Outputs viewer
# ──────────────────────────────────────────────────────────────────────────────

def _scan_training_outputs() -> str:
    lines: list[str] = []

    def _list_dirs(path: Path, label: str) -> None:
        if not path.is_dir():
            lines.append(f"📁 {label}/  (not created yet)")
            return
        entries = [p for p in sorted(path.iterdir()) if p.is_dir()]
        lines.append(f"📁 {label}/  ({len(entries)} item{'s' if len(entries) != 1 else ''})")
        for p in entries:
            lines.append(f"   • {p.name}")

    _list_dirs(_PROCESSED_DATA_DIR, "data/processed")
    lines.append("")
    _list_dirs(_INSTRUCTION_DATA_DIR, "data/instruction")
    lines.append("")
    _list_dirs(_PRETRAINED_MODELS_DIR, "models/pretrained")
    lines.append("")
    _list_dirs(_INSTRUCTION_MODELS_DIR, "models/instruction")

    return "\n".join(lines)


def _build_outputs_viewer() -> None:
    gr.Markdown(
        "### 📂 Training Outputs\n"
        "Quick view of datasets and trained models created by the full training pipeline."
    )

    outputs_box = gr.Textbox(
        value=_scan_training_outputs,
        label="Training artifacts",
        lines=18,
        max_lines=40,
        interactive=False,
    )
    refresh_btn = gr.Button("🔄 Refresh")
    refresh_btn.click(fn=_scan_training_outputs, outputs=[outputs_box])
    gr.Timer(value=10).tick(fn=_scan_training_outputs, outputs=[outputs_box])


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def build_training_tab() -> None:
    with gr.Tab("🏗️ Training"):
        with gr.Tabs():
            with gr.Tab("📦 1. Prepare CPT Data"):
                _build_prepare_pretrain_data()
            with gr.Tab("🔥 2. Continual Pretrain"):
                _build_pretrain()
            with gr.Tab("📝 3. Prepare Instruction Data"):
                _build_prepare_instruction_data()
            with gr.Tab("🎯 4. Instruction FT"):
                _build_instruction_finetune()
            with gr.Tab("📂 Outputs"):
                _build_outputs_viewer()
