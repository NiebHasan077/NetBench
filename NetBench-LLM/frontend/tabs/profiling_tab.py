"""
frontend/tabs/profiling_tab.py
──────────────────────────────
Profiling tab for the LLM Training Dashboard.

Four sub-tabs:

  Training Profiler   — wraps profiling/profile_training.py
                        Profiles a full continual-pretrain or instruction
                        fine-tune run and saves the resulting model to
                        models/profiled/<name>.

  Inference Profiler  — wraps profiling/profile_inference.py
                        Benchmarks one or more models (TTFT, tokens/s,
                        ms/token, VRAM) over a suite of prompts.

  Report Generator    — wraps profiling/generate_profile_report.py
                        Reads all JSON files in outputs/profiling_results/
                        and generates Excel workbooks + PNG plots.

  Results Viewer      — Scans outputs/profiling_results/ for JSON files,
                        displays an inline summary table of key metrics,
                        and renders throughput / memory bar charts.

Each profiler runs as a subprocess inside a JobRunner worker thread so its
model loading / unloading is fully isolated from the Dashboard process.
Logs are streamed to a polling Textbox via gr.Timer.

Must be called inside a gr.Blocks() context.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_MPL_CONFIG_DIR = Path("/tmp/llm_training_mpl")
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be set before any other matplotlib import
import matplotlib.pyplot as plt
import gradio as gr

from frontend.core.job_runner import make_runner

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

_ROOT         = Path(__file__).resolve().parent.parent.parent
_PYTHON       = sys.executable

_RESULTS_DIR  = _ROOT / "outputs" / "profiling_results"
_REPORTS_DIR  = _ROOT / "outputs" / "profiling_reports"
_MODELS_DIR   = _ROOT / "models"
_DATA_DIR     = _ROOT / "data"

# ──────────────────────────────────────────────────────────────────────────────
# JobRunner instances
# ──────────────────────────────────────────────────────────────────────────────

_runner_train  = make_runner("profile_training")
_runner_infer  = make_runner("profile_inference")
_runner_report = make_runner("profile_report")

# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _all_models(subtypes: tuple[str, ...] = ("base", "pretrained", "instruction", "lora-merged", "profiled")) -> list[str]:
    out = []
    for sub in subtypes:
        d = _MODELS_DIR / sub
        if d.is_dir():
            out += [str(p.relative_to(_ROOT)) for p in sorted(d.iterdir()) if p.is_dir()]
    return out or ["(no models found)"]


def _base_pretrained_models() -> list[str]:
    return _all_models(("base", "pretrained"))


def _data_dirs() -> list[str]:
    """Return processed + instruction dataset directories."""
    out = []
    for sub in ("processed", "instruction"):
        d = _DATA_DIR / sub
        if d.is_dir():
            out += [str(p.relative_to(_ROOT)) for p in sorted(d.iterdir()) if p.is_dir()]
    return out or ["(no data dirs found)"]


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess runner (same pattern as benchmark_tab)
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
    tail: list[str] = []   # rolling buffer of last 20 lines
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

def _training_job(
    mode: str,          # "pretrain" | "instruct"
    model_path: str,
    data_dir: str,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON, "profiling/profile_training.py",
        mode,
        "--model_path", model_path,
        "--data_dir",   data_dir,
        "--max_steps",  str(max_steps),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
    ]
    _run_subprocess(cmd, _cancel_event)


def _inference_job(
    model_paths: list[str],
    num_runs: int,
    warmup_runs: int,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON, "profiling/profile_inference.py",
        "--model_path", *model_paths,
        "--num_runs",    str(num_runs),
        "--warmup_runs", str(warmup_runs),
    ]
    _run_subprocess(cmd, _cancel_event)


def _report_job(
    filter_str: str,
    _cancel_event=None,
) -> None:
    cmd = [_PYTHON, "profiling/generate_profile_report.py"]
    if filter_str.strip():
        cmd += ["--filter", filter_str.strip()]
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


def _poll(runner) -> tuple[str, str]:
    runner.read_logs()
    return runner.get_full_log(), _status_label(runner)


# ──────────────────────────────────────────────────────────────────────────────
# Training profiler sub-tab
# ──────────────────────────────────────────────────────────────────────────────

def _build_training_profiler() -> None:
    gr.Markdown(
        "### Training Profiler\n"
        "Runs a short continual-pretraining or instruction fine-tuning pass "
        "and records throughput, VRAM, loss, and GPU utilisation. "
        "The profiled model is saved to `models/profiled/`."
    )

    with gr.Row():
        train_mode = gr.Radio(
            choices=["pretrain", "instruct"],
            value="pretrain",
            label="Mode",
            scale=2,
        )
        train_model = gr.Dropdown(
            choices=_base_pretrained_models(),
            value=None,
            label="Base / Pretrained Model",
            scale=4,
            interactive=True,
        )
        train_rescan_models = gr.Button("🔄", scale=1, min_width=48)

    with gr.Row():
        train_data = gr.Dropdown(
            choices=_data_dirs(),
            value=None,
            label="Dataset Directory",
            scale=4,
            interactive=True,
        )
        train_rescan_data = gr.Button("🔄", scale=1, min_width=48)

    with gr.Row():
        train_max_steps = gr.Slider(
            5, 500, value=50, step=5,
            label="Max steps (overrides epochs)", scale=3,
        )
        train_batch = gr.Slider(
            1, 8, value=2, step=1,
            label="Per-device batch size", scale=2,
        )
        train_grad_accum = gr.Slider(
            1, 32, value=8, step=1,
            label="Gradient accumulation steps", scale=2,
        )

    with gr.Row():
        train_run  = gr.Button("▶ Run Training Profiler", variant="primary", scale=3)
        train_stop = gr.Button("⏹ Cancel", variant="stop", scale=1)
        train_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2,
        )

    train_log = gr.Textbox(
        label="Log output", lines=14, max_lines=300,
        interactive=False, autoscroll=True,
    )

    def _submit_train(mode, model, data, max_steps, batch, grad_accum):
        if not model or model.startswith("("):
            gr.Warning("Select a model before running.")
            return "⬜ Idle", ""
        if not data or data.startswith("("):
            gr.Warning("Select a dataset directory before running.")
            return "⬜ Idle", ""
        if _runner_train.get_status().name == "RUNNING":
            gr.Warning("Training profiler is already running.")
            return _status_label(_runner_train), ""
        _runner_train.reset()
        _runner_train.submit(
            _training_job, mode, model, data,
            int(max_steps), int(batch), int(grad_accum),
        )
        return _status_label(_runner_train), ""

    train_run.click(
        fn=_submit_train,
        inputs=[train_mode, train_model, train_data,
                train_max_steps, train_batch, train_grad_accum],
        outputs=[train_status, train_log],
    )
    train_stop.click(
        fn=lambda: (_runner_train.cancel(), _status_label(_runner_train))[1],
        outputs=[train_status],
    )
    train_rescan_models.click(
        fn=lambda: gr.Dropdown(choices=_base_pretrained_models(), interactive=True),
        outputs=[train_model],
    )
    train_rescan_data.click(
        fn=lambda: gr.Dropdown(choices=_data_dirs(), interactive=True),
        outputs=[train_data],
    )
    gr.Timer(value=2).tick(
        fn=lambda: _poll(_runner_train),
        outputs=[train_log, train_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inference profiler sub-tab
# ──────────────────────────────────────────────────────────────────────────────

def _build_inference_profiler() -> None:
    gr.Markdown(
        "### Inference Profiler\n"
        "Benchmarks one or more models over a built-in prompt suite, measuring "
        "TTFT, tokens/s, ms/token, peak VRAM, and scaling behaviour vs. "
        "input and output length. Each model is loaded and unloaded in sequence."
    )

    with gr.Row():
        infer_models = gr.Dropdown(
            choices=_all_models(),
            value=None,
            label="Model(s) to benchmark",
            multiselect=True,
            scale=5,
            interactive=True,
        )
        infer_rescan = gr.Button("🔄 Rescan", scale=1)

    with gr.Row():
        infer_runs    = gr.Slider(1, 10, value=3, step=1,
                                  label="Runs per prompt", scale=3)
        infer_warmup  = gr.Slider(0, 5,  value=2, step=1,
                                  label="Warmup runs",     scale=3)

    with gr.Row():
        infer_run  = gr.Button("▶ Run Inference Profiler", variant="primary", scale=3)
        infer_stop = gr.Button("⏹ Cancel", variant="stop", scale=1)
        infer_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2,
        )

    infer_log = gr.Textbox(
        label="Log output", lines=14, max_lines=300,
        interactive=False, autoscroll=True,
    )

    def _submit_infer(models, num_runs, warmup):
        if not models or all(str(m).startswith("(") for m in models):
            gr.Warning("Select at least one model before running.")
            return "⬜ Idle", ""
        models = [m for m in models if not str(m).startswith("(")]
        if _runner_infer.get_status().name == "RUNNING":
            gr.Warning("Inference profiler is already running.")
            return _status_label(_runner_infer), ""
        _runner_infer.reset()
        _runner_infer.submit(
            _inference_job, models, int(num_runs), int(warmup),
        )
        return _status_label(_runner_infer), ""

    infer_run.click(
        fn=_submit_infer,
        inputs=[infer_models, infer_runs, infer_warmup],
        outputs=[infer_status, infer_log],
    )
    infer_stop.click(
        fn=lambda: (_runner_infer.cancel(), _status_label(_runner_infer))[1],
        outputs=[infer_status],
    )
    infer_rescan.click(
        fn=lambda: gr.Dropdown(choices=_all_models(), interactive=True),
        outputs=[infer_models],
    )
    gr.Timer(value=2).tick(
        fn=lambda: _poll(_runner_infer),
        outputs=[infer_log, infer_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Report generator sub-tab
# ──────────────────────────────────────────────────────────────────────────────

def _build_report_generator() -> None:
    gr.Markdown(
        "### Report Generator\n"
        "Reads all JSON files in `outputs/profiling_results/` and produces "
        "per-run Excel workbooks, PNG plots, and a cumulative "
        "`PROFILING_REPORT.md`.  Use the filter to target specific runs or "
        "model names."
    )

    rpt_filter = gr.Textbox(
        label="Filter (substring match on run/model name, leave blank for all)",
        placeholder="e.g. 8B  or  pretrain  or  leave blank",
        lines=1,
        interactive=True,
    )

    with gr.Row():
        rpt_run  = gr.Button("▶ Generate Report", variant="primary", scale=3)
        rpt_stop = gr.Button("⏹ Cancel", variant="stop", scale=1)
        rpt_status = gr.Textbox(
            value="⬜ Idle", label="Status", interactive=False, scale=2,
        )

    rpt_log = gr.Textbox(
        label="Log output", lines=12, max_lines=200,
        interactive=False, autoscroll=True,
    )

    def _submit_report(filter_str):
        if _runner_report.get_status().name == "RUNNING":
            gr.Warning("Report generator is already running.")
            return _status_label(_runner_report), ""
        _runner_report.reset()
        _runner_report.submit(_report_job, filter_str)
        return _status_label(_runner_report), ""

    rpt_run.click(
        fn=_submit_report,
        inputs=[rpt_filter],
        outputs=[rpt_status, rpt_log],
    )
    rpt_stop.click(
        fn=lambda: (_runner_report.cancel(), _status_label(_runner_report))[1],
        outputs=[rpt_status],
    )
    gr.Timer(value=2).tick(
        fn=lambda: _poll(_runner_report),
        outputs=[rpt_log, rpt_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Results viewer sub-tab
# ──────────────────────────────────────────────────────────────────────────────

def _load_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _summarise_inference(data: dict) -> dict:
    s = data.get("summary", {})
    ml = data.get("model_load", {})
    return {
        "type":           "inference",
        "model":          data.get("model_name", "?"),
        "device":         data.get("device", "?"),
        "timestamp":      data.get("timestamp", "")[:19],
        "load_time_s":    ml.get("load_time_s", ""),
        "avg_tok/s":      s.get("avg_tokens_per_second", ""),
        "avg_ms/tok":     s.get("avg_ms_per_token", ""),
        "avg_ttft_s":     s.get("avg_ttft_s", ""),
        "peak_vram_mb":   s.get("peak_gpu_vram_mb", ""),
        "peak_cpu_mb":    s.get("peak_cpu_ram_mb", ""),
    }


def _summarise_training(data: dict) -> dict:
    t = data.get("timing", {})
    tp = data.get("throughput", {})
    mem = data.get("memory", {})
    return {
        "type":           "training",
        "model":          data.get("config", {}).get("model_name", "?"),
        "mode":           data.get("mode", "?"),
        "timestamp":      data.get("timestamp", "")[:19],
        "total_train_s":  t.get("total_train_s", ""),
        "total_steps":    t.get("total_steps", ""),
        "tok/s":          tp.get("tokens_per_second", ""),
        "samples/s":      tp.get("samples_per_second", ""),
        "peak_vram_mb":   mem.get("peak_gpu_vram_mb", ""),
        "cpu_delta_mb":   mem.get("cpu_ram_delta_mb", ""),
    }


def _build_results_table() -> str:
    """Return a text table of all profiling results sorted by timestamp desc."""
    if not _RESULTS_DIR.is_dir():
        return "(No profiling results yet. Run a profiler to generate JSON files.)"

    files = sorted(_RESULTS_DIR.glob("*.json"), reverse=True)
    if not files:
        return "(No JSON files found in outputs/profiling_results/.)"

    rows = []
    for fp in files:
        d = _load_json_safe(fp)
        if d is None:
            continue
        if "summary" in d and "model_name" in d:     # inference
            s = _summarise_inference(d)
            rows.append(
                f"[inference] {s['model']}  |  device={s['device']}  |  "
                f"avg {s['avg_tok/s']} tok/s  |  {s['avg_ms/tok']} ms/tok  |  "
                f"ttft={s['avg_ttft_s']} s  |  vram={s['peak_vram_mb']} MB  |  "
                f"loaded in {s['load_time_s']} s  |  {s['timestamp']}"
            )
        elif "throughput" in d:                       # training
            s = _summarise_training(d)
            rows.append(
                f"[training/{s['mode']}] {s['model']}  |  "
                f"{s['total_steps']} steps in {s['total_train_s']} s  |  "
                f"{s['tok/s']} tok/s  |  {s['samples/s']} samples/s  |  "
                f"vram={s['peak_vram_mb']} MB  |  {s['timestamp']}"
            )
        else:
            rows.append(f"[unknown] {fp.name}")

    header = f"{'─'*90}\n"
    return header + f"\n{header}".join(rows) + f"\n{header}"


def _chart_data() -> tuple[Any, Any]:
    """
    Return two matplotlib Figure objects:
      fig_tput  — bar chart of avg tokens/s per model (inference runs)
      fig_vram  — bar chart of peak VRAM per profiling run

    Old figures are closed before creating new ones to prevent the matplotlib
    figure registry from growing unboundedly on each timer tick.
    """
    plt.close("all")   # release previously created figures — prevents memory leak

    if not _RESULTS_DIR.is_dir():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results yet", ha="center", va="center")
        return fig, fig

    files = sorted(_RESULTS_DIR.glob("*.json"))

    # Inference throughput
    infer_names: list[str] = []
    infer_tput:  list[float] = []
    # VRAM (all runs)
    vram_names: list[str] = []
    vram_vals:  list[float] = []

    for fp in files:
        d = _load_json_safe(fp)
        if d is None:
            continue
        short = fp.stem[:30]

        if "summary" in d and "model_name" in d:   # inference
            tps = d["summary"].get("avg_tokens_per_second")
            vram = d["summary"].get("peak_gpu_vram_mb")
            if isinstance(tps, (int, float)):
                infer_names.append(d.get("model_name", short)[:25])
                infer_tput.append(float(tps))
            if isinstance(vram, (int, float)):
                vram_names.append(short)
                vram_vals.append(float(vram))
        elif "throughput" in d:                    # training
            vram = d.get("memory", {}).get("peak_gpu_vram_mb")
            if isinstance(vram, (int, float)):
                vram_names.append(short)
                vram_vals.append(float(vram))

    # ── throughput chart ────────────────────────────────────────────────
    fig_tput, ax1 = plt.subplots(figsize=(max(5, len(infer_names) * 1.4 + 1), 4))
    if infer_names:
        bars = ax1.bar(range(len(infer_names)), infer_tput,
                       color="#4C72B0", edgecolor="white")
        ax1.set_xticks(range(len(infer_names)))
        ax1.set_xticklabels(infer_names, rotation=20, ha="right", fontsize=8)
        ax1.set_ylabel("Avg tokens / second")
        ax1.set_title("Inference Throughput by Model")
        ax1.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No inference results", ha="center", va="center")
        ax1.set_title("Inference Throughput (no data)")
    fig_tput.tight_layout()

    # ── VRAM chart ──────────────────────────────────────────────────────
    fig_vram, ax2 = plt.subplots(figsize=(max(5, len(vram_names) * 1.4 + 1), 4))
    if vram_names:
        colors = ["#DD8452" if n.startswith("training") else "#4C72B0"
                  for n in vram_names]
        bars2 = ax2.bar(range(len(vram_names)), vram_vals,
                        color=colors, edgecolor="white")
        ax2.set_xticks(range(len(vram_names)))
        ax2.set_xticklabels(vram_names, rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("Peak VRAM (MB)")
        ax2.set_title("Peak GPU VRAM per Profiling Run")
        ax2.bar_label(bars2, fmt="%.0f", padding=2, fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No VRAM data", ha="center", va="center")
        ax2.set_title("Peak GPU VRAM (no data)")
    fig_vram.tight_layout()

    return fig_tput, fig_vram


def _build_results_viewer() -> None:
    gr.Markdown(
        "### 📂 Results Viewer\n"
        "Summaries of all JSON files in `outputs/profiling_results/`.  "
        "Click **Refresh** after a profiler completes."
    )

    results_box = gr.Textbox(
        value=_build_results_table,
        label="Profiling results",
        lines=14,
        max_lines=40,
        interactive=False,
    )

    refresh_btn = gr.Button("🔄 Refresh", scale=1)

    with gr.Row():
        chart_tput = gr.Plot(label="Inference Throughput (tok/s)")
        chart_vram = gr.Plot(label="Peak GPU VRAM (MB)")

    def _refresh():
        fig_t, fig_v = _chart_data()
        return _build_results_table(), fig_t, fig_v

    # Populate charts on first load
    try:
        _ft, _fv = _chart_data()
        chart_tput.value = _ft
        chart_vram.value = _fv
    except Exception:
        pass

    refresh_btn.click(fn=_refresh, outputs=[results_box, chart_tput, chart_vram])

    # Auto-refresh every 10 s
    gr.Timer(value=10).tick(fn=_refresh, outputs=[results_box, chart_tput, chart_vram])


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint — called from app.py inside gr.Tabs()
# ──────────────────────────────────────────────────────────────────────────────

def build_profiling_tab() -> None:
    """
    Build the Profiling tab.
    Must be called inside a gr.Blocks() / gr.Tabs() context.
    """
    with gr.Tab("📈 Profiling"):
        with gr.Tabs():
            with gr.Tab("🏋 Training Profiler"):
                _build_training_profiler()
            with gr.Tab("⚡ Inference Profiler"):
                _build_inference_profiler()
            with gr.Tab("📄 Report Generator"):
                _build_report_generator()
            with gr.Tab("📊 Results Viewer"):
                _build_results_viewer()
