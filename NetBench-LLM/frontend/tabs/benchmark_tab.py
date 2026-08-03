"""
frontend/tabs/benchmark_tab.py
──────────────────────────────
Benchmark tab for the LLM Training Dashboard.

Wraps the 3-phase HPN Q&A benchmark pipeline:

  Phase 1 — evaluation/hpn_qa_benchmark.py
              Select model(s) — local HuggingFace paths OR API model names
              (gpt-4o, gemini-2.5-pro, etc.) — and run answer generation.
              Supports resume: re-running skips already-answered questions.
              Output: outputs/evaluations/answers/hpn_answers_<model>.xlsx

  Phase 2 — evaluation/judge_responses.py
              Select Phase 1 answer files, score with a Gemini or OpenAI judge.
              Output: outputs/evaluations/judged/hpn_judged_<model>_by_<judge>.xlsx

  Phase 3 — evaluation/benchmark_report.py
              Select two+ Phase 2 judged files (same judge), produce comparative report.
              Output: outputs/evaluations/reports/HPN_BENCHMARK_REPORT_by_<judge>.md + .xlsx

Each phase runs as a subprocess so that model loading/unloading is
self-contained and does not interfere with the ModelManager.  Stdout/stderr
are streamed line-by-line into a polling log box via JobRunner.

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

_ROOT        = Path(__file__).resolve().parent.parent.parent   # project root
_PYTHON      = sys.executable

_ANSWERS_DIR   = _ROOT / "outputs" / "evaluations" / "answers"
_JUDGED_DIR    = _ROOT / "outputs" / "evaluations" / "judged"
_FILTERED_DIR  = _ROOT / "outputs" / "evaluations" / "filtered"
_REPORTS_DIR   = _ROOT / "outputs" / "evaluations" / "reports"
_MODELS_DIR    = _ROOT / "models"

_DEFAULT_BENCHMARK       = str(_ROOT / "data" / "prompts" / "hpn_qa_benchmark_v4_general_skills.json")
_DEFAULT_GEMINI_KEYFILE  = str(_ROOT / "gemini_api_key.txt")
_DEFAULT_OPENAI_KEYFILE  = str(_ROOT / "openai_api_key.txt")

# Prefixes used to detect OpenAI models vs Gemini
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")

# ──────────────────────────────────────────────────────────────────────────────
# JobRunner instances (one per phase — allow concurrent phase runs)
# ──────────────────────────────────────────────────────────────────────────────

_runner_p1     = make_runner("bench_phase1")
_runner_p2     = make_runner("bench_phase2")
_runner_p3     = make_runner("bench_phase3")
_runner_filter = make_runner("bench_filter")

# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _instruction_models() -> list[str]:
    """Return model choices: common API model names first, then local models on disk."""
    # API model presets — shown at the top for easy selection
    api_presets = [
        "gpt-4o",
        "gpt-4.1",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    # Local instruction-tuned / LoRA-merged / pretrained / base models on disk
    local = []
    for sub in ("instruction", "lora-merged", "profiled", "pretrained", "base"):
        d = _MODELS_DIR / sub
        if d.is_dir():
            local += [
                str(p.relative_to(_ROOT))
                for p in sorted(d.iterdir())
                if p.is_dir()
            ]

    return api_presets + (local or ["(no local models found)"])


def _answer_files() -> list[str]:
    """Return relative paths of answer Excel files from Phase 1."""
    if not _ANSWERS_DIR.is_dir():
        return []
    return [
        str(p.relative_to(_ROOT))
        for p in sorted(_ANSWERS_DIR.glob("hpn_answers_*.xlsx"))
    ]


def _judged_files() -> list[str]:
    """Return relative paths of judged Excel files from Phase 2."""
    if not _JUDGED_DIR.is_dir():
        return []
    return [
        str(p.relative_to(_ROOT))
        for p in sorted(_JUDGED_DIR.glob("hpn_judged_*.xlsx"))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess-based job functions  (run inside JobRunner threads)
# ──────────────────────────────────────────────────────────────────────────────

def _run_subprocess(cmd: list[str], cancel_event=None) -> None:
    """
    Run *cmd* as a subprocess from the project root.
    Streams stdout+stderr line-by-line via print() so JobRunner captures them.
    On non-zero exit, echoes the last 20 lines with a ❌ prefix, then raises.
    """
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
        raise RuntimeError(
            f"Subprocess exited with code {proc.returncode}"
        )


def _phase1_job(
    model_paths: list[str],
    benchmark_file: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    gemini_key_file: str,
    openai_key_file: str,
    rpm: int,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON,
        "evaluation/hpn_qa_benchmark.py",
        "--model_path", *model_paths,
        "--benchmark",          benchmark_file,
        "--output_dir",         str(_ANSWERS_DIR),
        "--max_new_tokens",     str(max_new_tokens),
        "--temperature",        str(temperature),
        "--top_p",              str(top_p),
        "--gemini_api_key_file", gemini_key_file,
        "--openai_api_key_file", openai_key_file,
        "--requests_per_minute", str(rpm),
    ]
    _run_subprocess(cmd, _cancel_event)


def _phase2_job(
    answer_files: list[str],
    gemini_key_file: str,
    openai_key_file: str,
    judge_model: str,
    rpm: int,
    _cancel_event=None,
) -> None:
    # answer_files are relative paths — prepend root
    abs_answers = [str(_ROOT / f) for f in answer_files]
    cmd = [
        _PYTHON,
        "evaluation/judge_responses.py",
        "--answer_files",        *abs_answers,
        "--gemini_api_key_file",  gemini_key_file,
        "--openai_api_key_file",  openai_key_file,
        "--judge_model",          judge_model,
        "--output_dir",           str(_JUDGED_DIR),
        "--requests_per_minute",  str(rpm),
    ]
    _run_subprocess(cmd, _cancel_event)


def _phase3_job(
    judged_files: list[str],
    _cancel_event=None,
) -> None:
    abs_judged = [str(_ROOT / f) for f in judged_files]
    cmd = [
        _PYTHON,
        "evaluation/benchmark_report.py",
        "--judged_files", *abs_judged,
        "--output_dir",   str(_REPORTS_DIR),
    ]
    _run_subprocess(cmd, _cancel_event)


def _filter_job(
    judged_dir: str,
    output_dir: str,
    good_threshold: float,
    bad_threshold: float,
    _cancel_event=None,
) -> None:
    cmd = [
        _PYTHON,
        "evaluation/filter_judged.py",
        "--judged_dir",      judged_dir,
        "--output_dir",      output_dir,
        "--good_threshold",  str(good_threshold),
        "--bad_threshold",   str(bad_threshold),
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
    s = runner.get_status().name   # "IDLE" / "RUNNING" / "DONE" / "ERROR"
    return _STATUS_ICONS.get(s, s)


def _poll_logs(runner) -> tuple[str, str]:
    """
    Return (accumulated_log_text, status_label) for a runner.

    read_logs() drains the live queue so it stays bounded; the full
    accumulated log is retrieved from the internal deque via get_full_log()
    so the textbox always shows the COMPLETE log, not just the lines since
    the last poll.
    """
    runner.read_logs()          # drain queue → keeps it from filling up
    return runner.get_full_log(), _status_label(runner)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 panel
# ──────────────────────────────────────────────────────────────────────────────

def _build_phase1() -> None:
    gr.Markdown(
        "### Phase 1 — Answer Generation\n"
        "Supports **local HuggingFace models** and **API models** (GPT-4o, Gemini) "
        "in the same run.  Select from the dropdown — API presets (`gpt-4o`, "
        "`gemini-2.5-pro`, etc.) are listed at the top; local models discovered "
        "on disk follow.  You can also type any model name directly.  "
        "Local models are loaded and unloaded sequentially; API models use the "
        "configured key files with rate limiting.\n\n"
        "**Resume support**: re-running with the same model skips already-answered "
        "questions and regenerates only missing or empty answers."
    )

    with gr.Row():
        p1_models = gr.Dropdown(
            choices=_instruction_models(),
            value=None,
            label="Model(s) — local path or API name (gpt-4o, gemini-2.5-pro, …)",
            multiselect=True,
            allow_custom_value=True,
            scale=5,
            interactive=True,
        )
        p1_rescan = gr.Button("🔄 Rescan", scale=1)

    p1_benchmark = gr.Textbox(
        value=_DEFAULT_BENCHMARK,
        label="Benchmark JSON file",
        interactive=True,
    )

    with gr.Row():
        p1_max_tokens = gr.Slider(
            64, 1024, value=512, step=64,
            label="Max new tokens (local & OpenAI; Gemini adds 8192 thinking buffer)",
            scale=3,
        )
        p1_temperature = gr.Slider(
            0.0, 1.5, value=0.0, step=0.05,
            label="Temperature (0.0 = deterministic; ignored for Gemini thinking models)",
            scale=3,
        )
        p1_top_p = gr.Slider(
            0.1, 1.0, value=0.9, step=0.05,
            label="Top-p (local models only)",
            scale=2,
        )

    gr.Markdown("**API credentials** — required only when running API models:")
    with gr.Row():
        p1_gemini_key = gr.Textbox(
            value=_DEFAULT_GEMINI_KEYFILE,
            label="Gemini API key file",
            scale=3,
        )
        p1_openai_key = gr.Textbox(
            value=_DEFAULT_OPENAI_KEYFILE,
            label="OpenAI API key file",
            scale=3,
        )
        p1_rpm = gr.Slider(
            1, 60, value=15, step=1,
            label="API requests / minute",
            scale=2,
        )

    with gr.Row():
        p1_run    = gr.Button("▶ Run Phase 1", variant="primary", scale=3)
        p1_cancel = gr.Button("⏹ Cancel",      variant="stop",    scale=1)
        p1_status = gr.Textbox(
            value="⬜ Idle", label="Status",
            interactive=False, scale=2,
        )

    p1_log = gr.Textbox(
        label="Log output",
        lines=12,
        max_lines=200,
        interactive=False,
        autoscroll=True,
    )

    # ── Wire callbacks ────────────────────────────────────────────
    def _submit_p1(models, benchmark, max_tok, temp, tp, gemini_key, openai_key, rpm):
        if not models:
            gr.Warning("Select at least one model before running.")
            return "⬜ Idle", ""
        if _runner_p1.get_status().name == "RUNNING":
            gr.Warning("Phase 1 is already running.")
            return _status_label(_runner_p1), ""
        _runner_p1.reset()
        _runner_p1.submit(
            _phase1_job,
            models, benchmark, int(max_tok), float(temp), float(tp),
            gemini_key, openai_key, int(rpm),
        )
        return _status_label(_runner_p1), ""

    def _cancel_p1():
        _runner_p1.cancel()
        return _status_label(_runner_p1)

    p1_run.click(
        fn=_submit_p1,
        inputs=[
            p1_models, p1_benchmark, p1_max_tokens, p1_temperature, p1_top_p,
            p1_gemini_key, p1_openai_key, p1_rpm,
        ],
        outputs=[p1_status, p1_log],
    )
    p1_cancel.click(fn=_cancel_p1, outputs=[p1_status])

    p1_rescan.click(
        fn=lambda: gr.Dropdown(choices=_instruction_models(), interactive=True),
        outputs=[p1_models],
    )

    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_p1),
        outputs=[p1_log, p1_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 panel
# ──────────────────────────────────────────────────────────────────────────────

def _build_phase2() -> None:
    gr.Markdown(
        "### Phase 2 — LLM-as-Judge\n"
        "Sends each (question, reference answer, model answer) triple to a "
        "judge LLM (Gemini or OpenAI) for scoring on Correctness, Completeness, "
        "Clarity, and Conciseness (1–5 each).  Rate limiting and exponential "
        "back-off are handled automatically."
    )

    with gr.Row():
        p2_files = gr.Dropdown(
            choices=_answer_files(),
            value=None,
            label="Answer file(s) from Phase 1",
            multiselect=True,
            scale=5,
            interactive=True,
        )
        p2_rescan = gr.Button("🔄 Rescan", scale=1)

    with gr.Row():
        p2_gemini_key = gr.Textbox(
            value=_DEFAULT_GEMINI_KEYFILE,
            label="Gemini API key file",
            scale=3,
        )
        p2_openai_key = gr.Textbox(
            value=_DEFAULT_OPENAI_KEYFILE,
            label="OpenAI API key file",
            scale=3,
        )

    with gr.Row():
        p2_judge_model = gr.Dropdown(
            choices=[
                "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash",
                "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            ],
            value="gemini-2.5-flash",
            label="Judge model",
            allow_custom_value=True,
            scale=4,
        )
        p2_rpm = gr.Slider(
            1, 60, value=15, step=1,
            label="Requests / minute",
            scale=3,
        )

    with gr.Row():
        p2_run    = gr.Button("▶ Run Phase 2", variant="primary", scale=3)
        p2_cancel = gr.Button("⏹ Cancel",      variant="stop",    scale=1)
        p2_status = gr.Textbox(
            value="⬜ Idle", label="Status",
            interactive=False, scale=2,
        )

    p2_log = gr.Textbox(
        label="Log output",
        lines=12,
        max_lines=200,
        interactive=False,
        autoscroll=True,
    )

    # ── Wire callbacks ────────────────────────────────────────────
    def _submit_p2(files, gemini_key, openai_key, judge_model, rpm):
        if not files:
            gr.Warning("Select at least one answer file before running.")
            return "⬜ Idle", ""
        if _runner_p2.get_status().name == "RUNNING":
            gr.Warning("Phase 2 is already running.")
            return _status_label(_runner_p2), ""
        _runner_p2.reset()
        _runner_p2.submit(
            _phase2_job,
            files, gemini_key, openai_key, judge_model, int(rpm),
        )
        return _status_label(_runner_p2), ""

    def _cancel_p2():
        _runner_p2.cancel()
        return _status_label(_runner_p2)

    p2_run.click(
        fn=_submit_p2,
        inputs=[p2_files, p2_gemini_key, p2_openai_key, p2_judge_model, p2_rpm],
        outputs=[p2_status, p2_log],
    )
    p2_cancel.click(fn=_cancel_p2, outputs=[p2_status])

    p2_rescan.click(
        fn=lambda: gr.Dropdown(choices=_answer_files(), interactive=True),
        outputs=[p2_files],
    )

    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_p2),
        outputs=[p2_log, p2_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 panel
# ──────────────────────────────────────────────────────────────────────────────

def _build_phase3() -> None:
    gr.Markdown(
        "### Phase 3 — Comparative Report\n"
        "Reads two or more Phase 2 judged files and produces a side-by-side "
        "comparison report (Excel + Markdown) broken down by category and "
        "difficulty. Requires at least two judged files."
    )

    with gr.Row():
        p3_files = gr.Dropdown(
            choices=_judged_files(),
            value=None,
            label="Judged file(s) from Phase 2 (select ≥ 2)",
            multiselect=True,
            scale=5,
            interactive=True,
        )
        p3_rescan = gr.Button("🔄 Rescan", scale=1)

    with gr.Row():
        p3_run    = gr.Button("▶ Run Phase 3", variant="primary", scale=3)
        p3_cancel = gr.Button("⏹ Cancel",      variant="stop",    scale=1)
        p3_status = gr.Textbox(
            value="⬜ Idle", label="Status",
            interactive=False, scale=2,
        )

    p3_log = gr.Textbox(
        label="Log output",
        lines=12,
        max_lines=200,
        interactive=False,
        autoscroll=True,
    )

    # ── Wire callbacks ────────────────────────────────────────────
    def _submit_p3(files):
        if not files or len(files) < 2:
            gr.Warning("Select at least 2 judged files for a comparative report.")
            return "⬜ Idle", ""
        if _runner_p3.get_status().name == "RUNNING":
            gr.Warning("Phase 3 is already running.")
            return _status_label(_runner_p3), ""
        _runner_p3.reset()
        _runner_p3.submit(_phase3_job, files)
        return _status_label(_runner_p3), ""

    def _cancel_p3():
        _runner_p3.cancel()
        return _status_label(_runner_p3)

    p3_run.click(
        fn=_submit_p3,
        inputs=[p3_files],
        outputs=[p3_status, p3_log],
    )
    p3_cancel.click(fn=_cancel_p3, outputs=[p3_status])

    p3_rescan.click(
        fn=lambda: gr.Dropdown(choices=_judged_files(), interactive=True),
        outputs=[p3_files],
    )

    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_p3),
        outputs=[p3_log, p3_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Results panel — show existing output files + inline report preview
# ──────────────────────────────────────────────────────────────────────────────

def _build_results() -> None:
    gr.Markdown(
        "### 📂 Existing Output Files\n"
        "All files currently in `outputs/evaluations/`.  "
        "Click **Refresh** after a phase completes."
    )

    results_box = gr.Textbox(
        value=_scan_outputs,
        label="Output files",
        lines=16,
        max_lines=40,
        interactive=False,
    )

    with gr.Row():
        refresh_btn  = gr.Button("🔄 Refresh", scale=1)
        report_md_dd = gr.Dropdown(
            choices=_md_report_choices(),
            value=None,
            label="Preview Markdown report",
            scale=5,
            interactive=True,
        )

    report_preview = gr.Markdown(value="")

    refresh_btn.click(
        fn=lambda: (
            _scan_outputs(),
            gr.Dropdown(choices=_md_report_choices(), interactive=True),
        ),
        outputs=[results_box, report_md_dd],
    )

    report_md_dd.change(
        fn=_load_md_report,
        inputs=[report_md_dd],
        outputs=[report_preview],
    )

    # Auto-refresh every 10 s
    gr.Timer(value=10).tick(
        fn=lambda: (
            _scan_outputs(),
            gr.Dropdown(choices=_md_report_choices(), interactive=True),
        ),
        outputs=[results_box, report_md_dd],
    )


def _scan_outputs() -> str:
    """Return a formatted directory listing of evaluation outputs."""
    lines = []
    base = _ROOT / "outputs" / "evaluations"
    if not base.is_dir():
        return "(No outputs yet. Run the benchmark pipeline to generate results.)"

    for sub in ("answers", "judged", "filtered", "reports"):
        d = base / sub
        if not d.is_dir():
            continue
        files = sorted(d.iterdir())
        count = len(files)
        lines.append(
            f"📁 {sub}/  ({count} file{'s' if count != 1 else ''})"
        )
        for f in files:
            size_kb = f.stat().st_size / 1024
            lines.append(f"   • {f.name}  ({size_kb:.0f} KB)")
    return "\n".join(lines) if lines else "(No output files found.)"


def _md_report_choices() -> list[str]:
    if not _REPORTS_DIR.is_dir():
        return []
    return [
        str(p.relative_to(_ROOT))
        for p in sorted(_REPORTS_DIR.glob("*.md"))
    ]


def _load_md_report(rel_path: str) -> str:
    if not rel_path:
        return ""
    p = _ROOT / rel_path
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"_File not found: {rel_path}_"


# ──────────────────────────────────────────────────────────────────────────────
# Filter panel
# ──────────────────────────────────────────────────────────────────────────────

def _build_filter() -> None:
    gr.Markdown(
        "### 🔍 Filter Judged Results\n"
        "Reads judged Excel files and produces filtered copies with two "
        "sheets: **Good Answers** (overall score above threshold) and "
        "**Bad Answers** (overall score below threshold)."
    )

    with gr.Row():
        flt_judged_dir = gr.Textbox(
            value=str(_JUDGED_DIR),
            label="Judged directory (input)",
            scale=5,
        )
        flt_output_dir = gr.Textbox(
            value=str(_FILTERED_DIR),
            label="Filtered directory (output)",
            scale=5,
        )

    with gr.Row():
        flt_good = gr.Slider(
            1.0, 5.0, value=4.0, step=0.1,
            label="Good threshold (overall >)", scale=3,
        )
        flt_bad = gr.Slider(
            0.0, 3.0, value=1.3, step=0.1,
            label="Bad threshold (overall <)", scale=3,
        )

    with gr.Row():
        flt_run    = gr.Button("▶ Run Filter", variant="primary", scale=3)
        flt_cancel = gr.Button("⏹ Cancel",     variant="stop",    scale=1)
        flt_status = gr.Textbox(
            value="⬜ Idle", label="Status",
            interactive=False, scale=2,
        )

    flt_log = gr.Textbox(
        label="Log output",
        lines=12,
        max_lines=200,
        interactive=False,
        autoscroll=True,
    )

    # ── Wire callbacks ────────────────────────────────────────────
    def _submit_filter(judged_dir, output_dir, good_thr, bad_thr):
        if _runner_filter.get_status().name == "RUNNING":
            gr.Warning("Filter is already running.")
            return _status_label(_runner_filter), ""
        _runner_filter.reset()
        _runner_filter.submit(
            _filter_job,
            judged_dir, output_dir, float(good_thr), float(bad_thr),
        )
        return _status_label(_runner_filter), ""

    def _cancel_filter():
        _runner_filter.cancel()
        return _status_label(_runner_filter)

    flt_run.click(
        fn=_submit_filter,
        inputs=[flt_judged_dir, flt_output_dir, flt_good, flt_bad],
        outputs=[flt_status, flt_log],
    )
    flt_cancel.click(fn=_cancel_filter, outputs=[flt_status])

    gr.Timer(value=2).tick(
        fn=lambda: _poll_logs(_runner_filter),
        outputs=[flt_log, flt_status],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint — called from app.py inside gr.Tabs()
# ──────────────────────────────────────────────────────────────────────────────

def build_benchmark_tab() -> None:
    """
    Build the Benchmark tab.
    Must be called inside a gr.Blocks() / gr.Tabs() context.
    """
    with gr.Tab("📊 Benchmark"):
        with gr.Tabs():
            with gr.Tab("⚙ Phase 1 — Generate Answers"):
                _build_phase1()
            with gr.Tab("🤖 Phase 2 — Judge"):
                _build_phase2()
            with gr.Tab("📈 Phase 3 — Report"):
                _build_phase3()
            with gr.Tab("🔍 Filter"):
                _build_filter()
            with gr.Tab("📂 Results"):
                _build_results()
