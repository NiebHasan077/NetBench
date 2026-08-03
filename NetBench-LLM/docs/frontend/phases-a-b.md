# Frontend Development — Phase A & B

**Branch:** `integrate-frontend`  
**Date:** March 23, 2026  
**Stack:** Python 3.11, Gradio 6.9, PyTorch (CUDA 12.8), HuggingFace Transformers

---

## Directory Structure

```
frontend/
├── app.py                    ← Main entry point
├── test_phase_a.py           ← Phase A validation script
├── core/
│   ├── model_manager.py      ← Model scanning, load/unload, VRAM tracking
│   └── job_runner.py         ← Background thread runner with log capture
└── tabs/
    ├── system_tab.py         ← GPU stats + model loader UI (Phase B)
    ├── chat_tab.py           ← Stub (Phase C)
    ├── benchmark_tab.py      ← Stub (Phase D)
    └── profiling_tab.py      ← Stub (Phase E)
```

---

## Phase A — Core Infrastructure

### `frontend/core/model_manager.py`

Central registry for all LLM models. Exposes a module-level singleton `manager`.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `ModelType` (Enum) | `BASE`, `PRETRAINED`, `INSTRUCT`, `PROFILED_BASE`, `PROFILED_INSTRUCT` |
| `ModelInfo` | Metadata for a model on disk (nickname, path, type, category) |
| `LoadedModel` | A loaded model — holds `model`, `tokenizer`, `vram_bytes_on_load` |
| `VRAMInfo` | Per-GPU VRAM stats with `.used_gb`, `.free_gb`, `.used_pct` |
| `ModelManager` | Thread-safe registry (load/unload behind `threading.Lock`) |

**Type inference from directory path:**
```
models/base/        → BASE
models/pretrained/  → PRETRAINED
models/instruction/ → INSTRUCT
models/profiled/    → PROFILED_BASE  or  PROFILED_INSTRUCT (if "instruct" in folder name)
```

**Public API:**
```python
from frontend.core.model_manager import manager

manager.list_available_models()          # → list[ModelInfo]
manager.load_model(nickname, max_loaded=2)  # → LoadedModel  (max 2 for compare)
manager.unload_model(nickname)           # → None
manager.unload_all()                     # → None
manager.get_loaded_models()             # → dict[str, LoadedModel]
manager.get_vram_usage()                # → list[VRAMInfo]
manager.is_loaded(nickname)             # → bool
```

**Load config:** `torch_dtype=bfloat16`, `device_map="auto"`, `low_cpu_mem_usage=True`  
**VRAM source:** torch allocated bytes (per-process accurate) + nvidia-smi (total/temp/util)  
**Error handling:** OOM raises `RuntimeError` with free VRAM per GPU and fix hint; limit exceeded raises `RuntimeError` with names of loaded models.

---

### `frontend/core/job_runner.py`

Runs long callables (benchmark phases, profiling) in a background `threading.Thread` and makes stdout/stderr available line-by-line for Gradio's polling pattern.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `JobStatus` (Enum) | `IDLE`, `RUNNING`, `DONE`, `ERROR` |
| `JobResult` | Holds `status`, `return_value`, `error_message`, `traceback_str` |
| `JobRunner` | One runner per logical job slot |
| `_QueueWriter` | Redirects `sys.stdout/stderr` inside the thread to a `queue.Queue` + deque |

**Public API:**
```python
from frontend.core.job_runner import JobRunner

runner = JobRunner(name="benchmark")
runner.submit(my_fn, arg1, arg2)   # starts background thread
runner.read_logs()                  # → list[str]  drain queued lines
runner.get_status()                 # → JobStatus
runner.get_full_log()               # → str  entire log since last reset
runner.get_result()                 # → JobResult | None
runner.cancel()                     # sets cancel_event
runner.reset()                      # back to IDLE for reuse
```

**Cancel protocol:** If the callable accepts `_cancel_event` in its signature, the runner auto-injects `threading.Event`. The job should poll `_cancel_event.is_set()` and return early.

**Gradio polling pattern:**
```python
gr.Timer(1.0).tick(
    fn=lambda log: (log + "".join(runner.read_logs()), runner.get_status().value),
    inputs=[log_box],
    outputs=[log_box, status_txt],
)
```

---

## Phase B — App Shell + System Tab

### Launch

```bash
# From project root, with venv activated:
python frontend/app.py

# Options:
python frontend/app.py --host 0.0.0.0 --port 7860 --share
```

Opens at **http://localhost:7860**

### `frontend/app.py`

- `gr.Blocks(title="LLM Training Dashboard")` with `gr.themes.Soft()` passed to `launch()`
- 4 top-level `gr.Tabs`: ⚙️ System, 💬 Chat, 📊 Benchmark, 📈 Profiling
- CSS hides Gradio footer; container max-width 1400 px
- `build_system_tab()` wired in; other tabs are stubs

> **Gradio 6 note:** `theme` and `css` must be passed to `launch()`, not `gr.Blocks()`.

### `frontend/tabs/system_tab.py`

**GPU Stats group**
- `gr.DataFrame` — 7 columns: GPU, Used GB, Total GB, Free GB, Used %, Temp °C, Util %
- `gr.Timer(value=5)` auto-refreshes the table every 5 s
- Used bytes sourced from `torch.cuda.memory_allocated()` (accurate per-process); total/temp/util from `nvidia-smi`

**Loaded Models group**
- `gr.Markdown` — lists loaded model name, type label, and approximate VRAM on load
- Also refreshed every 5 s by the same timer

**Model Loader group**
- `gr.Dropdown` (available models) — 11 models grouped by type; populated by `_available_choices()`
- 🔄 Rescan button — re-scans `models/` directory without restarting the app
- ⬆️ Load Selected — calls `manager.load_model(nickname, max_loaded=2)`; up to 2 models allowed simultaneously
- `gr.Dropdown` (loaded models) + ⬇️ Unload Selected — calls `manager.unload_model()`
- 🗑️ Unload All — calls `manager.unload_all()`
- `gr.Markdown` status line — shows ✅/❌ result of last action

All four buttons share the same output list: `[status_md, loaded_md, unload_dd, vram_df]`.

---

## Validation Results

```
TEST 1 — list_available_models()
  11 models found (2 BASE, 2 PRETRAINED, 4 INSTRUCT, 2 PROFILED_BASE, 1 PROFILED_INSTRUCT)
  Type inference: all 11 correctly classified

TEST 2 — get_vram_usage()
  GPU 0 (RTX A6000): 0.72 / 47.99 GB  |  42°C  |  5% util
  GPU 1 (RTX A6000): 0.01 / 47.99 GB  |  30°C  |  0% util

TEST 3 — JobRunner dummy job       DONE  ✅
TEST 4 — JobRunner cancel          ERROR (cancelled) ✅
TEST 5 — JobRunner error capture   ERROR + traceback ✅

Phase B — HTTP 200 at localhost:7860 ✅
```

---

## Dependencies Added

```
gradio>=5.0.0   (installed: 6.9.0)
```

Added to `requirements.txt`.
