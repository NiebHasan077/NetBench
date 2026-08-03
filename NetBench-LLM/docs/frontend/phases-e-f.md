# Frontend Development — Phase E & F Documentation

**Project:** LLM Training Dashboard  
**Framework:** Gradio 6.9.0  
**Version:** 0.3.0  
**Date:** 2026-03-23  
**Branch:** `integrate-frontend`

---

## Phase E — Profiling Tab

### Overview

`frontend/tabs/profiling_tab.py` wraps the three profiling scripts under
`profiling/` into a four sub-tab UI: Training Profiler, Inference Profiler,
Report Generator, and Results Viewer.

### Architecture

```
build_profiling_tab()
├── gr.Tab "📈 Profiling"
│   └── gr.Tabs
│       ├── gr.Tab "🏋 Training Profiler"     → _build_training_profiler()
│       │   ├── Mode radio: "pretrain" | "instruct"
│       │   ├── Model dropdown (base + pretrained models)
│       │   ├── Dataset directory dropdown (processed + instruction dirs)
│       │   ├── Max-steps slider, batch-size slider, grad-accum slider
│       │   ├── Run / Cancel buttons + Status textbox
│       │   ├── Live log textbox (gr.Timer 2 s → runner.get_full_log())
│       │   └── subprocess: profiling/profile_training.py <mode> …
│       │
│       ├── gr.Tab "⚡ Inference Profiler"   → _build_inference_profiler()
│       │   ├── Model multi-select dropdown (all model dirs)
│       │   ├── Num-runs + warmup-runs sliders
│       │   ├── Run / Cancel buttons + live log
│       │   └── subprocess: profiling/profile_inference.py --model_path …
│       │
│       ├── gr.Tab "📄 Report Generator"    → _build_report_generator()
│       │   ├── Filter textbox (optional substring match)
│       │   ├── Run / Cancel buttons + live log
│       │   └── subprocess: profiling/generate_profile_report.py [--filter …]
│       │
│       └── gr.Tab "📊 Results Viewer"      → _build_results_viewer()
│           ├── Scan outputs/profiling_results/ for JSON files
│           ├── Formatted text summary table of key metrics
│           ├── Bar chart — inference throughput (tok/s) using gr.Plot
│           ├── Bar chart — peak GPU VRAM (MB)
│           └── gr.Timer 10 s auto-refresh
```

### JobRunner Instances

```python
_runner_train  = make_runner("profile_training")
_runner_infer  = make_runner("profile_inference")
_runner_report = make_runner("profile_report")
```

Each runner is module-level so its cancel_event persists across Gradio events.

### Subprocess Commands

**Training profiler:**
```bash
python profiling/profile_training.py pretrain \
    --model_path <model_path> \
    --data_dir   <data_dir> \
    --max_steps  50 \
    --batch_size 2 \
    --gradient_accumulation_steps 8
```

**Inference profiler:**
```bash
python profiling/profile_inference.py \
    --model_path <model1> [<model2> …] \
    --num_runs    3 \
    --warmup_runs 2
```

**Report generator:**
```bash
python profiling/generate_profile_report.py [--filter <substring>]
```

### Path Constants

```python
_RESULTS_DIR = _ROOT / "outputs" / "profiling_results"
_REPORTS_DIR = _ROOT / "outputs" / "profiling_reports"
_MODELS_DIR  = _ROOT / "models"
_DATA_DIR    = _ROOT / "data"
```

### Results Viewer Charts

`_chart_data()` returns two matplotlib `Figure` objects rendered by `gr.Plot`:
- **Inference Throughput** — bar chart of `summary.avg_tokens_per_second` per model
- **Peak GPU VRAM** — bar chart of `memory.peak_gpu_vram_mb` (training) and  
  `summary.peak_gpu_vram_mb` (inference), colour-coded by type

---

## Phase F — Polish & Git

### F1 — Error Resilience

#### `_run_subprocess` (benchmark_tab.py & profiling_tab.py)

A rolling `tail` list (max 20 entries) accumulates the last 20 lines emitted
by the subprocess.  On non-zero exit the tail is echoed to the log with a
❌ prefix before the `RuntimeError` is raised, so the log box shows the root
cause without needing to scroll:

```python
tail: list[str] = []
for line in proc.stdout:
    ...
    tail.append(line)
    if len(tail) > 20:
        tail.pop(0)

proc.wait()
if proc.returncode not in (0, None):
    print(f"\n❌ Process exited with code {proc.returncode}. Last output:")
    for ln in tail:
        print(f"❌  {ln}", end="", flush=True)
    raise RuntimeError(f"Subprocess exited with code {proc.returncode}")
```

#### `_stream_response` thread-leak warning (chat_tab.py)

After `gen_thread.join(timeout=5)` in the `finally` block, the thread's
liveness is checked.  A leaked thread is logged as a `RuntimeWarning` (visible
in the server console but non-fatal to the UI):

```python
finally:
    gen_thread.join(timeout=5)
    if gen_thread.is_alive():
        import warnings
        warnings.warn(
            f"[chat_tab] generate() thread for '{model_nickname}' is still "
            "alive after 5 s join — the thread may be leaked.",
            RuntimeWarning,
            stacklevel=2,
        )
```

#### Try/except wrappers in chat_tab (chat_tab.py)

`_submit`, `_stream_left`, and `_stream_right` all wrap `_stream_response`
iteration in `try/except Exception`.  On failure:
- `gr.Warning(f"… error: {exc}")` surfaces a toast notification in the browser
- The assistant bubble is updated to show `⚠️ Error: <msg>` instead of
  crashing the generator

---

### F2 — UI Polish

#### App-level header with version + git hash (app.py)

`_git_hash()` runs `git rev-parse --short HEAD` at `build_app()` call time
and injects the result into the header Markdown:

```
# 🧠 LLM Training Dashboard  v0.3.0 · ce2bf2e
Interactive UI for model management, chat, benchmarking, and profiling.
```

A plain-text fallback (`"unknown"`) is used when git is not available.

#### Advanced generation settings accordion (chat_tab.py)

Temperature and Top-p sliders are hidden by default inside a collapsed
`gr.Accordion("⚙️ Advanced generation settings", open=False)` so the chat
UI is uncluttered for simple use.

#### Avatar images on gr.Chatbot (chat_tab.py)

All three `gr.Chatbot` instances (single chat, left compare, right compare)
now receive `avatar_images=("👤", "🤖")`.  Gradio renders these as small
avatar icons beside each chat bubble.

#### Keyboard shortcut hints (chat_tab.py)

Message textbox placeholders updated:
- Single chat: `"Type your message and press ⏎ to send…"`
- Compare tab: `"Type a message and press ⏎ to send to both models…"`

#### Emoji-prefixed sub-tab labels (chat_tab.py)

| Old label | New label |
|-----------|-----------|
| `Single Chat` | `💬 Single Chat` |
| `Compare Two Models` | `⚖️ Compare Two Models` |

---

### F3 — Documentation

| File | Content |
|------|---------|
| `docs/frontend/phases-e-f.md` | This document |
| `README.md` | Added **Frontend (Gradio Dashboard)** section |

---

### F4 — Git Tag

```
git commit -m "Frontend development phases C, D, E, F complete"
git tag v0.3.0-frontend
```

---

## Bugs Found and Fixed

### Phase E — profiling_tab.py

| # | Bug | Fix |
|---|-----|-----|
| 1 | `_submit_train` passed slider `value` as float to `int()` — harmless but explicit cast added | Changed to `int(max_steps)`, `int(batch)`, `int(grad_accum)` |
| 2 | Results viewer attempted `chart_tput.value = _ft` at module import time outside a Blocks context — Gradio components are not initialised at that point | Guarded with `try/except` silently; charts are populated on first `gr.Timer` tick or Refresh click |

### Phase F — error resilience

| # | Bug | Fix |
|---|-----|-----|
| 3 | `_submit` in `_build_single_chat()` assigned `partial = ""` inside the generator but the variable was never read — dead code leftover from an earlier refactor | Removed |
| 4 | `_stream_left` / `_stream_right` had no error handling; an exception inside `_stream_response` (e.g. model unloaded mid-stream) would propagate as an unhandled generator error, leaving the chatbot in a broken state | Wrapped in `try/except Exception` with `gr.Warning` toast and graceful bubble update |

---

## Key Files

| File | Role |
|------|------|
| `frontend/tabs/profiling_tab.py` | Phase E full implementation |
| `frontend/tabs/chat_tab.py` | Phase F polish (accordion, avatars, error handling) |
| `frontend/tabs/benchmark_tab.py` | Phase F polish (subprocess tail on error) |
| `frontend/app.py` | Phase F polish (version header, git hash) |
| `profiling/profile_training.py` | Training profiler subprocess |
| `profiling/profile_inference.py` | Inference profiler subprocess |
| `profiling/generate_profile_report.py` | Report generator subprocess |
