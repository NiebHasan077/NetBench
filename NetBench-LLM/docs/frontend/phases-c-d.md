# Frontend Development — Phase C & D Documentation

**Project:** LLM Training Dashboard  
**Framework:** Gradio 6.9.0  
**Date:** 2026-03-23  
**Branch:** `integrate-frontend`

---

## Phase C — Chat Tab

### Overview

`frontend/tabs/chat_tab.py` implements a streaming chat interface over any
model loaded into the ModelManager. It provides two sub-tabs: **Single Chat**
(one model, full multi-turn) and **Compare Two Models** (two models receiving
the same input simultaneously).

### Architecture

```
build_chat_tab()                 ← gr.Blocks() entrypoint
├── gr.Tab "Single Chat"         → _build_single_chat()
│   ├── Model dropdown           ← manager.get_loaded_models(), auto-refreshed every 5s
│   ├── Mode radio               ← "Instruct" | "Text Completion"
│   ├── System prompt textbox    ← visible only when Mode=Instruct
│   ├── Sliders (tokens, temp, top-p)
│   ├── gr.Chatbot               ← messages format ({"role", "content"})
│   ├── Message input + Send / Clear
│   └── _submit() generator      ← yields history on each streamed token
│
└── gr.Tab "Compare Two Models"  → _build_compare()
    ├── Mode radio + sliders (shared)
    ├── Shared system prompt textbox
    ├── Left column  (model dropdown + gr.Chatbot)
    ├── Right column (model dropdown + gr.Chatbot)
    ├── Shared message input + Send to Both / Clear Both
    ├── _stream_left() generator  → outputs=[left_bot] only
    └── _stream_right() generator → outputs=[right_bot] only
```

### Streaming Mechanism

```
User sends message
       │
       ▼
_submit() / _stream_left/right()
       │
       ▼
_stream_response(model_nick, message, history, mode, ...)
       │
       ├─ tokenizer(prompt) → inputs on model.device
       │
       ├─ TextIteratorStreamer(skip_prompt=True, skip_special_tokens=True)
       │
       ├─ threading.Thread(target=model.generate, kwargs=gen_kwargs).start()
       │
       └─ for token in streamer:
              partial += token
              yield partial          ← Gradio re-renders chatbot each iteration
              token_count += 1
       │
       └─ yield partial + stats     ← "⏱ 4.2 s · 128 tokens · 30.5 tok/s"
```

### Prompt Templates

**Instruct mode** — Open-Orca template (matches training):
```
### System:
{system}

### User:
{turn_1_user}

### Assistant:
{turn_1_assistant}

### User:
{new_message}

### Assistant:
```

Multi-turn history is included for Instruct mode. Text Completion passes the
raw message directly.

### Generation Parameters

| Parameter           | Default | Source                         |
|---------------------|---------|--------------------------------|
| `max_new_tokens`    | 512     | slider (range 32–2048)         |
| `temperature`       | 0.7     | slider (range 0–2.0)           |
| `top_p`             | 0.9     | slider (range 0.1–1.0)         |
| `do_sample`         | True    | True when temperature > 0      |
| `repetition_penalty`| 1.1     | fixed (matches inference scripts) |
| `pad_token_id`      | eos_id  | tokenizer.eos_token_id         |
| `stop_strings`      | —       | Instruct mode only: `["### User:", "### System:", "\n### "]` |

> **Stop strings** (Instruct mode only): Generation halts immediately if the
> model emits `### User:`, `### System:`, or `\n### `.  This prevents
> instruction-tuned models from self-asking follow-up questions.  Any trailing
> stop-string fragments are stripped from the displayed response via
> `_StopStringHit` exception handling.

### Key Files

| File | Role |
|------|------|
| `frontend/tabs/chat_tab.py` | Full implementation |
| `frontend/core/model_manager.py` | Provides `manager.get_model()`, `loaded.is_instruct` |
| `inference/test_instruction_model.py` | Reference for Open-Orca template |
| `inference/chat_interactive.py` | Reference for BASE model decoding |

---

## Phase D — Benchmark Tab

### Overview

`frontend/tabs/benchmark_tab.py` wraps the 3-phase HPN Q&A benchmark pipeline
in a browser UI. Each phase runs as an isolated **subprocess**, keeping model
loading/unloading entirely separate from the ModelManager and Dashboard process.
Logs are streamed live into polling textboxes via `JobRunner`.

### Architecture

```
build_benchmark_tab()                 ← gr.Blocks() entrypoint
├── gr.Tab "⚙ Phase 1 — Generate Answers"  → _build_phase1()
│   └── subprocess: evaluation/hpn_qa_benchmark.py
│
├── gr.Tab "🤖 Phase 2 — Judge"            → _build_phase2()
│   └── subprocess: evaluation/judge_responses.py
│
├── gr.Tab "📈 Phase 3 — Report"           → _build_phase3()
│   └── subprocess: evaluation/benchmark_report.py
│
└── gr.Tab "📂 Results"                    → _build_results()
    └── directory scan + inline Markdown preview
```

### Subprocess Pipeline

```
Phase 1                          Phase 2                      Phase 3
──────────────────────────────   ──────────────────────────   ──────────────────
hpn_qa_benchmark.py              judge_responses.py            benchmark_report.py
  --model_path <paths...>          --answer_files <xlsx...>      --judged_files <xlsx...>
  --benchmark  <json>              --gemini_api_key_file          --output_dir <dir>
  --output_dir answers/            --openai_api_key_file
  --max_new_tokens 512             --judge_model <model>
  --temperature 0.3                --output_dir judged/
                                   --requests_per_minute 15
  
Output: answers/hpn_answers_     Output: judged/hpn_judged_   Output: reports/
  <model>_<bmk_tag>.xlsx           <model>_<bmk_tag>_by_          ├── HPN_BENCHMARK_REPORT_*.md
                                   <judge>.xlsx                   └── hpn_benchmark_report_*.xlsx
```

> **Default benchmark**: `data/prompts/hpn_qa_benchmark_v4_general_skills.json`
> (90 questions, 10 HPN categories). The benchmark tag (e.g. `v4_general_skills`)
> is embedded in all output filenames to keep results from different benchmark
> versions separate.

### Judge Provider Support

Phase 2 supports **two judge LLM providers**:
- **Gemini** (default): `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`

The provider is auto-detected from the model name prefix (`gpt-`, `o1-`, `o3-`, `o4-` → OpenAI).
The UI provides:
- Dual API key file inputs (Gemini + OpenAI), defaulting to `gemini_api_key.txt` and `openai_api_key.txt`
- A judge model dropdown with preset models + `allow_custom_value=True` for arbitrary model names

### Log Streaming

JobRunner from Phase A is reused with a subprocess adapter:

```
_phase1_job() / _phase2_job() / _phase3_job()
  └─ runs in JobRunner worker thread
        │
        ▼
_run_subprocess(cmd, cancel_event)
  └─ subprocess.Popen(stdout=PIPE, stderr=STDOUT)
        │
        ▼
  for line in proc.stdout:
      print(line, end="")     ← captured by _ThreadLocalProxy → _line_queue
        │
        ▼
gr.Timer(2s).tick → runner.read_logs() + runner.get_full_log()
                    → gr.Textbox (accumulated log, never truncated)
```

### JobRunner Instances

Three independent `make_runner()` instances — one per phase — so all three
phases can run in parallel and their logs are isolated:

```python
_runner_p1 = make_runner("bench_phase1")
_runner_p2 = make_runner("bench_phase2")
_runner_p3 = make_runner("bench_phase3")
```

Cancel works by setting the runner's `threading.Event`, which is checked
between subprocess stdout lines in `_run_subprocess`.

### Auto-Discovery

| Helper | Scans |
|--------|-------|
| `_instruction_models()` | `models/instruction/`, `models/profiled/` |
| `_answer_files()` | `outputs/evaluations/answers/hpn_answers_*.xlsx` |
| `_judged_files()` | `outputs/evaluations/judged/hpn_judged_*.xlsx` |
| `_md_report_choices()` | `outputs/evaluations/reports/*.md` |

All dropdowns have a "🔄 Rescan" button and auto-refresh via `gr.Timer`.

### Key Files

| File | Role |
|------|------|
| `frontend/tabs/benchmark_tab.py` | Full implementation |
| `frontend/core/job_runner.py` | Background thread runner, log streaming |
| `evaluation/hpn_qa_benchmark.py` | Phase 1 subprocess (answer generation) |
| `evaluation/judge_responses.py` | Phase 2 subprocess (Gemini / OpenAI judging) |
| `evaluation/benchmark_report.py` | Phase 3 subprocess (comparative report) |
| `gemini_api_key.txt` | Gemini API key file (default path) |
| `openai_api_key.txt` | OpenAI API key file (default path) |

---

## Bugs Found and Fixed

### Phase C — chat_tab.py

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | Dead variable `i = 0` in `_build_instruct_prompt` — leftover from a loop refactor | Dead code, no functional impact | Removed |
| 2 | `_submit()` used `return history, ""` inside a Python generator function — in Python 3.3+, `return value` in a generator raises `StopIteration(value)` which Gradio does not consume; empty messages silently skipped but the input box was not cleared | Empty message send silently dropped with no chat update | Changed to `yield history or [], ""; return` |
| 3 | Compare tab: `_stream_left` and `_stream_right` both wrote to `outputs=[left_bot, right_bot]`. Each generator captured the other chatbot's state at click time. Running concurrently, the left generator would overwrite the right chatbot with its original (pre-generation) state on every yield, erasing the right model's streamed output | Right model's streaming output erased by left generator's yields, and vice versa | Separated inputs and outputs for each generator: `_stream_left → outputs=[left_bot]` only, `_stream_right → outputs=[right_bot]` only |

### Phase D — benchmark_tab.py

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 4 | `_poll_logs(runner)` called `runner.read_logs()` which drains the live queue and returns only lines produced since the last poll. The textbox was set to `"\n".join(new_lines)` — on each 2 s tick this overwrote the entire log box with just the newest batch; all prior log output disappeared | Log textbox showed only the most recent 2 s of output; early lines scrolled away rather than accumulating | Changed to `runner.read_logs()` (drain queue to keep it bounded) + `runner.get_full_log()` (return full accumulated log from internal deque) |

---

## Phase E & F Plan (Remaining)

### Phase E — Profiling Tab

`frontend/tabs/profiling_tab.py`

**Sub-tabs:**

**Training Profiler** — wraps `profiling/profile_training.py`
- Dropdowns: base model, dataset (processed_data dirs), batch size, gradient accumulation steps, max steps
- Run button → `make_runner("profile_training")` → subprocess
- Live log box (same pattern as benchmark)
- Output: `outputs/profiling_results/training_profile_<model>_<timestamp>.json`

**Inference Profiler** — wraps `profiling/profile_inference.py`
- Model path dropdown (any model), mode (Instruct / Base), number of prompts, max tokens
- Run button → subprocess
- Live log box
- Output: `outputs/profiling_results/inference_profile_<model>_<timestamp>.json`

**Results Viewer**
- Auto-scan `outputs/profiling_results/` for JSON files
- Render key metrics inline: throughput (tokens/s), latency (ms/token), memory usage
- Line charts comparing multiple profiling runs using `gr.Plot` (matplotlib figure)
- If `outputs/profiling_reports/` has HTML/Markdown reports, link and preview them

**Wire-up:** Replace `profiling_tab.py` stub, import in `app.py`.

---

### Phase F — Polish & Git

**F1 — Error resilience**
- Wrap all model calls in `try/except` with `gr.Warning` messages; never let unhandled exceptions propagate into the Gradio response
- `_stream_response`: if `gen_thread` is still alive after `gen_thread.join(5)`, log a warning (thread leaked)
- Subprocess runner: if process exits non-zero, display the last 20 lines of stderr in the log box with a red ❌ prefix

**F2 — UI polish**
- App-level header with version string and git commit hash (`git rev-parse --short HEAD`)
- Consistent emoji-prefixed tab labels across all tabs
- `gr.Accordion` to collapse advanced sliders (temperature, top-p) by default
- Keyboard shortcut hint on message textboxes ("⏎ to send")
- `avatar_images` on `gr.Chatbot` for user (👤) and assistant (🤖)

**F3 — Documentation**
- `docs/frontend/phases-e-f.md` — same format as this document
- Update `README.md` with a "Frontend (Gradio Dashboard)" section covering installation, launch command, and a feature table

**F4 — Git commit**
- Commit message: `"Frontend development phases C, D, E, F complete"`
- Tag: `v0.3.0-frontend`
