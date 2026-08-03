# Frontend User Guide — LLM Training Dashboard

A step-by-step guide for using the Gradio-based dashboard (v0.4.0) to manage
models, chat with them, run HPN benchmarks, profile performance, and train LoRA
adapters — all from a single browser UI.

---

## Table of Contents

1.  [Launching the Dashboard](#1--launching-the-dashboard)
2.  [System Tab](#2--system-tab)
3.  [Chat Tab](#3--chat-tab)
4.  [Benchmark Tab](#4--benchmark-tab)
5.  [Profiling Tab](#5--profiling-tab)
6.  [Training Tab](#6--training-tab)
7.  [LoRA Tab](#7--lora-tab)
8.  [Tips & Troubleshooting](#8--tips--troubleshooting)

---

## 1 — Launching the Dashboard

```bash
# From the project root
python frontend/app.py

# Custom host / port / public link
python frontend/app.py --host 0.0.0.0 --port 7860 --share
```

The web UI opens at **http://localhost:7860**.  The title bar shows the app
version and short Git commit hash (e.g. `v0.4.0 · a1b2c3d`).

> **Requirements:** Python 3.11+, PyTorch with CUDA, the packages in
> `requirements.txt` installed, and at least one model directory under
> `models/`.

---

## 2 — System Tab

| Sub-section | What it does |
|-------------|--------------|
| **GPU Status** | Displays VRAM usage (used / total / percent) for every detected GPU. Auto-refreshes every 5 seconds. |
| **Model Loader** | Load or unload instruction-tuned models into GPU memory. Up to 2 models may be loaded simultaneously (for the Compare feature in the Chat tab). |

### Workflow

1. Open the **🖥 System** tab.
2. Review GPU VRAM usage.
3. Select a model from the dropdown (scans `models/` on disk).
4. Click **Load Model**. Progress appears in the status area.
5. To free VRAM, select the loaded model and click **Unload**, or use **Unload All**.

> **Tip:** Models are loaded in `bfloat16` with `device_map="auto"`.  A 1B
> model typically uses ≈ 2.5 GB; an 8B model uses ≈ 16 GB.

---

## 3 — Chat Tab

Two sub-tabs for interactive conversation:

### 3.1 — Single Chat

| Control | Description |
|---------|-------------|
| **Model** dropdown | Pick any model already loaded via the System tab.  Auto-refreshes every 5 s. |
| **Mode** radio | *Instruct* — uses the Open-Orca `### System: / ### User: / ### Assistant:` template.  *Text Completion* — raw continuation without a template. |
| **System prompt** | Visible only in Instruct mode.  Pre-filled with the HPN domain prompt. |
| **Max new tokens** | Upper bound on generated tokens (64 – 1024). |
| **Temperature** | Sampling temperature.  Lower → more deterministic. |
| **Top-p** | Nucleus sampling.  1.0 = disabled. |

> **Template auto-detection**: The chat tab automatically selects the correct
> prompt template (Open-Orca for Llama, ChatML for Qwen) based on the model
> path. No manual configuration is needed — Qwen3.5 models work out of the box.

Type a message, press **Send**, and watch the response stream token-by-token.
Each turn shows stats on completion: elapsed time, token count, and throughput
(`tok/s`).  Click **Clear** to reset conversation history.

### 3.2 — Compare Two Models

Side-by-side comparison.  Select a model for the left and right panes,
type a message, and click **Send to Both**.  Both models generate
simultaneously.  Useful for A/B testing pre-trained vs. fine-tuned variants.

> **Note:** Both models must be loaded first (System tab allows up to 2).

### Stop-String Behaviour

The chat applies automatic stop strings so models don't generate follow-up
questions or multi-turn continuations within a single turn:

| Model Family | Stop Strings |
|-------------|-------------|
| Llama | `### User:`, `### System:`, `\n### ` |
| Qwen | `<\|im_start\|>`, `<\|im_end\|>` |

---

## 4 — Benchmark Tab

Runs the full **HPN Q&A Benchmark** pipeline through five sub-tabs.

### 4.1 — Phase 1: Generate Answers

Loads each selected instruction model, prompts it with all 90 benchmark
questions, and saves answers to Excel.

| Control | Default | Notes |
|---------|---------|-------|
| **Instruction model(s)** | — | Multi-select.  Scans `models/instruction/`, `models/lora-merged/`, and `models/profiled/`.  Click 🔄 to rescan. |
| **Benchmark JSON file** | `data/prompts/hpn_qa_benchmark_v4_general_skills.json` | Path to the benchmark question set. |
| **Max new tokens** | 512 | Controls answer length ceiling. |
| **Temperature** | 0.3 | Low for deterministic technical answers. |
| **Top-p** | 0.9 | Standard nucleus sampling. |

Click **▶ Run Phase 1**.  Models are loaded and unloaded sequentially (only one
at a time).  Live log output scrolls in real-time and status shows
🔄 Running → ✅ Done or ❌ Error.

**Output:**
```
outputs/evaluations/answers/hpn_answers_<model>_<benchmark_tag>.xlsx
```

### 4.2 — Phase 2: Judge

Sends each (question, reference answer, model answer) triple to an LLM judge
for scoring on four dimensions (1–5): Correctness, Completeness, Clarity, and
Conciseness.

| Control | Default | Notes |
|---------|---------|-------|
| **Answer file(s)** | — | Multi-select from Phase 1 output.  Click 🔄 to rescan. |
| **Gemini API key file** | `gemini_api_key.txt` | Used when selecting a Gemini judge model. |
| **OpenAI API key file** | `openai_api_key.txt` | Used when selecting an OpenAI judge model (gpt-*, o1-*, o3-*, o4-*). |
| **Judge model** | `gemini-2.5-flash` | Dropdown with common choices + custom value.  Provider is auto-detected from the model name prefix. |
| **Requests / minute** | 15 | Rate limit sent to the judge script.  Increase for higher-tier API keys. |

Click **▶ Run Phase 2**.

**Output:**
```
outputs/evaluations/judged/hpn_judged_<model>_<benchmark_tag>_by_<judge>.xlsx
```

### 4.3 — Phase 3: Report

Reads **two or more** Phase 2 judged files and produces a side-by-side
comparison report broken down by category and difficulty.

| Control | Notes |
|---------|-------|
| **Judged file(s)** | Multi-select — must pick at least 2.  Click 🔄 to rescan. |

**Output:**
```
outputs/evaluations/reports/HPN_BENCHMARK_REPORT_<benchmark_tag>_by_<judge>.xlsx
outputs/evaluations/reports/HPN_BENCHMARK_REPORT_<benchmark_tag>_by_<judge>.md
```

### 4.4 — Filter

Reads all judged Excel files from `outputs/evaluations/judged/` and produces
filtered copies with two sheets:

* **Good Answers** — rows where the overall score exceeds the *good threshold*
  (default > 4.0).
* **Bad Answers** — rows where the overall score falls below the *bad threshold*
  (default < 1.3).

| Control | Default | Notes |
|---------|---------|-------|
| **Judged directory** | `outputs/evaluations/judged/` | Input directory containing judged Excel files. |
| **Filtered directory** | `outputs/evaluations/filtered/` | Where filtered files are written. |
| **Good threshold** | 4.0 | Overall score must be *above* this to qualify as "Good". |
| **Bad threshold** | 1.3 | Overall score must be *below* this to qualify as "Bad". |

Click **▶ Run Filter**.

**Output:**
```
outputs/evaluations/filtered/filtered_hpn_judged_<model>_<benchmark_tag>_by_<judge>.xlsx
```

### 4.5 — Results

A live directory listing of everything under `outputs/evaluations/`:

| Sub-folder | Content |
|------------|---------|
| `answers/` | Phase 1 answer Excel files |
| `judged/` | Phase 2 scored Excel files |
| `filtered/` | Filter utility output |
| `reports/` | Phase 3 comparison Excel + Markdown |

Click **🔄 Refresh** to update the file list.  The panel also auto-refreshes
every 10 seconds.

Use the **Preview Markdown report** dropdown to render any `.md` report inline
in the browser.

---

## 5 — Profiling Tab

Four sub-tabs for measuring training and inference performance:

### 5.1 — Training Profiler

Profiles a short training run (forward + backward pass) to measure throughput,
peak VRAM, and iteration time.

| Control | Default | Notes |
|---------|---------|-------|
| **Mode** | `pretrain` | `pretrain` or `instruct` — matches the training script used. |
| **Model** | — | Base or pretrained model.  Click 🔄 to rescan. |
| **Dataset directory** | — | Processed data directory (auto-scanned). |
| **Max steps** | 50 | Number of training steps to profile. |
| **Batch size** | 2 | Micro-batch size per GPU. |
| **Gradient accumulation** | 8 | Number of accumulation steps. |

### 5.2 — Inference Profiler

Benchmarks token generation speed across one or more models.

| Control | Default | Notes |
|---------|---------|-------|
| **Model(s)** | — | Multi-select.  Scans `models/base/`, `models/pretrained/`, `models/instruction/`, `models/lora-merged/`, and `models/profiled/`. |
| **Num runs** | 3 | Inference iterations (averaged). |
| **Warmup runs** | 2 | Warm-up iterations before measurement. |

### 5.3 — Report Generator

Aggregates profiling JSON results into a formatted Excel + Markdown report.

| Control | Notes |
|---------|-------|
| **Filter** | Optional substring to match only certain result filenames. |

### 5.4 — Results Viewer

Reads profiling JSON files from `outputs/profiling_results/` and displays:

* A summary table of key metrics (throughput, peak VRAM, latency).
* **Inference throughput** bar chart (tokens / second).
* **Peak GPU VRAM** bar chart (MB).

Auto-refreshes every 10 seconds.

---

## 6 — Training Tab

The dashboard now exposes the project's main full-weight training pipeline,
not just profiling and LoRA. It wraps the existing scripts under `training/`
and keeps them isolated from the dashboard process via background subprocesses
with live log streaming.

### 6.1 — Prepare CPT Data

Tokenizes the raw research corpus into packed causal-LM blocks for continual
pretraining.

| Control | Default | Notes |
|---------|---------|-------|
| **Base model / tokenizer** | — | Uses a local base model from `models/base/`. |
| **Input corpus JSON** | `data/raw/research_corpus_new.json` | Raw research corpus. |
| **Output dataset directory** | auto | Defaults to `data/processed/<model-slug>`. |
| **Sequence length** | 2048 | Packed block size in tokens. |
| **Validation split** | 0.05 | Held-out fraction for evaluation. |

### 6.2 — Continual Pretrain

Runs full-weight domain adaptation using the processed dataset from the prior step.

| Control | Default | Notes |
|---------|---------|-------|
| **Base model** | — | Source model from `models/base/`. |
| **Processed dataset** | — | Directory from `data/processed/`. |
| **Output trained-model directory** | auto | Defaults to `models/pretrained/<model>-trained-new`. |
| **Epochs** | 3 | Full training epochs. |
| **Train / Eval batch** | 2 / 2 | Per-device batch sizes. |
| **Grad accum** | 8 | Gradient accumulation steps. |
| **Learning rate** | `2e-5` | Passed through to the training script. |
| **Optimizer** | `adamw_torch` | Includes 8-bit options for memory-constrained runs. |

### 6.3 — Prepare Instruction Data

Builds the instruction-tuning dataset using either the default Orca + Dolly
mixture or a local pre-split JSONL dataset such as `data/Instruct-FTD/v3_run`.

| Control | Default | Notes |
|---------|---------|-------|
| **Target model / tokenizer** | — | Base or pretrained model. |
| **Output instruction-dataset directory** | auto | Defaults to `data/instruction/<model-name-lower>`. |
| **Optional local pre-split JSONL directory** | blank | If blank, downloads and mixes Orca + Dolly. |
| **Max token length** | 2048 | Per-sample truncation length. |
| **Validation split** | 0.05 | Used when generating the split from HF data. |
| **Max samples** | `5000` | Blank uses the full dataset. |

### 6.4 — Instruction FT

Runs full instruction fine-tuning on a selected base or pretrained model.

| Control | Default | Notes |
|---------|---------|-------|
| **Source model** | — | Base or pretrained model. |
| **Instruction dataset** | — | Directory under `data/instruction/`. |
| **Output instruction-model directory** | auto | Defaults to `models/instruction/<model>-instruct`. |
| **Epochs** | 3 | Full fine-tuning epochs. |
| **Batch size** | 2 | Per-device batch size. |
| **Grad accum** | 8 | Gradient accumulation steps. |
| **Learning rate** | `2e-5` | Passed through to the FT script. |

### 6.5 — Outputs

Shows the currently discovered directories under:

* `data/processed/`
* `data/instruction/`
* `models/pretrained/`
* `models/instruction/`

This view auto-refreshes every 10 seconds.

---

## 7 — LoRA Tab

End-to-end LoRA (Low-Rank Adaptation) pipeline in four panels:

### 7.1 — Prepare Data

Converts the raw HPN research corpus into LoRA-compatible QA pairs using the
same prompt extraction as instruction fine-tuning.

| Control | Default | Notes |
|---------|---------|-------|
| **Corpus JSON** | `data/raw/research_corpus_new.json` | Raw source material. |
| **Output directory** | `data/processed/lora/` | Where processed JSONL is saved. |
| **Max samples** | 0 (unlimited) | Limit for debugging. |

### 7.2 — Train LoRA

Launches LoRA training on a selected instruction model using the prepared data.

| Control | Default | Notes |
|---------|---------|-------|
| **Base instruction model** | — | Model from `models/instruction/`. |
| **Train data** | `data/processed/lora/lora_train.jsonl` | JSONL training split. |
| **Output directory** | `models/lora/` | Where the LoRA adapter checkpoint is saved. |
| **Epochs** | 3 | Number of training epochs. |
| **Batch size** | 4 | Micro-batch size. |
| **Learning rate** | 2e-4 | Initial learning rate. |
| **LoRA rank** | 64 | Rank of the low-rank matrices. |
| **LoRA alpha** | 128 | LoRA scaling factor (typically 2× rank). |
| **Target modules** | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Comma-separated list of linear layers to adapt. Same defaults for Llama and Qwen. |

### 7.3 — Merge Adapter

Merges the trained LoRA adapter back into the base instruction model, producing
a standalone model that can be used without PEFT.

| Control | Default | Notes |
|---------|---------|-------|
| **Base instruction model** | — | The same model used for training. |
| **Adapter directory** | — | Path to the LoRA checkpoint. |
| **Merged output directory** | `models/lora-merged/` | Fresh merged model. |

> **Note:** The original base model is **never** modified.  The merge creates a
> new directory.

### 7.4 — Adapters

Lists all LoRA adapter directories found under `models/lora/` with their file
sizes and metadata.

---

## 8 — Tips & Troubleshooting

### General

| Issue | Fix |
|-------|-----|
| UI not loading | Ensure all dependencies are installed: `pip install -r requirements.txt`. |
| Blank GPU panel | Check that `torch.cuda.is_available()` returns `True` and `nvidia-smi` is on `PATH`. |
| Port already in use | Pass a different port: `python frontend/app.py --port 7861`. |

### Models

| Issue | Fix |
|-------|-----|
| Dropdown is empty | Models must be placed under `models/<type>/`.  Click 🔄 Rescan. |
| OOM on load | Unload other models first (System tab → Unload All).  1B needs ~2.5 GB, 8B needs ~16 GB. |
| Chat output is truncated | Increase **Max new tokens** slider. |

### Benchmark

| Issue | Fix |
|-------|-----|
| "Select at least one model" | Choose a model from the Phase 1 dropdown first. |
| Phase 2 fails with API error | Verify that `gemini_api_key.txt` or `openai_api_key.txt` contains a valid key. |
| Rate limit errors | Lower **Requests / minute** in Phase 2 (try 5–10). |
| Report requires ≥ 2 files | Phase 3 compares models — select at least 2 judged files. |

### LoRA

| Issue | Fix |
|-------|-----|
| Merge fails | Ensure `peft>=0.10.0` is installed.  The adapter and base model architectures must match. |
| Training OOM | Reduce batch size, increase gradient accumulation, or try QLoRA (4-bit). |

### Accessing Outputs

All generated files reside in the `outputs/` directory:

```
outputs/
├── evaluations/
│   ├── answers/      ← Phase 1 Excel files
│   ├── judged/       ← Phase 2 scored Excel files
│   ├── filtered/     ← Filtered good/bad answers
│   └── reports/      ← Phase 3 comparison reports
├── profiling_reports/ ← Profiling Excel + Markdown reports
└── profiling_results/ ← Raw profiling JSON results
```

---

*Last updated for dashboard v0.4.0.*
