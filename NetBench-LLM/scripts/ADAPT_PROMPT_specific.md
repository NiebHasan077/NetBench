# Codex Adaptation Prompt — Qwen3.5-9B HPN Pipeline

Use this prompt to adapt `scripts/run_pipeline_specific.sh` for the Qwen3.5-9B
workflow on this machine. This is intentionally specific to the 9B Qwen model.

---

## PROMPT TO PASTE

```text
I have cloned the HPN LLM Training project on this machine. Adapt
scripts/run_pipeline_specific.sh so I can run the full workflow for Qwen3.5-9B
on this machine.

Target model:

MODEL_SHORTCUT: custom
MODEL_NAME: Qwen3.5-9B
MODEL_FAMILY: qwen
MODEL_HF_BASE: Qwen/Qwen3.5-9B-Base
MODEL_HF_INSTRUCT: Qwen/Qwen3.5-9B
MODEL_BASE_DIR: models/base/Qwen3.5-9B-Base
MODEL_INSTRUCT_DIR: models/instruct-official/Qwen3.5-9B
MODEL_PRETRAINED_DIR: models/pretrained/Qwen3.5-9B-trained-new

Important Qwen naming rules:

- My informal model name may be `qwen3.5:9B`, but that is an Ollama-style name.
  Do not put `qwen3.5:9B` into output paths or HuggingFace IDs.
- Do not use `MODEL_SHORTCUT="qwen-9b"` unless you also add full support for
  that shortcut in every parser and helper that needs it. The current codebase
  only registers `qwen-2b` and `qwen-4b`.
- Prefer `MODEL_SHORTCUT="custom"` for Qwen3.5-9B. In custom mode the pipeline
  downloads the base model with `snapshot_download`, prepares data with
  `--model_path`, and launches CPT with `--model_path` instead of the shortcut
  table.
- Do not use `Qwen/Qwen3.5-9B-Instruct`. That repo does not exist. Qwen3.5 uses
  `Qwen/Qwen3.5-9B` for the post-trained/instruct-side model and
  `Qwen/Qwen3.5-9B-Base` for the base model.

Workflow variants:

S1: official/post-trained Qwen3.5-9B baseline
S2: official/post-trained Qwen3.5-9B + LoRA HPN SFT
S3: official/post-trained Qwen3.5-9B + full HPN SFT
S4: Qwen3.5-9B-Base + full HPN CPT + full/LoRA HPN SFT
S5: official/post-trained Qwen3.5-9B + LoRA HPN CPT + LoRA HPN SFT
S6: official/post-trained Qwen3.5-9B + RAG
S7: S4 model + RAG
S8: S5 model + RAG

Your job:

1. Inspect the current repository before editing.
   - Run `git status --short`.
   - Read `scripts/run_pipeline_specific.sh`.
   - Check the relevant parser flags with `python training/<script>.py --help`
     when unsure. Do not add flags the Python scripts do not support.
   - Specifically confirm `training/pretrain_transformers.py --help` does not
     include `qwen-9b`; this is why this 9B adaptation should use `custom`.

2. Fill Section 1 for Qwen3.5-9B.
   - Set exactly:
     `MODEL_NAME="Qwen3.5-9B"`
     `MODEL_FAMILY="qwen"`
     `MODEL_SHORTCUT="custom"`
     `MODEL_HF_BASE="Qwen/Qwen3.5-9B-Base"`
     `MODEL_HF_INSTRUCT="Qwen/Qwen3.5-9B"`
     `MODEL_BASE_DIR="models/base/Qwen3.5-9B-Base"`
     `MODEL_INSTRUCT_DIR="models/instruct-official/Qwen3.5-9B"`
     `MODEL_PRETRAINED_DIR="models/pretrained/Qwen3.5-9B-trained-new"`
   - Verify `resolve_model_identity()` does not overwrite those custom paths.
   - Do not add a fake `qwen-9b` shortcut unless I explicitly ask for a broader
     codebase registration change.

3. Verify Qwen3.5 model loading compatibility without downloading full weights.
   - Run this cheap config check when network access is available:
     ```bash
     python - <<'PY'
     from transformers import AutoConfig, AutoModelForCausalLM
     from utils.setup_token import load_hf_token
     token = load_hf_token("hf_token.txt")
     for repo in ["Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B"]:
         cfg = AutoConfig.from_pretrained(repo, token=token, trust_remote_code=True)
         mapped = AutoModelForCausalLM._model_mapping[type(cfg)].__name__
         print(repo, getattr(cfg, "model_type", None), mapped)
     PY
     ```
   - Expected model type is `qwen3_5`, mapped to `Qwen3_5ForCausalLM`.
   - If `AutoModelForCausalLM` is not mapped, stop and report the issue instead
     of forcing the full pipeline.

4. Detect hardware and tune conservatively.
   - Run:
     `nvidia-smi --query-gpu=name,memory.total --format=csv`
     `nproc`
     `free -h`
   - The script requires at least 24 GB VRAM per GPU.
   - Classify VRAM from the MiB value reported by `nvidia-smi`; do not floor
     MiB/1024 too early. RTX A6000 reports about 49140 MiB and should use the
     48 GB profile.
   - For Qwen3.5-9B on 2x48 GB, keep full CPT/SFT conservative:
     CPT/SFT batch=1 grad_accum=16.
   - Use separate LoRA throughput profiles. A practical default is:
     LoRA-CPT batch=4 grad_accum=4 eval batch=1 epochs=3;
     LoRA-SFT batch=4 grad_accum=4 eval batch=1 epochs=3.
     This keeps effective batch size 16, avoids the known batch=8 instruction
     OOM, and improves corpus LoRA-CPT throughput without reducing epochs.
   - For smaller 48 GB+ models, CPT/SFT batch=4 grad_accum=4 and LoRA
     batch=8 grad_accum=2 can be reasonable.
   - For smaller 24-47 GB models, CPT/SFT batch=2 grad_accum=8 and LoRA
     batch=4 grad_accum=4 can be reasonable.
   - Keep effective batch size `batch * grad_accum * gpu_count >= 16`.
   - Keep `S4_USE_LORA_SFT=false` on 48 GB+ GPUs unless full SFT fails. Use
     `S4_USE_LORA_SFT=true` only when full SFT is too tight.
   - If resuming checkpoints that were created with the single-process
     `device_map="auto"` path, keep the compatible launcher:
     `USE_TORCHRUN_FULL_TRAINING=false` and
     `USE_DEEPSPEED_FULL_TRAINING=false`.
   - Only use distributed DeepSpeed for Qwen3.5-9B when starting a fresh full
     CPT/SFT run or resuming a checkpoint that was itself created by DeepSpeed.
     Do not mix non-DeepSpeed checkpoints with a DeepSpeed launcher.
   - Verify `configs/deepspeed_zero3_bf16.json` exists, `torchrun` is on PATH
     or `.venv/bin/torchrun` exists, and `python -c "import deepspeed"`
     succeeds when distributed full training is enabled.
   - Qwen3.5-9B on 2x48 GB generally needs optimizer offload for a fresh
     DeepSpeed run. Prefer `configs/deepspeed_zero3_bf16_cpuoffload.json`,
     which sets `zero_force_ds_cpu_optimizer=false` so DeepSpeed does not try
     to compile CPUAdam against a possibly mismatched system CUDA toolkit.
   - Keep `configs/deepspeed_zero3_bf16.json` as the no-CPU-offload ZeRO-3
     option for smaller models or larger GPU-memory machines.
   - Do not treat `device_map="auto"` as training throughput parallelism. It is
     only model sharding; full CPT/SFT should launch through `torchrun` when
     using multiple GPUs. Keep LoRA on the existing path unless the LoRA script
     explicitly supports distributed training.

5. Verify training data.
   - Corpus files may be JSON or JSONL. The script auto-selects the largest of:
     `data/raw/research_corpus_v3.json`
     `data/raw/research_corpus_v3.jsonl`
     `data/raw/research_corpus_new.json`
     `data/raw/research_corpus_new.jsonl`
   - Prefer `data/raw/research_corpus_v3.json` when it is the current bundled
     corpus; keep the other candidates as compatibility fallbacks.
   - HPN instruction data may be a JSON/JSONL file or a directory containing
     `train.jsonl` / `train.json` and optional validation JSON/JSONL.
   - Prefer the current bundled Instruct-FTD directory:
     `data/Instruct-FTD/v3_run_plus_json`
   - Keep `S4_FULL_SFT_DATASET="hpn"` so S4 full SFT uses the local Instruct-FTD
     data, not the OpenOrca+Dolly fallback.
   - Never use files in `data/prompts/` as CPT or SFT training data.

6. Verify benchmark data.
   - `BENCHMARK_FILE` should point to:
     `data/prompts/hpn_benchmark_v5.0.jsonl`
   - `BENCHMARK_FILE` may point to JSON or JSONL.
   - Keep benchmark files isolated from training data.
   - Ensure both the LLM and RAG evaluation paths can load the selected
     benchmark.
   - Configure answer-generation throughput separately from profiling:
     `BENCHMARK_GPU_WORKERS=0` should use all detected GPUs, and
     `BENCHMARK_CUDA_DEVICES` may restrict workers to a subset such as `0,1`.
   - Ensure direct LLM benchmark generation calls
     `evaluation/hpn_qa_benchmark.py --num_workers <N> --cuda_devices <ids>`.
   - Ensure RAG benchmark generation calls
     `../NetBench-RAG/evaluation/evaluate_rag.py --num_workers <N> --cuda_devices <ids>`.
     Each worker should load one local model replica on one GPU and answer a
     shard of questions; API-based models should ignore these worker flags.

7. Verify NetBench-RAG integration if S6/S7/S8 are enabled.
   - Check `../NetBench-RAG/` exists.
   - A single shared active Python venv may be used for both NetBench-LLM and
     NetBench-RAG. In that case, install both requirements files into the active
     venv and keep `RAG_VENV_DIR=""`.
   - If using a dedicated RAG environment, check `../NetBench-RAG/venv/` or
     `../NetBench-RAG/.venv/`, or set `RAG_VENV_DIR` explicitly.
   - Check:
     `../NetBench-RAG/index_corpus.py`
     `../NetBench-RAG/evaluation/evaluate_rag.py`
   - Read `../NetBench-RAG/config.yaml`.
   - The pipeline should use `paths.qdrant_dir` from config.yaml, not a hardcoded
     storage name.
   - RAG answer files must be routed to the model-scoped LLM output tree, not a
     shared `../NetBench-RAG/outputs/answers` directory.

8. Verify API keys and auth.
   - Check:
     `hf_token.txt`
     `openai_api_key.txt`
     `gemini_api_key.txt`
   - Missing `hf_token.txt` can break model downloads.
   - If `JUDGE_MODEL` is OpenAI, `openai_api_key.txt` is needed.
   - If `JUDGE_MODEL` is Gemini, `gemini_api_key.txt` is needed.
   - If no judge key exists, judging should be skipped, not allowed to crash
     after training completes.

9. Verify environment and dependencies.
   - Activate the project venv.
   - Run:
     `python -c "import torch, transformers, peft, trl, tensorboard; print('all ok')"`
   - If using one shared venv for LLM and RAG, also run:
     `python -c "import sentence_transformers, qdrant_client, rank_bm25, fitz, yaml; print('rag ok')"`
   - Qwen3.5 needs a Transformers version that exposes `qwen3_5` and
     `Qwen3_5ForCausalLM`; do not downgrade Transformers.
   - Install/verify `deepspeed>=0.14.0` if multi-GPU full CPT/SFT is enabled.
   - Do not run downloads or installs without my approval if they require
     network access.

10. Configure inference profiling.
    - Profiling should use benchmark questions, not only synthetic prompts.
    - Keep `PROFILE_BENCHMARK_LIMIT=10` unless I ask for a larger sample.
    - The default profiling comparison should include:
      direct GPU and CPU:
      `S1` base/post-trained Qwen3.5-9B, `S4` CPT-SFT, and `S5` LoRA-CPT-SFT;
      RAG GPU and CPU:
      `S7` RAG CPT-SFT and `S8` RAG LoRA-CPT-SFT.
    - Keep toggles for S2/S3/S6 profiling available, but off by default unless
      explicitly requested.
    - Summary CSV/Markdown should be written under:
      `outputs/by_model/qwen3.5-9b/profiling_results/`

11. Configure model-scoped outputs.
    - Multiple models may be run from the same repository.
    - Use this model-scoped root:
      `outputs/by_model/qwen3.5-9b/`
    - Keep answers, RAG answers, judged files, reports, adaptation reports,
      profiling files, and logs under that root.
    - Final benchmark reports should only consume judged files from this model's
      output tree unless I explicitly ask for a cross-model aggregate.

12. Configure long-run execution.
    - Prefer GNU screen for SSH or remote runs.
    - Keep foreground execution as the default unless I ask for detached launch.
    - If the script supports `USE_GNU_SCREEN=true`, verify it creates a unique
      model-based session name such as `hpn-qwen3.5-9b-<timestamp>`.
    - Include both commands in your final run instructions:
      foreground:
      `mkdir -p outputs/by_model/qwen3.5-9b/logs && bash scripts/run_pipeline_specific.sh 2>&1 | tee outputs/by_model/qwen3.5-9b/logs/pipeline_run_specific.log`
      detached screen:
      `USE_GNU_SCREEN=true bash scripts/run_pipeline_specific.sh`
    - Do not require a separate `SCREEN_SESSION_NAME` variable; derive the
      screen session name from `MODEL_NAME`.

13. Preserve workflow breadth.
    - Do not remove JSON support.
    - Do not remove JSONL support.
    - Do not remove local model support.
    - Do not remove OpenAI/Gemini API model or judge support.
    - Do not silently disable variants. Ask when a hard dependency is missing.

14. Validate before handing back.
    - Run `bash -n scripts/run_pipeline_specific.sh`.
    - Run Python syntax checks for modified Python files.
    - Run cheap `--help` checks for modified CLIs if any parser flags changed.
    - Confirm benchmark CLIs expose `--num_workers` and `--cuda_devices` when
      answer-generation worker support was touched.
    - Check that no `Qwen/Qwen3.5-9B-Instruct` or `qwen-9b` shortcut remains in
      `scripts/run_pipeline_specific.sh` unless you intentionally added full
      registered-shortcut support across the codebase.
    - Do not run the full pipeline unless I explicitly ask.

After adapting, summarize:

- variables changed
- dependency/data checks performed
- variants enabled/disabled and why
- exact command I should run
```
