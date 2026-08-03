# Codex Adaptation Prompt — Qwen3.5-27B HPN Pipeline

Use this prompt to adapt `scripts/run_pipeline_specific.sh` for the Qwen3.5-27B
workflow on this machine. This version intentionally proceeds with the
post-trained Qwen3.5-27B workflows that are possible without a separate base
checkpoint.

Machine hardware (confirmed):
- 4× NVIDIA H100 80GB HBM3 (81559 MiB each; ~80 GB usable per GPU;
  ~320 GB aggregate VRAM)
- CUDA 13.0, Driver 580.159.03
- GPUs 0, 1, 2, and 3 report 0 MiB used and 0% utilization at time of
  writing; no running GPU processes are present.
- MIG is disabled on all four GPUs.

---

## PROMPT TO PASTE

```text
I have cloned the HPN LLM Training project on this machine. Adapt
scripts/run_pipeline_specific.sh so I can run the supported Qwen3.5-27B
workflow variants on this machine without a separate Qwen3.5-27B base model.

Target model:

MODEL_SHORTCUT: custom
MODEL_NAME: Qwen3.5-27B
MODEL_FAMILY: qwen
MODEL_HF_BASE: empty / unused because base-CPT variants are disabled
MODEL_HF_INSTRUCT: Qwen/Qwen3.5-27B
MODEL_BASE_DIR: empty / unused because base-CPT variants are disabled
MODEL_INSTRUCT_DIR: models/instruct-official/Qwen3.5-27B
MODEL_PRETRAINED_DIR: models/pretrained/Qwen3.5-27B-trained-new

Machine hardware (confirmed before running):
- 4× NVIDIA H100 80GB HBM3 — 81559 MiB each (~80 GB usable;
  ~320 GB aggregate VRAM)
- CUDA 13.0, Driver 580.159.03
- GPUs 0, 1, 2, and 3 are free at time of writing; no running GPU processes.
- MIG is disabled.

Important Qwen naming rules:

- Do not use `MODEL_SHORTCUT="qwen-27b"` unless you also add full support for
  that shortcut in every parser and helper that needs it. The current codebase
  only registers `qwen-2b` and `qwen-4b`.
- Use `MODEL_SHORTCUT="custom"` for Qwen3.5-27B. In custom mode the pipeline
  uses explicit local/Hugging Face paths instead of a shortcut table entry.
- Do not use `Qwen/Qwen3.5-27B-Instruct`. That repo does not exist. Qwen3.5
  uses `Qwen/Qwen3.5-27B` for the post-trained/instruct-side model.
- Do not assume `Qwen/Qwen3.5-27B-Base` exists. A live Hugging Face config
  check on 2026-05-31 returned 404 for that repo. In this adaptation, disable
  S4 and S7 explicitly instead of substituting the post-trained model as a fake
  base model.
- The informal names `qwen3.5:27b` or `qwen3.5:27B` are Ollama-style. Do not
  use them in output paths or HuggingFace IDs.

Workflow variants:

S1: official/post-trained Qwen3.5-27B baseline
S2: official/post-trained Qwen3.5-27B + LoRA HPN SFT
S3: official/post-trained Qwen3.5-27B + full HPN SFT
S4: DISABLED — requires unavailable Qwen3.5-27B-Base
S5: official/post-trained Qwen3.5-27B + LoRA HPN CPT + LoRA HPN SFT
S6: official/post-trained Qwen3.5-27B + RAG
S7: DISABLED — depends on S4/base-CPT model
S8: S5 model + RAG

Enabled variants for this run:
- S1, S2, S3, S5, S6, S8

Disabled variants for this run:
- S4 and S7 only, because no verified Qwen3.5-27B base checkpoint is available.

GPU capacity expectation:

- From a VRAM-capacity perspective, this 4× H100 80GB node should support all
  enabled variants, provided dependencies are healthy and full-weight S3 uses
  torchrun + DeepSpeed ZeRO-3.
- S1/S6/S8 inference and RAG generation: a 27B bf16 model replica should fit
  on one H100 80GB. With four free GPUs, benchmark/RAG generation can use one
  worker per GPU.
- S2/S5 LoRA training: supported by capacity. Keep LoRA on the existing
  single-process/model-sharded path and use conservative micro-batches.
- S3 full-weight SFT: supported by capacity when launched through torchrun +
  DeepSpeed ZeRO-3 across all four GPUs. Do not use the non-DeepSpeed
  `device_map="auto"` path for full-weight 27B SFT except as a debugging
  fallback.
- S4/S7 remain unavailable for this run because the previously assumed
  `Qwen/Qwen3.5-27B-Base` repo returned 404.
- If any full-weight phase OOMs, first reduce `CPT_BATCH`/`SFT_BATCH` from 2 to
  1 and increase the corresponding gradient accumulation to preserve the
  effective batch.

Your job:

1. Inspect the current repository before editing.
   - Run `git status --short` if this checkout is a git repository. If it is
     not a git repository, record that fact and continue without trying to
     repair git metadata.
   - Read `scripts/run_pipeline_specific.sh`.
   - Check the relevant parser flags with `python training/<script>.py --help`
     when unsure. Do not add flags the Python scripts do not support.
   - Confirm `training/pretrain_transformers.py --help` does not include
     `qwen-27b`; this is why this adaptation must use `MODEL_SHORTCUT="custom"`.

2. Fill Section 1 for Qwen3.5-27B.
   - Set exactly:
     `MODEL_NAME="Qwen3.5-27B"`
     `MODEL_FAMILY="qwen"`
     `MODEL_SHORTCUT="custom"`
     `MODEL_HF_BASE=""`
     `MODEL_HF_INSTRUCT="Qwen/Qwen3.5-27B"`
     `MODEL_BASE_DIR=""`
     `MODEL_INSTRUCT_DIR="models/instruct-official/Qwen3.5-27B"`
     `MODEL_PRETRAINED_DIR="models/pretrained/Qwen3.5-27B-trained-new"`
   - Set workflow toggles exactly:
     `RUN_S1=true`
     `RUN_S2=true`
     `RUN_S3=true`
     `RUN_S4=false`
     `RUN_S5=true`
     `RUN_S6=true`
     `RUN_S7=false`
     `RUN_S8=true`
   - Because `MODEL_HF_BASE=""` is intentional, update preflight validation so
     `MODEL_HF_BASE` and `MODEL_BASE_DIR` are required only when `RUN_S4=true`
     or `RUN_S7=true`.
   - Update `resolve_model_identity()` so an empty `MODEL_BASE_DIR` is not
     defaulted to a real-looking base path while `RUN_S4=false` and
     `RUN_S7=false`. `M_BASE` may be empty in this run because no enabled
     variant should consume it.
   - Log the base directory only when `RUN_S4=true` or `RUN_S7=true`; otherwise
     log that base-CPT variants are disabled.
   - Keep all S4/S7 code paths in the script for future use, but they must be
     inactive by default in this Qwen3.5-27B run.
   - Verify `resolve_model_identity()` does not overwrite the post-trained
     model path or reintroduce a fake base repo.
   - Do not add a fake `qwen-27b` shortcut unless explicitly asked.

3. Verify Qwen3.5 model loading compatibility without downloading full weights.
   - Run this cheap config check when network access is available:
     ```bash
     python - <<'PY'
     from transformers import AutoConfig, AutoModelForCausalLM
     from utils.setup_token import load_hf_token
     token = load_hf_token("hf_token.txt")
     repo = "Qwen/Qwen3.5-27B"
     cfg = AutoConfig.from_pretrained(repo, token=token, trust_remote_code=True)
     mapped = AutoModelForCausalLM._model_mapping[type(cfg)].__name__
     print(repo, getattr(cfg, "model_type", None), mapped)
     PY
     ```
   - `Qwen/Qwen3.5-27B` was checked on 2026-05-31:
     model type `qwen3_5`, architectures `['Qwen3_5ForConditionalGeneration']`,
     and `AutoModelForCausalLM` maps it to `Qwen3_5ForCausalLM`.
   - `Qwen/Qwen3.5-27B-Base` was checked on 2026-05-31 and returned 404.
     Do not use it unless a later check proves the repo exists and is
     accessible with the available HF token.
   - If `AutoModelForCausalLM` is not mapped, stop and report the issue instead
     of forcing the full pipeline.

4. Detect hardware and set hyperparameters for 4× H100 80GB.
   - Confirmed hardware:
     `nvidia-smi` reports 4× NVIDIA H100 80GB HBM3, 81559 MiB each.
     CUDA 13.0, Driver 580.159.03.
     GPUs 0, 1, 2, and 3 are free; no running GPU processes are present.
   - VRAM per GPU: 81559 MiB → classify as 80 GB profile.
   - Aggregate VRAM: 326236 MiB (~320 GB).
   - At 80 GB per GPU with 4 GPUs, Qwen3.5-27B full-weight SFT should fit in
     GPU memory with ZeRO-3. Use the no-CPU-offload config:
     `DEEPSPEED_CONFIG="configs/deepspeed_zero3_bf16.json"`
   - Recommended first-run hyperparameters for 27B on 4× H100 80GB:
     CPT:  `CPT_BATCH=2`  `CPT_GRAD_ACCUM=4`
           (reserved for future S4 only; unused while RUN_S4=false)
     SFT:  `SFT_BATCH=2`  `SFT_GRAD_ACCUM=4`
           (distributed effective batch = 2 × 4 × 4 = 32)
     LoRA: `LORA_BATCH=4` `LORA_GRAD_ACCUM=4`
           (single-process effective batch = 16; use accumulation 8 if an
           effective batch of 32 is required after confirming memory headroom)
   - Keep full-weight distributed effective batch size
     `batch * grad_accum * gpu_count >= 32`.
   - Keep LoRA effective batch size `batch * grad_accum >= 16` because LoRA is
     still launched through the existing single-process path.
   - Keep `S4_USE_LORA_SFT=false`, but it is unused while `RUN_S4=false`.
   - Enable distributed full training:
     `TRAIN_GPU_WORKERS=0` (all GPUs)
     `USE_TORCHRUN_FULL_TRAINING=true`
     `USE_DEEPSPEED_FULL_TRAINING=true`
   - Keep `OPTIM_FULL="paged_adamw_8bit"` for S3 full SFT unless a tested
     full-precision AdamW configuration is deliberately chosen. The same value
     can be reused later if a real base model enables full CPT.
   - Add or update the 80 GB hardware branch in `detect_hardware()` so it does
     not reuse the older 48 GB profile. The 80 GB branch should assign the
     CPT/SFT/LoRA values above.
   - Verify `configs/deepspeed_zero3_bf16.json` exists (it does — already
     created in this project).
   - Verify `torchrun` is on PATH or available at `.venv/bin/torchrun`.
   - Verify `python -c "import deepspeed"` succeeds.
   - Keep `MAX_BUSY_GPU_MEMORY_MIB=4096` for this node. The GPUs are currently
     free, so heavy training should fail fast if another process starts using
     meaningful VRAM before launch.
   - Do not treat `device_map="auto"` as training throughput parallelism; it is
     only model sharding. Full SFT must go through `torchrun`.
   - Keep LoRA on its existing single-process path.
   - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — already present
     in `run_pipeline_specific.sh`; confirm it is not removed.

5. Verify training data.
   - Corpus files may be JSON or JSONL. The script auto-selects the largest of:
     `data/raw/research_corpus_v3.json`
     `data/raw/research_corpus_v3.jsonl`
     `data/raw/research_corpus_new.json`
     `data/raw/research_corpus_new.jsonl`
   - `data/raw/research_corpus_v3.json` is the current bundled corpus; keep
     the other candidates as compatibility fallbacks.
   - HPN instruction data may be a JSON/JSONL file or a directory containing
     `train.jsonl` / `train.json` and optional validation JSON/JSONL.
   - The current bundled Instruct-FTD directory is:
     `data/Instruct-FTD/v3_run_plus_json`
     It contains `train.jsonl` and `validation.jsonl` — confirmed present.
   - `S4_FULL_SFT_DATASET` is unused while `RUN_S4=false`; leave it unchanged
     for future base-model runs.
   - Never use files in `data/prompts/` as CPT or SFT training data.

6. Verify benchmark data.
   - `BENCHMARK_FILE` must point to:
     `data/prompts/hpn_benchmark_v5.0.jsonl` — confirmed present.
   - Keep benchmark files isolated from training data.
   - Configure answer-generation throughput separately from profiling:
     `BENCHMARK_GPU_WORKERS=0` (all four H100s), and
     `BENCHMARK_CUDA_DEVICES` may be set to `"0,1,2,3"` to be explicit.
   - Ensure direct LLM benchmark generation calls
     `evaluation/hpn_qa_benchmark.py --num_workers <N> --cuda_devices <ids>`.
   - Ensure RAG benchmark generation calls
     `../NetBench-RAG/evaluation/evaluate_rag.py --num_workers <N> --cuda_devices <ids>`.
     Each worker loads one local model replica on one GPU and answers a shard
     of questions; API-based models ignore these worker flags.

7. Verify NetBench-RAG integration if S6/S8 are enabled.
   - Check `../NetBench-RAG/` exists — confirmed present as sibling project.
   - A single shared `.venv` is used for both projects; keep `RAG_VENV_DIR=""`.
   - Verify:
     `../NetBench-RAG/index_corpus.py`
     `../NetBench-RAG/evaluation/evaluate_rag.py`
   - Read `../NetBench-RAG/config.yaml` and use `paths.qdrant_dir` from it,
     not a hardcoded storage path.
   - RAG answer files must be routed to the model-scoped LLM output tree, not a
     shared `../NetBench-RAG/outputs/answers` directory.

8. Verify API keys and auth.
   - All three key files are confirmed present and non-empty:
     `hf_token.txt`
     `openai_api_key.txt`
     `gemini_api_key.txt`
   - Default judge is OpenAI (`JUDGE_MODEL="gpt-5.1"`); OpenAI key is present.
   - If no judge key exists at runtime, judging must be skipped gracefully, not
     crash after training completes.

9. Verify environment and dependencies.
   - The project `.venv` is already created and requirements installed.
   - Activate with `source .venv/bin/activate`.
   - Run:
     `python -c "import torch, transformers, peft, trl, tensorboard; print('all ok')"`
   - Run (shared venv includes RAG deps):
     `python -c "import sentence_transformers, qdrant_client, rank_bm25, fitz, yaml; print('rag ok')"`
   - Qwen3.5-27B requires a Transformers version that exposes `qwen3_5` and
     `Qwen3_5ForCausalLM`; do not downgrade Transformers.
   - Verify `deepspeed>=0.14.0` is installed.
   - CUDA 13.0 is newer than common PyTorch CUDA builds (12.x). Verify
     `torch.cuda.is_available()` returns `True`; if not, report and stop.
   - Do not run downloads or installs without approval if they require network.

10. Configure inference profiling.
    - Profiling should use benchmark questions, not only synthetic prompts.
    - Keep `PROFILE_BENCHMARK_LIMIT=10` unless asked for a larger sample.
    - The default profiling comparison should include:
      direct GPU and CPU:
      `S1` post-trained Qwen3.5-27B, `S3` full HPN-SFT,
      `S5` LoRA-CPT-SFT;
      RAG GPU and CPU:
      `S6` RAG post-trained Qwen3.5-27B and `S8` RAG LoRA-CPT-SFT.
    - Keep toggles for S2 profiling available, but off by default.
    - Do not include S4/S7 in default profiling while base variants are
      disabled.
    - Summary CSV/Markdown should be written under:
      `outputs/by_model/qwen3.5-27b/profiling_results/`

11. Configure model-scoped outputs.
    - Use this model-scoped root:
      `outputs/by_model/qwen3.5-27b/`
    - Keep answers, RAG answers, judged files, reports, adaptation reports,
      profiling files, and logs under that root.
    - Final benchmark reports should only consume judged files from this model's
      output tree unless a cross-model aggregate is explicitly requested.

12. Configure long-run execution.
    - Prefer GNU screen for SSH or remote runs on the compute node.
    - Keep foreground execution as the default unless detached launch is asked.
    - If the script supports `USE_GNU_SCREEN=true`, verify it creates a session
      named `hpn-qwen3.5-27b-<timestamp>` derived from `MODEL_NAME`.
    - Include both commands in your final run instructions:
      foreground:
      `mkdir -p outputs/by_model/qwen3.5-27b/logs && bash scripts/run_pipeline_specific.sh 2>&1 | tee outputs/by_model/qwen3.5-27b/logs/pipeline_run_specific.log`
      detached screen:
      `USE_GNU_SCREEN=true bash scripts/run_pipeline_specific.sh`
    - Do not require a separate `SCREEN_SESSION_NAME` variable; derive the
      screen session name from `MODEL_NAME`.

13. Preserve workflow breadth.
    - Do not remove JSON support.
    - Do not remove JSONL support.
    - Do not remove local model support.
    - Do not remove OpenAI/Gemini API model or judge support.
    - S4 and S7 are intentionally disabled for this run because the base model
      is unavailable. Do not silently disable any other variant.

14. Validate before handing back.
    - Run `bash -n scripts/run_pipeline_specific.sh`.
    - Run Python syntax checks for modified Python files.
    - Run cheap `--help` checks for modified CLIs if any parser flags changed.
    - Confirm benchmark CLIs expose `--num_workers` and `--cuda_devices` when
      answer-generation worker support was touched.
    - Check that no `Qwen/Qwen3.5-27B-Instruct` or `qwen-27b` shortcut remains
      in `scripts/run_pipeline_specific.sh` unless full registered-shortcut
      support was intentionally added across the codebase.
    - Check that `Qwen/Qwen3.5-27B` is not used as `MODEL_HF_BASE`.
    - Check that `RUN_S4=false` and `RUN_S7=false`.
    - Do not run the full pipeline unless explicitly asked.

After adapting, summarize:

- variables changed
- dependency/data checks performed
- variants enabled/disabled and why
- exact command to run
```
