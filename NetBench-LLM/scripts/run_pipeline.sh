#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  HPN LLM Full Evaluation Pipeline
# =============================================================================
#
# Runs all 8 training / evaluation variants for one target LLM model:
#
#   S1:  Original Instruct (distribution organisation)
#   S2:  Original Instruct + LoRA HPN-SFT
#   S3:  Original Instruct + Full HPN-SFT
#   S4:  Base + Full HPN-CPT  + Full/LoRA HPN-SFT
#   S5:  Original Instruct + LoRA HPN-CPT + LoRA HPN-SFT
#   S6:  Original Instruct + RAG
#   S7:  Base + Full HPN-CPT + HPN-SFT + RAG
#   S8:  Original Instruct + LoRA HPN-CPT + LoRA HPN-SFT + RAG
#
# Pipeline phases (in order):
#   0.  Preflight checks
#   1.  Model downloads (HuggingFace → local)
#   2.  Data preparation (corpus tokenisation, LoRA packs, SFT pairs)
#   3.  Training (CPT → SFT → LoRA, in re-use order)
#   4.  RAG index (NetBench-RAG — built once, shared by S6/S7/S8)
#   5.  Benchmark generation (hpn_qa_benchmark.py + evaluate_rag.py)
#   6.  Judging (LLM-as-judge via Gemini / OpenAI)
#   7.  Inference profiling
#   8.  LM adaptation reports (CPT quality + forgetting metrics)
#   9.  Final benchmark comparison report
#
# Resume behaviour
# ────────────────
# Every training step checks for a completed final model (config.json).
# If a partial checkpoint-<N>/ dir exists the training script is re-launched
# with --resume_from_checkpoint so no epochs are wasted.
# All other steps skip if their output file / directory already exists.
# Re-running the script after any failure resumes exactly where it left off.
#
# Ollama management
# ─────────────────
# If the ollama process is running when training begins it is stopped
# automatically to free GPU VRAM.  It is restarted at pipeline exit
# (including on failure / Ctrl+C) via a trap.
#
# Requirements
# ────────────
# • GPU with >= 24 GB VRAM per card (script aborts if < 24 GB detected).
# • Instruct-FTD project present if S2/S3/S5/S8 are enabled.
# • NetBench-RAG project present if S6/S7/S8 are enabled.
# • API key files present for model download and LLM-as-judge.
#
# BEFORE RUNNING: fill in every PLACEHOLDER value in the configuration block
# below and tune the hyperparameters for your machine (see DEVELOPMENT.md).
#
# USAGE:
#   cd /path/to/LLM-Training
#   source .venv/bin/activate
#   bash scripts/run_pipeline.sh 2>&1 | tee outputs/pipeline_run.log
# =============================================================================

set -euo pipefail

# ─── Resolve project roots ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# NetBench-RAG is a sibling directory; resolve path without cd (may not exist).
RAG_DIR="$(realpath -m "${LLM_DIR}/../NetBench-RAG")"
cd "${LLM_DIR}"

# =============================================================================
# SECTION 1 — MODEL IDENTITY   (ADAPT FOR EACH MACHINE / MODEL)
# =============================================================================

MODEL_NAME="PLACEHOLDER"        # Short label used in output paths.
                                # e.g. "Qwen3.5-2B", "Llama-3.1-8B", "Gemma3-4B"

MODEL_FAMILY="PLACEHOLDER"      # "llama" | "qwen" | "gemma"
                                # Used to pick prompt templates and LoRA target modules.

MODEL_SHORTCUT="PLACEHOLDER"    # pretrain_transformers.py shortcut key, OR "custom".
                                # Registered: 1b | 8b | qwen-2b | qwen-4b | gemma-4b
                                #             gemma-e4b | gemma-e2b
                                # Use "custom" for unregistered models.

MODEL_HF_BASE="PLACEHOLDER"     # HuggingFace repo ID for the BASE model.
                                # e.g. "meta-llama/Llama-3.1-8B"
                                #      "Qwen/Qwen3.5-2B-Base"
                                #      "google/gemma-3-4b-pt"

MODEL_HF_INSTRUCT="PLACEHOLDER" # HuggingFace repo ID for the OFFICIAL instruct model.
                                # e.g. "meta-llama/Llama-3.1-8B-Instruct"
                                #      "Qwen/Qwen3.5-2B-Instruct"
                                #      "google/gemma-3-4b-it"

# =============================================================================
# SECTION 2 — VARIANT TOGGLES
# =============================================================================
# Set any to false to skip that variant entirely.
# NOTE: disabling S2/S3/S5/S8 is only valid if Instruct-FTD data is unavailable.
#       disabling S6/S7/S8 is only valid if NetBench-RAG project is absent.

RUN_S1=true    # Dist. Instruct  — download + benchmark only (no training)
RUN_S2=true    # Dist. Instruct  + LoRA HPN-SFT
RUN_S3=true    # Dist. Instruct  + Full HPN-SFT
RUN_S4=true    # Base + Full HPN-CPT + Full/LoRA HPN-SFT
S4_USE_LORA_SFT=false   # false = full SFT after CPT;  true = LoRA SFT after CPT
RUN_S5=true    # Dist. Instruct  + LoRA HPN-CPT + LoRA HPN-SFT
RUN_S6=true    # Dist. Instruct  + RAG
RUN_S7=true    # Base + Full HPN-CPT + HPN-SFT + RAG
RUN_S8=true    # Dist. Instruct  + LoRA HPN-CPT + LoRA HPN-SFT + RAG

# =============================================================================
# SECTION 3 — PATHS  (derived — change only if your layout differs)
# =============================================================================

# Source models
M_BASE="${LLM_DIR}/models/base/${MODEL_NAME}-Base"
M_INSTRUCT_OFFICIAL="${LLM_DIR}/models/instruct-official/${MODEL_NAME}-Instruct"
M_PRETRAINED="${LLM_DIR}/models/pretrained/${MODEL_NAME}-trained-new"

# Output models produced by this pipeline
M_S3_FULL_HPN_SFT="${LLM_DIR}/models/instruction/${MODEL_NAME}-instruct-full-hpn-sft"
M_S4_FULL_SFT="${LLM_DIR}/models/instruction/${MODEL_NAME}-cpt-full-sft"
M_S4_LORA_SFT_ADAPTER="${LLM_DIR}/models/lora/${MODEL_NAME}-cpt-lora-sft"
M_S4_LORA_SFT_MERGED="${LLM_DIR}/models/lora-merged/${MODEL_NAME}-cpt-lora-sft-merged"
M_S2_LORA_SFT_ADAPTER="${LLM_DIR}/models/lora/${MODEL_NAME}-instruct-lora-sft"
M_S2_LORA_SFT_MERGED="${LLM_DIR}/models/lora-merged/${MODEL_NAME}-instruct-lora-sft-merged"
M_S5_LORA_CPT_ADAPTER="${LLM_DIR}/models/lora/${MODEL_NAME}-instruct-lora-cpt"
M_S5_LORA_CPT_MERGED="${LLM_DIR}/models/lora-merged/${MODEL_NAME}-instruct-lora-cpt-merged"
M_S5_LORA_SFT_ADAPTER="${LLM_DIR}/models/lora/${MODEL_NAME}-instruct-lora-cpt-then-sft"
M_S5_LORA_SFT_MERGED="${LLM_DIR}/models/lora-merged/${MODEL_NAME}-instruct-lora-cpt-then-sft-merged"

# Data paths
# CORPUS_FILE is auto-selected by detect_corpus() — largest file in data/raw/
CORPUS_FILE=""
# Place your Instruct-FTD train.jsonl at this path (or update the variable).
# Only this file is needed — the full Instruct-FTD project is not required.
HPN_INSTRUCT_JSONL="${LLM_DIR}/data/raw/instruct_ftd/train.jsonl"
BENCHMARK_FILE="${LLM_DIR}/data/prompts/hpn_qa_benchmark_v4_general_skills.json"

DATA_CPT_DIR="${LLM_DIR}/data/processed/${MODEL_NAME,,}"
DATA_LORA_CPT_DIR="${LLM_DIR}/data/processed/lora/${MODEL_NAME,,}"
DATA_GENERAL_SFT_DIR="${LLM_DIR}/data/instruction/${MODEL_NAME,,}-general"
DATA_HPN_SFT_DIR="${LLM_DIR}/data/instruction/${MODEL_NAME,,}-hpn"

# Output directories
OUT_ANSWERS="${LLM_DIR}/outputs/evaluations/answers"
OUT_JUDGED="${LLM_DIR}/outputs/evaluations/judged"
OUT_REPORTS="${LLM_DIR}/outputs/evaluations/reports"
OUT_ADAPT="${LLM_DIR}/outputs/evaluations/adaptation_reports"
OUT_PROFILE="${LLM_DIR}/outputs/profiling_results"

# API / auth
JUDGE_MODEL="gpt-5.1"           # default OpenAI judge; switch to a Gemini model if no OpenAI key
GEMINI_KEY_FILE="${LLM_DIR}/gemini_api_key.txt"
OPENAI_KEY_FILE="${LLM_DIR}/openai_api_key.txt"
HF_TOKEN_FILE="${LLM_DIR}/hf_token.txt"

# =============================================================================
# SECTION 4 — HARDWARE HYPERPARAMETERS  (auto-tuned by detect_hardware)
# =============================================================================

GPU_COUNT=0    # set by detect_hardware()
VRAM_GB=0      # per-GPU VRAM in GB — set by detect_hardware()

# CPT hyperparameters (full-weight continual pre-training)
CPT_BATCH=2
CPT_GRAD_ACCUM=8
CPT_LR="2e-5"
CPT_EPOCHS=3

# SFT hyperparameters (full-weight instruction fine-tuning)
SFT_BATCH=2
SFT_GRAD_ACCUM=8
SFT_LR="2e-5"
SFT_EPOCHS=3
SFT_MAX_SAMPLES=5000

# LoRA hyperparameters
LORA_BATCH=4
LORA_GRAD_ACCUM=4
LORA_LR="2e-4"
LORA_EPOCHS=3
LORA_RANK=64
LORA_ALPHA=128
USE_QLORA=false

# =============================================================================
# SECTION 5 — UTILITIES
# =============================================================================

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] OK${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN${NC}  $*"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] FATAL:${NC} $*" >&2; exit 1; }
hr()   { echo -e "${BOLD}------------------------------------------------------------${NC}"; }

# Skip if the sentinel file / directory already exists (non-training steps).
skip_if_exists() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        warn "SKIP ${label} — already exists: ${path}"
        return 0
    fi
    return 1
}

# Run a command and die on failure with a clear label.
run() {
    local label="$1"; shift
    log "RUN ${label}"
    "$@" || die "${label} failed (exit $?)"
    ok "${label} done"
}

# ─── run_training: skip if complete, resume from last checkpoint if partial ───
#
#   run_training LABEL OUTPUT_DIR CMD [ARGS...]
#
#   Completion:  OUTPUT_DIR/config.json present  => skip entirely.
#   Partial:     OUTPUT_DIR/checkpoint-<N>/ present but no config.json
#                => appends --resume_from_checkpoint <last-ckpt> automatically.
#   Fresh start: neither => runs from scratch.
#
run_training() {
    local label="$1" outdir="$2"; shift 2

    if [[ -f "${outdir}/config.json" ]]; then
        warn "SKIP ${label} — already complete: ${outdir}"
        return 0
    fi

    local ckpt=""
    ckpt=$(ls -dt "${outdir}"/checkpoint-* 2>/dev/null | head -1 || true)

    if [[ -n "${ckpt}" ]]; then
        log "RESUME ${label} from checkpoint: ${ckpt}"
        run "${label}" "$@" --resume_from_checkpoint "${ckpt}"
    else
        run "${label}" "$@"
    fi
}

# Build common LoRA CLI flags from Section 4 variables.
lora_args() {
    local extra=""
    $USE_QLORA && extra="${extra} --quantize_4bit"
    echo "--lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
--epochs ${LORA_EPOCHS} --batch_size ${LORA_BATCH} \
--grad_accum ${LORA_GRAD_ACCUM} --lr ${LORA_LR} ${extra}"
}

# =============================================================================
# SECTION 6 — OLLAMA MANAGEMENT
# =============================================================================

OLLAMA_WAS_RUNNING=false

# Stop Ollama before training to free GPU VRAM.
stop_ollama() {
    if pgrep -x ollama &>/dev/null || systemctl is-active --quiet ollama 2>/dev/null; then
        warn "Ollama process detected — stopping it to free GPU VRAM for training."
        OLLAMA_WAS_RUNNING=true
        # Try systemctl (clean shutdown) before falling back to pkill.
        if systemctl stop ollama 2>/dev/null; then
            log "Ollama stopped via systemctl."
        elif pkill -x ollama 2>/dev/null; then
            log "Ollama stopped via pkill."
        else
            warn "Could not stop Ollama automatically — continuing (may cause OOM)."
            OLLAMA_WAS_RUNNING=false
        fi
        sleep 2  # allow VRAM to be released
    fi
}

# Restart Ollama if we stopped it.  Called by the EXIT trap.
restart_ollama() {
    if $OLLAMA_WAS_RUNNING; then
        log "Restarting Ollama (was running before pipeline)..."
        if systemctl start ollama 2>/dev/null; then
            ok "Ollama restarted via systemctl."
        else
            warn "Could not restart Ollama automatically — start it manually if needed."
        fi
    fi
}

# =============================================================================
# SECTION 7 — HARDWARE DETECTION
# =============================================================================

detect_hardware() {
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
              2>/dev/null | head -1 | awk '{printf "%d", $1/1024}' || echo 0)

    log "Hardware: ${GPU_COUNT}x GPU, ${VRAM_GB} GB VRAM per card"

    # Hard minimum: 24 GB VRAM required.
    if (( VRAM_GB < 24 )); then
        die "Insufficient VRAM: ${VRAM_GB} GB detected per GPU.
  This pipeline requires >= 24 GB VRAM per card.
  Use a machine with RTX 3090 / A5000 / A6000 or better."
    fi

    if (( VRAM_GB == 24 )); then
        warn "VRAM is exactly 24 GB (supported minimum). Full-weight CPT/SFT will be
  memory-tight — monitor GPU utilisation closely during training."
    fi

    # Auto-tune hyperparameters based on available VRAM.
    if (( VRAM_GB >= 48 )); then
        CPT_BATCH=4;  CPT_GRAD_ACCUM=4
        SFT_BATCH=4;  SFT_GRAD_ACCUM=4
        LORA_BATCH=8; LORA_GRAD_ACCUM=2
        LORA_RANK=64; USE_QLORA=false
    else
        # 24–47 GB
        CPT_BATCH=2;  CPT_GRAD_ACCUM=8
        SFT_BATCH=2;  SFT_GRAD_ACCUM=8
        LORA_BATCH=4; LORA_GRAD_ACCUM=4
        LORA_RANK=64; USE_QLORA=false
    fi

    log "Hyperparams: CPT batch=${CPT_BATCH}x${CPT_GRAD_ACCUM}  SFT batch=${SFT_BATCH}x${SFT_GRAD_ACCUM}  LoRA batch=${LORA_BATCH}x${LORA_GRAD_ACCUM}  rank=${LORA_RANK}"
}

# =============================================================================
# SECTION 8 — CORPUS AUTO-SELECTION
# =============================================================================

detect_corpus() {
    local raw_dir="${LLM_DIR}/data/raw"
    local v3="${raw_dir}/research_corpus_v3.json"
    local vn="${raw_dir}/research_corpus_new.json"

    if [[ -f "${v3}" ]] && [[ -f "${vn}" ]]; then
        local sz_v3 sz_new
        sz_v3=$(stat -c%s "${v3}")
        sz_new=$(stat -c%s "${vn}")
        if (( sz_v3 >= sz_new )); then
            CORPUS_FILE="${v3}"
            log "Corpus: ${v3} selected (${sz_v3} bytes — larger of two)"
        else
            CORPUS_FILE="${vn}"
            log "Corpus: ${vn} selected (${sz_new} bytes — larger of two)"
        fi
    elif [[ -f "${v3}" ]]; then
        CORPUS_FILE="${v3}"
        log "Corpus: ${v3}"
    elif [[ -f "${vn}" ]]; then
        CORPUS_FILE="${vn}"
        log "Corpus: ${vn}"
    else
        die "No research corpus found in data/raw/.
  Expected: research_corpus_v3.json or research_corpus_new.json"
    fi
}

# =============================================================================
# SECTION 9 — PREFLIGHT CHECKS
# =============================================================================

preflight() {
    echo -e "\n${BOLD}=== PHASE 0: Preflight ===${NC}"
    hr

    # ── PLACEHOLDER guard ─────────────────────────────────────────────────────
    for var in MODEL_NAME MODEL_FAMILY MODEL_SHORTCUT MODEL_HF_BASE MODEL_HF_INSTRUCT; do
        [[ "${!var}" == "PLACEHOLDER" ]] && \
            die "${var} is still PLACEHOLDER.
  Set the model variables at the top of this script; see DEVELOPMENT.md."
    done

    # ── Python / venv + required packages ───────────────────────────────────
    python -c "import torch" 2>/dev/null || \
        die "PyTorch not importable — activate the venv first:
  source .venv/bin/activate"
    python -c "import transformers" 2>/dev/null || die "transformers not installed."
    python -c "import peft"         2>/dev/null || die "peft not installed — run: pip install 'peft>=0.10.0'"
    python -c "import trl"          2>/dev/null || die "trl not installed  — run: pip install 'trl>=0.8.0'"

    # ── GPU / VRAM ────────────────────────────────────────────────────────────
    command -v nvidia-smi &>/dev/null || \
        die "nvidia-smi not found — no NVIDIA GPU detected.
  This pipeline requires a GPU with >= 24 GB VRAM."
    detect_hardware

    # ── Corpus auto-selection ─────────────────────────────────────────────────
    detect_corpus

    # ── Benchmark file ────────────────────────────────────────────────────────
    [[ -f "${BENCHMARK_FILE}" ]] || die "Benchmark file not found: ${BENCHMARK_FILE}"

    # ── Instruct-FTD HPN pairs (required if S2/S3/S5/S8 are enabled) ─────────
    local hpn_needed=false
    ($RUN_S2 || $RUN_S3 || $RUN_S5 || $RUN_S8) && hpn_needed=true
    if $hpn_needed && [[ ! -f "${HPN_INSTRUCT_JSONL}" ]]; then
        die "HPN instruction data not found: ${HPN_INSTRUCT_JSONL}

  Variants S2, S3, S5, and S8 require the HPN instruction pairs (train.jsonl).

  Options:
    (a) Place your train.jsonl at: ${HPN_INSTRUCT_JSONL}
        (mkdir -p data/raw/instruct_ftd  and copy train.jsonl there)
    (b) Update HPN_INSTRUCT_JSONL in Section 3 to wherever you placed the file.
    (c) Skip these variants: set RUN_S2=false RUN_S3=false RUN_S5=false RUN_S8=false
        in Section 2 of this script."
    fi

    # ── NetBench-RAG (required if S6/S7/S8 are enabled) ──────────────────────
    local rag_needed=false
    ($RUN_S6 || $RUN_S7 || $RUN_S8) && rag_needed=true
    if $rag_needed && [[ ! -d "${RAG_DIR}" ]]; then
        die "NetBench-RAG project not found: ${RAG_DIR}

  Variants S6, S7, and S8 require the NetBench-RAG sibling project.

  Options:
    (a) Clone it into the sibling directory:
          git clone <repo-url> $(dirname "${RAG_DIR}")/NetBench-RAG
    (b) Skip RAG variants: set RUN_S6=false RUN_S7=false RUN_S8=false
        in Section 2 of this script."
    fi

    # ── NetBench-RAG internal checks (venv and scripts) ───────────────────────
    if $rag_needed && [[ -d "${RAG_DIR}" ]]; then
        # Virtual environment
        if [[ ! -d "${RAG_DIR}/venv" ]] && [[ ! -d "${RAG_DIR}/.venv" ]]; then
            die "NetBench-RAG virtual environment not found.
  Expected: ${RAG_DIR}/venv/  or  ${RAG_DIR}/.venv/
  Set up the RAG project environment first (see NetBench-RAG README)."
        fi
        # Required scripts
        [[ -f "${RAG_DIR}/index_corpus.py" ]] || \
            die "NetBench-RAG index_corpus.py not found: ${RAG_DIR}/index_corpus.py"
        [[ -f "${RAG_DIR}/evaluation/evaluate_rag.py" ]] || \
            die "NetBench-RAG evaluate_rag.py not found: ${RAG_DIR}/evaluation/evaluate_rag.py"
    fi

    # ── API keys / auth tokens ────────────────────────────────────────────────
    if [[ ! -f "${HF_TOKEN_FILE}" ]]; then
        warn "HuggingFace token not found: ${HF_TOKEN_FILE}
  Model downloads (Phase 1) will fail.
  Fix: python utils/setup_token.py"
    fi

    # Default judge is OpenAI (gpt-5.1); Gemini is the fallback.
    if [[ ! -f "${OPENAI_KEY_FILE}" ]] && [[ ! -f "${GEMINI_KEY_FILE}" ]]; then
        warn "No LLM judge API key found.
  Checked: ${OPENAI_KEY_FILE}
           ${GEMINI_KEY_FILE}
  Phase 6 (LLM-as-judge) will fail unless one of these is added."
    elif [[ ! -f "${OPENAI_KEY_FILE}" ]]; then
        warn "OpenAI API key not found: ${OPENAI_KEY_FILE}
  Default judge (${JUDGE_MODEL}) requires an OpenAI key.
  Either add openai_api_key.txt, or set JUDGE_MODEL to a Gemini model
  (e.g. \"gemini-2.5-flash\") in Section 3."
    fi

    # ── Output dirs ───────────────────────────────────────────────────────────
    mkdir -p "${OUT_ANSWERS}" "${OUT_JUDGED}" "${OUT_REPORTS}" "${OUT_ADAPT}" "${OUT_PROFILE}"

    hr
    ok "Preflight passed — GPUs: ${GPU_COUNT}x${VRAM_GB}GB  Corpus: $(basename "${CORPUS_FILE}")"
}

# =============================================================================
# SECTION 10 — MODEL DOWNLOADS
# =============================================================================

download_models() {
    log "=== PHASE 1: Model Downloads ==="

    # Base model (needed for S4, S7)
    if $RUN_S4 || $RUN_S7; then
        if ! skip_if_exists "base model" "${M_BASE}"; then
            if [[ "${MODEL_SHORTCUT}" != "custom" ]]; then
                run "download base" python utils/download_models.py \
                    --model "${MODEL_SHORTCUT}"
            else
                run "download base (custom HF ID)" python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_HF_BASE}', local_dir='${M_BASE}',
    token=open('${HF_TOKEN_FILE}').read().strip())
print('Done.')
"
            fi
        fi
    fi

    # Official instruct model (needed for S1/S2/S3/S5/S6/S8)
    if $RUN_S1 || $RUN_S2 || $RUN_S3 || $RUN_S5 || $RUN_S6 || $RUN_S8; then
        if ! skip_if_exists "instruct-official model" "${M_INSTRUCT_OFFICIAL}"; then
            run "download official instruct" python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, pathlib
p = pathlib.Path('${M_INSTRUCT_OFFICIAL}'); p.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained('${MODEL_HF_INSTRUCT}',
    token=open('${HF_TOKEN_FILE}').read().strip())
tok.save_pretrained(str(p))
m = AutoModelForCausalLM.from_pretrained('${MODEL_HF_INSTRUCT}',
    torch_dtype=torch.bfloat16, token=open('${HF_TOKEN_FILE}').read().strip())
m.save_pretrained(str(p), safe_serialization=True)
print('Saved to ${M_INSTRUCT_OFFICIAL}')
"
        fi
    fi
}

# =============================================================================
# SECTION 11 — DATA PREPARATION
# =============================================================================

prepare_data() {
    log "=== PHASE 2: Data Preparation ==="

    # 11a. CPT corpus data (full tokenised blocks — for pretrain_transformers.py)
    if ($RUN_S4 || $RUN_S7) && ! skip_if_exists "CPT data" "${DATA_CPT_DIR}"; then
        if [[ "${MODEL_SHORTCUT}" != "custom" ]]; then
            run "prepare CPT data" python training/prepare_data.py \
                --input_file "${CORPUS_FILE}" \
                --model_name "${MODEL_HF_BASE}" \
                --output_dir "${DATA_CPT_DIR}"
        else
            run "prepare CPT data (custom path)" python training/prepare_data.py \
                --input_file "${CORPUS_FILE}" \
                --model_path "${M_BASE}" \
                --output_dir "${DATA_CPT_DIR}"
        fi
    fi

    # 11b. LoRA corpus data (packed CLM blocks — for S5/S8 LoRA CPT step)
    if ($RUN_S5 || $RUN_S8) && ! skip_if_exists "LoRA CPT data" "${DATA_LORA_CPT_DIR}"; then
        run "prepare LoRA CPT data" python training/prepare_lora_data.py \
            --model_path "${M_INSTRUCT_OFFICIAL}" \
            --corpus_file "${CORPUS_FILE}" \
            --output_dir "${DATA_LORA_CPT_DIR}"
    fi

    # 11c. General instruction data (Orca+Dolly — for full SFT in S4 when S4_USE_LORA_SFT=false)
    if ($RUN_S4 && ! $S4_USE_LORA_SFT) && \
       ! skip_if_exists "general SFT data" "${DATA_GENERAL_SFT_DIR}"; then
        run "prepare general SFT data" python training/prepare_instruction_data.py \
            --model_path "${M_BASE}" \
            --output_dir "${DATA_GENERAL_SFT_DIR}" \
            --max_samples "${SFT_MAX_SAMPLES}"
    fi

    # 11d. HPN SFT instruction data (from Instruct-FTD train.jsonl — for S2/S3/S5/S8)
    if ($RUN_S2 || $RUN_S3 || $RUN_S5 || $RUN_S8) && \
       ! skip_if_exists "HPN SFT data" "${DATA_HPN_SFT_DIR}"; then
        run "prepare HPN SFT data" python training/prepare_lora_data.py \
            --model_path "${M_INSTRUCT_OFFICIAL}" \
            --corpus_file "${HPN_INSTRUCT_JSONL}" \
            --output_dir "${DATA_HPN_SFT_DIR}"
    fi
}

# =============================================================================
# SECTION 12 — TRAINING
# =============================================================================

training() {
    log "=== PHASE 3: Training ==="
    stop_ollama  # free GPU VRAM; Ollama restarted at EXIT via trap

    # ── S4: Base -> Full HPN-CPT -> Full/LoRA HPN-SFT ────────────────────────
    if $RUN_S4; then
        # Step 4a: full continual pre-training
        if [[ "${MODEL_SHORTCUT}" != "custom" ]]; then
            run_training "S4 Full HPN-CPT" "${M_PRETRAINED}" \
                python training/pretrain_transformers.py \
                    --model "${MODEL_SHORTCUT}" \
                    --data_dir "${DATA_CPT_DIR}" \
                    --output_dir "${M_PRETRAINED}" \
                    --per_device_train_batch_size "${CPT_BATCH}" \
                    --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
                    --learning_rate "${CPT_LR}" \
                    --num_train_epochs "${CPT_EPOCHS}"
        else
            run_training "S4 Full HPN-CPT (custom)" "${M_PRETRAINED}" \
                python training/pretrain_transformers.py \
                    --model_path "${M_BASE}" \
                    --data_dir "${DATA_CPT_DIR}" \
                    --output_dir "${M_PRETRAINED}" \
                    --per_device_train_batch_size "${CPT_BATCH}" \
                    --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
                    --learning_rate "${CPT_LR}" \
                    --num_train_epochs "${CPT_EPOCHS}"
        fi

        if $S4_USE_LORA_SFT; then
            # Step 4b (LoRA path): LoRA SFT on top of CPT model
            if ! skip_if_exists "S4 LoRA SFT adapter" "${M_S4_LORA_SFT_ADAPTER}"; then
                run "S4 LoRA HPN-SFT" python training/lora_finetune.py \
                    --model_path "${M_PRETRAINED}" \
                    --data_dir "${DATA_HPN_SFT_DIR}" \
                    --output_dir "${M_S4_LORA_SFT_ADAPTER}" \
                    $(lora_args)
            fi
            if ! skip_if_exists "S4 LoRA SFT merged" "${M_S4_LORA_SFT_MERGED}"; then
                run "S4 merge LoRA SFT" python training/merge_lora_adapter.py \
                    --adapter_path "${M_S4_LORA_SFT_ADAPTER}" \
                    --base_model_path "${M_PRETRAINED}" \
                    --output_dir "${M_S4_LORA_SFT_MERGED}"
            fi
        else
            # Step 4b (full path): full SFT on top of CPT model
            run_training "S4 Full HPN-SFT" "${M_S4_FULL_SFT}" \
                python training/instruction_finetune.py \
                    --model_path "${M_PRETRAINED}" \
                    --data_dir "${DATA_GENERAL_SFT_DIR}" \
                    --output_dir "${M_S4_FULL_SFT}" \
                    --num_epochs "${SFT_EPOCHS}" \
                    --batch_size "${SFT_BATCH}" \
                    --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
                    --learning_rate "${SFT_LR}"
        fi
    fi

    # ── S3: Dist Instruct -> Full HPN-SFT ────────────────────────────────────
    if $RUN_S3; then
        run_training "S3 Full HPN-SFT" "${M_S3_FULL_HPN_SFT}" \
            python training/instruction_finetune.py \
                --model_path "${M_INSTRUCT_OFFICIAL}" \
                --data_dir "${DATA_HPN_SFT_DIR}" \
                --output_dir "${M_S3_FULL_HPN_SFT}" \
                --num_epochs "${SFT_EPOCHS}" \
                --batch_size "${SFT_BATCH}" \
                --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
                --learning_rate "${SFT_LR}"
    fi

    # ── S2: Dist Instruct -> LoRA HPN-SFT ────────────────────────────────────
    if $RUN_S2; then
        if ! skip_if_exists "S2 LoRA SFT adapter" "${M_S2_LORA_SFT_ADAPTER}"; then
            run "S2 LoRA HPN-SFT" python training/lora_finetune.py \
                --model_path "${M_INSTRUCT_OFFICIAL}" \
                --data_dir "${DATA_HPN_SFT_DIR}" \
                --output_dir "${M_S2_LORA_SFT_ADAPTER}" \
                $(lora_args)
        fi
        if ! skip_if_exists "S2 LoRA SFT merged" "${M_S2_LORA_SFT_MERGED}"; then
            run "S2 merge LoRA SFT" python training/merge_lora_adapter.py \
                --adapter_path "${M_S2_LORA_SFT_ADAPTER}" \
                --base_model_path "${M_INSTRUCT_OFFICIAL}" \
                --output_dir "${M_S2_LORA_SFT_MERGED}"
        fi
    fi

    # ── S5: Dist Instruct -> LoRA HPN-CPT -> merge -> LoRA HPN-SFT -> merge ──
    if $RUN_S5; then
        if ! skip_if_exists "S5 LoRA CPT adapter" "${M_S5_LORA_CPT_ADAPTER}"; then
            run "S5 LoRA HPN-CPT" python training/lora_finetune.py \
                --model_path "${M_INSTRUCT_OFFICIAL}" \
                --data_dir "${DATA_LORA_CPT_DIR}" \
                --output_dir "${M_S5_LORA_CPT_ADAPTER}" \
                $(lora_args)
        fi
        if ! skip_if_exists "S5 LoRA CPT merged" "${M_S5_LORA_CPT_MERGED}"; then
            run "S5 merge LoRA CPT" python training/merge_lora_adapter.py \
                --adapter_path "${M_S5_LORA_CPT_ADAPTER}" \
                --base_model_path "${M_INSTRUCT_OFFICIAL}" \
                --output_dir "${M_S5_LORA_CPT_MERGED}"
        fi
        if ! skip_if_exists "S5 LoRA SFT adapter" "${M_S5_LORA_SFT_ADAPTER}"; then
            run "S5 LoRA HPN-SFT" python training/lora_finetune.py \
                --model_path "${M_S5_LORA_CPT_MERGED}" \
                --data_dir "${DATA_HPN_SFT_DIR}" \
                --output_dir "${M_S5_LORA_SFT_ADAPTER}" \
                $(lora_args)
        fi
        if ! skip_if_exists "S5 LoRA SFT merged" "${M_S5_LORA_SFT_MERGED}"; then
            run "S5 merge LoRA SFT" python training/merge_lora_adapter.py \
                --adapter_path "${M_S5_LORA_SFT_ADAPTER}" \
                --base_model_path "${M_S5_LORA_CPT_MERGED}" \
                --output_dir "${M_S5_LORA_SFT_MERGED}"
        fi
    fi
}

# =============================================================================
# SECTION 13 — RAG INDEXING
# =============================================================================

rag_index() {
    if ! ($RUN_S6 || $RUN_S7 || $RUN_S8); then return; fi
    log "=== PHASE 4: RAG Index ==="

    [[ -d "${RAG_DIR}" ]] || die "NetBench-RAG not found: ${RAG_DIR}"

    # Skip if index already exists — check the sentinel marker AND the Qdrant
    # storage directory that index_corpus.py creates.
    local rag_marker="${RAG_DIR}/.index_done"
    local qdrant_dir="${RAG_DIR}/qdrant_storage"

    if [[ -f "${rag_marker}" ]] || [[ -d "${qdrant_dir}" ]]; then
        warn "SKIP RAG index — already built (marker or qdrant_storage found)"
        return
    fi

    pushd "${RAG_DIR}" > /dev/null
    source venv/bin/activate 2>/dev/null || source .venv/bin/activate
    run "RAG index corpus" python index_corpus.py
    touch "${rag_marker}"
    deactivate 2>/dev/null || true
    popd > /dev/null
}

# =============================================================================
# SECTION 14 — BENCHMARK GENERATION
# =============================================================================

run_benchmark() {
    local label="$1" model_path="$2"
    local out_file="${OUT_ANSWERS}/hpn_answers_${label}.xlsx"
    if skip_if_exists "benchmark ${label}" "${out_file}"; then return; fi

    run "benchmark ${label}" python evaluation/hpn_qa_benchmark.py \
        --model_path "${model_path}" \
        --benchmark "${BENCHMARK_FILE}" \
        --output_dir "${OUT_ANSWERS}" \
        --temperature 0 \
        --do_sample False
}

run_rag_benchmark() {
    local label="$1" model_path="$2"
    local out_file="${RAG_DIR}/outputs/answers/hpn_answers_RAG-${label}_v4_general_skills.xlsx"
    if skip_if_exists "RAG benchmark ${label}" "${out_file}"; then return; fi

    pushd "${RAG_DIR}" > /dev/null
    source venv/bin/activate 2>/dev/null || source .venv/bin/activate
    run "RAG benchmark ${label}" python evaluation/evaluate_rag.py \
        --model "${model_path}" \
        --benchmark "${BENCHMARK_FILE}"
    deactivate 2>/dev/null || true
    popd > /dev/null
}

benchmarks() {
    log "=== PHASE 5: HPN QA Benchmark Generation ==="

    $RUN_S1 && run_benchmark "S1-${MODEL_NAME}-Instruct-Official"     "${M_INSTRUCT_OFFICIAL}"
    $RUN_S2 && run_benchmark "S2-${MODEL_NAME}-Instruct-LoRA-SFT"     "${M_S2_LORA_SFT_MERGED}"
    $RUN_S3 && run_benchmark "S3-${MODEL_NAME}-Instruct-Full-HPN-SFT" "${M_S3_FULL_HPN_SFT}"

    if $RUN_S4; then
        local s4_final
        $S4_USE_LORA_SFT && s4_final="${M_S4_LORA_SFT_MERGED}" || s4_final="${M_S4_FULL_SFT}"
        run_benchmark "S4-${MODEL_NAME}-CPT-SFT" "${s4_final}"
    fi

    $RUN_S5 && run_benchmark "S5-${MODEL_NAME}-LoRA-CPT-SFT"          "${M_S5_LORA_SFT_MERGED}"

    $RUN_S6 && run_rag_benchmark "S6-${MODEL_NAME}-Instruct-Official"  "${M_INSTRUCT_OFFICIAL}"

    if $RUN_S7; then
        local s7_src
        $S4_USE_LORA_SFT && s7_src="${M_S4_LORA_SFT_MERGED}" || s7_src="${M_S4_FULL_SFT}"
        run_rag_benchmark "S7-${MODEL_NAME}-CPT-SFT" "${s7_src}"
    fi

    $RUN_S8 && run_rag_benchmark "S8-${MODEL_NAME}-LoRA-CPT-SFT"      "${M_S5_LORA_SFT_MERGED}"
}

# =============================================================================
# SECTION 15 — JUDGING
# =============================================================================

judge_all() {
    log "=== PHASE 6: LLM-as-Judge ==="

    # Collect answer files from both LLM-Training and NetBench-RAG outputs.
    local all_answers=()
    mapfile -t all_answers < <(
        find "${OUT_ANSWERS}" -name "hpn_answers_*.xlsx" 2>/dev/null
        [[ -d "${RAG_DIR}/outputs/answers" ]] && \
            find "${RAG_DIR}/outputs/answers" -name "hpn_answers_*.xlsx" 2>/dev/null \
            || true
    )

    if [[ ${#all_answers[@]} -eq 0 ]]; then
        warn "No answer files found — skipping judging."
        return
    fi

    run "judge all answers" python evaluation/judge_responses.py \
        --answer_files "${all_answers[@]}" \
        --judge_model "${JUDGE_MODEL}" \
        --gemini_api_key_file "${GEMINI_KEY_FILE}" \
        --openai_api_key_file "${OPENAI_KEY_FILE}" \
        --output_dir "${OUT_JUDGED}"
}

# =============================================================================
# SECTION 16 — INFERENCE PROFILING
# =============================================================================

profile_model() {
    local label="$1" model_path="$2"
    local out_file="${OUT_PROFILE}/inference_${label}.json"
    if skip_if_exists "profile ${label}" "${out_file}"; then return; fi

    run "profile ${label}" python profiling/profile_inference.py \
        --model_path "${model_path}" \
        --run_name "${label}"
}

profiling() {
    log "=== PHASE 7: Inference Profiling ==="

    $RUN_S1 && profile_model "S1-${MODEL_NAME}-Instruct-Official"   "${M_INSTRUCT_OFFICIAL}"
    $RUN_S2 && profile_model "S2-${MODEL_NAME}-Instruct-LoRA-SFT"   "${M_S2_LORA_SFT_MERGED}"
    $RUN_S3 && profile_model "S3-${MODEL_NAME}-Full-HPN-SFT"        "${M_S3_FULL_HPN_SFT}"

    if $RUN_S4; then
        local s4_final
        $S4_USE_LORA_SFT && s4_final="${M_S4_LORA_SFT_MERGED}" || s4_final="${M_S4_FULL_SFT}"
        profile_model "S4-${MODEL_NAME}-CPT-SFT" "${s4_final}"
    fi

    $RUN_S5 && profile_model "S5-${MODEL_NAME}-LoRA-CPT-SFT" "${M_S5_LORA_SFT_MERGED}"
}

# =============================================================================
# SECTION 17 — LM ADAPTATION REPORTS  (CPT quality + forgetting metrics)
# =============================================================================

adaptation_reports() {
    log "=== PHASE 8: LM Adaptation Reports ==="

    # S4: full CPT report (base -> pretrained)
    if $RUN_S4; then
        local out="${OUT_ADAPT}/adaptation_${MODEL_NAME}_S4_cpt.json"
        if ! skip_if_exists "S4 adaptation report" "${out}"; then
            run "S4 adaptation report" python evaluation/lm_adaptation_report.py \
                --base_model_path    "${M_BASE}" \
                --trained_model_path "${M_PRETRAINED}" \
                --hpn_data_dir       "${DATA_CPT_DIR}" \
                --output_dir         "${OUT_ADAPT}"
        fi
    fi

    # S5: LoRA CPT report (official instruct -> LoRA-CPT-merged)
    if $RUN_S5; then
        local out="${OUT_ADAPT}/adaptation_${MODEL_NAME}_S5_lora_cpt.json"
        if ! skip_if_exists "S5 LoRA-CPT adaptation report" "${out}"; then
            run "S5 LoRA-CPT adaptation report" python evaluation/lm_adaptation_report.py \
                --base_model_path    "${M_INSTRUCT_OFFICIAL}" \
                --trained_model_path "${M_S5_LORA_CPT_MERGED}" \
                --hpn_data_dir       "${DATA_LORA_CPT_DIR}" \
                --output_dir         "${OUT_ADAPT}"
        fi
    fi
}

# =============================================================================
# SECTION 18 — FINAL BENCHMARK COMPARISON REPORT
# =============================================================================

final_report() {
    log "=== PHASE 9: Final Benchmark Comparison Report ==="

    local all_judged=()
    mapfile -t all_judged < <(find "${OUT_JUDGED}" -name "*.xlsx" 2>/dev/null)

    if [[ ${#all_judged[@]} -eq 0 ]]; then
        warn "No judged result files found — skipping comparison report."
        return
    fi

    run "benchmark comparison report" python evaluation/benchmark_report.py \
        --judged_files "${all_judged[@]}" \
        --output_dir "${OUT_REPORTS}"

    ok "Reports written to: ${OUT_REPORTS}"
}

# =============================================================================
# SECTION 19 — MAIN
# =============================================================================

main() {
    echo -e "\n${BOLD}============================================================${NC}"
    echo -e "${BOLD}  HPN LLM Pipeline  —  Model: ${MODEL_NAME}${NC}"
    echo -e "${BOLD}============================================================${NC}\n"

    # Ensure Ollama is restarted at exit regardless of success, failure, or Ctrl+C.
    trap 'restart_ollama' EXIT

    preflight
    download_models
    prepare_data
    training           # stops Ollama internally; EXIT trap restarts it
    rag_index
    benchmarks
    judge_all
    profiling
    adaptation_reports
    final_report

    echo -e "\n${GREEN}${BOLD}=== PIPELINE COMPLETE ===${NC}"
    echo -e "Answers   : ${OUT_ANSWERS}"
    echo -e "Judged    : ${OUT_JUDGED}"
    echo -e "Reports   : ${OUT_REPORTS}"
    echo -e "Profiling : ${OUT_PROFILE}"
    echo -e "Adaptation: ${OUT_ADAPT}\n"
}

main
