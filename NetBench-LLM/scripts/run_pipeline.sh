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
# Inference profiling writes resumable JSON checkpoints after each completed
# prompt/question; reruns skip completed work and finish the same report file.
# Other non-training steps skip if their output file / directory already exists.
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
# • HPN instruction JSON/JSONL present if S2/S3/S5/S8 are enabled.
# • NetBench-RAG project present if S6/S7/S8 are enabled.
# • API key files present for model download and LLM-as-judge.
#
# BEFORE RUNNING: follow scripts/ADAPT_PROMPT.md to fill in all
# PLACEHOLDER values and tune hyperparameters for your machine.
#
# USAGE:
#   cd /path/to/LLM-Training
#   source .venv/bin/activate
#   bash scripts/run_pipeline.sh 2>&1 | tee outputs/pipeline_run.log
#
# Long runs: use GNU screen so the workflow survives SSH disconnects.
#   screen -S hpn-<MODEL_NAME>
#   bash scripts/run_pipeline.sh 2>&1 | tee outputs/pipeline_run.log
#
# Or let the script start a detached screen session named from the model:
# hpn-<model-name>-<timestamp>
#   USE_GNU_SCREEN=true bash scripts/run_pipeline.sh
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

MODEL_NAME="gemma-3-12b"         # Short label used in output paths.
                                # e.g. "Qwen3.5-2B", "Llama-3.1-8B", "Gemma3-4B"

MODEL_FAMILY="gemma"            # "llama" | "qwen" | "gemma"
                                # Used to pick prompt templates and LoRA target modules.

MODEL_SHORTCUT="custom"         # pretrain_transformers.py shortcut key, OR "custom".
                                # Registered: 1b | 8b | qwen-2b | qwen-4b | gemma-4b
                                #             gemma-e4b | gemma-e2b
                                # Use "custom" for unregistered models.

MODEL_HF_BASE="google/gemma-3-12b-pt"  # HuggingFace repo ID for the BASE model.
                                # e.g. "meta-llama/Llama-3.1-8B"
                                #      "Qwen/Qwen3.5-2B-Base"
                                #      "google/gemma-3-4b-pt"

MODEL_HF_INSTRUCT="google/gemma-3-12b-it"  # HuggingFace repo ID for the OFFICIAL instruct model.
                                # e.g. "meta-llama/Llama-3.1-8B-Instruct"
                                #      "Qwen/Qwen3.5-2B-Instruct"
                                #      "google/gemma-3-4b-it"

# Optional local path overrides. Leave empty for registered shortcuts.
MODEL_BASE_DIR="models/base/gemma-3-12b"
MODEL_INSTRUCT_DIR="models/instruct-official/gemma-3-12b-it"
MODEL_PRETRAINED_DIR="models/pretrained/gemma-3-12b-trained-new"

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
S4_FULL_SFT_DATASET="hpn" # hpn = Instruct-FTD; general = OpenOrca+Dolly fallback
RUN_S5=true    # Dist. Instruct  + LoRA HPN-CPT + LoRA HPN-SFT
RUN_S6=true    # Dist. Instruct  + RAG
RUN_S7=true    # Base + Full HPN-CPT + HPN-SFT + RAG
RUN_S8=true    # Dist. Instruct  + LoRA HPN-CPT + LoRA HPN-SFT + RAG

# =============================================================================
# SECTION 3 — PATHS  (initial values; resolve_model_identity recalculates them)
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
# Current bundled Instruct-FTD data. This may be a JSON/JSONL file or a
# directory containing train/validation JSON or JSONL files.
HPN_INSTRUCT_JSONL="${LLM_DIR}/data/Instruct-FTD/v3_run_plus_json"
BENCHMARK_FILE="${LLM_DIR}/data/prompts/hpn_benchmark_v5.0.jsonl"

DATA_CPT_DIR="${LLM_DIR}/data/processed/${MODEL_NAME,,}"
DATA_LORA_CPT_DIR="${LLM_DIR}/data/processed/lora/${MODEL_NAME,,}"
DATA_GENERAL_SFT_DIR="${LLM_DIR}/data/instruction/${MODEL_NAME,,}-general"
DATA_HPN_SFT_DIR="${LLM_DIR}/data/instruction/${MODEL_NAME,,}-hpn"

# Output directories — all scoped under outputs/<model-name>/ so runs for
# different models never intermix.
OUT_BASE="${LLM_DIR}/outputs/${MODEL_NAME,,}"
OUT_ANSWERS="${OUT_BASE}/evaluations/answers"
OUT_JUDGED="${OUT_BASE}/evaluations/judged"
OUT_REPORTS="${OUT_BASE}/evaluations/reports"
OUT_ADAPT="${OUT_BASE}/evaluations/adaptation_reports"
OUT_PROFILE="${OUT_BASE}/profiling_results"

# API / auth
JUDGE_MODEL="gpt-5.1"           # default OpenAI judge; switch to a Gemini model if no OpenAI key
GEMINI_KEY_FILE="${LLM_DIR}/gemini_api_key.txt"
OPENAI_KEY_FILE="${LLM_DIR}/openai_api_key.txt"
HF_TOKEN_FILE="${LLM_DIR}/hf_token.txt"

# Leave empty to reuse the currently active LLM virtualenv for NetBench-RAG.
# Set to an absolute or project-relative path if you want a separate RAG venv.
RAG_VENV_DIR=""
DS_ZERO3_CONFIG="${LLM_DIR}/configs/ds_zero3.json"

# GNU screen support for long runs. Keep false for normal foreground execution.
# Set USE_GNU_SCREEN=true to relaunch this script in a detached screen session.
# The session name is always generated from MODEL_NAME: hpn-<model-name>-<timestamp>.
USE_GNU_SCREEN="${USE_GNU_SCREEN:-false}"
SCREEN_LOG_FILE="${SCREEN_LOG_FILE:-}"

is_gemma_3_12b() {
    [[ "${MODEL_NAME,,}" == "gemma-3-12b" || \
       "${MODEL_HF_BASE}" == "google/gemma-3-12b-pt" || \
       "${MODEL_HF_INSTRUCT}" == "google/gemma-3-12b-it" ]]
}

use_distributed_full_training() {
    (( GPU_COUNT > 1 )) && ! is_gemma_3_12b
}

# Inference profiling controls. Profiling uses benchmark questions instead of
# synthetic prompts so direct and RAG reports are comparable.
PROFILE_BENCHMARK_LIMIT=5
PROFILE_MAX_NEW_TOKENS=256
PROFILE_NUM_RUNS=1
PROFILE_WARMUP_RUNS=1
RUN_DIRECT_GPU_PROFILING=true
RUN_DIRECT_CPU_PROFILING=true
RUN_RAG_GPU_PROFILING=true
RUN_RAG_CPU_PROFILING=true

# Selected variants for profiling. Enable S2/S3/S6 here if you need broader
# profiling, but the default comparison focuses on base instruct, S4, S5, S7,
# and S8.
PROFILE_DIRECT_S1=true
PROFILE_DIRECT_S2=false
PROFILE_DIRECT_S3=false
PROFILE_DIRECT_S4=true
PROFILE_DIRECT_S5=true
PROFILE_RAG_S6=false
PROFILE_RAG_S7=true
PROFILE_RAG_S8=true

# =============================================================================
# SECTION 4 — HARDWARE HYPERPARAMETERS  (auto-tuned by detect_hardware)
# =============================================================================

GPU_COUNT=0    # set by detect_hardware()
VRAM_GB=0      # per-GPU VRAM in GB — set by detect_hardware()

# CPT hyperparameters (full-weight continual pre-training)
# Defaults tuned conservatively for Gemma 3 12B; detect_hardware() refreshes
# them at runtime while preserving this 12B-safe profile.
CPT_BATCH=1
CPT_GRAD_ACCUM=4
CPT_LR="2e-5"
CPT_EPOCHS=3

# SFT hyperparameters (full-weight instruction fine-tuning)
SFT_BATCH=1
SFT_GRAD_ACCUM=4
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

# Batch size for benchmark answer generation (hpn_qa_benchmark.py --batch_size).
# With 4× H100 and small models (1B–8B) 16 works well. Gemma 3 12B uses a
# smaller answer-generation batch to leave room for long 512-token outputs.
BENCH_BATCH_SIZE=4

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

is_placeholder() {
    local value="${1:-}"
    [[ -z "${value}" || "${value}" == "PLACEHOLDER" || "${value}" == "auto" || "${value}" == "AUTO" ]]
}

set_if_placeholder() {
    local var="$1" value="$2"
    if is_placeholder "${!var:-}"; then
        printf -v "${var}" '%s' "${value}"
    fi
}

path_from_llm_root() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${LLM_DIR}/${path}"
    fi
}

path_from_project_root() {
    local root="$1" path="$2"
    if [[ -z "${path}" ]]; then
        printf '\n'
    elif [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${root}/${path}"
    fi
}

model_slug() {
    local slug="${MODEL_NAME,,}"
    slug="${slug// /-}"
    slug="${slug//\//-}"
    printf '%s\n' "${slug}"
}

default_screen_session_name() {
    printf 'hpn-%s-%s\n' "$(model_slug)" "$(date '+%Y%m%d-%H%M%S')"
}

screen_log_file() {
    local session="$1"
    if [[ -n "${SCREEN_LOG_FILE}" ]]; then
        path_from_llm_root "${SCREEN_LOG_FILE}"
    else
        printf '%s\n' "${LLM_DIR}/outputs/${MODEL_NAME,,}/${session}.log"
    fi
}

maybe_start_screen_session() {
    if [[ "${USE_GNU_SCREEN}" != "true" ]]; then
        if [[ -z "${STY:-}" && -z "${TMUX:-}" ]]; then
            warn "Long pipeline is not running inside screen/tmux.
  Recommended for SSH runs:
    USE_GNU_SCREEN=true bash ${BASH_SOURCE[0]}
  Or manually:
    screen -S hpn-$(model_slug)"
        fi
        return
    fi

    if [[ -n "${STY:-}" ]]; then
        log "Already inside GNU screen session: ${STY}"
        return
    fi

    command -v screen >/dev/null 2>&1 || \
        die "USE_GNU_SCREEN=true but GNU screen is not installed or not on PATH."

    local session log_file venv_activate module_cmds
    session="$(default_screen_session_name)"
    log_file="$(screen_log_file "${session}")"
    mkdir -p "$(dirname "${log_file}")"

    if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
        venv_activate="${VIRTUAL_ENV}/bin/activate"
    elif [[ -f "${LLM_DIR}/.venv/bin/activate" ]]; then
        venv_activate="${LLM_DIR}/.venv/bin/activate"
    else
        venv_activate=""
    fi

    # Reconstruct module load commands from the currently loaded modules so the
    # screen child (a non-login shell) gets the same HPC environment.
    # Uses LOADEDMODULES set by Lmod; safe no-op if Lmod is not present.
    module_cmds=""
    if [[ -n "${LOADEDMODULES:-}" && -n "${MODULESHOME:-}" ]]; then
        module_cmds="source '${MODULESHOME}/init/bash' 2>/dev/null || true"
        local mod
        while IFS= read -r mod; do
            [[ -n "${mod}" ]] && module_cmds="${module_cmds} && module load '${mod}'"
        done < <(tr ':' '\n' <<< "${LOADEDMODULES}")
        module_cmds="${module_cmds} &&"
    fi

    local venv_cmd=""
    [[ -n "${venv_activate}" ]] && venv_cmd="source '${venv_activate}' &&"

    log "Starting detached GNU screen session: ${session}"
    log "Log file: ${log_file}"
    # Use 'bash -c' (not login) so the child inherits the parent environment
    # unchanged. Module loads and venv activation are replayed explicitly above.
    screen -dmS "${session}" bash -c \
        "${module_cmds} ${venv_cmd} cd '${LLM_DIR}' && USE_GNU_SCREEN=false bash '${BASH_SOURCE[0]}' 2>&1 | tee '${log_file}'"
    ok "Detached. Reattach with: screen -r ${session}"
    exit 0
}

benchmark_tag() {
    local stem
    stem="$(basename "${BENCHMARK_FILE}")"
    stem="${stem%.*}"
    case "${stem}" in
        hpn_qa_benchmark_*) printf '_%s\n' "${stem#hpn_qa_benchmark_}" ;;
        hpn_benchmark_*)    printf '_%s\n' "${stem#hpn_benchmark_}" ;;
        *)                  printf '\n' ;;
    esac
}

# Resolve model identity and local paths from MODEL_SHORTCUT unless the user
# explicitly overrides paths in Section 1.
resolve_model_identity() {
    if is_placeholder "${MODEL_SHORTCUT}"; then
        MODEL_SHORTCUT="custom"
    fi

    local default_base="" default_pretrained="" default_instruct=""
    case "${MODEL_SHORTCUT}" in
        1b)
            set_if_placeholder MODEL_NAME "Llama-3.2-1B"
            set_if_placeholder MODEL_FAMILY "llama"
            set_if_placeholder MODEL_HF_BASE "meta-llama/Llama-3.2-1B"
            set_if_placeholder MODEL_HF_INSTRUCT "meta-llama/Llama-3.2-1B-Instruct"
            default_base="models/base/Llama-3.2-1B-base"
            default_pretrained="models/pretrained/Llama-3.2-1B-trained-new"
            default_instruct="models/instruct-official/Llama-3.2-1B-Instruct"
            ;;
        8b)
            set_if_placeholder MODEL_NAME "Llama-3.1-8B"
            set_if_placeholder MODEL_FAMILY "llama"
            set_if_placeholder MODEL_HF_BASE "meta-llama/Llama-3.1-8B"
            set_if_placeholder MODEL_HF_INSTRUCT "meta-llama/Llama-3.1-8B-Instruct"
            default_base="models/base/Llama-3.1-8B-base"
            default_pretrained="models/pretrained/Llama-3.1-8B-trained-new"
            default_instruct="models/instruct-official/Llama-3.1-8B-Instruct"
            ;;
        qwen-2b)
            set_if_placeholder MODEL_NAME "Qwen3.5-2B"
            set_if_placeholder MODEL_FAMILY "qwen"
            set_if_placeholder MODEL_HF_BASE "Qwen/Qwen3.5-2B-Base"
            set_if_placeholder MODEL_HF_INSTRUCT "Qwen/Qwen3.5-2B-Instruct"
            default_base="models/base/Qwen3.5-2B-Base"
            default_pretrained="models/pretrained/Qwen3.5-2B-trained-new"
            default_instruct="models/instruct-official/Qwen3.5-2B-Instruct"
            ;;
        qwen-4b)
            set_if_placeholder MODEL_NAME "Qwen3.5-4B"
            set_if_placeholder MODEL_FAMILY "qwen"
            set_if_placeholder MODEL_HF_BASE "Qwen/Qwen3.5-4B-Base"
            set_if_placeholder MODEL_HF_INSTRUCT "Qwen/Qwen3.5-4B-Instruct"
            default_base="models/base/Qwen3.5-4B-Base"
            default_pretrained="models/pretrained/Qwen3.5-4B-trained-new"
            default_instruct="models/instruct-official/Qwen3.5-4B-Instruct"
            ;;
        gemma-4b)
            set_if_placeholder MODEL_NAME "gemma-3-4b"
            set_if_placeholder MODEL_FAMILY "gemma"
            set_if_placeholder MODEL_HF_BASE "google/gemma-3-4b-pt"
            set_if_placeholder MODEL_HF_INSTRUCT "google/gemma-3-4b-it"
            default_base="models/base/gemma-3-4b"
            default_pretrained="models/pretrained/gemma-3-4b-trained-new"
            default_instruct="models/instruct-official/gemma-3-4b-it"
            ;;
        gemma-e4b)
            set_if_placeholder MODEL_NAME "gemma-4-e4b"
            set_if_placeholder MODEL_FAMILY "gemma"
            set_if_placeholder MODEL_HF_BASE "google/gemma-4-E4B"
            set_if_placeholder MODEL_HF_INSTRUCT "google/gemma-4-E4B-it"
            default_base="models/base/gemma-4-e4b"
            default_pretrained="models/pretrained/gemma-4-e4b-trained-new"
            default_instruct="models/instruct-official/gemma-4-e4b-it"
            ;;
        gemma-e2b)
            set_if_placeholder MODEL_NAME "gemma-4-e2b"
            set_if_placeholder MODEL_FAMILY "gemma"
            set_if_placeholder MODEL_HF_BASE "google/gemma-4-E2B"
            set_if_placeholder MODEL_HF_INSTRUCT "google/gemma-4-E2B-it"
            default_base="models/base/gemma-4-e2b"
            default_pretrained="models/pretrained/gemma-4-e2b-trained-new"
            default_instruct="models/instruct-official/gemma-4-e2b-it"
            ;;
        custom)
            default_base="models/base/${MODEL_NAME}-Base"
            default_pretrained="models/pretrained/${MODEL_NAME}-trained-new"
            default_instruct="models/instruct-official/${MODEL_NAME}-Instruct"
            ;;
        *)
            die "Unknown MODEL_SHORTCUT=${MODEL_SHORTCUT}. Use 1b, 8b, qwen-2b, qwen-4b, gemma-4b, gemma-e4b, gemma-e2b, or custom."
            ;;
    esac

    [[ -z "${MODEL_BASE_DIR}" ]] && MODEL_BASE_DIR="${default_base}"
    [[ -z "${MODEL_PRETRAINED_DIR}" ]] && MODEL_PRETRAINED_DIR="${default_pretrained}"
    [[ -z "${MODEL_INSTRUCT_DIR}" ]] && MODEL_INSTRUCT_DIR="${default_instruct}"

    M_BASE="$(path_from_llm_root "${MODEL_BASE_DIR}")"
    M_PRETRAINED="$(path_from_llm_root "${MODEL_PRETRAINED_DIR}")"
    M_INSTRUCT_OFFICIAL="$(path_from_llm_root "${MODEL_INSTRUCT_DIR}")"

    local slug
    slug="$(model_slug)"
    DATA_CPT_DIR="${LLM_DIR}/data/processed/${slug}"
    DATA_LORA_CPT_DIR="${LLM_DIR}/data/processed/lora/${slug}"
    DATA_GENERAL_SFT_DIR="${LLM_DIR}/data/instruction/${slug}-general"
    DATA_HPN_SFT_DIR="${LLM_DIR}/data/instruction/${slug}-hpn"

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
}

# Skip if the sentinel file / directory already exists (non-training steps).
skip_if_exists() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        warn "SKIP ${label} — already exists: ${path}"
        return 0
    fi
    return 1
}

skip_if_complete() {
    local label="$1" dir="$2" sentinel="$3"
    if [[ -f "${dir}/${sentinel}" ]]; then
        warn "SKIP ${label} — already complete: ${dir}"
        return 0
    fi
    return 1
}

profile_json_is_complete() {
    local path="$1"
    local expected_model_path="${2:-}"
    [[ -f "${path}" ]] || return 1
    python - "${path}" "${expected_model_path}" <<'PY'
import json
import sys
from pathlib import Path

path = sys.argv[1]
expected_model_path = sys.argv[2] if len(sys.argv) > 2 else ""
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

if expected_model_path:
    actual_model_path = data.get("model_path")
    if not actual_model_path:
        sys.exit(1)
    if Path(actual_model_path).resolve() != Path(expected_model_path).resolve():
        sys.exit(1)

status = data.get("status")
checkpoint = data.get("checkpoint")

if status == "complete":
    sys.exit(0)
if status in {"in_progress", "partial", "running", "failed"}:
    sys.exit(1)
if isinstance(checkpoint, dict) and checkpoint.get("complete") is False:
    sys.exit(1)

# Backward compatibility: reports created before checkpointing had no status.
if data.get("summary") is not None and isinstance(data.get("benchmark_prompts"), list):
    sys.exit(0)
sys.exit(1)
PY
}

promote_profile_checkpoint_if_available() {
    local target="$1"
    [[ -e "${target}" ]] && return 0

    local filename target_dir legacy_dir legacy_path slug
    filename="$(basename "${target}")"
    target_dir="$(dirname "${target}")"
    slug="$(model_slug)"

    for legacy_dir in \
        "${LLM_DIR}/outputs/profiling_results" \
        "${LLM_DIR}/outputs/by_model/${slug}/profiling_results"
    do
        [[ "${legacy_dir}" == "${target_dir}" ]] && continue
        legacy_path="${legacy_dir}/${filename}"
        if [[ -f "${legacy_path}" ]]; then
            mkdir -p "${target_dir}"
            cp -p "${legacy_path}" "${target}"
            warn "Imported existing profiling checkpoint: ${legacy_path} -> ${target}"
            return 0
        fi
    done
    return 1
}

skip_profile_if_complete() {
    local label="$1" path="$2" expected_model_path="${3:-}"
    promote_profile_checkpoint_if_available "${path}" || true

    if [[ ! -e "${path}" ]]; then
        return 1
    fi
    if profile_json_is_complete "${path}" "${expected_model_path}"; then
        warn "SKIP ${label} — already complete: ${path}"
        return 0
    fi
    warn "RESUME ${label} — partial profiling checkpoint: ${path}"
    return 1
}

activate_rag_env() {
    RAG_ENV_ACTIVATED=false

    local candidates=()
    if [[ -n "${RAG_VENV_DIR}" ]]; then
        candidates+=("$(path_from_project_root "${LLM_DIR}" "${RAG_VENV_DIR}")")
    fi
    candidates+=("${RAG_DIR}/venv" "${RAG_DIR}/.venv")

    local venv
    for venv in "${candidates[@]}"; do
        if [[ -f "${venv}/bin/activate" ]]; then
            # shellcheck disable=SC1090
            source "${venv}/bin/activate"
            RAG_ENV_ACTIVATED=true
            log "Using RAG Python environment: ${venv}"
            return 0
        fi
    done

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log "Using active Python environment for RAG: ${VIRTUAL_ENV}"
        return 0
    fi

    warn "No dedicated RAG virtualenv found and VIRTUAL_ENV is not set; using current shell Python."
}

deactivate_rag_env() {
    if ${RAG_ENV_ACTIVATED:-false}; then
        deactivate 2>/dev/null || true
        RAG_ENV_ACTIVATED=false
    fi
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
#   run_training_with_sentinel LABEL OUTPUT_DIR SENTINEL CMD [ARGS...]
#
#   Completion:  OUTPUT_DIR/<sentinel> present  => skip entirely.
#   Partial:     OUTPUT_DIR/checkpoints/checkpoint-<N>/ present but no sentinel
#                => appends --resume_from_checkpoint <last-ckpt> automatically.
#   Fresh start: neither => runs from scratch.
#
latest_checkpoint() {
    local outdir="$1"
    local ckpt=""
    local candidates=()

    mapfile -t candidates < <(
        find "${outdir}/checkpoints" "${outdir}" \
            -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null \
            | sort -V -r
    )

    for ckpt in "${candidates[@]}"; do
        if checkpoint_has_finite_metrics "${ckpt}"; then
            printf '%s\n' "${ckpt}"
            return 0
        fi
        warn "Ignoring checkpoint with non-finite metrics: ${ckpt}"
    done
    printf '\n'
}

checkpoint_has_finite_metrics() {
    local ckpt="$1"
    local state="${ckpt}/trainer_state.json"
    [[ -f "${state}" ]] || return 0
    python - "${state}" <<'PY'
import json
import math
import sys

state_path = sys.argv[1]
try:
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
except Exception:
    sys.exit(1)

for row in state.get("log_history", []):
    for key in ("loss", "eval_loss", "grad_norm"):
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            sys.exit(1)
sys.exit(0)
PY
}

output_has_nonfinite_checkpoint_metrics() {
    local outdir="$1"
    local ckpt=""
    local found_bad=false

    while IFS= read -r ckpt; do
        [[ -z "${ckpt}" ]] && continue
        if ! checkpoint_has_finite_metrics "${ckpt}"; then
            warn "Found checkpoint with non-finite metrics: ${ckpt}"
            found_bad=true
        fi
    done < <(
        find "${outdir}/checkpoints" "${outdir}" \
            -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null \
            | sort -V -r
    )

    ${found_bad}
}

run_training_with_sentinel() {
    local label="$1" outdir="$2" sentinel="$3"; shift 3

    if [[ -f "${outdir}/${sentinel}" ]]; then
        if output_has_nonfinite_checkpoint_metrics "${outdir}"; then
            warn "NOT skipping ${label} — existing output has non-finite checkpoint metrics and will be retrained."
        else
        warn "SKIP ${label} — already complete: ${outdir}"
        return 0
        fi
    fi

    local ckpt=""
    ckpt="$(latest_checkpoint "${outdir}")"

    if [[ -n "${ckpt}" ]]; then
        log "RESUME ${label} from checkpoint: ${ckpt}"
        run "${label}" "$@" --resume_from_checkpoint "${ckpt}"
    else
        run "${label}" "$@"
    fi
}

run_training() {
    local label="$1" outdir="$2"; shift 2
    run_training_with_sentinel "${label}" "${outdir}" "config.json" "$@"
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
    local gpu_names vram_values vram_mib
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    vram_values=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    GPU_COUNT=$(awk 'END { print NR }' <<< "${gpu_names}")
    vram_mib=$(awk 'NR == 1 { print int($1); exit }' <<< "${vram_values}")
    [[ -n "${vram_mib}" ]] || vram_mib=0
    VRAM_GB=$(( (vram_mib + 1023) / 1024 ))

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

    if [[ "${MODEL_FAMILY}" == "gemma" ]]; then
        LORA_BATCH=4
        LORA_GRAD_ACCUM=4
        warn "Gemma LoRA uses batch=${LORA_BATCH}x${LORA_GRAD_ACCUM} to avoid fp32 logits OOM during loss computation."
    fi

    if is_gemma_3_12b; then
        local safe_accum
        # Gemma 3 12B full CPT/SFT uses single-process model sharding because
        # torchrun+ZeRO-3 OOMs without Gemma gradient checkpointing.  Accumulate
        # to an actual effective batch of 16 in that non-data-parallel path.
        safe_accum=16
        (( safe_accum < 1 )) && safe_accum=1
        CPT_BATCH=1;  CPT_GRAD_ACCUM="${safe_accum}"
        SFT_BATCH=1;  SFT_GRAD_ACCUM="${safe_accum}"
        LORA_BATCH=4; LORA_GRAD_ACCUM=4
        BENCH_BATCH_SIZE=4
        warn "Gemma 3 12B profile active: full CPT/SFT uses model-sharded single-process training batch=${CPT_BATCH}x${CPT_GRAD_ACCUM}; LoRA batch=${LORA_BATCH}x${LORA_GRAD_ACCUM}; benchmark batch=${BENCH_BATCH_SIZE}."
    fi

    log "Hyperparams: CPT batch=${CPT_BATCH}x${CPT_GRAD_ACCUM}  SFT batch=${SFT_BATCH}x${SFT_GRAD_ACCUM}  LoRA batch=${LORA_BATCH}x${LORA_GRAD_ACCUM}  rank=${LORA_RANK}"
}

# =============================================================================
# SECTION 8 — CORPUS AUTO-SELECTION
# =============================================================================

detect_corpus() {
    local raw_dir="${LLM_DIR}/data/raw"
    local candidates=(
        "${raw_dir}/research_corpus_v3.json"
        "${raw_dir}/research_corpus_v3.jsonl"
        "${raw_dir}/research_corpus_new.json"
        "${raw_dir}/research_corpus_new.jsonl"
    )
    local best="" best_size=0 path size
    for path in "${candidates[@]}"; do
        if [[ -f "${path}" ]]; then
            size=$(stat -c%s "${path}")
            if (( size > best_size )); then
                best="${path}"
                best_size="${size}"
            fi
        fi
    done

    if [[ -n "${best}" ]]; then
        CORPUS_FILE="${best}"
        log "Corpus: ${best} selected (${best_size} bytes — largest candidate)"
    else
        die "No research corpus found in data/raw/.
  Expected one of: research_corpus_v3.json, research_corpus_v3.jsonl,
  research_corpus_new.json, research_corpus_new.jsonl"
    fi
}

# =============================================================================
# SECTION 9 — PREFLIGHT CHECKS
# =============================================================================

preflight() {
    echo -e "\n${BOLD}=== PHASE 0: Preflight ===${NC}"
    hr

    resolve_model_identity

    # ── PLACEHOLDER guard ─────────────────────────────────────────────────────
    for var in MODEL_NAME MODEL_FAMILY MODEL_SHORTCUT MODEL_HF_BASE MODEL_HF_INSTRUCT; do
        is_placeholder "${!var}" && \
            die "${var} is still PLACEHOLDER.
  Follow scripts/ADAPT_PROMPT.md with Claude Code to fill in all values."
    done
    [[ "${MODEL_FAMILY}" =~ ^(llama|qwen|gemma)$ ]] || \
        die "MODEL_FAMILY must be one of: llama, qwen, gemma. Got: ${MODEL_FAMILY}"
    log "Model: ${MODEL_NAME} (${MODEL_FAMILY}, shortcut=${MODEL_SHORTCUT})"
    log "Base dir: ${M_BASE}"
    log "Official instruct dir: ${M_INSTRUCT_OFFICIAL}"

    # ── Python / venv + required packages ───────────────────────────────────
    python -c "import torch" 2>/dev/null || \
        die "PyTorch not importable — activate the venv first:
  source .venv/bin/activate"
    python -c "import transformers" 2>/dev/null || die "transformers not installed."
    python -c "import peft"         2>/dev/null || die "peft not installed — run: pip install 'peft>=0.10.0'"
    python -c "import trl"          2>/dev/null || die "trl not installed  — run: pip install 'trl>=0.8.0'"
    python -c "import tensorboard"  2>/dev/null || die "tensorboard not installed — run: pip install 'tensorboard>=2.16.0'"

    # ── GPU / VRAM ────────────────────────────────────────────────────────────
    command -v nvidia-smi &>/dev/null || \
        die "nvidia-smi not found — no NVIDIA GPU detected.
  This pipeline requires a GPU with >= 24 GB VRAM."
    detect_hardware
    if use_distributed_full_training && { $RUN_S3 || $RUN_S4; }; then
        command -v torchrun &>/dev/null || die "torchrun not found in PATH; activate the project venv first."
        [[ -f "${DS_ZERO3_CONFIG}" ]] || die "DeepSpeed ZeRO-3 config not found: ${DS_ZERO3_CONFIG}"
    fi

    # ── Corpus auto-selection ─────────────────────────────────────────────────
    detect_corpus

    # ── Benchmark file ────────────────────────────────────────────────────────
    [[ -f "${BENCHMARK_FILE}" ]] || die "Benchmark file not found: ${BENCHMARK_FILE}"

    # ── Instruct-FTD HPN pairs (required if S2/S3/S5/S8 are enabled) ─────────
    local hpn_needed=false
    if $RUN_S2 || $RUN_S3 || { $RUN_S4 && { $S4_USE_LORA_SFT || [[ "${S4_FULL_SFT_DATASET}" == "hpn" ]]; }; } || $RUN_S5 || $RUN_S8; then
        hpn_needed=true
    fi
    if $hpn_needed && [[ ! -f "${HPN_INSTRUCT_JSONL}" && ! -d "${HPN_INSTRUCT_JSONL}" ]]; then
        die "HPN instruction data not found: ${HPN_INSTRUCT_JSONL}

  Variants S2, S3, S4-with-LoRA-SFT, S5, and S8 require HPN instruction pairs.

  Options:
    (a) Place train.jsonl at: ${HPN_INSTRUCT_JSONL}
    (b) Set HPN_INSTRUCT_JSONL to a JSON/JSONL file or a directory containing
        train.jsonl and optional validation.jsonl.
    (c) Skip affected variants: set RUN_S2=false RUN_S3=false RUN_S5=false RUN_S8=false
        and keep S4_USE_LORA_SFT=false if RUN_S4 remains enabled.
        in Section 2 of this script."
    fi

    # ── NetBench-RAG (required if S6/S7/S8 are enabled) ──────────────────────
    local rag_needed=false
    ($RUN_S6 || $RUN_S7 || $RUN_S8) && rag_needed=true
    if $RUN_RAG_GPU_PROFILING || $RUN_RAG_CPU_PROFILING; then
        ($PROFILE_RAG_S6 && $RUN_S6) && rag_needed=true
        ($PROFILE_RAG_S7 && $RUN_S7) && rag_needed=true
        ($PROFILE_RAG_S8 && $RUN_S8) && rag_needed=true
    fi
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
        local rag_venv_abs=""
        [[ -n "${RAG_VENV_DIR}" ]] && rag_venv_abs="$(path_from_project_root "${LLM_DIR}" "${RAG_VENV_DIR}")"
        if [[ -n "${rag_venv_abs}" && ! -f "${rag_venv_abs}/bin/activate" ]]; then
            die "Configured RAG_VENV_DIR does not contain bin/activate: ${rag_venv_abs}"
        elif [[ -z "${rag_venv_abs}" && ! -f "${RAG_DIR}/venv/bin/activate" && ! -f "${RAG_DIR}/.venv/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
            warn "No dedicated NetBench-RAG virtualenv found. The script will reuse the active LLM venv/current Python; install both requirements files into that environment."
        fi
        # Required scripts
        [[ -f "${RAG_DIR}/index_corpus.py" ]] || \
            die "NetBench-RAG index_corpus.py not found: ${RAG_DIR}/index_corpus.py"
        [[ -f "${RAG_DIR}/evaluation/evaluate_rag.py" ]] || \
            die "NetBench-RAG evaluate_rag.py not found: ${RAG_DIR}/evaluation/evaluate_rag.py"
    fi

    # ── Profiling scripts ────────────────────────────────────────────────────
    [[ -f "${LLM_DIR}/profiling/profile_inference.py" ]] || \
        die "Direct profiling script not found: ${LLM_DIR}/profiling/profile_inference.py"
    [[ -f "${LLM_DIR}/profiling/profile_rag_inference.py" ]] || \
        die "RAG profiling script not found: ${LLM_DIR}/profiling/profile_rag_inference.py"
    [[ -f "${LLM_DIR}/profiling/summarize_inference_profiles.py" ]] || \
        die "Profiling summary script not found: ${LLM_DIR}/profiling/summarize_inference_profiles.py"
    python -c "import yaml" 2>/dev/null || \
        warn "PyYAML not importable in active environment; RAG profiling requires pyYAML from NetBench-RAG requirements."

    # ── API keys / auth tokens ────────────────────────────────────────────────
    if [[ ! -f "${HF_TOKEN_FILE}" ]]; then
        warn "HuggingFace token not found: ${HF_TOKEN_FILE}
  Model downloads (Phase 1) will fail.
  Fix: python utils/setup_token.py"
    else
        python -c "from utils.setup_token import load_hf_token; load_hf_token('${HF_TOKEN_FILE}')" || \
            die "HuggingFace token file is present but invalid: ${HF_TOKEN_FILE}
  Keep only the token line, or leave comments on separate lines beginning with #."
    fi

    # Default judge is OpenAI (gpt-5.1); Gemini is the fallback.
    if [[ ! -f "${OPENAI_KEY_FILE}" ]] && [[ ! -f "${GEMINI_KEY_FILE}" ]]; then
        warn "No LLM judge API key found.
  Checked: ${OPENAI_KEY_FILE}
           ${GEMINI_KEY_FILE}
  Phase 6 (LLM-as-judge) will be skipped unless one of these is added."
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
        if ! skip_if_complete "base model" "${M_BASE}" "config.json"; then
            if [[ "${MODEL_SHORTCUT}" != "custom" ]]; then
                run "download base" python utils/download_models.py \
                    --model "${MODEL_SHORTCUT}"
            else
                run "download base (custom HF ID)" python -c "
from huggingface_hub import snapshot_download
from utils.setup_token import load_hf_token
token = load_hf_token('${HF_TOKEN_FILE}')
snapshot_download('${MODEL_HF_BASE}', local_dir='${M_BASE}',
    token=token)
print('Done.')
"
            fi
        fi
    fi

    # Official instruct model (needed for S1/S2/S3/S5/S6/S8)
    if $RUN_S1 || $RUN_S2 || $RUN_S3 || $RUN_S5 || $RUN_S6 || $RUN_S8; then
        if ! skip_if_complete "instruct-official model" "${M_INSTRUCT_OFFICIAL}" "config.json"; then
            run "download official instruct" python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.setup_token import load_hf_token
import torch, pathlib
p = pathlib.Path('${M_INSTRUCT_OFFICIAL}'); p.mkdir(parents=True, exist_ok=True)
token = load_hf_token('${HF_TOKEN_FILE}')
tok = AutoTokenizer.from_pretrained('${MODEL_HF_INSTRUCT}',
    token=token, trust_remote_code=True)
tok.save_pretrained(str(p))
m = AutoModelForCausalLM.from_pretrained('${MODEL_HF_INSTRUCT}',
    dtype=torch.bfloat16, token=token,
    trust_remote_code=True, low_cpu_mem_usage=True)
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
    if ($RUN_S4 || $RUN_S7) && ! skip_if_complete "CPT data" "${DATA_CPT_DIR}" "dataset_dict.json"; then
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
    if ($RUN_S5 || $RUN_S8) && ! skip_if_complete "LoRA CPT data" "${DATA_LORA_CPT_DIR}" "dataset_dict.json"; then
        run "prepare LoRA CPT data" python training/prepare_lora_data.py \
            --model_path "${M_INSTRUCT_OFFICIAL}" \
            --corpus_file "${CORPUS_FILE}" \
            --output_dir "${DATA_LORA_CPT_DIR}"
    fi

    # 11c. General instruction data (Orca+Dolly) is retained as an explicit
    # fallback for S4 full SFT. The default S4 full-SFT path uses HPN/Instruct-FTD.
    if ($RUN_S4 && ! $S4_USE_LORA_SFT && [[ "${S4_FULL_SFT_DATASET}" == "general" ]]) && \
       ! skip_if_complete "general SFT data" "${DATA_GENERAL_SFT_DIR}" "dataset_dict.json"; then
        run "prepare general SFT data" python training/prepare_instruction_data.py \
            --model_path "${M_BASE}" \
            --output_dir "${DATA_GENERAL_SFT_DIR}" \
            --max_samples "${SFT_MAX_SAMPLES}"
    fi

    # 11d. HPN SFT instruction data (from Instruct-FTD JSON/JSONL — for S2/S3/S4/S5/S8)
    if { $RUN_S2 || $RUN_S3 || { $RUN_S4 && { $S4_USE_LORA_SFT || [[ "${S4_FULL_SFT_DATASET}" == "hpn" ]]; }; } || $RUN_S5 || $RUN_S8; } && \
       ! skip_if_complete "HPN SFT data" "${DATA_HPN_SFT_DIR}" "dataset_dict.json"; then
        local hpn_input_args=()
        if [[ -d "${HPN_INSTRUCT_JSONL}" ]]; then
            hpn_input_args=(--input_dir "${HPN_INSTRUCT_JSONL}")
        else
            hpn_input_args=(--input_file "${HPN_INSTRUCT_JSONL}")
        fi
        run "prepare HPN SFT data" python training/prepare_instruction_data.py \
            --model_path "${M_INSTRUCT_OFFICIAL}" \
            "${hpn_input_args[@]}" \
            --output_dir "${DATA_HPN_SFT_DIR}" \
            --max_samples "${SFT_MAX_SAMPLES}"
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
            if use_distributed_full_training; then
                run_training "S4 Full HPN-CPT" "${M_PRETRAINED}" \
                    torchrun --standalone --nproc_per_node "${GPU_COUNT}" training/pretrain_transformers.py \
                        --model "${MODEL_SHORTCUT}" \
                        --data_dir "${DATA_CPT_DIR}" \
                        --output_dir "${M_PRETRAINED}" \
                        --per_device_train_batch_size "${CPT_BATCH}" \
                        --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
                        --learning_rate "${CPT_LR}" \
                        --num_train_epochs "${CPT_EPOCHS}" \
                        --deepspeed "${DS_ZERO3_CONFIG}"
            else
                run_training "S4 Full HPN-CPT" "${M_PRETRAINED}" \
                    python training/pretrain_transformers.py \
                        --model "${MODEL_SHORTCUT}" \
                        --data_dir "${DATA_CPT_DIR}" \
                        --output_dir "${M_PRETRAINED}" \
                        --per_device_train_batch_size "${CPT_BATCH}" \
                        --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
                        --learning_rate "${CPT_LR}" \
                        --num_train_epochs "${CPT_EPOCHS}"
            fi
        else
            if use_distributed_full_training; then
                run_training "S4 Full HPN-CPT (custom)" "${M_PRETRAINED}" \
                    torchrun --standalone --nproc_per_node "${GPU_COUNT}" training/pretrain_transformers.py \
                        --model_path "${M_BASE}" \
                        --data_dir "${DATA_CPT_DIR}" \
                        --output_dir "${M_PRETRAINED}" \
                        --per_device_train_batch_size "${CPT_BATCH}" \
                        --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
                        --learning_rate "${CPT_LR}" \
                        --num_train_epochs "${CPT_EPOCHS}" \
                        --deepspeed "${DS_ZERO3_CONFIG}"
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
        fi

        if $S4_USE_LORA_SFT; then
            # Step 4b (LoRA path): LoRA SFT on top of CPT model
            run_training_with_sentinel "S4 LoRA HPN-SFT" "${M_S4_LORA_SFT_ADAPTER}" "adapter_config.json" \
                python training/lora_finetune.py \
                    --model_path "${M_PRETRAINED}" \
                    --data_dir "${DATA_HPN_SFT_DIR}" \
                    --data_mode instruct \
                    --output_dir "${M_S4_LORA_SFT_ADAPTER}" \
                    $(lora_args)
            if ! skip_if_complete "S4 LoRA SFT merged" "${M_S4_LORA_SFT_MERGED}" "config.json"; then
                run "S4 merge LoRA SFT" python training/merge_lora_adapter.py \
                    --adapter_path "${M_S4_LORA_SFT_ADAPTER}" \
                    --base_model_path "${M_PRETRAINED}" \
                    --output_dir "${M_S4_LORA_SFT_MERGED}"
            fi
        else
            # Step 4b (full path): full SFT on top of CPT model
            local s4_full_sft_data_dir="${DATA_HPN_SFT_DIR}"
            if [[ "${S4_FULL_SFT_DATASET}" == "general" ]]; then
                s4_full_sft_data_dir="${DATA_GENERAL_SFT_DIR}"
            elif [[ "${S4_FULL_SFT_DATASET}" != "hpn" ]]; then
                die "S4_FULL_SFT_DATASET must be hpn or general. Got: ${S4_FULL_SFT_DATASET}"
            fi
            if use_distributed_full_training; then
                run_training "S4 Full HPN-SFT" "${M_S4_FULL_SFT}" \
                    torchrun --standalone --nproc_per_node "${GPU_COUNT}" training/instruction_finetune.py \
                        --model_path "${M_PRETRAINED}" \
                        --data_dir "${s4_full_sft_data_dir}" \
                        --output_dir "${M_S4_FULL_SFT}" \
                        --num_epochs "${SFT_EPOCHS}" \
                        --batch_size "${SFT_BATCH}" \
                        --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
                        --learning_rate "${SFT_LR}" \
                        --deepspeed "${DS_ZERO3_CONFIG}"
            else
                run_training "S4 Full HPN-SFT" "${M_S4_FULL_SFT}" \
                    python training/instruction_finetune.py \
                        --model_path "${M_PRETRAINED}" \
                        --data_dir "${s4_full_sft_data_dir}" \
                        --output_dir "${M_S4_FULL_SFT}" \
                        --num_epochs "${SFT_EPOCHS}" \
                        --batch_size "${SFT_BATCH}" \
                        --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
                        --learning_rate "${SFT_LR}"
            fi
        fi
    fi

    # ── S3: Dist Instruct -> Full HPN-SFT ────────────────────────────────────
    if $RUN_S3; then
        if use_distributed_full_training; then
            run_training "S3 Full HPN-SFT" "${M_S3_FULL_HPN_SFT}" \
                torchrun --standalone --nproc_per_node "${GPU_COUNT}" training/instruction_finetune.py \
                    --model_path "${M_INSTRUCT_OFFICIAL}" \
                    --data_dir "${DATA_HPN_SFT_DIR}" \
                    --output_dir "${M_S3_FULL_HPN_SFT}" \
                    --num_epochs "${SFT_EPOCHS}" \
                    --batch_size "${SFT_BATCH}" \
                    --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
                    --learning_rate "${SFT_LR}" \
                    --deepspeed "${DS_ZERO3_CONFIG}"
        else
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
    fi

    # ── S2: Dist Instruct -> LoRA HPN-SFT ────────────────────────────────────
    if $RUN_S2; then
        run_training_with_sentinel "S2 LoRA HPN-SFT" "${M_S2_LORA_SFT_ADAPTER}" "adapter_config.json" \
            python training/lora_finetune.py \
                --model_path "${M_INSTRUCT_OFFICIAL}" \
                --data_dir "${DATA_HPN_SFT_DIR}" \
                --data_mode instruct \
                --output_dir "${M_S2_LORA_SFT_ADAPTER}" \
                $(lora_args)
        if ! skip_if_complete "S2 LoRA SFT merged" "${M_S2_LORA_SFT_MERGED}" "config.json"; then
            run "S2 merge LoRA SFT" python training/merge_lora_adapter.py \
                --adapter_path "${M_S2_LORA_SFT_ADAPTER}" \
                --base_model_path "${M_INSTRUCT_OFFICIAL}" \
                --output_dir "${M_S2_LORA_SFT_MERGED}"
        fi
    fi

    # ── S5: Dist Instruct -> LoRA HPN-CPT -> merge -> LoRA HPN-SFT -> merge ──
    if $RUN_S5; then
        run_training_with_sentinel "S5 LoRA HPN-CPT" "${M_S5_LORA_CPT_ADAPTER}" "adapter_config.json" \
            python training/lora_finetune.py \
                --model_path "${M_INSTRUCT_OFFICIAL}" \
                --data_dir "${DATA_LORA_CPT_DIR}" \
                --data_mode corpus \
                --output_dir "${M_S5_LORA_CPT_ADAPTER}" \
                $(lora_args)
        if ! skip_if_complete "S5 LoRA CPT merged" "${M_S5_LORA_CPT_MERGED}" "config.json"; then
            run "S5 merge LoRA CPT" python training/merge_lora_adapter.py \
                --adapter_path "${M_S5_LORA_CPT_ADAPTER}" \
                --base_model_path "${M_INSTRUCT_OFFICIAL}" \
                --output_dir "${M_S5_LORA_CPT_MERGED}"
        fi
        run_training_with_sentinel "S5 LoRA HPN-SFT" "${M_S5_LORA_SFT_ADAPTER}" "adapter_config.json" \
            python training/lora_finetune.py \
                --model_path "${M_S5_LORA_CPT_MERGED}" \
                --data_dir "${DATA_HPN_SFT_DIR}" \
                --data_mode instruct \
                --output_dir "${M_S5_LORA_SFT_ADAPTER}" \
                $(lora_args)
        if ! skip_if_complete "S5 LoRA SFT merged" "${M_S5_LORA_SFT_MERGED}" "config.json"; then
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

rag_config_value() {
    local key="$1" default="$2"
    python -c "import sys, yaml; cfg=yaml.safe_load(open('config.yaml')) or {}; print((cfg.get('paths') or {}).get(sys.argv[1], sys.argv[2]))" "${key}" "${default}"
}

path_from_rag_root() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${RAG_DIR}/${path}"
    fi
}

rag_index() {
    if ! ($RUN_S6 || $RUN_S7 || $RUN_S8); then return; fi
    log "=== PHASE 4: RAG Index ==="

    [[ -d "${RAG_DIR}" ]] || die "NetBench-RAG not found: ${RAG_DIR}"

    pushd "${RAG_DIR}" > /dev/null
    activate_rag_env

    local rag_corpus_cfg rag_corpus_abs
    rag_corpus_cfg="$(rag_config_value corpus_json "data/raw/research_corpus_v3.json")"
    rag_corpus_abs="$(path_from_rag_root "${rag_corpus_cfg}")"
    if [[ ! -f "${rag_corpus_abs}" ]]; then
        mkdir -p "$(dirname "${rag_corpus_abs}")"
        ln -s "${CORPUS_FILE}" "${rag_corpus_abs}" 2>/dev/null || cp "${CORPUS_FILE}" "${rag_corpus_abs}"
        log "Linked/copied corpus for RAG: ${rag_corpus_abs}"
    fi

    # Skip if index already exists — check the sentinel marker AND the Qdrant
    # directory configured in NetBench-RAG/config.yaml.
    local rag_marker="${RAG_DIR}/.index_done"
    local qdrant_cfg qdrant_dir
    qdrant_cfg="$(rag_config_value qdrant_dir "data/qdrant_store")"
    qdrant_dir="$(path_from_rag_root "${qdrant_cfg}")"

    if [[ -f "${rag_marker}" ]] || [[ -d "${qdrant_dir}" ]]; then
        warn "SKIP RAG index — already built (marker or ${qdrant_dir} found)"
        deactivate_rag_env
        popd > /dev/null
        return
    fi

    run "RAG index corpus" python index_corpus.py
    touch "${rag_marker}"
    deactivate_rag_env
    popd > /dev/null
}

# =============================================================================
# SECTION 14 — BENCHMARK GENERATION
# =============================================================================

run_benchmark() {
    local label="$1" model_path="$2"
    local model_name tag_suffix out_file
    model_name="$(basename "${model_path}")"
    tag_suffix="$(benchmark_tag)"
    out_file="${OUT_ANSWERS}/hpn_answers_${model_name}${tag_suffix}.xlsx"
    if skip_if_exists "benchmark ${label}" "${out_file}"; then return; fi

    run "benchmark ${label}" python evaluation/hpn_qa_benchmark.py \
        --model_path "${model_path}" \
        --benchmark "${BENCHMARK_FILE}" \
        --output_dir "${OUT_ANSWERS}" \
        --temperature 0 \
        --batch_size "${BENCH_BATCH_SIZE}"
}

run_rag_benchmark() {
    local label="$1" model_path="$2"
    local model_name tag_suffix out_file
    model_name="$(basename "${model_path}")"
    tag_suffix="$(benchmark_tag)"
    out_file="${OUT_ANSWERS}/hpn_answers_RAG-${model_name}${tag_suffix}.xlsx"
    if skip_if_exists "RAG benchmark ${label}" "${out_file}"; then return; fi

    pushd "${RAG_DIR}" > /dev/null
    activate_rag_env
    run "RAG benchmark ${label}" python evaluation/evaluate_rag.py \
        --model "${model_path}" \
        --benchmark "${BENCHMARK_FILE}" \
        --output_dir "${OUT_ANSWERS}" \
        --batch_size "${BENCH_BATCH_SIZE}"
    deactivate_rag_env
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

    if [[ ! -f "${OPENAI_KEY_FILE}" && ! -f "${GEMINI_KEY_FILE}" ]]; then
        warn "No judge API key found — skipping judging."
        return
    fi
    case "${JUDGE_MODEL}" in
        gpt-*|o1-*|o3-*|o4-*|gpt5*|gpt-5*)
            if [[ ! -f "${OPENAI_KEY_FILE}" ]]; then
                warn "Judge model ${JUDGE_MODEL} needs ${OPENAI_KEY_FILE}; skipping judging."
                return
            fi
            ;;
        *)
            if [[ ! -f "${GEMINI_KEY_FILE}" ]]; then
                warn "Judge model ${JUDGE_MODEL} needs ${GEMINI_KEY_FILE}; skipping judging."
                return
            fi
            ;;
    esac

    # Collect all answer files — both direct and RAG answers are written to
    # OUT_ANSWERS (model-scoped), so a single find covers everything.
    local all_answers=()
    mapfile -t all_answers < <(
        find "${OUT_ANSWERS}" -name "hpn_answers_*.xlsx" 2>/dev/null
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
    local device="$1" label="$2" model_path="$3"
    local run_name="direct-${device}-benchmark${PROFILE_BENCHMARK_LIMIT}-${label}"
    local out_file="${OUT_PROFILE}/inference_${run_name}.json"
    if skip_profile_if_complete "profile ${label}" "${out_file}" "${model_path}"; then return; fi

    local device_args=()
    [[ "${device}" == "cpu" ]] && device_args=(--cpu)

    run "profile ${device} ${label}" python profiling/profile_inference.py \
        --model_path "${model_path}" \
        --output_dir "${OUT_PROFILE}" \
        --run_name "${run_name}" \
        --benchmark_file "${BENCHMARK_FILE}" \
        --benchmark_limit "${PROFILE_BENCHMARK_LIMIT}" \
        --benchmark_max_new_tokens "${PROFILE_MAX_NEW_TOKENS}" \
        --num_runs "${PROFILE_NUM_RUNS}" \
        --warmup_runs "${PROFILE_WARMUP_RUNS}" \
        --skip_scaling \
        "${device_args[@]}"
}

profile_rag_model() {
    local device="$1" label="$2" model_path="$3"
    local run_name="rag-${device}-benchmark${PROFILE_BENCHMARK_LIMIT}-${label}"
    local out_file="${OUT_PROFILE}/inference_${run_name}.json"
    if skip_profile_if_complete "RAG profile ${label}" "${out_file}" "${model_path}"; then return; fi

    local device_args=()
    [[ "${device}" == "cpu" ]] && device_args=(--cpu)

    activate_rag_env
    run "profile RAG ${device} ${label}" python "${LLM_DIR}/profiling/profile_rag_inference.py" \
        --rag_dir "${RAG_DIR}" \
        --config "${RAG_DIR}/config.yaml" \
        --benchmark_file "${BENCHMARK_FILE}" \
        --model_path "${model_path}" \
        --output_dir "${OUT_PROFILE}" \
        --run_name "${run_name}" \
        --limit "${PROFILE_BENCHMARK_LIMIT}" \
        "${device_args[@]}"
    deactivate_rag_env
}

profile_selected_direct_models() {
    local device="$1"
    $RUN_S1 && $PROFILE_DIRECT_S1 && profile_model "${device}" "S1-${MODEL_NAME}-Instruct-Official" "${M_INSTRUCT_OFFICIAL}"
    $RUN_S2 && $PROFILE_DIRECT_S2 && profile_model "${device}" "S2-${MODEL_NAME}-Instruct-LoRA-SFT" "${M_S2_LORA_SFT_MERGED}"
    $RUN_S3 && $PROFILE_DIRECT_S3 && profile_model "${device}" "S3-${MODEL_NAME}-Full-HPN-SFT" "${M_S3_FULL_HPN_SFT}"
    if $RUN_S4 && $PROFILE_DIRECT_S4; then
        local s4_final
        $S4_USE_LORA_SFT && s4_final="${M_S4_LORA_SFT_MERGED}" || s4_final="${M_S4_FULL_SFT}"
        profile_model "${device}" "S4-${MODEL_NAME}-CPT-SFT" "${s4_final}"
    fi
    $RUN_S5 && $PROFILE_DIRECT_S5 && profile_model "${device}" "S5-${MODEL_NAME}-LoRA-CPT-SFT" "${M_S5_LORA_SFT_MERGED}"
}

profile_selected_rag_models() {
    local device="$1"
    $RUN_S6 && $PROFILE_RAG_S6 && profile_rag_model "${device}" "S6-${MODEL_NAME}-RAG-Instruct-Official" "${M_INSTRUCT_OFFICIAL}"
    if $RUN_S7 && $PROFILE_RAG_S7; then
        local s7_src
        $S4_USE_LORA_SFT && s7_src="${M_S4_LORA_SFT_MERGED}" || s7_src="${M_S4_FULL_SFT}"
        profile_rag_model "${device}" "S7-${MODEL_NAME}-RAG-CPT-SFT" "${s7_src}"
    fi
    $RUN_S8 && $PROFILE_RAG_S8 && profile_rag_model "${device}" "S8-${MODEL_NAME}-RAG-LoRA-CPT-SFT" "${M_S5_LORA_SFT_MERGED}"
}

profile_report_path() {
    printf '%s/inference_%s-%s-benchmark%s-%s.json\n' "${OUT_PROFILE}" "$1" "$2" "${PROFILE_BENCHMARK_LIMIT}" "$3"
}

append_profile_report_if_enabled() {
    local -n target="$1"
    local type="$2" device="$3" label="$4"
    target+=("$(profile_report_path "${type}" "${device}" "${label}")")
}

profile_summary() {
    local reports=()
    local s1_label="S1-${MODEL_NAME}-Instruct-Official"
    local s2_label="S2-${MODEL_NAME}-Instruct-LoRA-SFT"
    local s3_label="S3-${MODEL_NAME}-Full-HPN-SFT"
    local s4_label="S4-${MODEL_NAME}-CPT-SFT"
    local s5_label="S5-${MODEL_NAME}-LoRA-CPT-SFT"
    local s6_label="S6-${MODEL_NAME}-RAG-Instruct-Official"
    local s7_label="S7-${MODEL_NAME}-RAG-CPT-SFT"
    local s8_label="S8-${MODEL_NAME}-RAG-LoRA-CPT-SFT"

    if $RUN_DIRECT_GPU_PROFILING; then
        $PROFILE_DIRECT_S1 && append_profile_report_if_enabled reports direct gpu "${s1_label}"
        $PROFILE_DIRECT_S2 && append_profile_report_if_enabled reports direct gpu "${s2_label}"
        $PROFILE_DIRECT_S3 && append_profile_report_if_enabled reports direct gpu "${s3_label}"
        $PROFILE_DIRECT_S4 && append_profile_report_if_enabled reports direct gpu "${s4_label}"
        $PROFILE_DIRECT_S5 && append_profile_report_if_enabled reports direct gpu "${s5_label}"
    fi
    if $RUN_DIRECT_CPU_PROFILING; then
        $PROFILE_DIRECT_S1 && append_profile_report_if_enabled reports direct cpu "${s1_label}"
        $PROFILE_DIRECT_S2 && append_profile_report_if_enabled reports direct cpu "${s2_label}"
        $PROFILE_DIRECT_S3 && append_profile_report_if_enabled reports direct cpu "${s3_label}"
        $PROFILE_DIRECT_S4 && append_profile_report_if_enabled reports direct cpu "${s4_label}"
        $PROFILE_DIRECT_S5 && append_profile_report_if_enabled reports direct cpu "${s5_label}"
    fi
    if $RUN_RAG_GPU_PROFILING; then
        $PROFILE_RAG_S6 && append_profile_report_if_enabled reports rag gpu "${s6_label}"
        $PROFILE_RAG_S7 && append_profile_report_if_enabled reports rag gpu "${s7_label}"
        $PROFILE_RAG_S8 && append_profile_report_if_enabled reports rag gpu "${s8_label}"
    fi
    if $RUN_RAG_CPU_PROFILING; then
        $PROFILE_RAG_S6 && append_profile_report_if_enabled reports rag cpu "${s6_label}"
        $PROFILE_RAG_S7 && append_profile_report_if_enabled reports rag cpu "${s7_label}"
        $PROFILE_RAG_S8 && append_profile_report_if_enabled reports rag cpu "${s8_label}"
    fi

    local existing=()
    local report
    for report in "${reports[@]}"; do
        if profile_json_is_complete "${report}"; then
            existing+=("${report}")
        elif [[ -f "${report}" ]]; then
            warn "Skipping incomplete profiling checkpoint in summary: ${report}"
        fi
    done

    if [[ ${#existing[@]} -eq 0 ]]; then
        warn "No profiling reports found — skipping profiling summary."
        return
    fi

    local slug
    slug="$(model_slug)"
    run "inference profiling summary" python profiling/summarize_inference_profiles.py \
        --reports "${existing[@]}" \
        --output_csv "${OUT_PROFILE}/inference_profile_summary_${slug}_benchmark${PROFILE_BENCHMARK_LIMIT}.csv" \
        --output_md "${OUT_PROFILE}/inference_profile_summary_${slug}_benchmark${PROFILE_BENCHMARK_LIMIT}.md"
}

profiling() {
    log "=== PHASE 7: Inference Profiling ==="
    log "Profiling prompt source: ${BENCHMARK_FILE} (first ${PROFILE_BENCHMARK_LIMIT} questions)"

    $RUN_DIRECT_GPU_PROFILING && profile_selected_direct_models "gpu"
    $RUN_DIRECT_CPU_PROFILING && profile_selected_direct_models "cpu"
    $RUN_RAG_GPU_PROFILING && profile_selected_rag_models "gpu"
    $RUN_RAG_CPU_PROFILING && profile_selected_rag_models "cpu"
    profile_summary
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
    resolve_model_identity
    maybe_start_screen_session

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
