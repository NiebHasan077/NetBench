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
#   mkdir -p outputs/by_model/qwen3.5-2b/logs
#   bash scripts/run_pipeline.sh 2>&1 | tee outputs/by_model/qwen3.5-2b/logs/pipeline_run.log
#
# Long runs: use GNU screen so the workflow survives SSH disconnects.
#   screen -S hpn-<MODEL_NAME>
#   bash scripts/run_pipeline.sh 2>&1 | tee outputs/by_model/qwen3.5-2b/logs/pipeline_run.log
#
# Or let the script start a detached screen session named from the model:
# hpn-<model-name>-<timestamp>
#   USE_GNU_SCREEN=true bash scripts/run_pipeline.sh
# =============================================================================

set -euo pipefail

# Reduce CUDA allocator fragmentation on long training runs. Some CUDA builds
# used on shared clusters do not support expandable_segments, so use the older
# split-size knob by default. The caller can still override this before launch.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# ─── Resolve project roots ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# NetBench-RAG is a sibling directory; resolve path without cd (may not exist).
RAG_DIR="$(realpath -m "${LLM_DIR}/../NetBench-RAG")"
cd "${LLM_DIR}"

# =============================================================================
# SECTION 1 — MODEL IDENTITY   (ADAPT FOR EACH MACHINE / MODEL)
# =============================================================================

MODEL_NAME="Llama-3.1-8B"        # Short label used in output paths.
                                # e.g. "Qwen3.5-2B", "Llama-3.1-8B", "Gemma3-4B"

MODEL_FAMILY="llama"             # "llama" | "qwen" | "gemma"
                                # Used to pick prompt templates and LoRA target modules.

MODEL_SHORTCUT="8b"             # pretrain_transformers.py shortcut key, OR "custom".
                                # Registered: 1b | 8b | qwen-2b | qwen-4b | gemma-4b
                                #             gemma-e4b | gemma-e2b
                                # Use "custom" for unregistered models.

MODEL_HF_BASE="meta-llama/Llama-3.1-8B"  # HuggingFace repo ID for the BASE model.
                                # e.g. "meta-llama/Llama-3.1-8B"
                                #      "Qwen/Qwen3.5-2B-Base"
                                #      "google/gemma-3-4b-pt"

MODEL_HF_INSTRUCT="meta-llama/Llama-3.1-8B-Instruct" # HuggingFace repo ID for the OFFICIAL instruct/post-trained model.
                                # e.g. "meta-llama/Llama-3.1-8B-Instruct"
                                #      "Qwen/Qwen3.5-2B"
                                #      "google/gemma-3-4b-it"

# Optional local path overrides. Leave empty for registered shortcuts.
MODEL_BASE_DIR=""
MODEL_INSTRUCT_DIR=""
MODEL_PRETRAINED_DIR=""

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

# Output directories. configure_output_dirs() recalculates these after
# MODEL_NAME is resolved so every model writes to a separate output tree.
OUT_MODEL_ROOT="${LLM_DIR}/outputs/by_model/${MODEL_NAME,,}"
OUT_ANSWERS="${OUT_MODEL_ROOT}/evaluations/answers"
OUT_RAG_ANSWERS="${OUT_MODEL_ROOT}/evaluations/rag_answers"
OUT_JUDGED="${OUT_MODEL_ROOT}/evaluations/judged"
OUT_REPORTS="${OUT_MODEL_ROOT}/evaluations/reports"
OUT_ADAPT="${OUT_MODEL_ROOT}/evaluations/adaptation_reports"
OUT_PROFILE="${OUT_MODEL_ROOT}/profiling_results"
OUT_LOGS="${OUT_MODEL_ROOT}/logs"

# API / auth
JUDGE_MODEL="gpt-5.1"            # default OpenAI judge
GEMINI_KEY_FILE="${LLM_DIR}/gemini_api_key.txt"
OPENAI_KEY_FILE="${LLM_DIR}/openai_api_key.txt"
HF_TOKEN_FILE="${LLM_DIR}/hf_token.txt"

# Leave empty to reuse the currently active LLM virtualenv for NetBench-RAG.
# Set to an absolute or project-relative path if you want a separate RAG venv.
RAG_VENV_DIR=""

# GNU screen support for long runs. Keep false for normal foreground execution.
# Set USE_GNU_SCREEN=true to relaunch this script in a detached screen session.
# The session name is always generated from MODEL_NAME: hpn-<model-name>-<timestamp>.
USE_GNU_SCREEN="${USE_GNU_SCREEN:-false}"
SCREEN_LOG_FILE="${SCREEN_LOG_FILE:-}"

# Inference profiling controls. Profiling uses benchmark questions instead of
# synthetic prompts so direct and RAG reports are comparable.
PROFILE_BENCHMARK_LIMIT="${PROFILE_BENCHMARK_LIMIT:-10}"
PROFILE_MAX_NEW_TOKENS="${PROFILE_MAX_NEW_TOKENS:-256}"
PROFILE_NUM_RUNS="${PROFILE_NUM_RUNS:-1}"
PROFILE_WARMUP_RUNS="${PROFILE_WARMUP_RUNS:-1}"
RUN_DIRECT_GPU_PROFILING="${RUN_DIRECT_GPU_PROFILING:-true}"
RUN_DIRECT_CPU_PROFILING="${RUN_DIRECT_CPU_PROFILING:-true}"
RUN_RAG_GPU_PROFILING="${RUN_RAG_GPU_PROFILING:-true}"
RUN_RAG_CPU_PROFILING="${RUN_RAG_CPU_PROFILING:-true}"

# Selected variants for profiling. Enable S2/S3/S6 here if you need broader
# profiling, but the default comparison focuses on base instruct, S4, S5, S7,
# and S8.
PROFILE_DIRECT_S1="${PROFILE_DIRECT_S1:-true}"
PROFILE_DIRECT_S2="${PROFILE_DIRECT_S2:-false}"
PROFILE_DIRECT_S3="${PROFILE_DIRECT_S3:-false}"
PROFILE_DIRECT_S4="${PROFILE_DIRECT_S4:-true}"
PROFILE_DIRECT_S5="${PROFILE_DIRECT_S5:-true}"
PROFILE_RAG_S6="${PROFILE_RAG_S6:-false}"
PROFILE_RAG_S7="${PROFILE_RAG_S7:-true}"
PROFILE_RAG_S8="${PROFILE_RAG_S8:-true}"

# Local benchmark answer generation can use throughput parallelism: one worker
# per visible GPU, each worker answers a shard of the benchmark, then shard
# workbooks are merged back into the normal Phase-1 answer file.
BENCHMARK_PARALLEL_LOCAL="${BENCHMARK_PARALLEL_LOCAL:-true}"
BENCHMARK_GPU_WORKERS="${BENCHMARK_GPU_WORKERS:-auto}"  # auto = GPU_COUNT
# NetBench-RAG currently uses local Qdrant disk mode, which permits only one
# process to open the store at a time. Keep RAG answer generation serial unless
# the RAG backend is changed to a concurrent-safe Qdrant server.
RAG_BENCHMARK_PARALLEL="${RAG_BENCHMARK_PARALLEL:-false}"
RAG_BENCHMARK_GPU_WORKERS="${RAG_BENCHMARK_GPU_WORKERS:-auto}"  # auto = GPU_COUNT when parallel

# Keep every project-related cache off the home filesystem. Override
# PROJECT_CACHE_DIR if you want these caches somewhere else on scratch.
PROJECT_CACHE_DIR="${PROJECT_CACHE_DIR:-${LLM_DIR}/.cache}"

# GPU selection for shared compute nodes.
#   auto       = use GPUs with <= GPU_IDLE_MEMORY_THRESHOLD_MIB allocated
#   all        = use every GPU reported by nvidia-smi
#   "1,2,3"    = use exactly those physical GPU IDs
# If CUDA_VISIBLE_DEVICES is already set by the caller and this remains auto,
# it is treated as a candidate list and still filtered for idle GPUs.
GPU_DEVICE_SELECTION="${GPU_DEVICE_SELECTION:-auto}"
GPU_IDLE_MEMORY_THRESHOLD_MIB="${GPU_IDLE_MEMORY_THRESHOLD_MIB:-4096}"
CUDA_DEVICE_LIST=""

# Full-weight training can use either:
#   auto     = torchrun on >=75 GiB cards, sharded otherwise
#   sharded  = single process with device_map="auto"
#   torchrun = one full trainable replica per GPU via HF Trainer/DDP
# H100/A100-80GB cards have enough memory for torchrun on 8B-class models and
# avoid the fragile autograd path caused by training through device_map shards.
TRAINING_LAUNCHER="${TRAINING_LAUNCHER:-auto}"          # auto | sharded | torchrun
TRAINING_TORCHRUN_GPUS="${TRAINING_TORCHRUN_GPUS:-auto}"
TRAINING_TORCHRUN_AUTO_GPUS="${TRAINING_TORCHRUN_AUTO_GPUS:-4}"
TRAINING_NCCL_STABLE="${TRAINING_NCCL_STABLE:-auto}"   # auto | true | false
FULL_TRAIN_DEEPSPEED_CONFIG="${FULL_TRAIN_DEEPSPEED_CONFIG:-auto}" # auto | none | path
FULL_TRAIN_FINAL_SAVE_ONLY="${FULL_TRAIN_FINAL_SAVE_ONLY:-auto}" # auto | true | false
FULL_TRAIN_PRECISION="${FULL_TRAIN_PRECISION:-auto}"
CPT_OPTIM="${CPT_OPTIM:-auto}"
SFT_OPTIM="${SFT_OPTIM:-auto}"
TRAINING_DATALOADER_WORKERS="${TRAINING_DATALOADER_WORKERS:-0}"
TRAINING_RESUME="${TRAINING_RESUME:-auto}"              # auto | true | false
TRAINING_RESTART_INCOMPATIBLE_CHECKPOINT="${TRAINING_RESTART_INCOMPATIBLE_CHECKPOINT:-true}"
TRAINING_TRUST_LOCAL_LORA_CHECKPOINTS="${TRAINING_TRUST_LOCAL_LORA_CHECKPOINTS:-true}"
ADAPTATION_BATCH_SIZE="${ADAPTATION_BATCH_SIZE:-auto}"

# =============================================================================
# SECTION 4 — HARDWARE HYPERPARAMETERS  (auto-tuned by detect_hardware)
# =============================================================================

PHYSICAL_GPU_COUNT=0  # all GPUs reported by nvidia-smi
GPU_COUNT=0           # selected/visible GPUs after detect_hardware()
VRAM_MIB=0            # per-GPU VRAM in MiB — set by detect_hardware()
VRAM_GB=0             # per-GPU VRAM in GiB — set by detect_hardware()

# CPT hyperparameters (full-weight continual pre-training)
CPT_BATCH=4
CPT_GRAD_ACCUM=4
CPT_LR="2e-5"
CPT_EPOCHS=3

# SFT hyperparameters (full-weight instruction fine-tuning)
SFT_BATCH=4
SFT_GRAD_ACCUM=4
SFT_LR="2e-5"
SFT_EPOCHS=3
SFT_MAX_SAMPLES=5000

# LoRA hyperparameters
LORA_BATCH=8
LORA_GRAD_ACCUM=2
LORA_LR="2e-4"
LORA_EPOCHS=3
LORA_RANK=64
LORA_ALPHA=128
USE_QLORA=false
LORA_OPTIM="${LORA_OPTIM:-auto}"
LORA_DATALOADER_WORKERS="${LORA_DATALOADER_WORKERS:-0}"
LORA_DATALOADER_PIN_MEMORY="${LORA_DATALOADER_PIN_MEMORY:-false}"
LORA_GRADIENT_CHECKPOINTING="${LORA_GRADIENT_CHECKPOINTING:-true}"
LORA_SAVE_BEST_ADAPTER="${LORA_SAVE_BEST_ADAPTER:-false}"
LORA_BATCH_OVERRIDE="${LORA_BATCH_OVERRIDE:-}"
LORA_GRAD_ACCUM_OVERRIDE="${LORA_GRAD_ACCUM_OVERRIDE:-}"

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

configure_project_caches() {
    PROJECT_CACHE_DIR="$(path_from_llm_root "${PROJECT_CACHE_DIR}")"
    export PROJECT_CACHE_DIR

    export XDG_CACHE_HOME="${PROJECT_CACHE_DIR}/xdg"
    export HF_HOME="${PROJECT_CACHE_DIR}/huggingface"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export HF_MODULES_CACHE="${HF_HOME}/modules"
    export HF_ASSETS_CACHE="${HF_HOME}/assets"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export DATASETS_CACHE="${HF_DATASETS_CACHE}"
    export TORCH_HOME="${PROJECT_CACHE_DIR}/torch"
    export TORCH_EXTENSIONS_DIR="${PROJECT_CACHE_DIR}/torch_extensions"
    export TRITON_CACHE_DIR="${PROJECT_CACHE_DIR}/triton"
    export PIP_CACHE_DIR="${PROJECT_CACHE_DIR}/pip"
    export NUMBA_CACHE_DIR="${PROJECT_CACHE_DIR}/numba"
    export MPLCONFIGDIR="${PROJECT_CACHE_DIR}/matplotlib"
    export PYTHONPYCACHEPREFIX="${PROJECT_CACHE_DIR}/pycache"
    export TMPDIR="${PROJECT_CACHE_DIR}/tmp"
    export WANDB_DIR="${PROJECT_CACHE_DIR}/wandb/run"
    export WANDB_CACHE_DIR="${PROJECT_CACHE_DIR}/wandb/cache"
    export WANDB_CONFIG_DIR="${PROJECT_CACHE_DIR}/wandb/config"
    export WANDB_DATA_DIR="${PROJECT_CACHE_DIR}/wandb/data"

    mkdir -p \
        "${PROJECT_CACHE_DIR}" \
        "${XDG_CACHE_HOME}" \
        "${HF_HUB_CACHE}" \
        "${HF_DATASETS_CACHE}" \
        "${HF_MODULES_CACHE}" \
        "${HF_ASSETS_CACHE}" \
        "${TRANSFORMERS_CACHE}" \
        "${TORCH_HOME}" \
        "${TORCH_EXTENSIONS_DIR}" \
        "${TRITON_CACHE_DIR}" \
        "${PIP_CACHE_DIR}" \
        "${NUMBA_CACHE_DIR}" \
        "${MPLCONFIGDIR}" \
        "${PYTHONPYCACHEPREFIX}" \
        "${TMPDIR}" \
        "${WANDB_DIR}" \
        "${WANDB_CACHE_DIR}" \
        "${WANDB_CONFIG_DIR}" \
        "${WANDB_DATA_DIR}"
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
        printf '%s\n' "${OUT_LOGS}/${session}.log"
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

    local session log_file venv_activate
    session="$(default_screen_session_name)"
    log_file="$(screen_log_file "${session}")"
    mkdir -p "$(dirname "${log_file}")"

    if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
        venv_activate="${VIRTUAL_ENV}/bin/activate"
    elif [[ -f "${LLM_DIR}/.venv/bin/activate" ]]; then
        venv_activate="${LLM_DIR}/.venv/bin/activate"
    elif [[ -f "/home/user/portables/NetBench-LLM/.venv/bin/activate" ]]; then
        venv_activate="/home/user/portables/NetBench-LLM/.venv/bin/activate"
    else
        venv_activate=""
    fi

    log "Starting detached GNU screen session: ${session}"
    log "Log file: ${log_file}"
    if [[ -n "${venv_activate}" ]]; then
        screen -dmS "${session}" bash -lc "cd '${LLM_DIR}' && source '${venv_activate}' && USE_GNU_SCREEN=false bash '${BASH_SOURCE[0]}' 2>&1 | tee '${log_file}'"
    else
        screen -dmS "${session}" bash -lc "cd '${LLM_DIR}' && USE_GNU_SCREEN=false bash '${BASH_SOURCE[0]}' 2>&1 | tee '${log_file}'"
    fi
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

csv_count() {
    local list="${1//[[:space:]]/}"
    if [[ -z "${list}" ]]; then
        printf '0\n'
        return
    fi
    local IFS=',' items=()
    read -r -a items <<< "${list}"
    printf '%s\n' "${#items[@]}"
}

csv_contains() {
    local needle="${1//[[:space:]]/}"
    local list="${2//[[:space:]]/}"
    local IFS=',' items=()
    read -r -a items <<< "${list}"
    local item
    for item in "${items[@]}"; do
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

filter_idle_devices() {
    local candidates="${1//[[:space:]]/}"
    local idle_list="${2//[[:space:]]/}"
    local IFS=',' candidate_ids=()
    read -r -a candidate_ids <<< "${candidates}"
    local selected=()
    local candidate
    for candidate in "${candidate_ids[@]}"; do
        [[ -z "${candidate}" ]] && continue
        if csv_contains "${candidate}" "${idle_list}"; then
            selected+=("${candidate}")
        else
            warn "GPU ${candidate} is not idle enough for auto selection — excluding from selected set." >&2
        fi
    done
    (IFS=,; printf '%s' "${selected[*]}")
}

cuda_device_at() {
    local pos="$1"
    local list="${CUDA_DEVICE_LIST//[[:space:]]/}"
    local IFS=',' devices=()
    read -r -a devices <<< "${list}"
    if [[ -z "${devices[$pos]:-}" ]]; then
        die "Requested CUDA worker ${pos}, but selected device list is: ${CUDA_DEVICE_LIST:-<empty>}"
    fi
    printf '%s\n' "${devices[$pos]}"
}

benchmark_worker_count() {
    local workers="${BENCHMARK_GPU_WORKERS}"
    if [[ "${BENCHMARK_PARALLEL_LOCAL}" != "true" ]]; then
        printf '1\n'
        return
    fi
    if [[ "${workers}" == "auto" ]]; then
        workers="${GPU_COUNT:-1}"
    fi
    [[ -z "${workers}" ]] && workers=1
    if (( workers < 1 )); then workers=1; fi
    if (( GPU_COUNT > 0 && workers > GPU_COUNT )); then workers="${GPU_COUNT}"; fi
    printf '%s\n' "${workers}"
}

rag_benchmark_worker_count() {
    local workers="${RAG_BENCHMARK_GPU_WORKERS}"
    if [[ "${RAG_BENCHMARK_PARALLEL}" != "true" ]]; then
        printf '1\n'
        return
    fi
    if [[ "${workers}" == "auto" ]]; then
        workers="${GPU_COUNT:-1}"
    fi
    [[ -z "${workers}" ]] && workers=1
    if (( workers < 1 )); then workers=1; fi
    if (( GPU_COUNT > 0 && workers > GPU_COUNT )); then workers="${GPU_COUNT}"; fi
    printf '%s\n' "${workers}"
}

training_torchrun_procs() {
    local procs="${TRAINING_TORCHRUN_GPUS}"
    if [[ "${TRAINING_LAUNCHER}" != "torchrun" ]]; then
        printf '1\n'
        return
    fi
    if [[ "${procs}" == "auto" ]]; then
        procs="${GPU_COUNT:-1}"
    fi
    [[ -z "${procs}" ]] && procs=1
    if (( procs < 1 )); then procs=1; fi
    if (( GPU_COUNT > 0 && procs > GPU_COUNT )); then procs="${GPU_COUNT}"; fi
    printf '%s\n' "${procs}"
}

maybe_torchrun() {
    if [[ "${TRAINING_LAUNCHER}" == "torchrun" ]]; then
        local procs
        procs="$(training_torchrun_procs)"
        # Keep the caller command shape as "python training/script.py ...".
        # torchrun normally prepends its own Python interpreter and expects the
        # script path directly; --no_python tells it to execute the following
        # binary as-is, so the active venv's python is used correctly.
        printf 'torchrun --standalone --nproc_per_node=%s --no_python\n' "${procs}"
    elif [[ "${TRAINING_LAUNCHER}" == "sharded" ]]; then
        printf '\n'
    else
        die "TRAINING_LAUNCHER must be sharded or torchrun. Got: ${TRAINING_LAUNCHER}"
    fi
}

full_train_extra_args() {
    if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
        printf -- '--deepspeed %s\n' "${FULL_TRAIN_DEEPSPEED_CONFIG}"
    fi
    if [[ "${FULL_TRAIN_FINAL_SAVE_ONLY}" == "true" ]]; then
        printf -- '--final_save_only\n'
    fi
}

write_benchmark_shards() {
    local benchmark_file="$1" num_shards="$2" out_dir="$3"
    mkdir -p "${out_dir}"
    python - "${benchmark_file}" "${num_shards}" "${out_dir}" <<'PY'
import json
import sys
from pathlib import Path

benchmark = Path(sys.argv[1])
num_shards = int(sys.argv[2])
out_dir = Path(sys.argv[3])

if benchmark.suffix.lower() == ".jsonl":
    with benchmark.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    data = {
        "metadata": {"title": f"Shard of {benchmark.name}"},
        "questions": questions,
    }
else:
    with benchmark.open(encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            questions = [json.loads(line) for line in f if line.strip()]
            data = {
                "metadata": {"title": f"Shard of {benchmark.name}"},
                "questions": questions,
            }
    if isinstance(data, list):
        data = {"metadata": {"title": f"Shard of {benchmark.name}"}, "questions": data}

questions = data.get("questions", [])
for shard_idx in range(num_shards):
    shard_data = dict(data)
    shard_data["metadata"] = dict(data.get("metadata", {}))
    shard_data["metadata"]["source_benchmark"] = str(benchmark)
    shard_data["metadata"]["shard_index"] = shard_idx
    shard_data["metadata"]["num_shards"] = num_shards
    shard_data["questions"] = [
        q for idx, q in enumerate(questions) if idx % num_shards == shard_idx
    ]
    out_path = out_dir / f"shard_{shard_idx}_of_{num_shards}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(shard_data, f, ensure_ascii=False, indent=2)
PY
}

wait_for_workers() {
    local label="$1"; shift
    local pid failed=0
    for pid in "$@"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    (( failed == 0 )) || die "${label} failed — inspect worker logs under the .shards directory"
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
            set_if_placeholder MODEL_HF_INSTRUCT "Qwen/Qwen3.5-2B"
            default_base="models/base/Qwen3.5-2B-Base"
            default_pretrained="models/pretrained/Qwen3.5-2B-trained-new"
            default_instruct="models/instruct-official/Qwen3.5-2B"
            ;;
        qwen-4b)
            set_if_placeholder MODEL_NAME "Qwen3.5-4B"
            set_if_placeholder MODEL_FAMILY "qwen"
            set_if_placeholder MODEL_HF_BASE "Qwen/Qwen3.5-4B-Base"
            set_if_placeholder MODEL_HF_INSTRUCT "Qwen/Qwen3.5-4B"
            default_base="models/base/Qwen3.5-4B-Base"
            default_pretrained="models/pretrained/Qwen3.5-4B-trained-new"
            default_instruct="models/instruct-official/Qwen3.5-4B"
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

configure_output_dirs() {
    local slug
    slug="$(model_slug)"
    OUT_MODEL_ROOT="${LLM_DIR}/outputs/by_model/${slug}"
    OUT_ANSWERS="${OUT_MODEL_ROOT}/evaluations/answers"
    OUT_RAG_ANSWERS="${OUT_MODEL_ROOT}/evaluations/rag_answers"
    OUT_JUDGED="${OUT_MODEL_ROOT}/evaluations/judged"
    OUT_REPORTS="${OUT_MODEL_ROOT}/evaluations/reports"
    OUT_ADAPT="${OUT_MODEL_ROOT}/evaluations/adaptation_reports"
    OUT_PROFILE="${OUT_MODEL_ROOT}/profiling_results"
    OUT_LOGS="${OUT_MODEL_ROOT}/logs"
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
    ckpt=$(ls -dt "${outdir}"/checkpoints/checkpoint-* 2>/dev/null | head -1 || true)
    if [[ -z "${ckpt}" ]]; then
        ckpt=$(ls -dt "${outdir}"/checkpoint-* 2>/dev/null | head -1 || true)
    fi
    printf '%s\n' "${ckpt}"
}

training_output_has_nonfinite_metrics() {
    local outdir="$1"
    python - "${outdir}" <<'PY'
import json
import math
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
paths = []
for rel in (
    "train_results.json",
    "all_results.json",
    "checkpoints/train_results.json",
    "checkpoints/all_results.json",
):
    p = outdir / rel
    if p.exists():
        paths.append(p)

for root in (outdir, outdir / "checkpoints"):
    if root.exists():
        paths.extend(sorted(root.glob("checkpoint-*/trainer_state.json")))

bad = []

def walk(obj, where):
    if isinstance(obj, dict):
        for key, value in obj.items():
            walk(value, f"{where}.{key}" if where else str(key))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            walk(value, f"{where}[{idx}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(f"{where}={obj}")

for path in paths:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        bad.append(f"{path}: unreadable JSON ({exc})")
        continue
    before = len(bad)
    walk(data, path.name)
    if len(bad) > before:
        bad[-1] = f"{path}: {bad[-1]}"

if bad:
    print(bad[0])
    sys.exit(0)
sys.exit(1)
PY
}

checkpoint_has_torch_pickle_state() {
    local ckpt="$1"
    [[ -f "${ckpt}/optimizer.pt" ||
       -f "${ckpt}/scheduler.pt" ||
       -f "${ckpt}/rng_state.pth" ||
       -f "${ckpt}/training_args.bin" ]]
}

torch_load_checkpoint_resume_supported() {
    python - <<'PY'
import re
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    torch_version = version("torch")
except PackageNotFoundError:
    sys.exit(1)

match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)
if not match:
    sys.exit(1)

major, minor, patch = (int(part or 0) for part in match.groups())
sys.exit(0 if (major, minor, patch) >= (2, 6, 0) else 1)
PY
}

torch_package_version() {
    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
try:
    print(version("torch"))
except PackageNotFoundError:
    print("not-installed")
PY
}

move_incompatible_training_output() {
    local label="$1" outdir="$2" ckpt="$3" torch_ver="$4"
    local safe_ver stamp backup
    safe_ver="${torch_ver//[^A-Za-z0-9_.-]/_}"
    stamp="$(date '+%Y%m%d-%H%M%S')"
    backup="${outdir}.incompatible-torch-${safe_ver}.${stamp}"
    if [[ -e "${backup}" ]]; then
        backup="${backup}.$$"
    fi
    warn "${label} cannot resume ${ckpt} with torch ${torch_ver}; moving partial output aside and restarting this step."
    warn "Partial output backup: ${backup}"
    mv "${outdir}" "${backup}" || die "Could not move partial output directory aside: ${outdir}"
}

run_training_with_sentinel() {
    local label="$1" outdir="$2" sentinel="$3"; shift 3

    if [[ -f "${outdir}/${sentinel}" ]]; then
        local bad_metric=""
        if bad_metric="$(training_output_has_nonfinite_metrics "${outdir}")"; then
            warn "BAD ${label} output has non-finite metrics: ${bad_metric}"
            die "${label} output exists but is numerically invalid. Move it aside or delete it before rerunning: ${outdir}"
        fi
        warn "SKIP ${label} — already complete: ${outdir}"
        return 0
    fi

    local ckpt=""
    ckpt="$(latest_checkpoint "${outdir}")"

    if [[ -n "${ckpt}" ]]; then
        local bad_metric=""
        if bad_metric="$(training_output_has_nonfinite_metrics "${outdir}")"; then
            warn "BAD ${label} checkpoint has non-finite metrics: ${bad_metric}"
            die "${label} has invalid checkpoints. Move the output directory aside before rerunning: ${outdir}"
        fi

        if [[ "${TRAINING_RESUME}" == "false" ]]; then
            die "${label} has a partial checkpoint but TRAINING_RESUME=false. Move the output directory aside before rerunning: ${outdir}"
        elif [[ "${TRAINING_RESUME}" != "auto" && "${TRAINING_RESUME}" != "true" ]]; then
            die "TRAINING_RESUME must be auto, true, or false. Got: ${TRAINING_RESUME}"
        fi

        if checkpoint_has_torch_pickle_state "${ckpt}" && ! torch_load_checkpoint_resume_supported; then
            local torch_ver
            torch_ver="$(torch_package_version)"
            if [[ "${label}" == *"LoRA"* && "${TRAINING_TRUST_LOCAL_LORA_CHECKPOINTS}" == "true" ]]; then
                warn "${label} will resume trusted local checkpoint ${ckpt} with torch ${torch_ver}; LoRA training will allow torch.load for this local checkpoint."
                log "RESUME ${label} from checkpoint: ${ckpt}"
                run "${label}" "$@" --resume_from_checkpoint "${ckpt}"
                return
            fi
            if [[ "${TRAINING_RESUME}" == "auto" && "${TRAINING_RESTART_INCOMPATIBLE_CHECKPOINT}" == "true" ]]; then
                move_incompatible_training_output "${label}" "${outdir}" "${ckpt}" "${torch_ver}"
                run "${label}" "$@"
                return
            fi
            die "${label} cannot safely resume ${ckpt} with torch ${torch_ver}. Transformers now requires torch >=2.6 to torch.load checkpoint state files. Move the output directory aside to restart this training step, upgrade torch to >=2.6, or rerun with TRAINING_RESTART_INCOMPATIBLE_CHECKPOINT=true and TRAINING_RESUME=auto."
        fi

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
    [[ "${LORA_GRADIENT_CHECKPOINTING}" == "false" ]] && extra="${extra} --no-gradient_checkpointing"
    [[ "${LORA_DATALOADER_PIN_MEMORY}" == "true" ]] && extra="${extra} --dataloader_pin_memory"
    [[ "${LORA_SAVE_BEST_ADAPTER}" == "false" ]] && extra="${extra} --no-save_best_adapter"
    echo "--lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
--epochs ${LORA_EPOCHS} --batch_size ${LORA_BATCH} \
--grad_accum ${LORA_GRAD_ACCUM} --lr ${LORA_LR} \
--optim ${LORA_OPTIM} --dataloader_num_workers ${LORA_DATALOADER_WORKERS} ${extra}"
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
    local rows=()
    mapfile -t rows < <(
        nvidia-smi --query-gpu=index,name,memory.total,memory.used \
            --format=csv,noheader,nounits 2>/dev/null
    )
    PHYSICAL_GPU_COUNT="${#rows[@]}"

    if (( PHYSICAL_GPU_COUNT < 1 )); then
        die "No NVIDIA GPUs reported by nvidia-smi."
    fi

    local physical_ids=()
    local idle_ids=()
    local row idx name total_mib used_mib
    VRAM_MIB=0

    for row in "${rows[@]}"; do
        IFS=',' read -r idx name total_mib used_mib <<< "${row}"
        idx="${idx//[[:space:]]/}"
        total_mib="${total_mib//[!0-9]/}"
        used_mib="${used_mib//[!0-9]/}"
        [[ -z "${total_mib}" ]] && total_mib=0
        [[ -z "${used_mib}" ]] && used_mib=0

        physical_ids+=("${idx}")
        if (( VRAM_MIB == 0 )); then
            VRAM_MIB="${total_mib}"
        fi

        if (( used_mib <= GPU_IDLE_MEMORY_THRESHOLD_MIB )); then
            idle_ids+=("${idx}")
        else
            warn "GPU ${idx} is busy (${used_mib} MiB used) — excluding from auto selection."
        fi
    done

    local idle_list
    idle_list="$(IFS=,; printf '%s' "${idle_ids[*]}")"

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${GPU_DEVICE_SELECTION}" == "auto" ]]; then
        log "Filtering caller CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} for idle GPUs."
        CUDA_DEVICE_LIST="$(filter_idle_devices "${CUDA_VISIBLE_DEVICES}" "${idle_list}")"
        if [[ -z "${CUDA_DEVICE_LIST}" ]]; then
            die "No idle GPUs remain after filtering caller CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}.
  Free a GPU, raise GPU_IDLE_MEMORY_THRESHOLD_MIB, or set GPU_DEVICE_SELECTION=all / GPU_DEVICE_SELECTION=1,2,3."
        fi
    elif [[ "${GPU_DEVICE_SELECTION}" == "auto" ]]; then
        if (( ${#idle_ids[@]} < 1 )); then
            die "No idle GPUs found under GPU_IDLE_MEMORY_THRESHOLD_MIB=${GPU_IDLE_MEMORY_THRESHOLD_MIB}.
  Free a GPU, raise GPU_IDLE_MEMORY_THRESHOLD_MIB, or set GPU_DEVICE_SELECTION=all / GPU_DEVICE_SELECTION=1,2,3."
        fi
        CUDA_DEVICE_LIST="${idle_list}"
    elif [[ "${GPU_DEVICE_SELECTION}" == "all" ]]; then
        CUDA_DEVICE_LIST="$(IFS=,; printf '%s' "${physical_ids[*]}")"
    else
        CUDA_DEVICE_LIST="${GPU_DEVICE_SELECTION//[[:space:]]/}"
    fi

    GPU_COUNT="$(csv_count "${CUDA_DEVICE_LIST}")"
    if (( GPU_COUNT < 1 )); then
        die "Selected CUDA device list is empty. GPU_DEVICE_SELECTION=${GPU_DEVICE_SELECTION}"
    fi

    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}"

    # Round to nearest GiB; H100 80GB reports 81559 MiB, which should classify
    # as an 80GB profile rather than being floored to 79GB.
    VRAM_GB=$(( (VRAM_MIB + 512) / 1024 ))

    log "Hardware: ${PHYSICAL_GPU_COUNT} physical GPU(s), selected ${GPU_COUNT}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    log "Selected GPU class: ~${VRAM_GB} GiB per card (${VRAM_MIB} MiB reported)"

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
    if (( VRAM_GB >= 75 )); then
        # H100/A100-80GB class. Full-model DDP still keeps one trainable 8B
        # replica, gradients, optimizer state, and activations on each GPU.
        # batch=4 was observed to use ~75.6 GiB and OOM on the first backward
        # pass at 2048-token SFT, so keep per-GPU micro-batches small and use
        # accumulation to preserve the effective batch.
        CPT_BATCH=1;   CPT_GRAD_ACCUM=16
        SFT_BATCH=1;   SFT_GRAD_ACCUM=16
        LORA_BATCH=4;  LORA_GRAD_ACCUM=4
        LORA_RANK=64;  USE_QLORA=false
    elif (( VRAM_GB >= 48 )); then
        CPT_BATCH=4;  CPT_GRAD_ACCUM=4
        SFT_BATCH=4;  SFT_GRAD_ACCUM=4
        LORA_BATCH=8; LORA_GRAD_ACCUM=2
        LORA_RANK=64; USE_QLORA=false
    else
        # 24–47 GB: batch=1 for full SFT to avoid OOM (model+grads=32 GB peak)
        CPT_BATCH=2;  CPT_GRAD_ACCUM=8
        SFT_BATCH=1;  SFT_GRAD_ACCUM=16
        LORA_BATCH=4; LORA_GRAD_ACCUM=4
        LORA_RANK=64; USE_QLORA=false
    fi
    [[ -n "${LORA_BATCH_OVERRIDE}" ]] && LORA_BATCH="${LORA_BATCH_OVERRIDE}"
    [[ -n "${LORA_GRAD_ACCUM_OVERRIDE}" ]] && LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM_OVERRIDE}"
    if [[ "${ADAPTATION_BATCH_SIZE}" == "auto" ]]; then
        if (( VRAM_GB < 48 )); then
            ADAPTATION_BATCH_SIZE=1
        else
            ADAPTATION_BATCH_SIZE=4
        fi
    fi

    if [[ "${TRAINING_LAUNCHER}" == "auto" ]]; then
        if (( VRAM_GB >= 75 )); then
            TRAINING_LAUNCHER="torchrun"
        else
            TRAINING_LAUNCHER="sharded"
        fi
    fi
    if [[ "${FULL_TRAIN_DEEPSPEED_CONFIG}" == "auto" ]]; then
        if [[ "${TRAINING_LAUNCHER}" == "torchrun" && "${VRAM_GB}" -ge 75 ]]; then
            if (( GPU_COUNT < 4 )); then
                FULL_TRAIN_DEEPSPEED_CONFIG="${LLM_DIR}/scripts/deepspeed_zero2_h100_cpuoffload.json"
            else
                FULL_TRAIN_DEEPSPEED_CONFIG="${LLM_DIR}/scripts/deepspeed_zero2_h100.json"
            fi
        else
            FULL_TRAIN_DEEPSPEED_CONFIG="none"
        fi
    elif [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
        FULL_TRAIN_DEEPSPEED_CONFIG="$(path_from_llm_root "${FULL_TRAIN_DEEPSPEED_CONFIG}")"
    fi
    if [[ "${TRAINING_LAUNCHER}" != "torchrun" && -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
        warn "TRAINING_LAUNCHER=${TRAINING_LAUNCHER} on ~${VRAM_GB} GiB GPUs; disabling DeepSpeed config ${FULL_TRAIN_DEEPSPEED_CONFIG} for sharded/device_map training."
        FULL_TRAIN_DEEPSPEED_CONFIG="none"
    fi
    if (( VRAM_GB < 75 )) && [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" && "$(basename "${FULL_TRAIN_DEEPSPEED_CONFIG}")" == deepspeed_zero2_h100* ]]; then
        warn "Selected GPUs are ~${VRAM_GB} GiB, not H100/A100-80GB class; disabling H100 DeepSpeed config ${FULL_TRAIN_DEEPSPEED_CONFIG}."
        FULL_TRAIN_DEEPSPEED_CONFIG="none"
    fi
    if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" && ! -f "${FULL_TRAIN_DEEPSPEED_CONFIG}" ]]; then
        die "DeepSpeed config not found: ${FULL_TRAIN_DEEPSPEED_CONFIG}"
    fi
    if [[ "${TRAINING_TORCHRUN_GPUS}" == "auto" && "${TRAINING_LAUNCHER}" == "torchrun" ]]; then
        if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
            TRAINING_TORCHRUN_GPUS="${TRAINING_TORCHRUN_AUTO_GPUS}"
        else
            TRAINING_TORCHRUN_GPUS="${GPU_COUNT:-1}"
        fi
        if (( GPU_COUNT > 0 && TRAINING_TORCHRUN_GPUS > GPU_COUNT )); then
            TRAINING_TORCHRUN_GPUS="${GPU_COUNT}"
        fi
    fi
    if [[ "${TRAINING_TORCHRUN_GPUS}" != "auto" ]]; then
        if ! [[ "${TRAINING_TORCHRUN_GPUS}" =~ ^[0-9]+$ ]]; then
            die "TRAINING_TORCHRUN_GPUS must be auto or a positive integer. Got: ${TRAINING_TORCHRUN_GPUS}"
        fi
        if (( TRAINING_TORCHRUN_GPUS < 1 )); then
            die "TRAINING_TORCHRUN_GPUS must be >= 1. Got: ${TRAINING_TORCHRUN_GPUS}"
        fi
        if (( GPU_COUNT > 0 && TRAINING_TORCHRUN_GPUS > GPU_COUNT )); then
            die "TRAINING_TORCHRUN_GPUS=${TRAINING_TORCHRUN_GPUS} was requested, but only ${GPU_COUNT} GPU(s) are selected: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}.
  Expose enough GPUs before launching, for example:
    unset CUDA_VISIBLE_DEVICES
  or:
    CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINING_TORCHRUN_GPUS=${TRAINING_TORCHRUN_GPUS} bash scripts/run_pipeline.sh"
        fi
    fi
    if [[ "${TRAINING_NCCL_STABLE}" == "auto" ]]; then
        if [[ "${TRAINING_LAUNCHER}" == "torchrun" ]]; then
            TRAINING_NCCL_STABLE="true"
        else
            TRAINING_NCCL_STABLE="false"
        fi
    elif [[ "${TRAINING_NCCL_STABLE}" != "true" && "${TRAINING_NCCL_STABLE}" != "false" ]]; then
        die "TRAINING_NCCL_STABLE must be auto, true, or false. Got: ${TRAINING_NCCL_STABLE}"
    fi
    if [[ "${FULL_TRAIN_FINAL_SAVE_ONLY}" == "auto" ]]; then
        if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
            FULL_TRAIN_FINAL_SAVE_ONLY="true"
        else
            FULL_TRAIN_FINAL_SAVE_ONLY="false"
        fi
    elif [[ "${FULL_TRAIN_FINAL_SAVE_ONLY}" != "true" && "${FULL_TRAIN_FINAL_SAVE_ONLY}" != "false" ]]; then
        die "FULL_TRAIN_FINAL_SAVE_ONLY must be auto, true, or false. Got: ${FULL_TRAIN_FINAL_SAVE_ONLY}"
    fi
    if [[ "${CPT_OPTIM}" == "auto" ]]; then
        if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
            CPT_OPTIM="adamw_torch"
        elif (( VRAM_GB >= 75 )); then
            CPT_OPTIM="adamw_bnb_8bit"
        else
            CPT_OPTIM="paged_adamw_8bit"
        fi
    fi
    if [[ "${SFT_OPTIM}" == "auto" ]]; then
        if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
            SFT_OPTIM="adamw_torch"
        elif (( VRAM_GB >= 75 )); then
            SFT_OPTIM="adamw_bnb_8bit"
        else
            SFT_OPTIM="paged_adamw_8bit"
        fi
    fi
    if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
        case "${CPT_OPTIM}" in
            *bnb*|paged_*)
                warn "DeepSpeed full-training is enabled; overriding CPT_OPTIM=${CPT_OPTIM} to adamw_torch to avoid bitsandbytes/DDP native crashes."
                CPT_OPTIM="adamw_torch"
                ;;
        esac
        case "${SFT_OPTIM}" in
            *bnb*|paged_*)
                warn "DeepSpeed full-training is enabled; overriding SFT_OPTIM=${SFT_OPTIM} to adamw_torch to avoid bitsandbytes/DDP native crashes."
                SFT_OPTIM="adamw_torch"
                ;;
        esac
    fi

    log "Hyperparams: CPT batch=${CPT_BATCH}x${CPT_GRAD_ACCUM}  SFT batch=${SFT_BATCH}x${SFT_GRAD_ACCUM}  LoRA batch=${LORA_BATCH}x${LORA_GRAD_ACCUM}  rank=${LORA_RANK}"
    log "Full-training precision=${FULL_TRAIN_PRECISION}  CPT optim=${CPT_OPTIM}  SFT optim=${SFT_OPTIM}  dataloader_workers=${TRAINING_DATALOADER_WORKERS}"
    log "Adaptation report batch=${ADAPTATION_BATCH_SIZE}"
    log "Full-training DeepSpeed: ${FULL_TRAIN_DEEPSPEED_CONFIG}"
    log "Full-training final-save-only: ${FULL_TRAIN_FINAL_SAVE_ONLY}"
    log "Torchrun training processes: $(training_torchrun_procs)  NCCL stable env: ${TRAINING_NCCL_STABLE}"
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
    configure_output_dirs
    configure_project_caches

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
    log "Project cache dir: ${PROJECT_CACHE_DIR}"

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
    if [[ -n "${FULL_TRAIN_DEEPSPEED_CONFIG}" && "${FULL_TRAIN_DEEPSPEED_CONFIG}" != "none" ]]; then
        python -c "import deepspeed" 2>/dev/null || \
            die "DeepSpeed config is enabled but deepspeed is not importable in the active environment: ${FULL_TRAIN_DEEPSPEED_CONFIG}"
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
    [[ -f "${LLM_DIR}/evaluation/merge_answer_shards.py" ]] || \
        die "Answer-shard merge helper not found: ${LLM_DIR}/evaluation/merge_answer_shards.py"
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
    mkdir -p "${OUT_ANSWERS}" "${OUT_RAG_ANSWERS}" "${OUT_JUDGED}" "${OUT_REPORTS}" "${OUT_ADAPT}" "${OUT_PROFILE}" "${OUT_LOGS}"

    hr
    ok "Preflight passed — selected GPUs: ${GPU_COUNT}x${VRAM_GB}GB (${CUDA_VISIBLE_DEVICES})  Corpus: $(basename "${CORPUS_FILE}")"
    log "Training launcher: ${TRAINING_LAUNCHER}  Benchmark GPU workers: $(benchmark_worker_count)  RAG benchmark workers: $(rag_benchmark_worker_count)"
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
    dtype=torch.float16, token=token,
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
    # Keep the CUDA allocator split size bounded to reduce fragmentation on
    # older cluster CUDA/PyTorch combinations.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_CACHE_DIR}/triton}"
    mkdir -p "${TRITON_CACHE_DIR}"
    if [[ "${TRAINING_NCCL_STABLE}" == "true" ]]; then
        export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
        export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
        export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    fi
    if [[ "${TRAINING_LAUNCHER}" == "torchrun" ]]; then
        warn "TRAINING_LAUNCHER=torchrun: full-weight training will launch $(training_torchrun_procs) process(es).
  This improves data-parallel utilization only if each process can fit a full
  trainable model replica. On 24-32 GB cards, full AdamW may OOM; switch back
  to TRAINING_LAUNCHER=sharded if that happens."
    fi
    stop_ollama  # free GPU VRAM; Ollama restarted at EXIT via trap

    # ── S4: Base -> Full HPN-CPT -> Full/LoRA HPN-SFT ────────────────────────
    if $RUN_S4; then
        # Step 4a: full continual pre-training
        if [[ "${MODEL_SHORTCUT}" != "custom" ]]; then
            run_training "S4 Full HPN-CPT" "${M_PRETRAINED}" \
                $(maybe_torchrun) python training/pretrain_transformers.py \
                    --model "${MODEL_SHORTCUT}" \
                    --data_dir "${DATA_CPT_DIR}" \
                    --output_dir "${M_PRETRAINED}" \
                    --per_device_train_batch_size "${CPT_BATCH}" \
                    --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
                    --learning_rate "${CPT_LR}" \
                    --num_train_epochs "${CPT_EPOCHS}" \
                    --precision "${FULL_TRAIN_PRECISION}" \
                    --optim "${CPT_OPTIM}" \
                    --dataloader_num_workers "${TRAINING_DATALOADER_WORKERS}" \
                    $(full_train_extra_args)
        else
            run_training "S4 Full HPN-CPT (custom)" "${M_PRETRAINED}" \
                $(maybe_torchrun) python training/pretrain_transformers.py \
                    --model_path "${M_BASE}" \
                    --data_dir "${DATA_CPT_DIR}" \
                    --output_dir "${M_PRETRAINED}" \
                    --per_device_train_batch_size "${CPT_BATCH}" \
                    --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
                    --learning_rate "${CPT_LR}" \
                    --num_train_epochs "${CPT_EPOCHS}" \
                    --precision "${FULL_TRAIN_PRECISION}" \
                    --optim "${CPT_OPTIM}" \
                    --dataloader_num_workers "${TRAINING_DATALOADER_WORKERS}" \
                    $(full_train_extra_args)
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
            run_training "S4 Full HPN-SFT" "${M_S4_FULL_SFT}" \
                $(maybe_torchrun) python training/instruction_finetune.py \
                    --model_path "${M_PRETRAINED}" \
                    --data_dir "${s4_full_sft_data_dir}" \
                    --output_dir "${M_S4_FULL_SFT}" \
                    --num_epochs "${SFT_EPOCHS}" \
                    --batch_size "${SFT_BATCH}" \
                    --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
                    --learning_rate "${SFT_LR}" \
                    --precision "${FULL_TRAIN_PRECISION}" \
                    --optim "${SFT_OPTIM}" \
                    --dataloader_num_workers "${TRAINING_DATALOADER_WORKERS}" \
                    $(full_train_extra_args)
        fi
    fi

    # ── S3: Dist Instruct -> Full HPN-SFT ────────────────────────────────────
    if $RUN_S3; then
        run_training "S3 Full HPN-SFT" "${M_S3_FULL_HPN_SFT}" \
            $(maybe_torchrun) python training/instruction_finetune.py \
                --model_path "${M_INSTRUCT_OFFICIAL}" \
                --data_dir "${DATA_HPN_SFT_DIR}" \
                --output_dir "${M_S3_FULL_HPN_SFT}" \
                --num_epochs "${SFT_EPOCHS}" \
                --batch_size "${SFT_BATCH}" \
                --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
                --learning_rate "${SFT_LR}" \
                --precision "${FULL_TRAIN_PRECISION}" \
                --optim "${SFT_OPTIM}" \
                --dataloader_num_workers "${TRAINING_DATALOADER_WORKERS}" \
                $(full_train_extra_args)
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
    local model_name tag_suffix out_file workers
    model_name="$(basename "${model_path}")"
    tag_suffix="$(benchmark_tag)"
    out_file="${OUT_ANSWERS}/hpn_answers_${model_name}${tag_suffix}.xlsx"
    if skip_if_exists "benchmark ${label}" "${out_file}"; then return; fi

    workers="$(benchmark_worker_count)"
    if (( workers <= 1 )); then
        run "benchmark ${label}" python evaluation/hpn_qa_benchmark.py \
            --model_path "${model_path}" \
            --benchmark "${BENCHMARK_FILE}" \
            --output_dir "${OUT_ANSWERS}" \
            --temperature 0
        return
    fi

    local shard_root shard_bench_dir shard_ans_root
    shard_root="${OUT_ANSWERS}/.shards/${model_name}${tag_suffix}"
    shard_bench_dir="${shard_root}/benchmarks"
    shard_ans_root="${shard_root}/answers"
    mkdir -p "${shard_bench_dir}" "${shard_ans_root}"
    write_benchmark_shards "${BENCHMARK_FILE}" "${workers}" "${shard_bench_dir}"

    log "Parallel benchmark ${label}: ${workers} GPU worker(s), one benchmark shard per worker"
    local pids=()
    local gpu device_id shard_bench shard_out log_file
    for ((gpu=0; gpu<workers; gpu++)); do
        device_id="$(cuda_device_at "${gpu}")"
        shard_bench="${shard_bench_dir}/shard_${gpu}_of_${workers}.json"
        shard_out="${shard_ans_root}/gpu_${gpu}"
        log_file="${shard_root}/worker_${gpu}_cuda_${device_id}.log"
        mkdir -p "${shard_out}"
        (
            cd "${LLM_DIR}"
            export CUDA_VISIBLE_DEVICES="${device_id}"
            export PYTHONUNBUFFERED=1
            python evaluation/hpn_qa_benchmark.py \
                --model_path "${model_path}" \
                --benchmark "${shard_bench}" \
                --output_dir "${shard_out}" \
                --temperature 0
        ) > "${log_file}" 2>&1 &
        pids+=("$!")
        log "  worker ${gpu} on CUDA device ${device_id}: ${log_file}"
    done
    wait_for_workers "parallel benchmark ${label}" "${pids[@]}"

    local shard_files=()
    local shard_file
    for ((gpu=0; gpu<workers; gpu++)); do
        shard_file="${shard_ans_root}/gpu_${gpu}/hpn_answers_${model_name}.xlsx"
        [[ -f "${shard_file}" ]] || die "Missing benchmark shard output: ${shard_file}"
        shard_files+=("${shard_file}")
    done

    run "merge benchmark shards ${label}" python evaluation/merge_answer_shards.py \
        --shard_files "${shard_files[@]}" \
        --output_file "${out_file}" \
        --benchmark "${BENCHMARK_FILE}"
}

run_rag_benchmark() {
    local label="$1" model_path="$2"
    local model_name tag_suffix out_dir out_file workers
    model_name="$(basename "${model_path}")"
    tag_suffix="$(benchmark_tag)"
    out_dir="${OUT_RAG_ANSWERS}"
    out_file="${out_dir}/hpn_answers_RAG-${model_name}${tag_suffix}.xlsx"
    if skip_if_exists "RAG benchmark ${label}" "${out_file}"; then return; fi

    if [[ "${RAG_BENCHMARK_PARALLEL}" != "true" ]]; then
        local direct_workers
        direct_workers="$(benchmark_worker_count)"
        if (( direct_workers > 1 )); then
            local qdrant_cfg qdrant_dir
            qdrant_cfg="$(cd "${RAG_DIR}" && rag_config_value qdrant_dir "data/qdrant_store")"
            qdrant_dir="$(path_from_rag_root "${qdrant_cfg}")"
            warn "RAG benchmark ${label} will run serially because local Qdrant disk mode locks ${qdrant_dir}. Set RAG_BENCHMARK_PARALLEL=true only after switching NetBench-RAG to a concurrent-safe Qdrant backend."
        fi
    fi

    workers="$(rag_benchmark_worker_count)"
    if (( workers > 1 )); then
        local shard_root shard_bench_dir shard_ans_root
        shard_root="${OUT_RAG_ANSWERS}/.shards/RAG-${model_name}${tag_suffix}"
        shard_bench_dir="${shard_root}/benchmarks"
        shard_ans_root="${shard_root}/answers"
        mkdir -p "${shard_bench_dir}" "${shard_ans_root}"
        write_benchmark_shards "${BENCHMARK_FILE}" "${workers}" "${shard_bench_dir}"

        log "Parallel RAG benchmark ${label}: ${workers} GPU worker(s), one benchmark shard per worker"
        local pids=()
        local gpu device_id shard_bench shard_out log_file
        for ((gpu=0; gpu<workers; gpu++)); do
            device_id="$(cuda_device_at "${gpu}")"
            shard_bench="${shard_bench_dir}/shard_${gpu}_of_${workers}.json"
            shard_out="${shard_ans_root}/gpu_${gpu}"
            log_file="${shard_root}/worker_${gpu}_cuda_${device_id}.log"
            mkdir -p "${shard_out}"
            (
                export CUDA_VISIBLE_DEVICES="${device_id}"
                export PYTHONUNBUFFERED=1
                cd "${RAG_DIR}"
                activate_rag_env
                python evaluation/evaluate_rag.py \
                    --model "${model_path}" \
                    --benchmark "${shard_bench}" \
                    --output_dir "${shard_out}"
                deactivate_rag_env
            ) > "${log_file}" 2>&1 &
            pids+=("$!")
            log "  RAG worker ${gpu} on CUDA device ${device_id}: ${log_file}"
        done
        wait_for_workers "parallel RAG benchmark ${label}" "${pids[@]}"

        local shard_files=()
        local shard_file
        for ((gpu=0; gpu<workers; gpu++)); do
            shard_file="${shard_ans_root}/gpu_${gpu}/hpn_answers_RAG-${model_name}.xlsx"
            [[ -f "${shard_file}" ]] || die "Missing RAG benchmark shard output: ${shard_file}"
            shard_files+=("${shard_file}")
        done

        run "merge RAG benchmark shards ${label}" python evaluation/merge_answer_shards.py \
            --shard_files "${shard_files[@]}" \
            --output_file "${out_file}" \
            --benchmark "${BENCHMARK_FILE}"
        return
    fi

    pushd "${RAG_DIR}" > /dev/null
    activate_rag_env
    run "RAG benchmark ${label}" python evaluation/evaluate_rag.py \
        --model "${model_path}" \
        --benchmark "${BENCHMARK_FILE}" \
        --output_dir "${out_dir}"
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

    # Collect only this model's direct and RAG answer files.
    local all_answers=()
    mapfile -t all_answers < <(
        find "${OUT_ANSWERS}" -name "hpn_answers_*.xlsx" 2>/dev/null
        [[ -d "${OUT_RAG_ANSWERS}" ]] && \
            find "${OUT_RAG_ANSWERS}" -name "hpn_answers_*.xlsx" 2>/dev/null \
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
    local device="$1" label="$2" model_path="$3"
    local run_name="direct-${device}-benchmark${PROFILE_BENCHMARK_LIMIT}-${label}"
    local out_file="${OUT_PROFILE}/inference_${run_name}.json"
    if skip_if_exists "profile ${label}" "${out_file}"; then return; fi

    local device_args=()
    [[ "${device}" == "cpu" ]] && device_args=(--cpu)

    run "profile ${device} ${label}" python profiling/profile_inference.py \
        --model_path "${model_path}" \
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
    if skip_if_exists "RAG profile ${label}" "${out_file}"; then return; fi

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
        [[ -f "${report}" ]] && existing+=("${report}")
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
                --batch_size         "${ADAPTATION_BATCH_SIZE}" \
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
                --batch_size         "${ADAPTATION_BATCH_SIZE}" \
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
    configure_output_dirs
    configure_project_caches
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
    echo -e "Output root: ${OUT_MODEL_ROOT}"
    echo -e "Answers   : ${OUT_ANSWERS}"
    echo -e "RAG answers: ${OUT_RAG_ANSWERS}"
    echo -e "Judged    : ${OUT_JUDGED}"
    echo -e "Reports   : ${OUT_REPORTS}"
    echo -e "Profiling : ${OUT_PROFILE}"
    echo -e "Adaptation: ${OUT_ADAPT}\n"
}

main
