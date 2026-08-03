#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Interactive Chat – BASE & TRAINED model completions
#  Uses chat_interactive.py with a model-selection menu.
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtualenv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# ── colour helpers ──────────────────────────────────────────
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ── discover available models ───────────────────────────────
declare -a MODEL_PATHS=()
declare -a MODEL_LABELS=()

add_model() {
    local path="$1" label="$2"
    if [ -d "$path" ]; then
        MODEL_PATHS+=("$path")
        MODEL_LABELS+=("$label")
    fi
}

# Base models
add_model "models/base/Llama-3.2-1B-base"                     "Llama-3.2-1B  BASE"
add_model "models/base/Llama-3.1-8B-base"                     "Llama-3.1-8B  BASE"
add_model "models/base/gemma-3-4b"                             "Gemma-3-4B    BASE"
# Trained (new corpus)
add_model "models/pretrained/Llama-3.2-1B-trained-new"   "Llama-3.2-1B  TRAINED (new corpus)"
add_model "models/pretrained/Llama-3.1-8B-trained-new"   "Llama-3.1-8B  TRAINED (new corpus)"
add_model "models/pretrained/gemma-3-4b-trained-new"     "Gemma-3-4B    TRAINED (new corpus)"

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo -e "${RED}No models found. Run utils/download_models.py first.${NC}"
    exit 1
fi

# ── allow passing model path directly ───────────────────────
if [ -n "$1" ]; then
    if [ -d "$1" ]; then
        echo -e "${GREEN}Using model: $1${NC}"
        python inference/chat_interactive.py --model_name "$1"
        exit 0
    else
        echo -e "${RED}Model directory not found: $1${NC}"
        exit 1
    fi
fi

# ── interactive menu ────────────────────────────────────────
echo -e "${BLUE}"
echo "═══════════════════════════════════════════════"
echo "   Interactive BASE Model Chat"
echo "   (completion-style, NOT instruction-following)"
echo "═══════════════════════════════════════════════"
echo -e "${NC}"

for i in "${!MODEL_LABELS[@]}"; do
    echo -e "  ${GREEN}$((i+1)))${NC} ${MODEL_LABELS[$i]}"
done
echo ""

read -p "Select model [1-${#MODEL_PATHS[@]}]: " choice

idx=$((choice - 1))
if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#MODEL_PATHS[@]}" ]; then
    echo -e "${RED}Invalid selection.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Loading: ${MODEL_LABELS[$idx]}${NC}\n"
python inference/chat_interactive.py --model_name "${MODEL_PATHS[$idx]}"
