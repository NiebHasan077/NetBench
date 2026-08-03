#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Test Instruction-Tuned Models
#  Uses test_instruction_model.py with a model-selection menu.
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

# ── discover available instruction models ───────────────────
declare -a MODEL_PATHS=()
declare -a MODEL_LABELS=()

add_model() {
    local path="$1" label="$2"
    if [ -d "$path" ]; then
        MODEL_PATHS+=("$path")
        MODEL_LABELS+=("$label")
    fi
}

add_model "models/instruction/Llama-3.1-8B-base-instruct"          "Llama-3.1-8B  Base → Instruct"
add_model "models/instruction/Llama-3.1-8B-trained-new-instruct"   "Llama-3.1-8B  Trained-New → Instruct"
add_model "models/instruction/gemma-3-4b-instruct"                 "Gemma-3-4B    Trained-New → Instruct"

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo -e "${RED}No instruction models found.${NC}"
    echo "Run training/instruction_finetune.py first."
    exit 1
fi

# ── allow passing model path(s) directly ────────────────────
if [ -n "$1" ]; then
    # Pass all CLI args straight through to the Python script
    python inference/test_instruction_model.py "$@"
    exit 0
fi

# ── interactive menu ────────────────────────────────────────
echo -e "${BLUE}"
echo "═══════════════════════════════════════════════"
echo "   Instruction-Tuned Model Testing"
echo "═══════════════════════════════════════════════"
echo -e "${NC}"

for i in "${!MODEL_LABELS[@]}"; do
    echo -e "  ${GREEN}$((i+1)))${NC} ${MODEL_LABELS[$i]}"
done

if [ ${#MODEL_PATHS[@]} -ge 2 ]; then
    echo -e "  ${GREEN}$((${#MODEL_PATHS[@]}+1)))${NC} Compare ALL models (side-by-side)"
fi
echo ""

read -p "Select option [1-$((${#MODEL_PATHS[@]}+1))]: " choice

# Compare mode
if [ "$choice" -eq "$((${#MODEL_PATHS[@]}+1))" ] 2>/dev/null && [ ${#MODEL_PATHS[@]} -ge 2 ]; then
    read -p "Enter instruction: " instruction
    echo -e "\n${GREEN}Comparing ${#MODEL_PATHS[@]} models...${NC}\n"
    python inference/test_instruction_model.py \
        --model_path "${MODEL_PATHS[@]}" \
        --instruction "$instruction"
    exit 0
fi

# Single model interactive
idx=$((choice - 1))
if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#MODEL_PATHS[@]}" ]; then
    echo -e "${RED}Invalid selection.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Loading: ${MODEL_LABELS[$idx]}${NC}\n"
python inference/test_instruction_model.py --model_path "${MODEL_PATHS[$idx]}"
