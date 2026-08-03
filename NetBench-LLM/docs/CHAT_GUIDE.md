# Interactive Testing Guide

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `./chat.sh` | Interactive **BASE / TRAINED** model completions |
| `./test_instruct.sh` | Interactive **instruction-tuned** model testing |

Both scripts show a menu of all available models.  You can also pass a
model path directly:

```bash
# Direct model path
./chat.sh models/base/Llama-3.1-8B-base

# Direct instruction model path
./test_instruct.sh --model_path models/instruction/Llama-3.1-8B-base-instruct
```

---

## BASE Model Chat (`chat_interactive.py`)

BASE models **complete text** — they do **not** follow instructions.

### Prompt Modes

Type `mode` during the interactive session to switch:

| Mode | Template | Example |
|------|----------|---------|
| `continue` | `{text}` | Direct continuation |
| `explain` | `{text} \n Response: …` | Academic explanation |
| `define` | `{text} is defined as` | Definition |
| `describe` | `The main characteristics of {text} include` | Characteristics |
| `technical` | `From a technical perspective, {text} works by` | Technical detail |

### Available Models

| Model | Path |
|-------|------|
| Llama-3.2-1B BASE | `models/base/Llama-3.2-1B-base` |
| Llama-3.1-8B BASE | `models/base/Llama-3.1-8B-base` |
| Qwen3.5-2B BASE | `models/base/Qwen3.5-2B-Base` |
| Qwen3.5-4B BASE | `models/base/Qwen3.5-4B-Base` |
| Gemma-3-4B BASE | `models/base/gemma-3-4b` |
| Gemma-4-E4B BASE | `models/base/gemma-4-e4b` |
| Gemma-4-E2B BASE | `models/base/gemma-4-e2b` |
| Llama-3.2-1B TRAINED | `models/pretrained/Llama-3.2-1B-trained-new` |
| Llama-3.1-8B TRAINED | `models/pretrained/Llama-3.1-8B-trained-new` |
| Qwen3.5-2B TRAINED | `models/pretrained/Qwen3.5-2B-trained-new` |
| Qwen3.5-4B TRAINED | `models/pretrained/Qwen3.5-4B-trained-new` |
| Gemma-3-4B TRAINED | `models/pretrained/gemma-3-4b-trained-new` |
| Gemma-4-E4B TRAINED | `models/pretrained/gemma-4-e4b-trained-new` |
| Gemma-4-E2B TRAINED | `models/pretrained/gemma-4-e2b-trained-new` |

### Controls

- `mode` — change prompt template
- `settings` — adjust temperature / max tokens
- `quit` / `exit` — leave the session

---

## Instruction Model Testing (`test_instruction_model.py`)

Instruction-tuned models follow the **Open-Orca** chat prompt template.

**Stop strings** are applied during generation to prevent the model from
continuing past its answer (e.g. self-asking follow-up questions).  Any
trailing stop-string fragments are automatically stripped from the output.

| Model Family | Stop Strings |
|-------------|-------------|
| Llama | `### User:`, `### System:`, `\n### ` |
| Qwen | `<\|im_start\|>`, `<\|im_end\|>` |
| Gemma 3 | `<start_of_turn>`, `<end_of_turn>` |
| Gemma 4 | `<start_of_turn>`, `<end_of_turn>` |

The correct set is auto-detected from the model path — no manual config needed.

### Modes

| Mode | How to invoke |
|------|---------------|
| Interactive REPL | omit `--instruction` |
| Single shot | `--instruction "your question"` |
| Compare models | pass multiple `--model_path` values |

### Multiline Input (Interactive Mode)

Type `<<<` then Enter to start multiline mode.  
Type `>>>` on its own line to submit.

```
📝  Instruction: <<<
... You are a network tuning assistant.
... Predict the optimal number of threads for this transfer:
... source: ANL, destination: NERSC, file size: 10 GB
... >>>
```

### REPL Commands

| Command | Action |
|---------|--------|
| `:q` / `:quit` | Exit |
| `:temp 0.5` | Change temperature |
| `:tokens 512` | Change max new tokens |
| `:help` | Show help |

### Available Models

| Model | Path |
|-------|------|
| 8B Base → Instruct | `models/instruction/Llama-3.1-8B-base-instruct` |
| 8B Trained-New → Instruct | `models/instruction/Llama-3.1-8B-trained-new-instruct` |
| 1B Base → Instruct | `models/instruction/Llama-3.2-1B-base-instruct` |
| 1B Trained-New → Instruct | `models/instruction/Llama-3.2-1B-trained-new-instruct` |
| 1B LoRA-Merged | `models/lora-merged/Llama-3.2-1B-base-instruct-lora-merged` |
| Qwen3.5-2B → Instruct | `models/instruction/Qwen3.5-2B-trained-new-instruct` |
| Qwen3.5-4B → Instruct | `models/instruction/Qwen3.5-4B-trained-new-instruct` |
| Gemma-3-4B → Instruct | `models/instruction/gemma-3-4b-instruct` |
| Gemma-4-E4B → Instruct | `models/instruction/gemma-4-e4b-instruct` |
| Gemma-4-E2B → Instruct | `models/instruction/gemma-4-e2b-instruct` |
| Gemma-4-E4B Official IT | `models/instruct-official/gemma-4-e4b-Instruct` |

---

## Tips

- **BASE models**: use completion-style prompts (NOT questions)
- **Instruction models**: use clear, specific instructions
- TRAINED models should show better domain knowledge (GridFTP, NWS, data-aware scheduling, etc.)
- For 8B models, ensure ≥ 48 GB VRAM available
- Stop strings prevent models from generating follow-up questions after the answer
