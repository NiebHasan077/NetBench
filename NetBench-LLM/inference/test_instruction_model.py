#!/usr/bin/env python3
"""
Test instruction-tuned Llama models.

Supports two modes:
  1. Single-shot: pass --instruction on the command line.
  2. Interactive: launch a REPL to chat with the model.

Usage:
    # Interactive mode
    python test_instruction_model.py \
        --model_path models/instruction/Llama-3.1-8B-base-instruct

    # Single instruction
    python test_instruction_model.py \
        --model_path models/instruction/Llama-3.1-8B-base-instruct \
        --instruction "Explain what GridFTP is."

    # Compare two instruction-tuned models
    python test_instruction_model.py \
        --model_path models/instruction/Llama-3.1-8B-base-instruct \
                     models/instruction/Llama-3.1-8B-trained-new-instruct \
        --instruction "What is the Network Weather Service?"
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import detect_model_family, fix_tokenizer_padding


# ─── Llama / Open-Orca inference templates ─────────────────────────────
ORCA_INFERENCE_TEMPLATE_WITH_SYSTEM = (
    "### System:\n{system}\n\n"
    "### User:\n{instruction}\n\n"
    "### Assistant:\n"
)

ORCA_INFERENCE_TEMPLATE_NO_SYSTEM = (
    "### System:\nYou are a helpful assistant.\n\n"
    "### User:\n{instruction}\n\n"
    "### Assistant:\n"
)

# ─── Qwen / ChatML inference templates ─────────────────────────────────
CHATML_INFERENCE_TEMPLATE_WITH_SYSTEM = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

CHATML_INFERENCE_TEMPLATE_NO_SYSTEM = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# ─── Gemma inference templates ─────────────────────────────────────────
# Gemma 3 folds the system prompt into the first user turn.
GEMMA_INFERENCE_TEMPLATE_WITH_SYSTEM = (
    "<start_of_turn>user\n{system}\n\n{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

GEMMA_INFERENCE_TEMPLATE_NO_SYSTEM = (
    "<start_of_turn>user\n{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# Stop strings per model family
_STOP_STRINGS = {
    "llama": ["### User:", "### System:", "\n### "],
    "qwen":  ["<|im_start|>", "<|im_end|>"],
    "gemma": ["<start_of_turn>", "<end_of_turn>"],
}


def build_prompt(instruction: str, input_text: str = "", model_family: str = "llama") -> str:
    """Build an inference prompt (response left blank for generation)."""
    if model_family == "qwen":
        if input_text.strip():
            return CHATML_INFERENCE_TEMPLATE_WITH_SYSTEM.format(
                system=input_text, instruction=instruction
            )
        return CHATML_INFERENCE_TEMPLATE_NO_SYSTEM.format(instruction=instruction)
    elif model_family == "gemma":
        if input_text.strip():
            return GEMMA_INFERENCE_TEMPLATE_WITH_SYSTEM.format(
                system=input_text, instruction=instruction
            )
        return GEMMA_INFERENCE_TEMPLATE_NO_SYSTEM.format(instruction=instruction)
    else:
        if input_text.strip():
            return ORCA_INFERENCE_TEMPLATE_WITH_SYSTEM.format(
                system=input_text, instruction=instruction
            )
        return ORCA_INFERENCE_TEMPLATE_NO_SYSTEM.format(instruction=instruction)


def load_model(model_path: str):
    """Load model + tokenizer from a local directory.

    Returns:
        (model, tokenizer, model_family)
    """
    print(f"\n{'=' * 70}")
    print(f"Loading: {model_path}")
    print(f"{'=' * 70}")

    model_family = detect_model_family(model_path)
    print(f"  Model family: {model_family}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    fix_tokenizer_padding(tokenizer, model_family)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("✅ Model loaded\n")
    return model, tokenizer, model_family


@torch.no_grad()
def generate(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_family: str = "llama",
):
    """Generate a response for a single instruction."""
    prompt = build_prompt(instruction, input_text, model_family=model_family)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stop_strings = _STOP_STRINGS.get(model_family, _STOP_STRINGS["llama"])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        stop_strings=stop_strings,
        tokenizer=tokenizer,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if model_family == "qwen":
        marker = "<|im_start|>assistant\n"
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
        for tag in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            decoded = decoded.split(tag)[0]
        decoded = decoded.strip()
    elif model_family == "gemma":
        marker = "<start_of_turn>model\n"
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
        for tag in ("<end_of_turn>", "<start_of_turn>", "<eos>"):
            decoded = decoded.split(tag)[0]
        decoded = decoded.strip()
    else:
        if "### Assistant:" in decoded:
            decoded = decoded.split("### Assistant:")[-1].strip()
        for ss in stop_strings:
            if ss in decoded:
                decoded = decoded.split(ss)[0].strip()

    return decoded


# ─── Multiline input helper ────────────────────────────────────────────
def read_multiline(prompt: str) -> str:
    """
    Read multiline input from the terminal.

    - Single-line mode: type your text and press Enter.
    - Multiline mode:   type  <<<  on the first line, then enter as many
      lines as you like.  Finish with  >>>  on its own line.

    Example:
        📝  Instruction: <<<
        ... You are a network tuning assistant.
        ... Predict the optimal number of threads.
        ... >>>
    """
    first = input(prompt).strip()

    # Single-line shortcut
    if first != "<<<":
        return first

    # Multiline mode
    print("    (multiline mode – type >>> on its own line to finish)")
    lines = []
    while True:
        line = input("... ")
        if line.strip() == ">>>":
            break
        lines.append(line)
    return "\n".join(lines)


# ─── Interactive REPL ──────────────────────────────────────────────────
def interactive(model, tokenizer, max_new_tokens, temperature, model_family="llama"):
    """Run an interactive instruction loop."""
    print("=" * 70)
    print("Interactive Instruction Testing")
    print("=" * 70)
    print("Commands:")
    print("  :q / :quit           – exit")
    print("  :temp <float>        – change temperature")
    print("  :tokens <int>        – change max new tokens")
    print("  :help                – show this help")
    print()
    print("Multiline input:")
    print("  Type  <<<  then Enter to start multiline mode.")
    print("  Type  >>>  then Enter to finish and submit.")
    print()

    while True:
        try:
            instruction = read_multiline("📝  Instruction: ")

            if not instruction:
                continue
            if instruction.lower() in (":q", ":quit"):
                print("\n👋 Goodbye!\n")
                break
            if instruction.startswith(":temp"):
                try:
                    temperature = float(instruction.split()[1])
                    print(f"✅ temperature = {temperature}\n")
                except (IndexError, ValueError):
                    print("Usage: :temp 0.7\n")
                continue
            if instruction.startswith(":tokens"):
                try:
                    max_new_tokens = int(instruction.split()[1])
                    print(f"✅ max_new_tokens = {max_new_tokens}\n")
                except (IndexError, ValueError):
                    print("Usage: :tokens 256\n")
                continue
            if instruction == ":help":
                print("  :q / :quit   – exit")
                print("  :temp <f>    – temperature")
                print("  :tokens <n>  – max new tokens")
                print("  <<<          – start multiline input")
                print("  >>>          – end multiline input\n")
                continue

            input_text = read_multiline("💬  Input (optional, Enter to skip): ")

            print(f"\n⏳ Generating (temp={temperature}, "
                  f"tokens={max_new_tokens})...\n")
            resp = generate(
                model, tokenizer, instruction, input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                model_family=model_family,
            )
            print("🤖  Response:")
            print("-" * 70)
            print(resp)
            print("-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as exc:
            print(f"\n❌ Error: {exc}\n")


# ─── Compare multiple models ──────────────────────────────────────────
def compare(model_paths, instruction, input_text, max_new_tokens, temperature):
    """Load several models and show their responses side by side."""
    print(f"\n{'=' * 70}")
    print(f"Comparing {len(model_paths)} model(s)")
    print(f"Instruction: {instruction}")
    if input_text:
        print(f"Input:       {input_text}")
    print(f"{'=' * 70}\n")

    for path in model_paths:
        model, tokenizer, model_family = load_model(path)
        resp = generate(
            model, tokenizer, instruction, input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_family=model_family,
        )
        print(f"── {Path(path).name} ──")
        print(resp)
        print()

        # Free memory before loading the next model
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# ─── CLI ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Test instruction-tuned Llama models"
    )
    p.add_argument(
        "--model_path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to instruction-tuned model(s)",
    )
    p.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Instruction to test (omit for interactive mode)",
    )
    p.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional input / context for the instruction",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)

    args = p.parse_args()

    if args.instruction:
        if len(args.model_path) > 1:
            compare(
                args.model_path, args.instruction, args.input,
                args.max_new_tokens, args.temperature,
            )
        else:
            model, tokenizer, model_family = load_model(args.model_path[0])
            resp = generate(
                model, tokenizer, args.instruction, args.input,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                model_family=model_family,
            )
            print(f"📝 Instruction: {args.instruction}")
            if args.input:
                print(f"💬 Input:       {args.input}")
            print()
            print("🤖 Response:")
            print("-" * 70)
            print(resp)
            print("-" * 70 + "\n")
    else:
        if len(args.model_path) > 1:
            print("⚠️  Interactive mode uses only the first model.")
        model, tokenizer, model_family = load_model(args.model_path[0])
        interactive(
            model, tokenizer, args.max_new_tokens, args.temperature,
            model_family=model_family,
        )


if __name__ == "__main__":
    main()
