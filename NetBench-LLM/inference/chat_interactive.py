#!/usr/bin/env python3
"""
Interactive Chat for BASE LLMs
Uses completion-style prompts for testing base models.

BASE models complete text - they don't follow instructions or have conversations.
This script helps you test model completions interactively.
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import detect_model_family, fix_tokenizer_padding

warnings.filterwarnings("ignore")


class InteractiveCompletion:
    """Interactive completion testing for BASE models."""
    
    # Prompt modes for BASE models
    PROMPT_MODES = {
        "continue": "{}",  # Direct continuation
        "explain": " {} \n Response: The parameter that helps the most with dealing small files ",
        "define": "{} is defined as",
        "describe": "The main characteristics of {} include",
        "technical": "From a technical perspective, {} works by",
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.model_family = detect_model_family(model_name)
        self.current_mode = "explain"

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        print(f"Model family:  {self.model_family}")
        print("This may take a moment...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        fix_tokenizer_padding(self.tokenizer, self.model_family)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()
        print(f"Model loaded successfully!")
        
    def generate(
        self,
        text: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Generate completion."""
        
        # Apply prompt template
        prompt = self.PROMPT_MODES[self.current_mode].format(text)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Get only new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return prompt, response.strip()
    
    def run(self):
        """Run interactive session."""
        print("\n" + "=" * 60)
        print("Interactive BASE Model Completion")
        print("=" * 60)
        print("\nThis is a BASE model - it completes text, not follows instructions.")
        print("\nCommands:")
        print("  :mode <name>  - Change prompt mode (continue/explain/define/describe/technical)")
        print("  :modes        - Show all available modes")
        print("  :temp <val>   - Set temperature (0.1-2.0)")
        print("  :tokens <n>   - Set max tokens (10-500)")
        print("  :help         - Show this help")
        print("  :quit         - Exit")
        print("\nCurrent mode: " + self.current_mode)
        print("-" * 60)
        
        max_tokens = 200
        temperature = 0.7
        
        while True:
            try:
                user_input = input("\n Enter topic/text: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith(":"):
                    parts = user_input[1:].split()
                    cmd = parts[0].lower()
                    
                    if cmd == "quit" or cmd == "exit":
                        print("Goodbye!")
                        break
                    elif cmd == "help":
                        print("\nCommands:")
                        print("  :mode <name>  - Change mode")
                        print("  :modes        - List modes")
                        print("  :temp <val>   - Set temperature")
                        print("  :tokens <n>   - Set max tokens")
                        print("  :quit         - Exit")
                    elif cmd == "modes":
                        print("\nAvailable modes:")
                        for name, template in self.PROMPT_MODES.items():
                            example = template.format("<your_text>")
                            marker = " (current)" if name == self.current_mode else ""
                            print(f"  {name}{marker}: \"{example}\"")
                    elif cmd == "mode" and len(parts) > 1:
                        mode = parts[1].lower()
                        if mode in self.PROMPT_MODES:
                            self.current_mode = mode
                            print(f"Mode changed to: {mode}")
                        else:
                            print(f"Unknown mode. Available: {list(self.PROMPT_MODES.keys())}")
                    elif cmd == "temp" and len(parts) > 1:
                        try:
                            temperature = float(parts[1])
                            temperature = max(0.1, min(2.0, temperature))
                            print(f"Temperature set to: {temperature}")
                        except ValueError:
                            print("Invalid temperature value")
                    elif cmd == "tokens" and len(parts) > 1:
                        try:
                            max_tokens = int(parts[1])
                            max_tokens = max(10, min(500, max_tokens))
                            print(f"Max tokens set to: {max_tokens}")
                        except ValueError:
                            print("Invalid token value")
                    else:
                        print("Unknown command. Type :help for help.")
                    continue
                
                # Generate completion
                print("\n Generating...")
                prompt, response = self.generate(
                    user_input,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                print(f"\n Prompt: {prompt}")
                print(f"\n Completion: {response}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type :quit to exit.")
            except Exception as e:
                print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive completion testing for BASE LLMs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name or path"
    )
    
    args = parser.parse_args()
    
    session = InteractiveCompletion(args.model_name)
    session.load_model()
    session.run()


if __name__ == "__main__":
    main()
