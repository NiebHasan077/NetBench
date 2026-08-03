#!/usr/bin/env python3
"""
Setup HuggingFace token from file.
Reads token from hf_token.txt and configures HuggingFace Hub.
"""

import os
from pathlib import Path

def load_hf_token(token_file: str = "hf_token.txt") -> str:
    """Load HuggingFace token from file."""
    token_path = Path(token_file)
    
    if not token_path.exists():
        raise FileNotFoundError(
            f"Token file not found: {token_file}\n"
            f"Please create {token_file} with your HuggingFace token.\n"
            f"Get your token from: https://huggingface.co/settings/tokens"
        )
    
    with open(token_path, 'r') as f:
        lines = f.readlines()
    
    # Find first non-comment, non-empty line
    token = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            token = line
            break
    
    if not token or token == "YOUR_TOKEN_HERE":
        raise ValueError(
            f"Please add your HuggingFace token to {token_file}\n"
            f"Replace 'YOUR_TOKEN_HERE' with your actual token.\n"
            f"Get your token from: https://huggingface.co/settings/tokens"
        )
    
    return token


def setup_hf_auth(token_file: str = "hf_token.txt"):
    """Setup HuggingFace authentication from token file."""
    token = load_hf_token(token_file)
    
    # Set environment variable
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    
    # Login to HuggingFace Hub
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ HuggingFace authentication successful!")
        return True
    except Exception as e:
        print(f"❌ HuggingFace authentication failed: {e}")
        return False


if __name__ == "__main__":
    setup_hf_auth()
