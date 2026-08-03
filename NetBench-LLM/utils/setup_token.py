#!/usr/bin/env python3
"""
Setup HuggingFace token from file.
Reads token from hf_token.txt and configures HuggingFace Hub.
"""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the NetBench-LLM project root."""
    return Path(__file__).resolve().parents[1]


def configure_project_caches(cache_dir: str | None = None) -> Path:
    """Keep HuggingFace/Torch/tool caches under the project scratch tree."""
    root = project_root()
    cache_root = Path(
        cache_dir or os.environ.get("PROJECT_CACHE_DIR") or root / ".cache"
    ).expanduser().resolve()

    hf_home = cache_root / "huggingface"
    defaults = {
        "PROJECT_CACHE_DIR": cache_root,
        "XDG_CACHE_HOME": cache_root / "xdg",
        "HF_HUB_CACHE": hf_home / "hub",
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "HF_DATASETS_CACHE": hf_home / "datasets",
        "HF_MODULES_CACHE": hf_home / "modules",
        "HF_ASSETS_CACHE": hf_home / "assets",
        "TRANSFORMERS_CACHE": hf_home / "transformers",
        "DATASETS_CACHE": hf_home / "datasets",
        "TORCH_HOME": cache_root / "torch",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "PIP_CACHE_DIR": cache_root / "pip",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        "TMPDIR": cache_root / "tmp",
        "WANDB_DIR": cache_root / "wandb" / "run",
        "WANDB_CACHE_DIR": cache_root / "wandb" / "cache",
        "WANDB_CONFIG_DIR": cache_root / "wandb" / "config",
        "WANDB_DATA_DIR": cache_root / "wandb" / "data",
    }
    for key, value in defaults.items():
        os.environ[key] = str(value)
    os.environ["HF_HOME"] = str(hf_home)

    for key in defaults:
        path = Path(os.environ[key])
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Do not block token loading if an optional cache directory cannot
            # be created; the caller will surface any later write failure.
            pass

    _patch_loaded_cache_constants()
    return cache_root


def _patch_loaded_cache_constants() -> None:
    """Update cache constants in libraries that may already be imported."""
    try:
        import huggingface_hub.constants as hf_constants

        for attr, env_key in (
            ("HF_HOME", "HF_HOME"),
            ("HF_HUB_CACHE", "HF_HUB_CACHE"),
            ("HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"),
            ("HF_ASSETS_CACHE", "HF_ASSETS_CACHE"),
        ):
            if hasattr(hf_constants, attr):
                setattr(hf_constants, attr, os.environ[env_key])
        if hasattr(hf_constants, "HF_TOKEN_PATH"):
            hf_constants.HF_TOKEN_PATH = os.path.join(os.environ["HF_HOME"], "token")
        if hasattr(hf_constants, "HF_STORED_TOKENS_PATH"):
            hf_constants.HF_STORED_TOKENS_PATH = os.path.join(
                os.environ["HF_HOME"], "stored_tokens"
            )
    except Exception:
        pass

    try:
        import datasets.config as datasets_config

        datasets_config.HF_DATASETS_CACHE = Path(os.environ["HF_DATASETS_CACHE"])
        if hasattr(datasets_config, "DOWNLOADED_DATASETS_PATH"):
            datasets_config.DOWNLOADED_DATASETS_PATH = Path(
                os.environ["HF_DATASETS_CACHE"]
            ) / "downloads"
    except Exception:
        pass

    try:
        import transformers.utils.hub as transformers_hub

        if hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
            transformers_hub.TRANSFORMERS_CACHE = os.environ["TRANSFORMERS_CACHE"]
        if hasattr(transformers_hub, "HF_MODULES_CACHE"):
            transformers_hub.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
    except Exception:
        pass


configure_project_caches()


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
