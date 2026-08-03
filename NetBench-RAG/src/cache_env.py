"""Project-local cache defaults for NetBench-RAG."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_project_caches(cache_dir: str | None = None) -> Path:
    cache_root = Path(
        cache_dir or os.environ.get("PROJECT_CACHE_DIR") or project_root() / ".cache"
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
        try:
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    _patch_loaded_cache_constants()
    return cache_root


def _patch_loaded_cache_constants() -> None:
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
