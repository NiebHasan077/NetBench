#!/usr/bin/env python3
"""Convert a local pytorch_model.bin checkpoint to model.safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    models_root = (project_root / "models").resolve()

    if models_root not in model_dir.parents and model_dir != models_root:
        raise SystemExit(f"Refusing to convert outside project models dir: {model_dir}")

    bin_path = model_dir / "pytorch_model.bin"
    safe_path = model_dir / "model.safetensors"
    if not bin_path.is_file():
        raise SystemExit(f"Missing {bin_path}")
    if safe_path.exists() and not args.force:
        raise SystemExit(f"Already exists: {safe_path}")

    print(f"Loading local tensor checkpoint: {bin_path}")
    try:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict):
        raise SystemExit(f"Expected state dict, got: {type(state_dict).__name__}")
    bad = [key for key, value in state_dict.items() if not torch.is_tensor(value)]
    if bad:
        preview = ", ".join(bad[:5])
        raise SystemExit(f"State dict contains non-tensor values: {preview}")

    print(f"Saving safetensors checkpoint: {safe_path}")
    save_file(state_dict, safe_path, metadata={"format": "pt"})
    print("Done")


if __name__ == "__main__":
    main()
