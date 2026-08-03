"""
frontend/core/model_manager.py
──────────────────────────────
Central registry for all available and currently loaded LLM models.

Responsibilities:
- Scan the `models/` directory and return structured metadata for every model found.
- Load / unload models via HuggingFace transformers with bfloat16 + device_map="auto".
- Enforce a configurable maximum number of simultaneously loaded models (default 1,
  bumped to 2 only when side-by-side Chat Compare is active).
- Report current VRAM usage per GPU via torch.cuda memory stats.
- Thread-safe: load/unload operations are serialised with a threading.Lock.

Model type is inferred from the model's parent directory inside `models/`:
  models/base/        → BASE
  models/pretrained/  → PRETRAINED
  models/instruction/ → INSTRUCT  (also if folder name contains "instruct")
  models/profiled/    → PROFILED_BASE  or  PROFILED_INSTRUCT
                        (resolved by whether folder name contains "instruct")

Public API
──────────
  list_available_models() → list[ModelInfo]
  load_model(path_or_nickname, *, max_loaded=1) → LoadedModel
  unload_model(nickname) → None
  unload_all() → None
  get_loaded_models() → dict[str, LoadedModel]
  get_vram_usage() → list[VRAMInfo]
  get_model(nickname) → LoadedModel | None
"""

from __future__ import annotations

import gc
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model_utils import detect_model_family, fix_tokenizer_padding

# ──────────────────────────────────────────────────────────────────────────────
# Root of the project (two levels up from this file: frontend/core/ → project/)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes / Enums
# ──────────────────────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    BASE             = "BASE"
    PRETRAINED       = "PRETRAINED"
    INSTRUCT         = "INSTRUCT"
    PROFILED_BASE    = "PROFILED_BASE"
    PROFILED_INSTRUCT = "PROFILED_INSTRUCT"

    @property
    def label(self) -> str:
        return {
            ModelType.BASE:              "Base",
            ModelType.PRETRAINED:        "Pretrained",
            ModelType.INSTRUCT:          "Instruct",
            ModelType.PROFILED_BASE:     "Profiled (Base)",
            ModelType.PROFILED_INSTRUCT: "Profiled (Instruct)",
        }[self]

    @property
    def is_instruct(self) -> bool:
        return self in (ModelType.INSTRUCT, ModelType.PROFILED_INSTRUCT)


@dataclass
class ModelInfo:
    """Metadata about a model on disk (not yet loaded)."""
    nickname: str           # Unique human-readable ID, e.g. "Llama-3.2-1B-base"
    name: str               # Folder name, e.g. "Llama-3.2-1B-base"
    path: Path              # Absolute path to the model directory
    model_type: ModelType
    category: str           # Parent directory name, e.g. "base"

    @property
    def display_name(self) -> str:
        return f"[{self.model_type.label}] {self.nickname}"

    def to_dict(self) -> dict:
        return {
            "nickname":   self.nickname,
            "name":       self.name,
            "path":       str(self.path),
            "type":       self.model_type.value,
            "type_label": self.model_type.label,
            "category":   self.category,
        }


@dataclass
class LoadedModel:
    """A model that has been loaded into GPU memory."""
    info: ModelInfo
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    vram_bytes_on_load: int = 0   # Peak VRAM allocated right after loading

    @property
    def nickname(self) -> str:
        return self.info.nickname

    @property
    def model_type(self) -> ModelType:
        return self.info.model_type

    @property
    def is_instruct(self) -> bool:
        return self.info.model_type.is_instruct


@dataclass
class VRAMInfo:
    """VRAM usage for one GPU."""
    gpu_index: int
    gpu_name: str
    used_bytes: int
    total_bytes: int
    temperature_c: Optional[int] = None
    utilization_pct: Optional[int] = None

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return (self.total_bytes - self.used_bytes) / (1024 ** 3)

    @property
    def used_pct(self) -> float:
        return (self.used_bytes / self.total_bytes * 100) if self.total_bytes else 0.0

    def to_dict(self) -> dict:
        return {
            "GPU":           f"GPU {self.gpu_index}: {self.gpu_name}",
            "Used (GB)":     f"{self.used_gb:.2f}",
            "Total (GB)":    f"{self.total_gb:.2f}",
            "Free (GB)":     f"{self.free_gb:.2f}",
            "Used %":        f"{self.used_pct:.1f}%",
            "Temp (°C)":     str(self.temperature_c) if self.temperature_c is not None else "N/A",
            "GPU Util %":    f"{self.utilization_pct}%" if self.utilization_pct is not None else "N/A",
        }


# ──────────────────────────────────────────────────────────────────────────────
# ModelManager
# ──────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Thread-safe registry for scanning, loading, and unloading LLM models.

    Usage (singleton via module-level `manager` instance at bottom of file):

        from frontend.core.model_manager import manager

        for m in manager.list_available_models():
            print(m.display_name)

        loaded = manager.load_model("Llama-3.2-1B-base")
        manager.unload_model("Llama-3.2-1B-base")
    """

    def __init__(self, models_root: Path = MODELS_ROOT, max_loaded: int = 1):
        self._models_root = models_root
        self._max_loaded = max_loaded
        self._lock = threading.Lock()
        self._loaded: dict[str, LoadedModel] = {}   # nickname → LoadedModel

    # ── Directory scanning ────────────────────────────────────────────────────

    def list_available_models(self) -> list[ModelInfo]:
        """
        Scan the `models/` directory and return a list of ModelInfo objects,
        sorted by (category, nickname).

        A directory is considered a valid model if it contains a `config.json`.
        """
        if not self._models_root.exists():
            return []

        results: list[ModelInfo] = []

        for category_dir in sorted(self._models_root.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name  # "base", "pretrained", "instruction", "profiled"

            for model_dir in sorted(category_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                if not (model_dir / "config.json").exists():
                    continue  # Not a HuggingFace model directory

                model_type = self._infer_model_type(category, model_dir.name)
                nickname   = model_dir.name

                results.append(ModelInfo(
                    nickname=nickname,
                    name=model_dir.name,
                    path=model_dir,
                    model_type=model_type,
                    category=category,
                ))

        return results

    def get_available_model(self, nickname: str) -> Optional[ModelInfo]:
        """Return ModelInfo for a given nickname, or None if not found."""
        for m in self.list_available_models():
            if m.nickname == nickname:
                return m
        return None

    # ── Load / Unload ─────────────────────────────────────────────────────────

    def load_model(self, nickname: str, *, max_loaded: Optional[int] = None) -> LoadedModel:
        """
        Load a model into GPU memory.

        Parameters
        ----------
        nickname      : The folder name of the model (e.g. "Llama-3.2-1B-base").
        max_loaded    : Override the instance-level maximum. Use max_loaded=2
                        when the Chat Compare tab needs two models simultaneously.

        Raises
        ------
        ValueError    : Model not found on disk.
        RuntimeError  : Already at maximum loaded-model limit — unload one first.
        RuntimeError  : CUDA OOM or other load failure — message includes fix hint.

        Design notes
        ------------
        The lock is held only for fast dictionary checks, never during the slow
        HuggingFace model load (which can take 30–60 s).  This keeps the VRAM
        timer and unload button responsive while a load is in progress.
        """
        limit = max_loaded if max_loaded is not None else self._max_loaded

        # ── Phase 1: fast pre-checks under lock ───────────────────────────
        with self._lock:
            if nickname in self._loaded:
                return self._loaded[nickname]
            if len(self._loaded) >= limit:
                loaded_names = ", ".join(self._loaded.keys())
                raise RuntimeError(
                    f"Cannot load '{nickname}': already {len(self._loaded)}/{limit} model(s) loaded "
                    f"({loaded_names}). Unload a model first via the System tab."
                )

        # ── Phase 2: disk scan — no lock needed (read-only) ───────────────
        info = self.get_available_model(nickname)
        if info is None:
            raise ValueError(
                f"Model '{nickname}' not found under {self._models_root}. "
                f"Run utils/download_models.py or check the models/ directory."
            )

        # ── Phase 3: slow model load — lock released ───────────────────────
        try:
            vram_before = self._total_vram_used_bytes()
            model, tokenizer = self._load_hf_model(info.path)
            vram_after = self._total_vram_used_bytes()
            new_loaded = LoadedModel(
                info=info,
                model=model,
                tokenizer=tokenizer,
                vram_bytes_on_load=max(0, vram_after - vram_before),
            )
        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            vram = self.get_vram_usage()
            free_str = " | ".join(f"GPU{v.gpu_index}: {v.free_gb:.1f} GB free" for v in vram)
            raise RuntimeError(
                f"CUDA out of memory while loading '{nickname}'.\n"
                f"Current VRAM: {free_str}\n"
                f"Fix: Unload other models first, or use a smaller model (1B instead of 8B).\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{nickname}': {e}\n"
                f"Fix: Check that the model directory is complete (config.json + model.safetensors)."
            ) from e

        # ── Phase 4: commit under lock — double-check for races ───────────
        with self._lock:
            if nickname in self._loaded:
                # Another thread loaded the same model while we were loading.
                # Discard our copy and return the existing one.
                del new_loaded.model
                del new_loaded.tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                return self._loaded[nickname]
            if len(self._loaded) >= limit:
                # Limit changed while we were loading (another model was added).
                del new_loaded.model
                del new_loaded.tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                loaded_names = ", ".join(self._loaded.keys())
                raise RuntimeError(
                    f"Cannot load '{nickname}': model limit reached while loading "
                    f"({loaded_names}). Unload a model first via the System tab."
                )
            self._loaded[nickname] = new_loaded
            return new_loaded

    def unload_model(self, nickname: str) -> None:
        """
        Unload a model and free GPU memory.

        Raises
        ------
        KeyError : Model is not currently loaded.
        """
        with self._lock:
            if nickname not in self._loaded:
                raise KeyError(
                    f"Model '{nickname}' is not loaded. "
                    f"Currently loaded: {list(self._loaded.keys()) or 'none'}."
                )
            loaded = self._loaded.pop(nickname)
            del loaded.model
            del loaded.tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """Unload every currently loaded model."""
        with self._lock:
            for nickname in list(self._loaded.keys()):
                loaded = self._loaded.pop(nickname)
                del loaded.model
                del loaded.tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_loaded_models(self) -> dict[str, LoadedModel]:
        """Return a snapshot dict of {nickname: LoadedModel} for loaded models."""
        with self._lock:
            return dict(self._loaded)

    def get_model(self, nickname: str) -> Optional[LoadedModel]:
        """Return a LoadedModel by nickname, or None if not loaded."""
        with self._lock:
            return self._loaded.get(nickname)

    def is_loaded(self, nickname: str) -> bool:
        with self._lock:
            return nickname in self._loaded

    def loaded_count(self) -> int:
        with self._lock:
            return len(self._loaded)

    # ── VRAM ──────────────────────────────────────────────────────────────────

    def get_vram_usage(self) -> list[VRAMInfo]:
        """
        Return per-GPU VRAM usage.

        Uses nvidia-smi for total/free/temperature/utilization and
        torch.cuda for allocated bytes (more accurate for current process).
        Falls back gracefully if nvidia-smi is unavailable.
        """
        if not torch.cuda.is_available():
            return []

        gpu_count = torch.cuda.device_count()
        results: list[VRAMInfo] = []

        # Fetch extra info from nvidia-smi
        smi_data: dict[int, dict] = {}
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(cmd, text=True, timeout=5)
            for line in output.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    idx = int(parts[0])
                    smi_data[idx] = {
                        "name":         parts[1],
                        "total_mb":     int(parts[2]),
                        "used_mb":      int(parts[3]),
                        "temperature":  int(parts[4]) if parts[4].isdigit() else None,
                        "utilization":  int(parts[5]) if parts[5].isdigit() else None,
                    }
        except Exception:
            pass  # nvidia-smi not available; fall back to torch

        for i in range(gpu_count):
            # Prefer torch for used memory (reflects actual process allocation)
            torch_used  = torch.cuda.memory_allocated(i)
            torch_total = torch.cuda.get_device_properties(i).total_memory

            if i in smi_data:
                d = smi_data[i]
                gpu_name  = d["name"]
                total_bytes = d["total_mb"] * (1024 ** 2)
                # Use torch's allocated bytes for "used" when possible
                # (nvidia-smi includes display driver overhead)
                used_bytes  = torch_used if torch_used > 0 else d["used_mb"] * (1024 ** 2)
                temperature = d["temperature"]
                utilization = d["utilization"]
            else:
                gpu_name    = torch.cuda.get_device_name(i)
                total_bytes = torch_total
                used_bytes  = torch_used
                temperature = None
                utilization = None

            results.append(VRAMInfo(
                gpu_index=i,
                gpu_name=gpu_name,
                used_bytes=used_bytes,
                total_bytes=total_bytes,
                temperature_c=temperature,
                utilization_pct=utilization,
            ))

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _infer_model_type(category: str, folder_name: str) -> ModelType:
        """
        Infer ModelType from the parent category directory and folder name.

            category=base          → BASE
            category=pretrained    → PRETRAINED
            category=instruction   → INSTRUCT
            category=profiled      → PROFILED_INSTRUCT if "instruct" in folder_name
                                     else PROFILED_BASE
        """
        folder_lower = folder_name.lower()

        if category == "base":
            return ModelType.BASE
        if category == "pretrained":
            return ModelType.PRETRAINED
        if category == "instruction":
            return ModelType.INSTRUCT
        if category == "profiled":
            return (
                ModelType.PROFILED_INSTRUCT if "instruct" in folder_lower
                else ModelType.PROFILED_BASE
            )

        # Fallback: use folder name keywords
        if "instruct" in folder_lower:
            return ModelType.INSTRUCT
        if "pretrain" in folder_lower or "trained" in folder_lower:
            return ModelType.PRETRAINED
        return ModelType.BASE

    @staticmethod
    def _load_hf_model(
        path: Path,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a HuggingFace model from a local directory."""
        model_family = detect_model_family(str(path))
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=True,
        )
        fix_tokenizer_padding(tokenizer, model_family)

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"

        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            **model_kwargs,
        )
        model.eval()
        return model, tokenizer

    def _total_vram_used_bytes(self) -> int:
        """Sum of torch-allocated VRAM across all GPUs."""
        if not torch.cuda.is_available():
            return 0
        return sum(
            torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton — import and use directly
# ──────────────────────────────────────────────────────────────────────────────

manager = ModelManager()
