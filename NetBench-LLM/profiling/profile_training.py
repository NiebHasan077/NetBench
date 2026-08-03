#!/usr/bin/env python3
"""
Profile Training Pipeline — Continual Pre-training & Instruction Fine-tuning.

Two separate profiling modes, each with its own defaults that match the
real training scripts (pretrain_transformers.py / instruction_finetune.py).

Saves the trained model to --output_dir so that downstream profiling
(inference benchmarks, etc.) can be run on the profiled model.  Also
writes a JSON report to  outputs/profiling_results/training_<run_name>.json .

Auto-defaults:
  - --output_dir  (optional): if omitted, defaults to
        models/profiled/<model_name>-<mode>
  - --run_name    (optional): if omitted, defaults to
        <mode>_<model_name>_<YYYYMMDD_HHMMSS>

Auto-tokenization:
  If --data_dir does not point to a pre-tokenized HuggingFace dataset
  (i.e. no dataset_dict.json), the script will automatically tokenize
  the data before training:
    * pretrain mode — expects a JSON corpus file (e.g. research_corpus.json)
      and calls the same logic as prepare_data.py
    * instruct mode — downloads Orca + Dolly from HuggingFace Hub
      and calls the same logic as prepare_instruction_data.py

Collected metrics:
  - Wall-clock timing  (total, per-epoch, per-step)
  - Throughput          (samples/s, tokens/s)
  - Peak GPU VRAM & CPU RAM
  - GPU utilization & temperature  (background nvidia-smi polling)
  - Loss & learning-rate curves
  - Full hardware + config metadata
  - Dataset provenance  (source file, corpus name, format, sequence length)

Usage:
    # ── Continual pre-training (explicit paths) ────────────────────
    python profile_training.py pretrain \
        --model_path models/base/Llama-3.1-8B-base \
        --data_dir data/processed/llama-3.1-8b \
        --output_dir models/profiled/Llama-3.1-8B-pretrain \
        --run_name pretrain_8b

    # ── Minimal (auto output_dir & run_name) ───────────────────────
    python profile_training.py pretrain \
        --model_path models/base/Llama-3.2-1B-base \
        --data_dir data/processed/llama-3.2-1b \
        --max_steps 50

    # ── Auto-tokenize from raw JSON corpus ─────────────────────────
    python profile_training.py pretrain \
        --model_path models/base/Llama-3.2-1B-base \
        --data_dir data/raw/research_corpus_new.json \
        --max_steps 50

    # ── Instruction fine-tuning ────────────────────────────────────
    python profile_training.py instruct \
        --model_path models/base/Llama-3.1-8B-base \
        --data_dir data/instruction/llama-3.1-8b-base \
        --output_dir models/profiled/Llama-3.1-8B-instruct \
        --run_name instruct_8b_base

    # ── Instruct with auto-tokenization (downloads Orca + Dolly) ──
    python profile_training.py instruct \
        --model_path models/pretrained/Llama-3.1-8B-trained-new \
        --data_dir auto \
        --max_steps 50
"""

import os
import sys
import json
import time
import random
import argparse
import logging
import platform
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
import psutil
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_token import setup_hf_auth
from utils.model_utils import detect_model_family, get_gemma_token_type_key

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

REPORT_DIR = "outputs/profiling_results"


# ═══════════════════════════════════════════════════════════════════
# 1.  HARDWARE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════

def get_gpu_info() -> List[Dict[str, Any]]:
    gpus = []
    if not torch.cuda.is_available():
        return gpus
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        total_mem = getattr(p, "total_memory", None) or getattr(p, "total_mem", 0)
        gpus.append({
            "index": i,
            "name": p.name,
            "total_memory_mb": round(total_mem / 1024**2),
            "compute_capability": f"{p.major}.{p.minor}",
            "multi_processor_count": p.multi_processor_count,
        })
    return gpus


def get_cpu_info() -> Dict[str, Any]:
    model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    return {
        "model": model,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "architecture": platform.machine(),
    }


def system_snapshot() -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": get_gpu_info(),
        "cpu": get_cpu_info(),
    }


# ═══════════════════════════════════════════════════════════════════
# 2.  GPU MONITOR (background nvidia-smi thread)
# ═══════════════════════════════════════════════════════════════════

class GPUMonitor:
    """Poll nvidia-smi every *interval* seconds in a daemon thread."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._snapshots: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,"
                        "memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                ts = time.time()
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    self._snapshots.append({
                        "t": round(ts, 2),
                        "gpu": int(parts[0]),
                        "util_pct": float(parts[1]),
                        "mem_used_mb": float(parts[2]),
                        "mem_total_mb": float(parts[3]),
                        "temp_c": float(parts[4]),
                        "power_w": float(parts[5]) if len(parts) > 5 and parts[5] not in ("[N/A]", "") else None,
                    })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> Dict[str, Any]:
        if not self._snapshots:
            return {}
        gpu_ids = sorted({s["gpu"] for s in self._snapshots})
        out: Dict[str, Any] = {}
        for gid in gpu_ids:
            snaps = [s for s in self._snapshots if s["gpu"] == gid]
            utils = [s["util_pct"] for s in snaps]
            mems  = [s["mem_used_mb"] for s in snaps]
            temps = [s["temp_c"] for s in snaps]
            powers = [s["power_w"] for s in snaps if s["power_w"] is not None]
            out[f"gpu_{gid}"] = {
                "utilization_pct_avg": round(sum(utils) / len(utils), 1),
                "utilization_pct_max": round(max(utils), 1),
                "memory_used_mb_avg": round(sum(mems) / len(mems)),
                "memory_used_mb_max": round(max(mems)),
                "memory_total_mb": round(snaps[0]["mem_total_mb"]),
                "temperature_c_avg": round(sum(temps) / len(temps), 1),
                "temperature_c_max": round(max(temps), 1),
                "power_w_avg": round(sum(powers) / len(powers), 1) if powers else None,
                "power_w_max": round(max(powers), 1) if powers else None,
                "num_snapshots": len(snaps),
            }
        return out

    @property
    def raw(self) -> List[Dict[str, Any]]:
        return list(self._snapshots)


# ═══════════════════════════════════════════════════════════════════
# 3.  TRAINER CALLBACK — per-step metrics
# ═══════════════════════════════════════════════════════════════════

class ProfilingCallback(TrainerCallback):
    """Records loss / lr / memory at every logging step."""

    def __init__(self):
        self.step_logs: List[Dict[str, Any]] = []
        self.eval_logs: List[Dict[str, Any]] = []
        self.epoch_wall: List[float] = []
        self._epoch_t0 = 0.0
        self._train_t0 = 0.0

    def on_train_begin(self, args, state, control, **kw):
        self._train_t0 = time.time()

    def on_epoch_begin(self, args, state, control, **kw):
        self._epoch_t0 = time.time()

    def on_epoch_end(self, args, state, control, **kw):
        self.epoch_wall.append(round(time.time() - self._epoch_t0, 2))

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None or "loss" not in logs:
            return
        mem_alloc = mem_resv = 0.0
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.max_memory_allocated() / 1024**2
            mem_resv  = torch.cuda.max_memory_reserved()  / 1024**2
        self.step_logs.append({
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "loss": round(logs["loss"], 5),
            "lr": logs.get("learning_rate", 0),
            "wall_s": round(time.time() - self._train_t0, 2),
            "gpu_alloc_mb": round(mem_alloc),
            "gpu_resv_mb": round(mem_resv),
        })

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics:
            self.eval_logs.append({
                "step": state.global_step,
                "wall_s": round(time.time() - self._train_t0, 2),
                **{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in metrics.items()},
            })


# ═══════════════════════════════════════════════════════════════════
# 4.  DATASET INFO HELPERS
# ═══════════════════════════════════════════════════════════════════

def _detect_corpus_source(data_dir: str) -> Dict[str, Any]:
    """Infer provenance from the dataset directory path and metadata."""
    data_path = Path(data_dir)
    info: Dict[str, Any] = {
        "data_dir": str(data_path),
        "data_dir_name": data_path.name,
    }

    # Check dataset_info.json produced by the datasets library
    ds_info_path = data_path / "dataset_info.json"
    if ds_info_path.exists():
        try:
            with open(ds_info_path) as f:
                ds_meta = json.load(f)
            info["datasets_lib_info"] = ds_meta
        except Exception:
            pass

    # Heuristic: instruction data vs pre-training data
    dir_str = str(data_path).lower()
    if "instruction" in dir_str or "instruct" in dir_str:
        info["dataset_type"] = "instruction"
        info["template"] = "Open-Orca chat (### System / ### User / ### Assistant)"
        info["source_datasets"] = "Open-Orca/OpenOrca + databricks/databricks-dolly-15k"
    elif "processed" in dir_str or "processed_data" in dir_str:
        info["dataset_type"] = "pretrain_corpus"
        if "new" in dir_str or "data/processed" in dir_str:
            info["source_file"] = "data/raw/research_corpus_new.json"
            info["description"] = "Research corpus (753 documents, ~9.5M tokens)"
        else:
            info["source_file"] = "data/raw/research_corpus.json"
            info["description"] = "Original research corpus (373 documents)"
    else:
        info["dataset_type"] = "unknown"

    return info


# ═══════════════════════════════════════════════════════════════════
# 4a.  AUTO-TOKENIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def _is_tokenized_dataset(data_dir: str) -> bool:
    """Return True if data_dir is already a tokenized HuggingFace dataset."""
    p = Path(data_dir)
    if not p.is_dir():
        return False
    # HF datasets saved with .save_to_disk() always contain dataset_dict.json
    # (DatasetDict) or dataset_info.json (single Dataset).
    return (p / "dataset_dict.json").exists() or (p / "dataset_info.json").exists()


def _tokenize_pretrain_corpus(
    raw_json_path: str,
    model_path: str,
    max_length: int = 2048,
    test_size: float = 0.05,
) -> str:
    """
    Tokenize a raw JSON corpus for pre-training (mirrors prepare_data.py).

    Returns the path to the tokenized dataset directory.
    """
    from tqdm import tqdm

    raw_path = Path(raw_json_path)
    # Output alongside the JSON:  <stem>_tokenized_<model_name>/
    model_name = Path(model_path).name
    output_dir = str(raw_path.parent / f"{raw_path.stem}_tokenized_{model_name}")

    # If we already tokenized this combination, skip
    if _is_tokenized_dataset(output_dir):
        logger.info(f"Tokenized dataset already exists at {output_dir} — reusing.")
        return output_dir

    logger.info(f"Auto-tokenizing pre-training corpus: {raw_json_path}")
    logger.info(f"  model tokenizer : {model_path}")
    logger.info(f"  output          : {output_dir}")

    # ── load raw JSON ──────────────────────────────────────────────
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"  Loaded {len(data)} documents")

    texts = []
    for item in tqdm(data, desc="Extracting texts"):
        if isinstance(item, dict) and "text" in item:
            text = item["text"].strip()
            if text:
                texts.append(text)
        elif isinstance(item, str):
            text = item.strip()
            if text:
                texts.append(text)
    logger.info(f"  Prepared {len(texts)} text documents")
    if not texts:
        raise ValueError(f"No texts found in {raw_json_path}")

    # ── tokenizer ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── dataset → tokenize → group into blocks ────────────────────
    ds = Dataset.from_dict({"text": texts})
    split = ds.train_test_split(test_size=test_size, seed=42)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

    def group_texts(examples, block_size=max_length):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated[list(examples.keys())[0]])
        if total >= block_size:
            total = (total // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tok_train = split["train"].map(tokenize_fn, batched=True,
                                    remove_columns=split["train"].column_names,
                                    desc="Tokenizing train", num_proc=4)
    tok_val = split["test"].map(tokenize_fn, batched=True,
                                 remove_columns=split["test"].column_names,
                                 desc="Tokenizing validation", num_proc=4)

    lm_train = tok_train.map(lambda ex: group_texts(ex, max_length),
                              batched=True, desc="Grouping train")
    lm_val = tok_val.map(lambda ex: group_texts(ex, max_length),
                          batched=True, desc="Grouping validation")

    final = DatasetDict({"train": lm_train, "validation": lm_val})

    os.makedirs(output_dir, exist_ok=True)
    final.save_to_disk(output_dir)
    logger.info(f"  ✅ Tokenized dataset saved — train: {len(lm_train):,}, "
                f"val: {len(lm_val):,}")
    return output_dir


# ── Instruction templates (same as prepare_instruction_data.py) ────
# Open-Orca chat template — unified for both Orca and Dolly samples.
_ORCA_TEMPLATE_WITH_SYSTEM = (
    "### System:\n{system}\n\n"
    "### User:\n{question}\n\n"
    "### Assistant:\n{response}"
)
_ORCA_TEMPLATE_NO_SYSTEM = (
    "### System:\nYou are a helpful assistant.\n\n"
    "### User:\n{question}\n\n"
    "### Assistant:\n{response}"
)


def _format_sample(sample: dict) -> str:
    """Format a single sample (unified schema) into the Open-Orca template."""
    question = sample.get("question", "").strip()
    system = sample.get("system", "").strip()
    response = sample.get("response", "").strip()
    if not question or not response:
        return ""
    if system:
        return _ORCA_TEMPLATE_WITH_SYSTEM.format(
            system=system, question=question, response=response)
    return _ORCA_TEMPLATE_NO_SYSTEM.format(
        question=question, response=response)


def _tokenize_instruction_data(
    model_path: str,
    max_length: int = 512,
    max_samples: int = 5000,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> str:
    """
    Download Orca + Dolly, tokenize, and save (mirrors prepare_instruction_data.py).

    Defaults to 5000 samples (matching prepare_instruction_data.py --max_samples 5000).
    Returns the path to the tokenized dataset directory.
    """
    model_name = Path(model_path).name
    output_dir = os.path.join("data/instruction", f"{model_name}-auto")

    if _is_tokenized_dataset(output_dir):
        logger.info(f"Tokenized instruction dataset already exists at {output_dir} — reusing.")
        return output_dir

    logger.info("Auto-tokenizing instruction dataset (Orca + Dolly)…")
    logger.info(f"  model tokenizer : {model_path}")
    logger.info(f"  output          : {output_dir}")

    # ── tokenizer ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── download ───────────────────────────────────────────────────
    logger.info("  Downloading Open-Orca/OpenOrca (10k subset)…")
    orca_ds = load_dataset("Open-Orca/OpenOrca", split="train[:10000]")
    logger.info("  Downloading databricks/databricks-dolly-15k…")
    dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    def normalize_orca(sample):
        return {
            "system": sample.get("system_prompt", "")[:500],
            "question": sample.get("question", ""),
            "response": sample.get("response", ""),
        }

    def normalize_dolly(sample):
        return {
            "system": sample.get("context", "")[:500],
            "question": sample.get("instruction", ""),
            "response": sample.get("response", ""),
        }

    orca_norm = orca_ds.map(normalize_orca, remove_columns=orca_ds.column_names)
    dolly_norm = dolly_ds.map(normalize_dolly, remove_columns=dolly_ds.column_names)
    dataset = concatenate_datasets([orca_norm, dolly_norm]).shuffle(seed=seed)
    logger.info(f"  Combined: {len(dataset):,} samples")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"  Trimmed to {len(dataset):,} samples")

    # ── format with Open-Orca template ─────────────────────────────
    def add_text(sample):
        sample["text"] = _format_sample(sample)
        return sample

    dataset = dataset.map(add_text, desc="Formatting")
    dataset = dataset.filter(lambda x: bool(x.get("text")))

    # ── tokenize ───────────────────────────────────────────────────
    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"], truncation=True, max_length=max_length,
            padding="max_length", return_attention_mask=True)
        tokens["labels"] = [
            [tid if m == 1 else -100 for tid, m in zip(ids, att)]
            for ids, att in zip(tokens["input_ids"], tokens["attention_mask"])
        ]
        return tokens

    tokenized = dataset.map(tokenize_fn, batched=True,
                             remove_columns=dataset.column_names,
                             desc="Tokenizing", num_proc=4)

    split = tokenized.train_test_split(test_size=val_ratio, seed=seed)
    split["validation"] = split.pop("test")

    os.makedirs(output_dir, exist_ok=True)
    split.save_to_disk(output_dir)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    logger.info(f"  ✅ Instruction dataset saved — train: {len(split['train']):,}, "
                f"val: {len(split['validation']):,}")
    return output_dir


def _auto_tokenize_if_needed(args, mode: str) -> str:
    """
    Check whether args.data_dir is already tokenized.

    If not:
      - pretrain mode: tokenize the raw JSON file pointed to by --data_dir
      - instruct mode: download Orca+Dolly and tokenize (--data_dir may be
        "auto" or a raw JSON that we ignore for instruction data)

    Returns the (possibly updated) data_dir path.
    """
    data_dir = args.data_dir

    # Already tokenized — nothing to do
    if _is_tokenized_dataset(data_dir):
        logger.info(f"Dataset at '{data_dir}' is already tokenized.")
        return data_dir

    logger.info(f"Dataset at '{data_dir}' is NOT a tokenized HF dataset — "
                "auto-tokenizing…")

    if mode == "pretrain":
        # --data_dir should point to a raw JSON file
        raw_path = Path(data_dir)
        if not raw_path.is_file():
            logger.error(
                f"For pre-training auto-tokenization, --data_dir must point "
                f"to a JSON corpus file, but '{data_dir}' is not a file."
            )
            sys.exit(1)
        if raw_path.suffix.lower() not in (".json",):
            logger.warning(f"Expected a .json file but got '{raw_path.suffix}'. "
                           "Attempting to tokenize anyway…")
        return _tokenize_pretrain_corpus(str(raw_path), args.model_path)

    else:  # instruct
        # For instruction mode, we always download Orca+Dolly from the Hub
        if data_dir.lower() == "auto":
            logger.info("--data_dir=auto → downloading Orca+Dolly from HuggingFace Hub.")
        else:
            logger.info(f"'{data_dir}' is not tokenized; downloading Orca+Dolly instead.")
        return _tokenize_instruction_data(args.model_path)


def _count_tokens(dataset, pad_id: int) -> int:
    """Count real (non-pad) tokens in a HF dataset split."""
    count = 0
    for ids in dataset["input_ids"]:
        count += sum(1 for t in ids if t != pad_id)
    return count


def _get_sequence_lengths(dataset, pad_id: int, sample_size: int = 500) -> Dict[str, Any]:
    """Compute sequence length statistics from a dataset split."""
    indices = list(range(len(dataset)))
    if len(indices) > sample_size:
        random.seed(42)
        indices = random.sample(indices, sample_size)

    lengths = []
    for idx in indices:
        ids = dataset[idx]["input_ids"]
        real_len = sum(1 for t in ids if t != pad_id)
        lengths.append(real_len)

    if not lengths:
        return {}
    lengths.sort()
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 1),
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(len(lengths) * 0.95)],
        "sampled_n": len(lengths),
    }


# ═══════════════════════════════════════════════════════════════════
# 5.  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════

def _build_training_args(args, output_dir: str, has_kv_sharing: bool = False) -> TrainingArguments:
    """Build TrainingArguments — model is saved manually after training."""
    ta_kwargs: Dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=9999999,            # avoid mid-training checkpoints
        save_total_limit=1,
        eval_strategy="steps",
        save_strategy="no",            # we save manually at the end
        bf16=True,
        gradient_checkpointing=not has_kv_sharing,
        use_cache=has_kv_sharing,
        optim="adamw_torch",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[],                  # disable wandb / tensorboard
        load_best_model_at_end=False,
    )
    if args.max_steps and args.max_steps > 0:
        ta_kwargs["max_steps"] = args.max_steps

    return TrainingArguments(**ta_kwargs)


def _load_model_and_tokenizer(model_path: str):
    """Load model + tokenizer. Returns (model, tokenizer, load_time_s, has_kv_sharing)."""
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("Loading model …")
    load_t0 = time.time()

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }

    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        logger.info("Flash Attention not available, using SDPA")

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Detect KV-sharing (Gemma 4 E2B/E4B) — use_cache=False breaks these models
    # (HF issue #45242) and gradient checkpointing is also incompatible with them.
    _text_cfg = getattr(model.config, "text_config", model.config)
    _has_kv_sharing = getattr(_text_cfg, "num_kv_shared_layers", 0) > 0
    if _has_kv_sharing:
        model.config.use_cache = True
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True
        logger.info(
            f"Model has {_text_cfg.num_kv_shared_layers} KV-shared layers — "
            "keeping use_cache=True, disabling gradient checkpointing"
        )
    else:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    load_s = round(time.time() - load_t0, 2)
    logger.info(f"Model loaded in {load_s}s")

    return model, tokenizer, load_s, _has_kv_sharing


def _run_profiled(
    mode: str,
    args,
    model,
    tokenizer,
    train_ds,
    eval_ds,
    model_load_s: float,
    sys_info: Dict,
    dataset_info: Dict,
    has_kv_sharing: bool = False,
) -> Dict[str, Any]:
    """Run the Trainer with profiling and return the full report dict."""

    gpu_mon = GPUMonitor(interval=args.gpu_poll_interval)
    prof_cb = ProfilingCallback()

    # token counting
    pad_id = tokenizer.pad_token_id
    train_tok_count = _count_tokens(train_ds, pad_id)
    seq_len_stats = _get_sequence_lengths(train_ds, pad_id)

    logger.info(f"Train: {len(train_ds):,} samples  ({train_tok_count:,} tokens)")
    if eval_ds:
        logger.info(f"Eval : {len(eval_ds):,} samples")

    # collator — Gemma models require token_type_ids / mm_token_type_ids
    _base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model_family = detect_model_family(args.model_path)
    if model_family == "gemma":
        _tt_key = get_gemma_token_type_key(model)
        def collator(features):
            batch = _base_collator(features)
            batch[_tt_key] = torch.zeros_like(batch["input_ids"])
            return batch
        logger.info(f"Gemma model detected — injecting {_tt_key} into batches")
    else:
        collator = _base_collator

    # output directory for trained model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    training_args = _build_training_args(args, output_dir, has_kv_sharing=has_kv_sharing)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[prof_cb],
    )

    # Some model forward methods have **kwargs, which makes the Trainer infer
    # model_accepts_loss_kwargs=True.  Override to False so the Trainer
    # normalises the loss correctly across gradient-accumulation steps.
    if trainer.model_accepts_loss_kwargs:
        trainer.model_accepts_loss_kwargs = False
        logger.info(
            "Overrode model_accepts_loss_kwargs → False "
            "(model forward has **kwargs but does not scale loss by num_items_in_batch)"
        )

    # ── run ────────────────────────────────────────────────────────
    gpu_mon.start()
    cpu_before = psutil.Process().memory_info().rss / 1024**2

    logger.info("\n🚀  Starting profiled training …\n")
    t0 = time.time()
    result = trainer.train()
    wall_s = round(time.time() - t0, 2)

    gpu_mon.stop()
    cpu_after = psutil.Process().memory_info().rss / 1024**2

    # peak GPU memory
    peak_gpu_mb = 0
    per_gpu_peak: Dict[str, int] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.max_memory_allocated(i) / 1024**2
            per_gpu_peak[f"gpu_{i}"] = round(p)
            peak_gpu_mb = max(peak_gpu_mb, p)

    # ── save trained model & tokenizer ─────────────────────────────
    logger.info(f"\n💾  Saving model to {output_dir} …")
    save_t0 = time.time()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_s = round(time.time() - save_t0, 2)
    logger.info(f"    Model saved in {save_s}s")
    saved_model_path = str(Path(output_dir).resolve())

    # ── compile report ─────────────────────────────────────────────
    total_steps = result.global_step
    effective_batch = (
        args.batch_size
        * args.gradient_accumulation_steps
        * max(sys_info["gpu_count"], 1)
    )
    samples_per_sec = len(train_ds) * args.num_epochs / wall_s if wall_s else 0
    tokens_per_sec  = train_tok_count * args.num_epochs / wall_s if wall_s else 0

    report: Dict[str, Any] = {
        "run_name": args.run_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "system": sys_info,

        "config": {
            "model_path": args.model_path,
            "model_name": Path(args.model_path).name,
            "data_dir": args.data_dir,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
        },

        "dataset": {
            **dataset_info,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds else 0,
            "train_tokens": train_tok_count,
            "sequence_length_stats": seq_len_stats,
        },

        "timing": {
            "model_load_s": model_load_s,
            "total_train_s": wall_s,
            "total_train_min": round(wall_s / 60, 2),
            "per_step_s": round(wall_s / total_steps, 3) if total_steps else 0,
            "per_epoch_s": prof_cb.epoch_wall,
            "total_steps": total_steps,
            "model_save_s": save_s,
        },

        "throughput": {
            "samples_per_second": round(samples_per_sec, 2),
            "tokens_per_second": round(tokens_per_sec, 1),
            "steps_per_second": round(total_steps / wall_s, 2) if wall_s else 0,
        },

        "memory": {
            "peak_gpu_vram_mb": round(peak_gpu_mb),
            "per_gpu_peak_mb": per_gpu_peak,
            "cpu_ram_before_mb": round(cpu_before),
            "cpu_ram_after_mb": round(cpu_after),
            "cpu_ram_delta_mb": round(cpu_after - cpu_before),
        },

        "gpu_monitor": gpu_mon.summary(),
        "gpu_monitor_timeseries": gpu_mon.raw,

        "training_loss": {
            "final_loss": result.metrics.get("train_loss", 0),
            "loss_curve": prof_cb.step_logs,
        },
        "eval_metrics": prof_cb.eval_logs,
        "saved_model_path": saved_model_path,
    }

    return report


# ═══════════════════════════════════════════════════════════════════
# 6.  PHASE 1 — CONTINUAL PRE-TRAINING
# ═══════════════════════════════════════════════════════════════════

def profile_pretrain(args):
    """Profile continual pre-training (mirrors pretrain_transformers.py)."""
    setup_hf_auth()
    sys_info = system_snapshot()

    logger.info("=" * 70)
    logger.info("  PROFILED CONTINUAL PRE-TRAINING")
    logger.info("=" * 70)
    logger.info(f"  Model       : {args.model_path}")
    logger.info(f"  Data        : {args.data_dir}")
    logger.info(f"  Run name    : {args.run_name}")
    logger.info(f"  Epochs      : {args.num_epochs}")
    logger.info(f"  Batch       : {args.batch_size} × {args.gradient_accumulation_steps} grad-accum")
    logger.info(f"  LR          : {args.learning_rate}")
    logger.info(f"  Max steps   : {args.max_steps or 'all'}")
    logger.info(f"  GPUs        : {sys_info['gpu_count']}")
    logger.info(f"  Output dir  : {args.output_dir}")
    logger.info("=" * 70)

    model, tokenizer, load_s, has_kv_sharing = _load_model_and_tokenizer(args.model_path)

    # ── dataset ────────────────────────────────────────────────────
    logger.info("Loading pre-training dataset …")
    dataset = load_from_disk(args.data_dir)
    train_ds = dataset["train"]
    eval_ds  = dataset.get("validation", dataset.get("test"))

    dataset_info = _detect_corpus_source(args.data_dir)
    dataset_info["format"] = "pre-training CLM (causal language modelling)"
    dataset_info["tokenizer"] = Path(args.model_path).name

    report = _run_profiled(
        mode="pretrain",
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        model_load_s=load_s,
        sys_info=sys_info,
        dataset_info=dataset_info,
        has_kv_sharing=has_kv_sharing,
    )

    _save_and_print(report)
    return report


# ═══════════════════════════════════════════════════════════════════
# 7.  PHASE 2 — INSTRUCTION FINE-TUNING
# ═══════════════════════════════════════════════════════════════════

def profile_instruct(args):
    """Profile instruction fine-tuning (mirrors instruction_finetune.py)."""
    setup_hf_auth()
    sys_info = system_snapshot()

    logger.info("=" * 70)
    logger.info("  PROFILED INSTRUCTION FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"  Model       : {args.model_path}")
    logger.info(f"  Data        : {args.data_dir}")
    logger.info(f"  Run name    : {args.run_name}")
    logger.info(f"  Epochs      : {args.num_epochs}")
    logger.info(f"  Batch       : {args.batch_size} × {args.gradient_accumulation_steps} grad-accum")
    logger.info(f"  LR          : {args.learning_rate}")
    logger.info(f"  Max steps   : {args.max_steps or 'all'}")
    logger.info(f"  GPUs        : {sys_info['gpu_count']}")
    logger.info(f"  Output dir  : {args.output_dir}")
    logger.info("=" * 70)

    model, tokenizer, load_s, has_kv_sharing = _load_model_and_tokenizer(args.model_path)

    # ── dataset ────────────────────────────────────────────────────
    logger.info("Loading instruction dataset …")
    dataset = load_from_disk(args.data_dir)
    train_ds = dataset["train"]
    eval_ds  = dataset.get("validation", dataset.get("test"))

    dataset_info = _detect_corpus_source(args.data_dir)
    dataset_info["format"] = "instruction SFT (Open-Orca template)"
    dataset_info["tokenizer"] = Path(args.model_path).name

    report = _run_profiled(
        mode="instruct",
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        model_load_s=load_s,
        sys_info=sys_info,
        dataset_info=dataset_info,
        has_kv_sharing=has_kv_sharing,
    )

    _save_and_print(report)
    return report


# ═══════════════════════════════════════════════════════════════════
# 8.  SAVE & PRINT
# ═══════════════════════════════════════════════════════════════════

def _save_and_print(report: Dict[str, Any]):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"training_{report['run_name']}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    t = report["timing"]
    m = report["memory"]
    th = report["throughput"]
    d = report["dataset"]
    wall_s = t["total_train_s"]
    total_steps = t["total_steps"]

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  ✅  PROFILED {report['mode'].upper()} COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Mode               : {report['mode']}")
    logger.info(f"  Model              : {report['config']['model_name']}")
    logger.info(f"  Dataset dir        : {d.get('data_dir_name', '—')}")
    logger.info(f"  Dataset type       : {d.get('dataset_type', '—')}")
    if d.get("source_file"):
        logger.info(f"  Source file        : {d['source_file']}")
    if d.get("source_datasets"):
        logger.info(f"  Source datasets    : {d['source_datasets']}")
    logger.info(f"  Train samples      : {d['train_samples']:,}")
    logger.info(f"  Train tokens       : {d['train_tokens']:,}")
    seq = d.get("sequence_length_stats", {})
    if seq:
        logger.info(f"  Seq lengths        : min={seq['min']}, mean={seq['mean']}, "
                     f"max={seq['max']}, p95={seq['p95']}")
    logger.info(f"  Total wall time    : {wall_s/60:.1f} min  ({wall_s:.0f}s)")
    logger.info(f"  Steps              : {total_steps}")
    if total_steps:
        logger.info(f"  Seconds / step     : {wall_s/total_steps:.2f}")
    logger.info(f"  Samples / sec      : {th['samples_per_second']:.2f}")
    logger.info(f"  Tokens  / sec      : {th['tokens_per_second']:.0f}")
    logger.info(f"  Peak GPU VRAM      : {m['peak_gpu_vram_mb']} MB")
    logger.info(f"  CPU RAM (Δ)        : {m['cpu_ram_delta_mb']:+d} MB")
    logger.info(f"  Final loss         : {report['training_loss']['final_loss']:.4f}")
    logger.info(f"  Report             : {path}")
    logger.info(f"  Saved model        : {report.get('saved_model_path', '—')}")
    if t.get("model_save_s"):
        logger.info(f"  Model save time    : {t['model_save_s']}s")
    logger.info("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# 9.  CLI — subcommands: pretrain / instruct
# ═══════════════════════════════════════════════════════════════════

def _add_common_args(sub):
    """Add arguments shared between both subcommands."""
    sub.add_argument("--model_path", required=True,
                     help="Local model directory")
    sub.add_argument("--data_dir", required=True,
                     help="Tokenized dataset directory, raw JSON corpus file, "
                          "or 'auto' (instruct mode downloads Orca+Dolly)")
    sub.add_argument("--output_dir", default=None,
                     help="Directory to save the trained model. "
                          "Default: models/profiled/<model_name>-<mode>")
    sub.add_argument("--run_name", default=None,
                     help="Name for this profiling run (used in output filename). "
                          "Default: <mode>_<model_name>_<YYYYMMDD_HHMMSS>")

    # hyper-params (None = use mode-specific defaults)
    sub.add_argument("--num_epochs", type=int, default=None,
                     help="Number of epochs (default: 3)")
    sub.add_argument("--batch_size", type=int, default=None,
                     help="Per-device batch size (default: 2)")
    sub.add_argument("--gradient_accumulation_steps", type=int, default=None,
                     help="Gradient accumulation steps (default: 8)")
    sub.add_argument("--learning_rate", type=float, default=None,
                     help="Learning rate (default: 2e-5)")
    sub.add_argument("--warmup_ratio", type=float, default=None,
                     help="Warmup ratio (default: 0.1)")
    sub.add_argument("--weight_decay", type=float, default=None,
                     help="Weight decay (default: 0.01)")
    sub.add_argument("--max_grad_norm", type=float, default=None,
                     help="Max gradient norm (default: 1.0)")
    sub.add_argument("--max_steps", type=int, default=None,
                     help="Stop after N optimizer steps (overrides epochs)")

    # profiling knobs
    sub.add_argument("--logging_steps", type=int, default=5,
                     help="Log loss every N steps (default: 5)")
    sub.add_argument("--eval_steps", type=int, default=100,
                     help="Run eval every N steps (default: 100)")
    sub.add_argument("--gpu_poll_interval", type=float, default=5.0,
                     help="nvidia-smi poll interval in seconds (default: 5)")


def _apply_defaults(args, mode: str):
    """Fill in None values with mode-specific defaults."""
    PRETRAIN_DEFAULTS = {
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    }
    INSTRUCT_DEFAULTS = {
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    }
    defaults = PRETRAIN_DEFAULTS if mode == "pretrain" else INSTRUCT_DEFAULTS
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)


def main():
    p = argparse.ArgumentParser(
        description="Profile training — collects metrics AND saves the trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = p.add_subparsers(dest="mode", help="Training mode")
    subparsers.required = True

    # ── pretrain ───────────────────────────────────────────────────
    sub_pt = subparsers.add_parser(
        "pretrain",
        help="Profile continual pre-training",
        description=(
            "Profile a continual pre-training run.\n"
            "Mirrors pretrain_transformers.py hyperparameters.\n"
            "Saves the trained model to --output_dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(sub_pt)
    sub_pt.set_defaults(func=profile_pretrain)

    # ── instruct ───────────────────────────────────────────────────
    sub_ft = subparsers.add_parser(
        "instruct",
        help="Profile instruction fine-tuning",
        description=(
            "Profile an instruction fine-tuning run.\n"
            "Mirrors instruction_finetune.py hyperparameters.\n"
            "Saves the trained model to --output_dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(sub_ft)
    sub_ft.set_defaults(func=profile_instruct)

    args = p.parse_args()

    # Validate model path
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    # Validate data_dir — allow "auto" for instruct mode, otherwise check
    if args.data_dir.lower() != "auto" and not Path(args.data_dir).exists():
        logger.error(f"Data not found: {args.data_dir}")
        sys.exit(1)

    # ── auto-generate --output_dir if not provided ─────────────────
    if args.output_dir is None:
        model_name = Path(args.model_path).name
        args.output_dir = os.path.join("models/profiled",
                                       f"{model_name}-{args.mode}")
        logger.info(f"Auto output_dir: {args.output_dir}")

    # ── auto-generate --run_name if not provided ───────────────────
    if args.run_name is None:
        model_name = Path(args.model_path).name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.mode}_{model_name}_{ts}"
        logger.info(f"Auto run_name: {args.run_name}")

    # ── auto-tokenize if dataset is not already tokenized ──────────
    args.data_dir = _auto_tokenize_if_needed(args, args.mode)

    _apply_defaults(args, args.mode)
    args.func(args)


if __name__ == "__main__":
    main()
