#!/usr/bin/env python3
"""
Profile Inference Pipeline — GPU and CPU benchmarks.

Measures for every model × prompt combination:
  - Time to first token  (TTFT)
  - Per-token generation latency  (ms / token)
  - End-to-end generation time
  - Throughput  (tokens / sec)
  - Peak GPU VRAM / CPU RAM
  - Model load time

Scaling tests:
  - Latency vs output length  (fixed prompt, growing max_new_tokens)
  - Latency vs input length   (growing prompt, fixed max_new_tokens)

Saves JSON to  outputs/profiling_results/inference_<run_name>.json

Usage:
    # GPU — single model
    python profile_inference.py \
        --model_path models/instruction/Llama-3.1-8B-base-instruct

    # GPU — all discovered models
    python profile_inference.py --all

    # CPU only
    python profile_inference.py --cpu \
        --model_path models/instruction/Llama-3.1-8B-base-instruct

    # Custom prompts file
    python profile_inference.py \
        --model_path models/instruction/Llama-3.1-8B-base-instruct \
        --prompts_file my_prompts.json
"""

import os
import sys
import gc
import json
import time
import argparse
import logging
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

REPORT_DIR = "outputs/profiling_results"

# ═══════════════════════════════════════════════════════════════════
# 1.  DEFAULT BENCHMARK PROMPTS
# ═══════════════════════════════════════════════════════════════════

DEFAULT_PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "minimal",
        "instruction": "Define throughput in one sentence.",
        "max_new_tokens": 30,
    },
    {
        "name": "short_qa",
        "instruction": "What is pipelining in high-performance data transfer?",
        "max_new_tokens": 100,
    },
    {
        "name": "medium_technical",
        "instruction": (
            "Explain the differences between GridFTP and standard FTP "
            "for large-scale scientific data movement. Include discussion of "
            "parallel streams, third-party transfers, and security mechanisms."
        ),
        "max_new_tokens": 256,
    },
    {
        "name": "json_structured",
        "instruction": (
            "You are a network concurrency tuning assistant. Predict the "
            "optimal number of concurrent network threads.\n"
            'STATE: {"throughput_mbps": 450.5, "rtt_ms": 12.3, '
            '"current_threads": 4, "cpu_usage_pct": 35.2}\n\n'
            "Respond with only a JSON object: "
            '{"reasoning": "<string>", "network_threads": <int>}'
        ),
        "max_new_tokens": 150,
    },
    {
        "name": "long_generation",
        "instruction": (
            "Write a detailed technical overview of network optimisation "
            "strategies for wide-area data transfers, covering congestion "
            "control, parallelism, pipelining, and concurrency tuning."
        ),
        "max_new_tokens": 512,
    },
]

# For the output-length scaling test
OUTPUT_SCALING_TOKENS = [32, 64, 128, 256, 512]

# For the input-length scaling test  (chars ≈ 4× tokens)
INPUT_SCALING_CHARS = [200, 800, 2000, 4000]


# ═══════════════════════════════════════════════════════════════════
# 2.  INSTRUCTION PROMPT TEMPLATE  (must match training — Open-Orca)
# ═══════════════════════════════════════════════════════════════════

def format_prompt(instruction: str) -> str:
    return (
        "### System:\nYou are a helpful assistant.\n\n"
        f"### User:\n{instruction}\n\n"
        "### Assistant:\n"
    )


# ═══════════════════════════════════════════════════════════════════
# 3.  SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════

def system_info(use_cpu: bool) -> Dict[str, Any]:
    cpu_model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu" if use_cpu else "cuda",
        "cpu": {
            "model": cpu_model,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        },
    }
    if not use_cpu and torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            total_mem = getattr(p, "total_memory", None) or getattr(p, "total_mem", 0)
            gpus.append({
                "index": i,
                "name": p.name,
                "total_memory_mb": round(total_mem / 1024**2),
            })
        info["gpus"] = gpus
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    return info


# ═══════════════════════════════════════════════════════════════════
# 4.  SINGLE GENERATION BENCHMARK
# ═══════════════════════════════════════════════════════════════════

class _FirstTokenStreamer:
    """Tiny streamer that records the wall-clock of the first generated token.

    Implements the put()/end() interface required by Transformers ≥ 5.0.
    """

    def __init__(self):
        self.first_token_time: Optional[float] = None
        self._count = 0

    def put(self, token_ids):
        self._count += 1
        if self._count == 1:
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


@torch.no_grad()
def benchmark_one(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run a single generation and return timing + memory stats.
    """
    device = next(model.parameters()).device
    formatted = format_prompt(prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # Reset GPU counters
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # ── generate ───────────────────────────────────────────────────
    gen_start = time.perf_counter()

    # We use a trivial streamer to capture TTFT
    first_tok = _FirstTokenStreamer()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        streamer=first_tok,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gen_end = time.perf_counter()

    # ── stats ──────────────────────────────────────────────────────
    total_s    = gen_end - gen_start
    new_tokens = outputs.shape[1] - input_len
    ttft       = (first_tok.first_token_time - gen_start) if first_tok.first_token_time else None

    peak_gpu = 0.0
    if device.type == "cuda":
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**2

    cpu_mb = psutil.Process().memory_info().rss / 1024**2

    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return {
        "input_tokens": input_len,
        "new_tokens": new_tokens,
        "total_time_s": round(total_s, 4),
        "ttft_s": round(ttft, 4) if ttft else None,
        "tokens_per_second": round(new_tokens / total_s, 2) if total_s else 0,
        "ms_per_token": round((total_s / new_tokens) * 1000, 2) if new_tokens else 0,
        "peak_gpu_vram_mb": round(peak_gpu),
        "cpu_ram_mb": round(cpu_mb),
        "text_preview": decoded[:200],
    }


# ═══════════════════════════════════════════════════════════════════
# 5.  FULL BENCHMARK SUITE FOR ONE MODEL
# ═══════════════════════════════════════════════════════════════════

def benchmark_model(
    model_path: str,
    use_cpu: bool = False,
    prompts: Optional[List[Dict]] = None,
    num_runs: int = 3,
    warmup_runs: int = 2,
    skip_scaling: bool = False,
    prompt_source: str = "default",
) -> Dict[str, Any]:
    prompts = prompts or DEFAULT_PROMPTS
    sysinfo = system_info(use_cpu)

    logger.info("=" * 70)
    logger.info(f"  INFERENCE PROFILING : {model_path}")
    logger.info(f"  Device              : {'CPU' if use_cpu else 'GPU'}")
    logger.info(f"  Runs per prompt     : {num_runs}")
    logger.info("=" * 70)

    # ── load model (timed) ─────────────────────────────────────────
    logger.info("\nLoading model …")
    if torch.cuda.is_available() and not use_cpu:
        torch.cuda.reset_peak_memory_stats()

    load_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: Dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if use_cpu:
        load_kw["device_map"] = "cpu"
        load_kw["torch_dtype"] = torch.float32
    else:
        load_kw["device_map"] = "auto"
        load_kw["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
    model.eval()
    load_s = round(time.perf_counter() - load_t0, 2)
    logger.info(f"Model loaded in {load_s}s")

    load_gpu_mb = 0.0
    if torch.cuda.is_available() and not use_cpu:
        load_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2

    # ── warmup ─────────────────────────────────────────────────────
    logger.info(f"\nWarmup ({warmup_runs} runs) …")
    for _ in range(warmup_runs):
        benchmark_one(model, tokenizer, "Hello world.", max_new_tokens=10)

    # ── main benchmark ─────────────────────────────────────────────
    logger.info(f"\nBenchmarking {len(prompts)} prompts × {num_runs} runs …\n")
    prompt_results: List[Dict[str, Any]] = []

    for p in prompts:
        name = p["name"]
        instruction = p["instruction"]
        mnt = p["max_new_tokens"]

        logger.info(f"  📝 {name}  (max_new_tokens={mnt})")
        runs = [benchmark_one(model, tokenizer, instruction, mnt) for _ in range(num_runs)]

        # aggregate
        agg = _aggregate_runs(runs)
        prompt_results.append({
            "name": name,
            "instruction_preview": instruction[:120],
            "max_new_tokens": mnt,
            "num_runs": num_runs,
            **agg,
            "runs": runs,
        })
        logger.info(
            f"     → {agg['avg_tokens_per_second']:.1f} tok/s   "
            f"{agg['avg_ms_per_token']:.1f} ms/tok   "
            f"ttft={agg['avg_ttft_s']:.3f}s\n"
            if agg["avg_ttft_s"] else
            f"     → {agg['avg_tokens_per_second']:.1f} tok/s   "
            f"{agg['avg_ms_per_token']:.1f} ms/tok\n"
        )

    ols_results: List[Dict[str, Any]] = []
    ils_results: List[Dict[str, Any]] = []
    if skip_scaling:
        logger.info("Skipping synthetic scaling tests.")
    else:
        # ── output-length scaling ──────────────────────────────────────
        logger.info("Output-length scaling test …")
        scaling_instruction = "Explain network data transfer optimization strategies in detail."
        for mnt in OUTPUT_SCALING_TOKENS:
            logger.info(f"  max_new_tokens={mnt}")
            r = benchmark_one(model, tokenizer, scaling_instruction, max_new_tokens=mnt)
            ols_results.append({"max_new_tokens": mnt, **r})

        # ── input-length scaling ───────────────────────────────────────
        logger.info("\nInput-length scaling test …")
        filler = ("Explain network data transfer optimization. " * 200)
        for nchars in INPUT_SCALING_CHARS:
            text = filler[:nchars]
            logger.info(f"  ~{nchars} chars")
            r = benchmark_one(model, tokenizer, text, max_new_tokens=64)
            ils_results.append({"input_chars": nchars, **r})

    # ── summary ────────────────────────────────────────────────────
    summary = {
        "avg_tokens_per_second": round(
            sum(p["avg_tokens_per_second"] for p in prompt_results)
            / len(prompt_results), 2,
        ),
        "avg_ms_per_token": round(
            sum(p["avg_ms_per_token"] for p in prompt_results)
            / len(prompt_results), 2,
        ),
        "peak_gpu_vram_mb": max(
            (p["peak_gpu_vram_mb"] for p in prompt_results), default=0
        ),
        "peak_cpu_ram_mb": round(psutil.Process().memory_info().rss / 1024**2),
    }
    if any(p.get("avg_ttft_s") for p in prompt_results):
        ttft_vals = [p["avg_ttft_s"] for p in prompt_results if p.get("avg_ttft_s")]
        summary["avg_ttft_s"] = round(sum(ttft_vals) / len(ttft_vals), 4)

    report = {
        "model_path": model_path,
        "model_name": Path(model_path).name,
        "device": "cpu" if use_cpu else "cuda",
        "profile_type": "direct",
        "prompt_source": prompt_source,
        "timestamp": datetime.now().isoformat(),
        "system": sysinfo,
        "model_load": {
            "load_time_s": load_s,
            "gpu_vram_after_load_mb": round(load_gpu_mb),
        },
        "benchmark_prompts": prompt_results,
        "output_length_scaling": ols_results,
        "input_length_scaling": ils_results,
        "summary": summary,
    }

    # ── free memory ────────────────────────────────────────────────
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available() and not use_cpu:
        torch.cuda.empty_cache()

    return report


def _aggregate_runs(runs: List[Dict]) -> Dict[str, Any]:
    n = len(runs)
    return {
        "avg_total_time_s": round(sum(r["total_time_s"] for r in runs) / n, 4),
        "avg_tokens_per_second": round(sum(r["tokens_per_second"] for r in runs) / n, 2),
        "avg_ms_per_token": round(sum(r["ms_per_token"] for r in runs) / n, 2),
        "avg_ttft_s": round(
            sum(r["ttft_s"] for r in runs if r["ttft_s"] is not None)
            / max(sum(1 for r in runs if r["ttft_s"] is not None), 1), 4
        ) if any(r["ttft_s"] is not None for r in runs) else None,
        "peak_gpu_vram_mb": max(r["peak_gpu_vram_mb"] for r in runs),
    }


# ═══════════════════════════════════════════════════════════════════
# 6.  MODEL DISCOVERY
# ═══════════════════════════════════════════════════════════════════

_MODEL_PARENTS = ["models/base", "models/pretrained", "models/instruction", "models/profiled"]


def discover_models() -> List[str]:
    found = []
    for parent in _MODEL_PARENTS:
        pp = Path(parent)
        if not pp.exists():
            continue
        for child in sorted(pp.iterdir()):
            if (child / "config.json").exists():
                found.append(str(child))
    return found


def _normalise_benchmark_question(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "name": str(raw.get("id") or f"q{index:03d}"),
        "instruction": str(raw.get("question") or raw.get("prompt") or raw.get("instruction") or ""),
    }


def load_benchmark_prompts(
    benchmark_file: str,
    limit: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """Load benchmark questions as profiling prompts.

    Supports the JSONL benchmark format and the legacy JSON format with a
    top-level ``questions`` list. Prompts are taken deterministically from the
    start of the file so repeated profiling runs compare the same questions.
    """
    path = Path(benchmark_file)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        else:
            rows = data.get("questions", [])

    prompts: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        prompt = _normalise_benchmark_question(row, idx)
        if not prompt["instruction"]:
            continue
        prompt["max_new_tokens"] = max_new_tokens
        prompts.append(prompt)
        if limit > 0 and len(prompts) >= limit:
            break

    if not prompts:
        raise ValueError(f"No questions found in benchmark file: {benchmark_file}")
    return prompts


# ═══════════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Profile inference (GPU / CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--model_path", nargs="*", help="Model dir(s) to benchmark")
    p.add_argument("--all", action="store_true", help="Benchmark all discovered models")
    p.add_argument("--cpu", action="store_true", help="CPU-only mode (no GPU)")
    p.add_argument("--num_runs", type=int, default=3, help="Runs per prompt (default: 3)")
    p.add_argument("--warmup_runs", type=int, default=2, help="Warmup runs (default: 2)")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="JSON file with custom prompts "
                        '(list of {"name","instruction","max_new_tokens"})')
    p.add_argument("--benchmark_file", type=str, default=None,
                   help="JSON/JSONL benchmark file to use as profiling prompts")
    p.add_argument("--benchmark_limit", type=int, default=10,
                   help="Number of benchmark questions to profile (default: 10)")
    p.add_argument("--benchmark_max_new_tokens", type=int, default=256,
                   help="max_new_tokens for benchmark prompts (default: 256)")
    p.add_argument("--skip_scaling", action="store_true",
                   help="Skip synthetic output/input length scaling tests")
    p.add_argument("--run_name", type=str, default=None,
                   help="Report filename suffix (auto-generated if omitted)")

    args = p.parse_args()

    # resolve model list
    if args.all:
        model_paths = discover_models()
        if not model_paths:
            logger.error("No models discovered.  Check directory structure.")
            sys.exit(1)
        logger.info(f"Discovered {len(model_paths)} model(s)")
    elif args.model_path:
        model_paths = args.model_path
    else:
        p.print_help()
        sys.exit(1)

    if args.prompts_file and args.benchmark_file:
        logger.error("Use only one of --prompts_file or --benchmark_file.")
        sys.exit(1)

    # custom prompts
    prompts = None
    prompt_source = "default"
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        prompt_source = args.prompts_file
    elif args.benchmark_file:
        prompts = load_benchmark_prompts(
            args.benchmark_file,
            limit=args.benchmark_limit,
            max_new_tokens=args.benchmark_max_new_tokens,
        )
        prompt_source = args.benchmark_file
        logger.info(
            "Loaded %d benchmark profiling prompts from %s",
            len(prompts),
            args.benchmark_file,
        )

    os.makedirs(REPORT_DIR, exist_ok=True)
    all_reports: List[Dict] = []

    for mp in model_paths:
        if not Path(mp).exists():
            logger.warning(f"Skipping (not found): {mp}")
            continue

        report = benchmark_model(
            model_path=mp,
            use_cpu=args.cpu,
            prompts=prompts,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            skip_scaling=args.skip_scaling,
            prompt_source=prompt_source,
        )
        all_reports.append(report)

        # save individual report
        tag = args.run_name or f"{Path(mp).name}_{'cpu' if args.cpu else 'gpu'}"
        path = os.path.join(REPORT_DIR, f"inference_{tag}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"  Report → {path}\n")

    # combined report when benchmarking multiple models
    if len(all_reports) > 1:
        path = os.path.join(
            REPORT_DIR,
            f"inference_comparison_{'cpu' if args.cpu else 'gpu'}.json",
        )
        with open(path, "w") as f:
            json.dump(all_reports, f, indent=2, default=str)
        logger.info(f"Combined report → {path}")

    logger.info("\n✅  Inference profiling complete.\n")


if __name__ == "__main__":
    main()
