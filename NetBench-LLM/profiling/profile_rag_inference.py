#!/usr/bin/env python3
"""Profile local NetBench-RAG inference on benchmark questions.

This measures the full local RAG path:

    question -> retrieval/reranking -> context prompt -> local model generation

It writes JSON reports compatible with the LLM profiling summary script.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _system_info(use_cpu: bool) -> dict[str, Any]:
    cpu_model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    info: dict[str, Any] = {
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
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i,
                "name": p.name,
                "total_memory_mb": round(p.total_memory / 1024**2),
            })
    return info


def _force_cpu_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("embedding", {})["device"] = "cpu"
    cfg.setdefault("retrieval", {})["reranker_device"] = "cpu"
    local = cfg.setdefault("generation", {}).setdefault("local", {})
    local["device_map"] = "cpu"
    local["dtype"] = "float32"
    return cfg


def _chunk_labels(chunks: list[Any]) -> str:
    labels = []
    for chunk in chunks:
        title = chunk.paper_title[:40] + "..." if len(chunk.paper_title) > 40 else chunk.paper_title
        if chunk.section_number:
            sec = f"{chunk.section_number} {chunk.section_heading}".strip()
        else:
            sec = chunk.section_heading or "Body"
        labels.append(f"{title} §{sec}")
    return " | ".join(labels)


def _answer_token_count(generator: Any, answer: str) -> int:
    tokenizer = getattr(getattr(generator, "_backend", None), "_tokenizer", None)
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer(answer, add_special_tokens=False)["input_ids"])
    except Exception:
        return 0


def _aggregate(results: list[dict[str, Any]], use_cpu: bool) -> dict[str, Any]:
    n = max(len(results), 1)
    output_tokens = [r.get("output_tokens", 0) for r in results]
    gen_times = [r.get("generation_time_s", 0.0) for r in results]
    total_output_tokens = sum(output_tokens)
    total_generation_time = sum(gen_times)

    summary = {
        "question_count": len(results),
        "avg_total_time_s": round(sum(r["total_time_s"] for r in results) / n, 4),
        "avg_retrieval_time_s": round(sum(r["retrieval_time_s"] for r in results) / n, 4),
        "avg_generation_time_s": round(total_generation_time / n, 4),
        "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in results) / n, 2),
        "avg_output_tokens": round(total_output_tokens / n, 2),
        "avg_generation_tokens_per_second": round(
            total_output_tokens / total_generation_time, 2
        ) if total_generation_time else 0,
        "peak_cpu_ram_mb": round(psutil.Process().memory_info().rss / 1024**2),
        "peak_gpu_vram_mb": 0,
    }
    if not use_cpu and torch.cuda.is_available():
        summary["peak_gpu_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2)
    return summary


def profile_rag(
    rag_dir: Path,
    config_path: Path,
    benchmark_file: Path,
    model_path: Path,
    output_dir: Path,
    run_name: str,
    limit: int,
    use_cpu: bool,
    top_k: int | None,
) -> Path:
    rag_dir = rag_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    os.chdir(rag_dir)
    sys.path.insert(0, str(rag_dir))

    from evaluation.evaluate_rag import load_benchmark  # type: ignore
    from src.generator import generator_from_config  # type: ignore
    from src.retriever import retriever_from_config  # type: ignore

    cfg = _load_yaml(config_path)
    cfg = {**cfg, "paths": {**cfg.get("paths", {}), "local_model": str(model_path.resolve())}}
    if use_cpu:
        cfg = _force_cpu_config(cfg)

    questions = load_benchmark(str(benchmark_file.resolve())).get("questions", [])
    if limit > 0:
        questions = questions[:limit]
    if not questions:
        raise ValueError(f"No benchmark questions found: {benchmark_file}")

    if not use_cpu and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'=' * 70}")
    print(f"RAG INFERENCE PROFILING: {run_name}")
    print(f"Device   : {'CPU' if use_cpu else 'GPU'}")
    print(f"Model    : {model_path}")
    print(f"Questions: {len(questions)}")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    retriever = retriever_from_config(cfg)
    if top_k is not None:
        retriever.reranker_top_n = top_k
    retriever_load_s = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    generator = generator_from_config(cfg, backend_override="local")
    generator_load_s = round(time.perf_counter() - t0, 2)

    results: list[dict[str, Any]] = []
    for idx, q in enumerate(questions, start=1):
        qid = q.get("id") or f"q{idx:03d}"
        question = q["question"]
        print(f"[{idx:02d}/{len(questions):02d}] {qid}", flush=True)

        total_start = time.perf_counter()
        ret_start = time.perf_counter()
        retrieval = retriever.retrieve(question)
        ret_s = time.perf_counter() - ret_start

        gen_start = time.perf_counter()
        generated = generator.generate(question, retrieval.chunks)
        gen_s = time.perf_counter() - gen_start
        total_s = time.perf_counter() - total_start

        output_tokens = _answer_token_count(generator, generated.answer)
        results.append({
            "id": qid,
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "question_type": q.get("question_type", ""),
            "question": question,
            "retrieval_time_s": round(ret_s, 4),
            "generation_time_s": round(gen_s, 4),
            "total_time_s": round(total_s, 4),
            "prompt_tokens": generated.prompt_tokens,
            "output_tokens": output_tokens,
            "generation_tokens_per_second": round(output_tokens / gen_s, 2) if gen_s else 0,
            "retrieved_chunks": _chunk_labels(retrieval.chunks),
            "reranker_scores": [round(float(s), 4) for s in retrieval.reranker_scores],
            "answer_preview": generated.answer[:300],
        })

    report = {
        "profile_type": "rag",
        "model_path": str(model_path.resolve()),
        "model_name": model_path.name,
        "run_name": run_name,
        "device": "cpu" if use_cpu else "cuda",
        "timestamp": datetime.now().isoformat(),
        "benchmark_file": str(benchmark_file.resolve()),
        "question_limit": limit,
        "system": _system_info(use_cpu),
        "model_load": {
            "retriever_load_time_s": retriever_load_s,
            "generator_load_time_s": generator_load_s,
        },
        "benchmark_prompts": results,
        "summary": _aggregate(results, use_cpu),
    }

    path = output_dir / f"inference_{run_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    del retriever, generator
    gc.collect()
    if torch.cuda.is_available() and not use_cpu:
        torch.cuda.empty_cache()
    os.chdir(original_cwd)
    print(f"Report -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile local NetBench-RAG inference")
    parser.add_argument("--rag_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="outputs/profiling_results")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    config = Path(args.config) if args.config else rag_dir / "config.yaml"
    profile_rag(
        rag_dir=rag_dir,
        config_path=config,
        benchmark_file=Path(args.benchmark_file),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        limit=args.limit,
        use_cpu=args.cpu,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
