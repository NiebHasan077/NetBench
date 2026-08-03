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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _load_json_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"Ignoring unreadable RAG profiling checkpoint {path}: {exc}", flush=True)
        return None


def _resolved_path(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return path


def _question_id(question: dict[str, Any], index: int) -> str:
    return str(question.get("id") or f"q{index:03d}")


def _question_signature(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": _question_id(question, index),
            "question": str(question.get("question", "")),
        }
        for index, question in enumerate(questions, start=1)
    ]


def _checkpoint_is_compatible(
    report: dict[str, Any],
    model_path: Path,
    benchmark_file: Path,
    questions: list[dict[str, Any]],
    use_cpu: bool,
    top_k: int | None,
) -> bool:
    if report.get("profile_type") != "rag":
        return False
    if report.get("device") != ("cpu" if use_cpu else "cuda"):
        return False
    if _resolved_path(str(report.get("model_path", ""))) != _resolved_path(str(model_path.resolve())):
        return False
    if _resolved_path(str(report.get("benchmark_file", ""))) != _resolved_path(str(benchmark_file.resolve())):
        return False

    checkpoint = report.get("checkpoint")
    if isinstance(checkpoint, dict):
        if checkpoint.get("question_signature") != _question_signature(questions):
            return False
        if checkpoint.get("top_k") != top_k:
            return False
    return True


def _ordered_results(
    questions: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = []
    for index, question in enumerate(questions, start=1):
        qid = _question_id(question, index)
        if qid in by_id:
            ordered.append(by_id[qid])
    return ordered


def _build_report(
    *,
    model_path: Path,
    run_name: str,
    use_cpu: bool,
    benchmark_file: Path,
    limit: int,
    questions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    model_load: dict[str, Any],
    top_k: int | None,
    status: str,
) -> dict[str, Any]:
    return {
        "profile_type": "rag",
        "model_path": str(model_path.resolve()),
        "model_name": model_path.name,
        "run_name": run_name,
        "device": "cpu" if use_cpu else "cuda",
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "benchmark_file": str(benchmark_file.resolve()),
        "question_limit": limit,
        "checkpoint": {
            "complete": status == "complete",
            "completed_questions": len(results),
            "total_questions": len(questions),
            "question_signature": _question_signature(questions),
            "top_k": top_k,
            "updated_at": datetime.now().isoformat(),
        },
        "system": _system_info(use_cpu),
        "model_load": model_load,
        "benchmark_prompts": results,
        "summary": _aggregate(results, use_cpu),
    }


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
    path = output_dir / f"inference_{run_name}.json"

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

    results_by_id: dict[str, dict[str, Any]] = {}
    checkpoint = _load_json_report(path)
    if checkpoint and _checkpoint_is_compatible(
        checkpoint,
        model_path=model_path,
        benchmark_file=benchmark_file,
        questions=questions,
        use_cpu=use_cpu,
        top_k=top_k,
    ):
        results_by_id = {
            str(r["id"]): r
            for r in checkpoint.get("benchmark_prompts", [])
            if isinstance(r, dict) and r.get("id")
        }
        results = _ordered_results(questions, results_by_id)
        if len(results) >= len(questions):
            report = _build_report(
                model_path=model_path,
                run_name=run_name,
                use_cpu=use_cpu,
                benchmark_file=benchmark_file,
                limit=limit,
                questions=questions,
                results=results,
                model_load=checkpoint.get("model_load", {}),
                top_k=top_k,
                status="complete",
            )
            _write_json_atomic(path, report)
            os.chdir(original_cwd)
            print(f"Report -> {path}")
            return path
        print(
            f"Resuming RAG profiling checkpoint {path} "
            f"({len(results)}/{len(questions)} questions complete)",
            flush=True,
        )
    elif checkpoint:
        print(f"Existing RAG profiling checkpoint is incompatible; starting fresh: {path}", flush=True)

    t0 = time.perf_counter()
    retriever = retriever_from_config(cfg)
    if top_k is not None:
        retriever.reranker_top_n = top_k
    retriever_load_s = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    generator = generator_from_config(cfg, backend_override="local")
    generator_load_s = round(time.perf_counter() - t0, 2)
    model_load = {
        "retriever_load_time_s": retriever_load_s,
        "generator_load_time_s": generator_load_s,
    }

    for idx, q in enumerate(questions, start=1):
        qid = _question_id(q, idx)
        if qid in results_by_id:
            print(f"[{idx:02d}/{len(questions):02d}] {qid} (checkpoint hit)", flush=True)
            continue

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
        results_by_id[qid] = {
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
        }
        results = _ordered_results(questions, results_by_id)
        _write_json_atomic(path, _build_report(
            model_path=model_path,
            run_name=run_name,
            use_cpu=use_cpu,
            benchmark_file=benchmark_file,
            limit=limit,
            questions=questions,
            results=results,
            model_load=model_load,
            top_k=top_k,
            status="in_progress",
        ))

    results = _ordered_results(questions, results_by_id)
    report = _build_report(
        model_path=model_path,
        run_name=run_name,
        use_cpu=use_cpu,
        benchmark_file=benchmark_file,
        limit=limit,
        questions=questions,
        results=results,
        model_load=model_load,
        top_k=top_k,
        status="complete",
    )
    _write_json_atomic(path, report)

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
