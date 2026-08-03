#!/usr/bin/env python3
"""Summarize direct and RAG inference profiling JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "label",
    "profile_type",
    "device",
    "model_name",
    "question_count",
    "avg_total_time_s",
    "avg_retrieval_time_s",
    "avg_generation_time_s",
    "avg_tokens_per_second",
    "avg_ms_per_token",
    "avg_ttft_s",
    "peak_gpu_vram_mb",
    "peak_cpu_ram_mb",
    "load_time_s",
]


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _direct_avg_total(report: dict[str, Any]) -> float:
    prompts = report.get("benchmark_prompts", [])
    values = [float(p.get("avg_total_time_s", 0.0)) for p in prompts if p.get("avg_total_time_s") is not None]
    return _avg(values)


def _row(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    profile_type = report.get("profile_type", "direct")
    label = report.get("run_name") or path.stem.removeprefix("inference_")
    model_load = report.get("model_load", {})

    if profile_type == "rag":
        load_time = (
            float(model_load.get("retriever_load_time_s", 0.0))
            + float(model_load.get("generator_load_time_s", 0.0))
        )
        return {
            "label": label,
            "profile_type": "rag",
            "device": report.get("device", ""),
            "model_name": report.get("model_name", ""),
            "question_count": summary.get("question_count", len(report.get("benchmark_prompts", []))),
            "avg_total_time_s": summary.get("avg_total_time_s", 0),
            "avg_retrieval_time_s": summary.get("avg_retrieval_time_s", 0),
            "avg_generation_time_s": summary.get("avg_generation_time_s", 0),
            "avg_tokens_per_second": summary.get("avg_generation_tokens_per_second", 0),
            "avg_ms_per_token": "",
            "avg_ttft_s": "",
            "peak_gpu_vram_mb": summary.get("peak_gpu_vram_mb", 0),
            "peak_cpu_ram_mb": summary.get("peak_cpu_ram_mb", 0),
            "load_time_s": round(load_time, 2),
        }

    return {
        "label": label,
        "profile_type": "direct",
        "device": report.get("device", ""),
        "model_name": report.get("model_name", ""),
        "question_count": len(report.get("benchmark_prompts", [])),
        "avg_total_time_s": _direct_avg_total(report),
        "avg_retrieval_time_s": "",
        "avg_generation_time_s": _direct_avg_total(report),
        "avg_tokens_per_second": summary.get("avg_tokens_per_second", 0),
        "avg_ms_per_token": summary.get("avg_ms_per_token", 0),
        "avg_ttft_s": summary.get("avg_ttft_s", ""),
        "peak_gpu_vram_mb": summary.get("peak_gpu_vram_mb", 0),
        "peak_cpu_ram_mb": summary.get("peak_cpu_ram_mb", 0),
        "load_time_s": model_load.get("load_time_s", 0),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(FIELDS) + " |"
    sep = "| " + " | ".join("---" for _ in FIELDS) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize inference profile reports")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    rows = [_row(Path(path)) for path in args.reports if Path(path).exists()]
    if not rows:
        raise SystemExit("No existing reports found.")

    csv_path = Path(args.output_csv)
    md_path = Path(args.output_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown(rows))

    print(f"CSV summary: {csv_path}")
    print(f"Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
