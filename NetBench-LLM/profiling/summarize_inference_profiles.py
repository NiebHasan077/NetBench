#!/usr/bin/env python3
"""Build inference_profile_summary_<model>_<benchmark>.{csv,md} from the
inference_*.json files in a by_model/<model>/profiling_results/ directory.

One summary row per profiling JSON, in the order direct-gpu, direct-cpu,
rag-gpu, rag-cpu (filename order within each group). Direct rows take their
numbers from the JSON's `summary` block (question_count and avg_total_time_s
from `benchmark_prompts`); RAG rows use the RAG profiler's richer `summary`
(tok/s = avg_generation_tokens_per_second) and sum the retriever+generator
load times. This reproduces the format of the summaries generated on the
profiling machines byte-for-byte (validated against untouched models).

Usage:
    python profiling/summarize_inference_profiles.py \
        outputs/by_model/<model>/profiling_results [...more dirs]
"""

import glob
import json
import os
import re
import sys

COLUMNS = ["label", "profile_type", "device", "model_name", "question_count",
           "avg_total_time_s", "avg_retrieval_time_s", "avg_generation_time_s",
           "avg_tokens_per_second", "avg_ms_per_token", "avg_ttft_s",
           "peak_gpu_vram_mb", "peak_cpu_ram_mb", "load_time_s"]


def row_from_json(path):
    d = json.load(open(path))
    label = os.path.basename(path)[len("inference_"):-len(".json")]
    s = d.get("summary", {})
    ml = d.get("model_load", {})
    row = {"label": label, "profile_type": d.get("profile_type"),
           "device": d.get("device"), "model_name": d.get("model_name")}
    if d.get("profile_type") == "rag":
        row.update({
            "question_count": s.get("question_count"),
            "avg_total_time_s": s.get("avg_total_time_s"),
            "avg_retrieval_time_s": s.get("avg_retrieval_time_s"),
            "avg_generation_time_s": s.get("avg_generation_time_s"),
            "avg_tokens_per_second": s.get("avg_generation_tokens_per_second"),
            "avg_ms_per_token": None, "avg_ttft_s": None,
            "peak_gpu_vram_mb": s.get("peak_gpu_vram_mb"),
            "peak_cpu_ram_mb": s.get("peak_cpu_ram_mb"),
            "load_time_s": round(ml.get("retriever_load_time_s", 0)
                                 + ml.get("generator_load_time_s", 0), 2),
        })
    else:
        bp = d.get("benchmark_prompts", [])
        total = round(sum(p["avg_total_time_s"] for p in bp) / len(bp), 4) if bp else None
        row.update({
            "question_count": len(bp),
            "avg_total_time_s": total,
            "avg_retrieval_time_s": None,
            "avg_generation_time_s": total,
            "avg_tokens_per_second": s.get("avg_tokens_per_second"),
            "avg_ms_per_token": s.get("avg_ms_per_token"),
            "avg_ttft_s": s.get("avg_ttft_s"),
            "peak_gpu_vram_mb": s.get("peak_gpu_vram_mb"),
            "peak_cpu_ram_mb": s.get("peak_cpu_ram_mb"),
            "load_time_s": ml.get("load_time_s"),
        })
    return row


def fmt(v):
    return "" if v is None else str(v)


def group_key(row):
    order = {("direct", "cuda"): 0, ("direct", "cpu"): 1,
             ("rag", "cuda"): 2, ("rag", "cpu"): 3}
    return order.get((row["profile_type"], row["device"]), 9)


def summarize(results_dir):
    model = os.path.basename(os.path.dirname(os.path.abspath(results_dir)))
    files = sorted(glob.glob(os.path.join(results_dir, "inference_*.json")))
    by_bench = {}
    for f in files:
        m = re.search(r"(benchmark\d+)", os.path.basename(f))
        if m:
            by_bench.setdefault(m.group(1), []).append(f)
    for bench, fs in sorted(by_bench.items()):
        rows = sorted((row_from_json(f) for f in fs),
                      key=lambda r: (group_key(r), r["label"]))
        stem = os.path.join(results_dir, f"inference_profile_summary_{model}_{bench}")
        # \r\n matches the CSVs produced on the profiling machines
        with open(stem + ".csv", "w", newline="") as f:
            f.write(",".join(COLUMNS) + "\r\n")
            for r in rows:
                f.write(",".join(fmt(r[c]) for c in COLUMNS) + "\r\n")
        with open(stem + ".md", "w") as f:
            f.write("| " + " | ".join(COLUMNS) + " |\n")
            f.write("|" + "|".join([" --- "] * len(COLUMNS)) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |\n")
        print(f"wrote {stem}.csv/.md ({len(rows)} rows)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        summarize(d)
