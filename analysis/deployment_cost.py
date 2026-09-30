"""R7: restate the deployment claims on what the profiling actually measured.

Phase 12, answering the reviewer's "cost and deployment claims exceed the
profiling evidence". Three objections, all fair:

  1. Table 14 reports peak VRAM *per GPU* under multi-way sharding and the text
     concludes memory is not the bottleneck. Reading `profile_inference.py`
     settles it against us: the peak is taken with
     `torch.cuda.max_memory_allocated(device)` where
     `device = next(model.parameters()).device` -- the device holding the first
     shard. It is one shard's allocation, not the maximum across devices and
     certainly not the total. Total accelerator memory and GPU count are the
     numbers a reader needs, and both are recoverable.
  2. "8-13B models form a single workstation-class GPU tier" is not shown by
     profiling an 8B on 4xV100, a 9B on 2xA6000 and a 12B on 4xH100.
  3. "1-2.5 s of retrieval rules out real-time use" is a range with no
     distribution behind it.

All three are answerable from committed artifacts. The profiling JSONs carry a
full `system.gpus` inventory, and the 22 RAG answer workbooks carry
per-question `retrieval_time_sec` and `generation_time_sec` for all 242 items
-- roughly 5,300 timings, enough for percentiles rather than a range.

  4. "1-2B fits CPU or edge hardware" is a *footprint* claim, and the paper
     offers it as a deployment tier without saying what serving on that tier
     costs in latency. The profiling summaries carry a CPU run beside every
     GPU run, so the price is recoverable -- and it is not small.

The even-sharding assumption behind "total = per-shard x GPUs" is not asserted
here; it is checked against the bf16 weight footprint each model must occupy,
and the check is reported so a reader can see how far to trust the total.

Objection 4 is answered with a *within-model* ratio rather than absolute
seconds. Each model was profiled on its own hardware and on its own question
subset (10 questions for six models, 5 for the two Gemmas), so absolute times
do not compare across models; a model's GPU and CPU runs share both, so their
ratio does.

Reads only committed outputs. Nothing is re-profiled.

    analysis/.venv/bin/python analysis/deployment_cost.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# Parameter counts as reported in the roster (Table 6), used only to check the
# even-sharding assumption against the bf16 footprint (2 bytes/parameter).
PARAMS_B = {"gemma-3-1b": 1.0, "llama-3.2-1b": 1.24, "qwen3.5-2b": 2.0,
            "llama-3.1-8b": 8.0, "qwen3.5-9b": 9.0, "gemma-3-12b": 12.0,
            "gemma-3-27b": 27.0, "qwen3.5-27b": 27.0}
RAG_GLOB = ("NetBench-LLM/outputs/**/hpn_answers_RAG*.xlsx",
            "NetBench-RAG/outputs/answers/*.xlsx")


def gpu_inventory(by_model: str) -> pd.DataFrame:
    rows = {}
    for path in glob.glob(os.path.join(by_model, "*/profiling_results/inference_*.json")):
        if "-gpu-" not in path:
            continue
        d = json.load(open(path))
        gpus = d.get("system", {}).get("gpus")
        if not gpus:
            continue
        model = path.split("/by_model/")[1].split("/")[0]
        load_mb = (d.get("model_load") or {}).get("gpu_vram_after_load_mb")
        peak_mb = (d.get("summary") or {}).get("peak_gpu_vram_mb")
        cur = rows.setdefault(model, dict(
            base_model=model, n_gpus=len(gpus), gpu_name=gpus[0]["name"],
            gpu_memory_gb=round(gpus[0]["total_memory_mb"] / 1024, 1),
            shard_load_mb=0, shard_peak_mb=0))
        cur["shard_load_mb"] = max(cur["shard_load_mb"], load_mb or 0)
        cur["shard_peak_mb"] = max(cur["shard_peak_mb"], peak_mb or 0)
    df = pd.DataFrame(rows.values())
    df["params_b"] = df.base_model.map(PARAMS_B)
    df["bf16_weights_gb"] = (2.0 * df.params_b).round(1)
    df["aggregate_gpu_memory_gb"] = (df.gpu_memory_gb * df.n_gpus).round(0)
    df["shard_load_gb"] = (df.shard_load_mb / 1024).round(1)
    df["shard_peak_gb"] = (df.shard_peak_mb / 1024).round(1)
    # Weights, summed back across the shards. Load-time allocation on one device
    # is close to pure weights, so weights/shard_load recovers the shard count --
    # and it lands on n_gpus for every model where load was recorded, i.e.
    # device_map="auto" spread even the 1-2B models over every visible card.
    # gemma-3-1b's load allocation was not recorded (it rounds to 0), so its
    # weight total is unknown rather than zero. Propagate that as NaN: a table
    # cell reading "0.0 GB" for a 1B model would be a fabricated measurement.
    load = df.shard_load_gb.replace(0, np.nan)
    df["implied_shards"] = (df.bf16_weights_gb / load).round(1)
    df["weights_total_gb"] = (load * df.n_gpus).round(1)
    # Deliberately NOT reported: peak x n_gpus. The peak is measured on the
    # first shard's device and includes activations and KV cache, which flow
    # through the pipeline rather than being replicated per device, so
    # multiplying it by the GPU count would invent memory that never existed.
    return df.sort_values("params_b").reset_index(drop=True)


def rag_latency() -> pd.DataFrame:
    seen, frames = set(), []
    for pat in RAG_GLOB:
        for path in glob.glob(pat, recursive=True):
            name = os.path.basename(path)
            if name in seen:
                continue
            seen.add(name)
            d = pd.read_excel(path, sheet_name="Answers")
            if "retrieval_time_sec" not in d:
                continue
            d["run"] = name
            frames.append(d[["run", "id", "difficulty", "retrieval_time_sec",
                             "generation_time_sec", "prompt_tokens"]])
    return pd.concat(frames, ignore_index=True)


def serving_mode_cost(by_model: str) -> pd.DataFrame:
    """GPU-versus-CPU serving cost per model, closed-book and open-book.

    One row per (model, profile_type). Absolute seconds are means over that
    model's variants; the load-bearing column is cpu_gpu_ratio, which is
    matched on hardware and question subset because it divides one model's CPU
    run by its own GPU run.
    """
    frames = []
    for path in glob.glob(os.path.join(by_model, "*/profiling_results/"
                                       "inference_profile_summary_*.csv")):
        d = pd.read_csv(path)
        d["base_model"] = path.split("/by_model/")[1].split("/")[0]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["device"] = df.device.map({"cuda": "gpu", "cpu": "cpu"})

    g = (df.groupby(["base_model", "profile_type", "device"])
           .agg(total_s=("avg_total_time_s", "mean"),
                retrieval_s=("avg_retrieval_time_s", "mean"),
                generation_s=("avg_generation_time_s", "mean"),
                tok_s=("avg_tokens_per_second", "mean"),
                n_runs=("label", "size"),
                n_questions=("question_count", "max"))
           .round(2).reset_index())
    wide = g.pivot(index=["base_model", "profile_type"], columns="device")
    wide.columns = [f"{dev}_{col}" for col, dev in wide.columns]
    wide = wide.reset_index()
    wide["cpu_gpu_ratio"] = (wide.cpu_total_s / wide.gpu_total_s).round(2)
    wide["n_questions"] = wide.gpu_n_questions.fillna(wide.cpu_n_questions).astype(int)
    keep = ["base_model", "profile_type", "n_questions", "gpu_total_s",
            "cpu_total_s", "cpu_gpu_ratio", "gpu_retrieval_s",
            "cpu_retrieval_s", "gpu_tok_s", "cpu_tok_s",
            "gpu_n_runs", "cpu_n_runs"]
    wide["params_b"] = wide.base_model.map(PARAMS_B)
    return (wide.sort_values(["profile_type", "params_b"])[keep]
                .reset_index(drop=True))


def pcts(s: pd.Series) -> dict[str, float]:
    a = s.dropna().to_numpy(dtype=float)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 90, 95, 99)} | \
           {"mean": round(float(a.mean()), 2), "max": round(float(a.max()), 2), "n": len(a)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-model", default="NetBench-LLM/outputs/by_model")
    ap.add_argument("--out", default="analysis/outputs")
    args = ap.parse_args()

    inv = gpu_inventory(args.by_model)
    print("Per-model accelerator footprint (profiling JSONs, system.gpus):\n")
    print(f"{'model':14s} {'GPUs':>17s} {'aggr':>6s} {'wt/shard':>9s} "
          f"{'wt total':>9s} {'bf16':>6s} {'shards':>7s} {'peak/shard':>11s}")
    for _, r in inv.iterrows():
        g = f"{r.n_gpus}x {r.gpu_name.replace('NVIDIA ', '').replace('Tesla ', '')[:12]}"
        print(f"{r.base_model:14s} {g:>17s} {r.aggregate_gpu_memory_gb:5.0f}G "
              f"{r.shard_load_gb:8.1f}G {r.weights_total_gb:8.1f}G "
              f"{r.bf16_weights_gb:5.1f}G {r.implied_shards:7.1f} "
              f"{r.shard_peak_gb:10.1f}G")
    print("\n'shards' = bf16 weights / per-shard load. It lands on the GPU count "
          "for every\nmodel where load was recorded, so device_map spread even the "
          "1-2B models across\nevery visible card -- no model here was profiled on "
          "a single GPU.")
    print("\nWhat this does and does not license:")
    print("  * the binding memory constraint is the weight footprint, 2 bytes x "
          "parameters:\n    1-2B need 2-4GB, 8-9B need 16-18GB, 12B needs 24GB, "
          "27B needs 54GB;")
    print("  * the peak column is one shard's allocation, not a per-model total, "
          "so it\n    cannot support 'memory is not the bottleneck';")
    print("  * single-GPU serving was never profiled, so a 'single "
          "workstation-class GPU\n    tier' is not measurable from this evidence "
          "and should be stated as a\n    footprint argument, not a measurement.")

    lat = rag_latency()
    r, g = pcts(lat.retrieval_time_sec), pcts(lat.generation_time_sec)
    e2e = pcts(lat.retrieval_time_sec + lat.generation_time_sec)
    print(f"\nOpen-book latency over {r['n']} logged per-question timings "
          f"({lat.run.nunique()} RAG runs x 242 items):")
    print(f"{'':14s} {'p50':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'max':>8s}")
    for name, d in (("retrieval", r), ("generation", g), ("end-to-end", e2e)):
        print(f"{name:14s} {d['p50']:6.2f}s {d['p90']:6.2f}s {d['p95']:6.2f}s "
              f"{d['p99']:6.2f}s {d['max']:7.2f}s")

    svc = serving_mode_cost(args.by_model)
    print("\nGPU vs CPU serving, per model. Absolute seconds are not comparable "
          "across\nmodels (own hardware, own question subset); the ratio is, "
          "because it divides a\nmodel's CPU run by its own GPU run.\n")
    print(f"{'model':14s} {'mode':>10s} {'GPU s':>8s} {'CPU s':>8s} "
          f"{'CPU/GPU':>8s} {'GPU tok/s':>10s} {'CPU tok/s':>10s}")
    for _, x in svc.iterrows():
        mode = "open-book" if x.profile_type == "rag" else "closed-book"
        print(f"{x.base_model:14s} {mode:>10s} {x.gpu_total_s:7.1f}s "
              f"{x.cpu_total_s:7.1f}s {x.cpu_gpu_ratio:7.1f}x "
              f"{x.gpu_tok_s:10.2f} {x.cpu_tok_s:10.2f}")
    rag = svc[svc.profile_type == "rag"]
    print(f"\nOpen-book CPU penalty spans {rag.cpu_gpu_ratio.min():.1f}x-"
          f"{rag.cpu_gpu_ratio.max():.1f}x, and the cheapest CPU round trip in "
          f"the roster\nis {rag.cpu_total_s.min():.1f}s "
          f"({rag.loc[rag.cpu_total_s.idxmin()].base_model}), already past "
          "interactive. The CPU/edge tier is a\nfootprint claim that the "
          "retrieval configuration does not inherit.")

    os.makedirs(args.out, exist_ok=True)
    inv.to_csv(os.path.join(args.out, "deployment_footprint.csv"), index=False)
    pd.DataFrame([dict(stage=k, **v) for k, v in
                  (("retrieval", r), ("generation", g), ("end_to_end", e2e))]
                 ).to_csv(os.path.join(args.out, "rag_latency.csv"), index=False)
    svc.to_csv(os.path.join(args.out, "serving_mode_cost.csv"), index=False)
    print(f"\nwrote {args.out}/deployment_footprint.csv, rag_latency.csv "
          "and serving_mode_cost.csv")


if __name__ == "__main__":
    main()
