#!/usr/bin/env python3
"""
Plotting module for profiling reports.

Generates PNG figures from profiling JSON data:
  - Per-run:  loss curve, GPU utilization, memory timeline
  - Cumulative: loss comparison, throughput bars, inference bars,
                scaling charts, GPU-vs-CPU comparison
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)

# ── Style defaults ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLORS = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
    "#FF9800", "#00BCD4", "#E91E63", "#8BC34A",
]


def _save(fig, path: str):
    """Save figure and close."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"    📊  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 1.  PER-RUN TRAINING PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_loss_curve(report: Dict, out_path: str):
    """Loss + learning-rate dual-axis plot for one training run."""
    curve = report.get("training_loss", {}).get("loss_curve", [])
    if not curve:
        return

    steps = [e["step"] for e in curve]
    losses = [e["loss"] for e in curve]
    lrs = [e.get("lr", 0) for e in curve]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color=COLORS[0])
    ax1.plot(steps, losses, color=COLORS[0], linewidth=1.5, label="Loss")
    ax1.tick_params(axis="y", labelcolor=COLORS[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate", color=COLORS[1])
    ax2.plot(steps, lrs, color=COLORS[1], linewidth=1.0, alpha=0.7,
             linestyle="--", label="LR")
    ax2.tick_params(axis="y", labelcolor=COLORS[1])
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    run_name = report.get("run_name", "unknown")
    model = report.get("config", {}).get("model_name", "")
    final_loss = report.get("training_loss", {}).get("final_loss", "?")
    ax1.set_title(f"Loss Curve — {model}\n(run: {run_name}, final loss: {final_loss:.4f})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    _save(fig, out_path)


def plot_gpu_utilization(report: Dict, out_path: str):
    """GPU utilization %, VRAM, temperature from nvidia-smi timeseries."""
    ts = report.get("gpu_monitor_timeseries", [])
    if not ts:
        return

    # group by GPU
    gpus: Dict[int, List[Dict]] = {}
    for s in ts:
        gpus.setdefault(s["gpu"], []).append(s)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    t0 = ts[0]["t"]

    for gid, snaps in sorted(gpus.items()):
        times = [(s["t"] - t0) / 60 for s in snaps]  # minutes
        util = [s["util_pct"] for s in snaps]
        mem = [s["mem_used_mb"] / 1024 for s in snaps]  # GB
        temp = [s["temp_c"] for s in snaps]
        color = COLORS[gid % len(COLORS)]

        axes[0].plot(times, util, color=color, linewidth=1, label=f"GPU {gid}", alpha=0.8)
        axes[1].plot(times, mem, color=color, linewidth=1, label=f"GPU {gid}", alpha=0.8)
        axes[2].plot(times, temp, color=color, linewidth=1, label=f"GPU {gid}", alpha=0.8)

    axes[0].set_ylabel("Utilization (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[1].set_ylabel("VRAM Used (GB)")
    axes[1].legend()
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Time (min)")
    axes[2].legend()

    run_name = report.get("run_name", "unknown")
    fig.suptitle(f"GPU Metrics — {run_name}", fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


def plot_memory_usage(report: Dict, out_path: str):
    """Bar chart: peak GPU VRAM per GPU + CPU RAM delta."""
    mem = report.get("memory", {})
    if not mem:
        return

    per_gpu = mem.get("per_gpu_peak_mb", {})
    cpu_delta = mem.get("cpu_ram_delta_mb", 0)

    labels = list(per_gpu.keys()) + ["CPU Δ"]
    values = [v / 1024 for v in per_gpu.values()] + [cpu_delta / 1024]
    colors = [COLORS[i % len(COLORS)] for i in range(len(per_gpu))] + ["#607D8B"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel("Memory (GB)")
    ax.set_title(f"Peak Memory — {report.get('run_name', '?')}")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f} GB", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════
# 2.  PER-RUN INFERENCE PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_inference_per_prompt(report: Dict, out_path: str):
    """Bar chart of tok/s per prompt for one inference run."""
    prompts = report.get("benchmark_prompts", [])
    if not prompts:
        return

    names = [p["name"] for p in prompts]
    tps = [p.get("avg_tokens_per_second", 0) for p in prompts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, tps, color=COLORS[0], height=0.5)
    ax.set_xlabel("Tokens / second")
    device = report.get("device", "?").upper()
    model = report.get("model_name", "?")
    ax.set_title(f"Inference Throughput — {model} ({device})")

    for bar, val in zip(bars, tps):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, out_path)


def plot_inference_scaling(report: Dict, out_path: str):
    """Output-length + input-length scaling as two subplots."""
    ols = report.get("output_length_scaling", [])
    ils = report.get("input_length_scaling", [])
    if not ols and not ils:
        return

    n_plots = sum([bool(ols), bool(ils)])
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    idx = 0

    if ols:
        ax = axes[idx]
        x = [o["max_new_tokens"] for o in ols]
        y = [o.get("tokens_per_second", 0) for o in ols]
        ax.plot(x, y, marker="o", color=COLORS[0], linewidth=2)
        ax.set_xlabel("max_new_tokens")
        ax.set_ylabel("Tokens / second")
        ax.set_title("Output-Length Scaling")
        idx += 1

    if ils:
        ax = axes[idx]
        x = [o["input_tokens"] for o in ils]
        y = [o.get("tokens_per_second", 0) for o in ils]
        ax.plot(x, y, marker="s", color=COLORS[1], linewidth=2)
        ax.set_xlabel("Input tokens")
        ax.set_ylabel("Tokens / second")
        ax.set_title("Input-Length Scaling")

    model = report.get("model_name", "?")
    device = report.get("device", "?").upper()
    fig.suptitle(f"Scaling — {model} ({device})", fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════
# 3.  CUMULATIVE / COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_loss_comparison(reports: List[Dict], out_path: str):
    """Overlay loss curves from multiple training runs."""
    if not reports:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, r in enumerate(reports):
        curve = r.get("training_loss", {}).get("loss_curve", [])
        if not curve:
            continue
        steps = [e["step"] for e in curve]
        losses = [e["loss"] for e in curve]
        label = r.get("run_name", f"run_{i}")
        # shorten label for readability
        parts = label.split("_")
        if len(parts) > 3:
            label = "_".join(parts[:3])
        ax.plot(steps, losses, color=COLORS[i % len(COLORS)],
                linewidth=1.5, label=label, alpha=0.85)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_path)


def plot_training_throughput(reports: List[Dict], out_path: str):
    """Grouped bar chart: samples/s and tokens/s per training run."""
    if not reports:
        return

    names = []
    samples_s = []
    tokens_s = []
    for r in reports:
        th = r.get("throughput", {})
        label = r.get("run_name", "?")
        parts = label.split("_")
        if len(parts) > 3:
            label = "_".join(parts[:3])
        names.append(label)
        samples_s.append(th.get("samples_per_second", 0))
        tokens_s.append(th.get("tokens_per_second", 0))

    import numpy as np
    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(x - width / 2, samples_s, width, color=COLORS[0],
                     label="Samples/s", alpha=0.85)
    ax1.set_ylabel("Samples / second", color=COLORS[0])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, tokens_s, width, color=COLORS[1],
                     label="Tokens/s", alpha=0.85)
    ax2.set_ylabel("Tokens / second", color=COLORS[1])
    ax2.tick_params(axis="y", labelcolor=COLORS[1])

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax1.set_title("Training Throughput Comparison")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    _save(fig, out_path)


def plot_memory_comparison(reports: List[Dict], out_path: str):
    """Bar chart comparing peak GPU VRAM across training runs."""
    if not reports:
        return

    names = []
    peak_vram = []
    for r in reports:
        m = r.get("memory", {})
        label = r.get("run_name", "?")
        parts = label.split("_")
        if len(parts) > 3:
            label = "_".join(parts[:3])
        names.append(label)
        peak_vram.append(m.get("peak_gpu_vram_mb", 0) / 1024)  # GB

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, peak_vram,
                   color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                   width=0.5)
    ax.set_ylabel("Peak GPU VRAM (GB)")
    ax.set_title("Peak GPU Memory Comparison")
    plt.xticks(rotation=25, ha="right", fontsize=9)

    for bar, val in zip(bars, peak_vram):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save(fig, out_path)


def plot_inference_comparison(reports: List[Dict], out_path: str):
    """Bar chart comparing avg tok/s across inference runs (one device)."""
    if not reports:
        return

    names = []
    tps = []
    for r in reports:
        s = r.get("summary", {})
        names.append(f"{r.get('model_name','?')}\n({r.get('device','?')})")
        tps.append(s.get("avg_tokens_per_second", 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, tps,
                   color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                   width=0.5)
    ax.set_ylabel("Avg Tokens / second")
    ax.set_title("Inference Throughput Comparison")

    for bar, val in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save(fig, out_path)


def plot_gpu_vs_cpu(gpu_reports: List[Dict], cpu_reports: List[Dict],
                    out_path: str):
    """Side-by-side bar chart for models benchmarked on both GPU and CPU."""
    gpu_map = {r.get("model_name"): r for r in gpu_reports}
    cpu_map = {r.get("model_name"): r for r in cpu_reports}
    common = sorted(set(gpu_map) & set(cpu_map))
    if not common:
        return

    import numpy as np

    names = common
    gpu_tps = [gpu_map[n].get("summary", {}).get("avg_tokens_per_second", 0)
               for n in names]
    cpu_tps = [cpu_map[n].get("summary", {}).get("avg_tokens_per_second", 0)
               for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, gpu_tps, width, color=COLORS[0], label="GPU")
    ax.bar(x + width / 2, cpu_tps, width, color=COLORS[1], label="CPU")

    ax.set_ylabel("Avg Tokens / second")
    ax.set_title("GPU vs CPU Inference")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()

    # annotate speedup
    for i in range(len(names)):
        speedup = gpu_tps[i] / cpu_tps[i] if cpu_tps[i] else 0
        ax.text(x[i], max(gpu_tps[i], cpu_tps[i]) + 1,
                f"{speedup:.1f}×", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, out_path)
