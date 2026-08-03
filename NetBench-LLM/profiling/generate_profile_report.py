#!/usr/bin/env python3
"""
Generate profiling reports from JSON files.

Orchestrator script that reads outputs/profiling_results/*.json and produces:
  - Per-run directories  with Excel workbooks + PNG plots
  - Cumulative directory with comparison Excel + comparison plots
  - A lightweight  PROFILING_REPORT.md  with embedded image links

The report includes ALL JSON files found in outputs/profiling_results/.
Use --filter to narrow down to specific run names or model names.
Use --list  to see what JSON files are available without generating anything.

Usage:
    # Generate all reports (Markdown + Excel + plots)
    python generate_profile_report.py

    # List available profiling JSON files without generating
    python generate_profile_report.py --list

    # Generate report for a specific run only
    python generate_profile_report.py --filter pretrain_1b_quick

    # Filter by model name substring
    python generate_profile_report.py --filter 8B

    # Custom output directory
    python generate_profile_report.py --output_dir my_reports
"""

import os
import sys
import json
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path so `from profiling import ...` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = "outputs/profiling_results"
REPORTS_DIR = "outputs/profiling_reports"


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _load(pattern: str, results_dir: str = RESULTS_DIR) -> List[Dict[str, Any]]:
    out: List[Dict] = []
    for fp in sorted(glob.glob(os.path.join(results_dir, pattern))):
        with open(fp) as f:
            data = json.load(f)
        if isinstance(data, list):
            out.extend(data)
        else:
            out.append(data)
    return out


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s/60:.1f} min"
    return f"{s/3600:.1f} h"


def _n(v: Any, decimals: int = 2) -> str:
    """Format a number nicely, with comma-thousands when it's int-ish."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == int(v) and v > 100:
            return f"{int(v):,}"
        return f"{v:,.{decimals}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# ═══════════════════════════════════════════════════════════════════
# sections
# ═══════════════════════════════════════════════════════════════════

def _section_training(reports: List[Dict], output_dir: str = REPORTS_DIR) -> str:
    if not reports:
        return "## 1 · Training Profiling\n\n_No training profiling data found._\n"

    lines = ["## 1 · Training Profiling\n"]

    # ── 1a. summary table ──────────────────────────────────────────
    lines.append("### 1.1  Summary\n")
    lines.append(
        "| Run | Mode | Model | Epochs | Steps | Samples | Tokens "
        "| Wall time | s/step | Samples/s | Tokens/s | Peak GPU MB | Final Loss "
        "| Saved Model |"
    )
    lines.append("|" + "|".join(["---"] * 14) + "|")

    for r in reports:
        c = r.get("config", {})
        t = r.get("timing", {})
        th = r.get("throughput", {})
        m = r.get("memory", {})
        d = r.get("dataset", {})
        l = r.get("training_loss", {})
        saved = r.get("saved_model_path", "—")
        # show relative path if possible, otherwise full path
        if saved and saved != "—":
            try:
                saved = f"`{Path(saved).relative_to(Path.cwd())}`"
            except ValueError:
                saved = f"`{saved}`"
        lines.append(
            f"| {r.get('run_name','—')} "
            f"| {r.get('mode','—')} "
            f"| `{Path(c.get('model_path','')).name}` "
            f"| {c.get('num_epochs','—')} "
            f"| {_n(t.get('total_steps'))} "
            f"| {_n(d.get('train_samples'))} "
            f"| {_n(d.get('train_tokens'))} "
            f"| {_fmt_time(t.get('total_train_s',0))} "
            f"| {_n(t.get('per_step_s'),3)} "
            f"| {_n(th.get('samples_per_second'))} "
            f"| {_n(th.get('tokens_per_second'),0)} "
            f"| {_n(m.get('peak_gpu_vram_mb'))} "
            f"| {_n(l.get('final_loss'),4)} "
            f"| {saved} |"
        )
    lines.append("")

    # ── 1b. hardware (from first report) ──────────────────────────
    sys0 = reports[0].get("system", {})
    if sys0:
        lines.append("### 1.2  Hardware\n")
        cpu = sys0.get("cpu", {})
        lines.append(f"- **Host:** {sys0.get('hostname','—')}")
        lines.append(f"- **OS:** {sys0.get('os','—')}")
        lines.append(f"- **Python:** {sys0.get('python','—')}  ·  **PyTorch:** {sys0.get('torch','—')}  ·  **CUDA:** {sys0.get('cuda_version','—')}")
        lines.append(f"- **CPU:** {cpu.get('model','—')} ({cpu.get('physical_cores','?')} cores / {cpu.get('logical_cores','?')} threads)")
        lines.append(f"- **RAM:** {cpu.get('total_ram_gb','?')} GB")
        for g in sys0.get("gpus", []):
            lines.append(f"- **GPU {g['index']}:** {g['name']} — {_n(g['total_memory_mb'])} MB")
        lines.append("")

    # ── 1c. config details ─────────────────────────────────────────
    lines.append("### 1.3  Training Configurations\n")
    lines.append("| Run | Batch | Grad Accum | Eff. Batch | LR | Dtype | Grad Ckpt | Optimizer | Scheduler |")
    lines.append("|" + "|".join(["---"] * 9) + "|")
    for r in reports:
        c = r.get("config", {})
        lines.append(
            f"| {r.get('run_name','—')} "
            f"| {c.get('per_device_batch_size','—')} "
            f"| {c.get('gradient_accumulation_steps','—')} "
            f"| {c.get('effective_batch_size','—')} "
            f"| {c.get('learning_rate','—')} "
            f"| {c.get('dtype','—')} "
            f"| {c.get('gradient_checkpointing','—')} "
            f"| {c.get('optimizer','—')} "
            f"| {c.get('lr_scheduler','—')} |"
        )
    lines.append("")

    # ── 1d. dataset info ───────────────────────────────────────────
    lines.append("### 1.4  Dataset Info\n")
    lines.append(
        "| Run | Mode | Dataset Type | Source | Samples (train) | Samples (eval) "
        "| Tokens (train) | Seq Len (mean) | Seq Len (p95) | Seq Len (max) |"
    )
    lines.append("|" + "|".join(["---"] * 10) + "|")
    for r in reports:
        d = r.get("dataset", {})
        sl = d.get("sequence_length_stats", {})
        source = d.get("source_file") or d.get("source_datasets") or d.get("data_dir_name", "—")
        lines.append(
            f"| {r.get('run_name','—')} "
            f"| {r.get('mode','—')} "
            f"| {d.get('dataset_type','—')} "
            f"| {source} "
            f"| {_n(d.get('train_samples'))} "
            f"| {_n(d.get('eval_samples'))} "
            f"| {_n(d.get('train_tokens'))} "
            f"| {_n(sl.get('mean'))} "
            f"| {_n(sl.get('p95'))} "
            f"| {_n(sl.get('max'))} |"
        )
    lines.append("")

    # ── 1e. GPU utilization ────────────────────────────────────────
    for r in reports:
        gm = r.get("gpu_monitor", {})
        if not gm:
            continue
        lines.append(f"### 1.5  GPU Utilization — `{r.get('run_name','—')}`\n")
        lines.append("| GPU | Util % (avg) | Util % (max) | VRAM MB (avg) | VRAM MB (max) | Temp °C (avg) | Temp °C (max) | Power W (avg) | Power W (max) |")
        lines.append("|" + "|".join(["---"] * 9) + "|")
        for gid, s in gm.items():
            lines.append(
                f"| {gid} "
                f"| {s.get('utilization_pct_avg','—')} "
                f"| {s.get('utilization_pct_max','—')} "
                f"| {_n(s.get('memory_used_mb_avg'))} "
                f"| {_n(s.get('memory_used_mb_max'))} "
                f"| {s.get('temperature_c_avg','—')} "
                f"| {s.get('temperature_c_max','—')} "
                f"| {_n(s.get('power_w_avg'))} "
                f"| {_n(s.get('power_w_max'))} |"
            )
        lines.append("")

    # ── 1f. memory ─────────────────────────────────────────────────
    lines.append("### 1.6  Memory Usage\n")
    lines.append("| Run | Peak GPU VRAM (MB) | Per-GPU Peak | CPU before (MB) | CPU after (MB) | CPU Δ (MB) |")
    lines.append("|" + "|".join(["---"] * 6) + "|")
    for r in reports:
        m = r.get("memory", {})
        pgp = m.get("per_gpu_peak_mb", {})
        pgp_str = ", ".join(f"{k}={_n(v)}" for k, v in pgp.items()) if pgp else "—"
        lines.append(
            f"| {r.get('run_name','—')} "
            f"| {_n(m.get('peak_gpu_vram_mb'))} "
            f"| {pgp_str} "
            f"| {_n(m.get('cpu_ram_before_mb'))} "
            f"| {_n(m.get('cpu_ram_after_mb'))} "
            f"| {_n(m.get('cpu_ram_delta_mb'))} |"
        )
    lines.append("")

    # ── 1g. loss curves (image links instead of tables) ──────────
    for r in reports:
        curve = r.get("training_loss", {}).get("loss_curve", [])
        if not curve:
            continue
        run_name = r.get("run_name", "unknown")
        lines.append(f"### 1.7  Loss Curve — `{run_name}`\n")
        # Link to per-run plot
        img_rel = os.path.relpath(
            os.path.join(output_dir, "per_run", run_name, "loss_curve.png"),
            output_dir,
        )
        if os.path.exists(os.path.join(output_dir, "per_run", run_name, "loss_curve.png")):
            lines.append(f"![Loss Curve — {run_name}]({img_rel})\n")
        else:
            # Fallback: show representative rows in a table
            lines.append("| Step | Epoch | Loss | LR | GPU Alloc MB |")
            lines.append("|" + "|".join(["---"] * 5) + "|")
            step = max(1, len(curve) // 25)
            for i, e in enumerate(curve):
                if i % step == 0 or i == len(curve) - 1:
                    lines.append(
                        f"| {e['step']} "
                        f"| {e.get('epoch','—')} "
                        f"| {_n(e['loss'],5)} "
                        f"| {e['lr']:.2e} "
                        f"| {_n(e.get('gpu_alloc_mb'))} |"
                    )
        lines.append(f"Final loss: **{_n(r.get('training_loss',{}).get('final_loss'),4)}**\n")
        lines.append(f"📄 Full data: [`{run_name}.xlsx`](per_run/{run_name}/{run_name}.xlsx)\n")
        lines.append("")

    return "\n".join(lines) + "\n"


def _section_inference(reports: List[Dict], output_dir: str = REPORTS_DIR) -> str:
    if not reports:
        return "## 2 · Inference Profiling\n\n_No inference profiling data found._\n"

    lines = ["## 2 · Inference Profiling\n"]

    gpu_reps = [r for r in reports if r.get("device") != "cpu"]
    cpu_reps = [r for r in reports if r.get("device") == "cpu"]

    for label, reps in [("GPU", gpu_reps), ("CPU", cpu_reps)]:
        if not reps:
            continue

        lines.append(f"### 2.{1 if label=='GPU' else 2}  {label} Inference\n")

        # model load
        lines.append(f"#### Model Loading ({label})\n")
        lines.append("| Model | Load time | VRAM after load |")
        lines.append("|---|---|---|")
        for r in reps:
            ml = r.get("model_load", {})
            lines.append(
                f"| `{r.get('model_name','—')}` "
                f"| {ml.get('load_time_s','—')} s "
                f"| {_n(ml.get('gpu_vram_after_load_mb'))} MB |"
            )
        lines.append("")

        # per-prompt
        lines.append(f"#### Benchmark Results ({label})\n")
        lines.append("| Model | Prompt | Max Tok | Avg time | Tok/s | ms/tok | TTFT (s) | Peak GPU MB |")
        lines.append("|" + "|".join(["---"] * 8) + "|")
        for r in reps:
            mn = r.get("model_name", "—")
            for bp in r.get("benchmark_prompts", []):
                lines.append(
                    f"| `{mn}` "
                    f"| {bp['name']} "
                    f"| {bp['max_new_tokens']} "
                    f"| {_n(bp.get('avg_total_time_s'),2)} s "
                    f"| {_n(bp.get('avg_tokens_per_second'),1)} "
                    f"| {_n(bp.get('avg_ms_per_token'),1)} "
                    f"| {_n(bp.get('avg_ttft_s'),3)} "
                    f"| {_n(bp.get('peak_gpu_vram_mb'))} |"
                )
        lines.append("")

        # overall summary
        lines.append(f"#### Overall ({label})\n")
        lines.append("| Model | Avg Tok/s | Avg ms/tok | Avg TTFT | Peak GPU MB | Peak CPU MB |")
        lines.append("|" + "|".join(["---"] * 6) + "|")
        for r in reps:
            s = r.get("summary", {})
            lines.append(
                f"| `{r.get('model_name','—')}` "
                f"| {_n(s.get('avg_tokens_per_second'),1)} "
                f"| {_n(s.get('avg_ms_per_token'),1)} "
                f"| {_n(s.get('avg_ttft_s'),3)} "
                f"| {_n(s.get('peak_gpu_vram_mb'))} "
                f"| {_n(s.get('peak_cpu_ram_mb'))} |"
            )
        lines.append("")

        # output-length scaling
        has_ols = any(r.get("output_length_scaling") for r in reps)
        if has_ols:
            lines.append(f"#### Output-Length Scaling ({label})\n")
            lines.append("| Model | max_new_tokens | Actual new | Time (s) | Tok/s | ms/tok |")
            lines.append("|" + "|".join(["---"] * 6) + "|")
            for r in reps:
                for o in r.get("output_length_scaling", []):
                    lines.append(
                        f"| `{r.get('model_name','—')}` "
                        f"| {o.get('max_new_tokens','—')} "
                        f"| {o.get('new_tokens','—')} "
                        f"| {_n(o.get('total_time_s'),2)} "
                        f"| {_n(o.get('tokens_per_second'),1)} "
                        f"| {_n(o.get('ms_per_token'),1)} |"
                    )
            lines.append("")

        # input-length scaling
        has_ils = any(r.get("input_length_scaling") for r in reps)
        if has_ils:
            lines.append(f"#### Input-Length Scaling ({label})\n")
            lines.append("| Model | Input tokens | New tokens | Time (s) | Tok/s | ms/tok |")
            lines.append("|" + "|".join(["---"] * 6) + "|")
            for r in reps:
                for o in r.get("input_length_scaling", []):
                    lines.append(
                        f"| `{r.get('model_name','—')}` "
                        f"| {o.get('input_tokens','—')} "
                        f"| {o.get('new_tokens','—')} "
                        f"| {_n(o.get('total_time_s'),2)} "
                        f"| {_n(o.get('tokens_per_second'),1)} "
                        f"| {_n(o.get('ms_per_token'),1)} |"
                    )
            lines.append("")

    return "\n".join(lines) + "\n"


def _section_gpu_vs_cpu(reports: List[Dict]) -> str:
    gpu = {Path(r["model_path"]).name: r for r in reports if r.get("device") != "cpu"}
    cpu = {Path(r["model_path"]).name: r for r in reports if r.get("device") == "cpu"}
    common = sorted(set(gpu) & set(cpu))
    if not common:
        return ""

    lines = ["## 3 · GPU vs CPU Comparison\n"]
    lines.append("| Model | Metric | GPU | CPU | GPU Speedup |")
    lines.append("|---|---|---|---|---|")

    for mn in common:
        gs = gpu[mn].get("summary", {})
        cs = cpu[mn].get("summary", {})
        g_tps = gs.get("avg_tokens_per_second", 0)
        c_tps = cs.get("avg_tokens_per_second", 0)
        speedup = g_tps / c_tps if c_tps else 0
        lines.append(f"| `{mn}` | Tokens/s | {_n(g_tps,1)} | {_n(c_tps,1)} | **{speedup:.1f}×** |")
        lines.append(f"| | ms/token | {_n(gs.get('avg_ms_per_token'),1)} | {_n(cs.get('avg_ms_per_token'),1)} | |")
        lines.append(
            f"| | Peak memory | {_n(gs.get('peak_gpu_vram_mb'))} MB VRAM "
            f"| {_n(cs.get('peak_cpu_ram_mb'))} MB RAM | |"
        )

    lines.append("")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════
# per-run report generation (plots + Excel)
# ═══════════════════════════════════════════════════════════════════

def _generate_per_run_training(report: Dict, base_dir: str):
    """Generate per-run plots + Excel for one training run."""
    from profiling import report_plots
    from profiling import report_excel

    run_name = report.get("run_name", "unknown")
    run_dir = os.path.join(base_dir, "per_run", run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"  🔧  Training run: {run_name}")

    # Plots
    report_plots.plot_loss_curve(report, os.path.join(run_dir, "loss_curve.png"))
    report_plots.plot_gpu_utilization(report, os.path.join(run_dir, "gpu_utilization.png"))
    report_plots.plot_memory_usage(report, os.path.join(run_dir, "memory_usage.png"))

    # Excel
    report_excel.write_training_run_excel(report, os.path.join(run_dir, f"{run_name}.xlsx"))


def _generate_per_run_inference(report: Dict, base_dir: str):
    """Generate per-run plots + Excel for one inference run."""
    from profiling import report_plots
    from profiling import report_excel

    model = report.get("model_name", "unknown")
    device = report.get("device", "unknown")
    run_name = f"{model}_{device}"
    run_dir = os.path.join(base_dir, "per_run", run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"  🔧  Inference run: {run_name}")

    # Plots
    report_plots.plot_inference_per_prompt(report, os.path.join(run_dir, "inference_throughput.png"))
    report_plots.plot_inference_scaling(report, os.path.join(run_dir, "inference_scaling.png"))

    # Excel
    report_excel.write_inference_run_excel(report, os.path.join(run_dir, f"{run_name}.xlsx"))


def _generate_cumulative(train_reps: List[Dict], infer_reps: List[Dict],
                          base_dir: str):
    """Generate cumulative comparison plots + Excel."""
    from profiling import report_plots
    from profiling import report_excel

    cum_dir = os.path.join(base_dir, "cumulative")
    os.makedirs(cum_dir, exist_ok=True)

    logger.info("  📊  Cumulative comparison reports")

    # Training comparisons
    if train_reps:
        report_plots.plot_loss_comparison(train_reps, os.path.join(cum_dir, "loss_comparison.png"))
        report_plots.plot_training_throughput(train_reps, os.path.join(cum_dir, "training_throughput.png"))
        report_plots.plot_memory_comparison(train_reps, os.path.join(cum_dir, "memory_comparison.png"))

    # Inference comparisons
    if infer_reps:
        report_plots.plot_inference_comparison(infer_reps, os.path.join(cum_dir, "inference_comparison.png"))

    # GPU vs CPU
    gpu_reps = [r for r in infer_reps if r.get("device") != "cpu"]
    cpu_reps = [r for r in infer_reps if r.get("device") == "cpu"]
    if gpu_reps and cpu_reps:
        report_plots.plot_gpu_vs_cpu(gpu_reps, cpu_reps, os.path.join(cum_dir, "gpu_vs_cpu.png"))

    # Cumulative Excel
    report_excel.write_cumulative_excel(train_reps, infer_reps,
                                         os.path.join(cum_dir, "cumulative_report.xlsx"))


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Generate profiling reports (Markdown + Excel + plots)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The report includes ALL JSON files in outputs/profiling_results/.\n"
            "Each JSON records which model was profiled (model_path, run_name).\n"
            "Use --filter to narrow results to specific runs or models.\n"
            "Use --list  to see available JSON files without generating.\n\n"
            "Output structure:\n"
            "  outputs/profiling_reports/\n"
            "  ├── cumulative/          ← comparison Excel + comparison plots\n"
            "  ├── per_run/<name>/      ← per-run Excel + per-run plots\n"
            "  └── PROFILING_REPORT.md  ← lightweight Markdown with image links\n"
        ),
    )
    p.add_argument("--output_dir", default=REPORTS_DIR,
                   help=f"Output directory for all reports (default: {REPORTS_DIR})")
    p.add_argument("--results_dir", default=RESULTS_DIR,
                   help="Directory with profiling JSON files (default: outputs/profiling_results)")
    p.add_argument("--list", action="store_true", dest="list_files",
                   help="List available JSON files and exit (no report generated)")
    p.add_argument("--filter", type=str, default=None, metavar="SUBSTR",
                   help="Only include reports whose run_name, model_name, or "
                        "model_path contains this substring (case-insensitive)")
    p.add_argument("--cumulative-only", action="store_true",
                   help="Only generate cumulative comparison (skip per-run)")
    args = p.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir

    if not Path(results_dir).exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.error("Run profile_training.py / profile_inference.py first.")
        sys.exit(1)

    train_reps = _load("training_*.json", results_dir)
    infer_reps = _load("inference_*.json", results_dir)

    # Deduplicate inference reports
    seen = set()
    deduped_infer: List[Dict] = []
    for r in infer_reps:
        key = (r.get("model_path", ""), r.get("device", ""), r.get("timestamp", ""))
        if key not in seen:
            seen.add(key)
            deduped_infer.append(r)
    infer_reps = deduped_infer

    # ── --list mode ────────────────────────────────────────────────
    if args.list_files:
        print(f"\n📂  Profiling results in: {results_dir}/\n")
        print(f"  Training reports ({len(train_reps)}):")
        for r in train_reps:
            model_name = r.get("config", {}).get("model_name", "?")
            print(f"    • {r.get('run_name','?'):30s}  mode={r.get('mode','?'):10s}  "
                  f"model={model_name}")
        print(f"\n  Inference reports ({len(infer_reps)}):")
        for r in infer_reps:
            print(f"    • {r.get('model_name','?'):30s}  device={r.get('device','?'):5s}  "
                  f"path={r.get('model_path','?')}")
        print()
        return

    # ── --filter ───────────────────────────────────────────────────
    if args.filter:
        filt = args.filter.lower()

        def _match(r: Dict) -> bool:
            searchable = " ".join([
                r.get("run_name", ""),
                r.get("model_name", ""),
                r.get("model_path", ""),
                r.get("config", {}).get("model_name", ""),
                r.get("config", {}).get("model_path", ""),
                r.get("mode", ""),
            ]).lower()
            return filt in searchable

        before_t, before_i = len(train_reps), len(infer_reps)
        train_reps = [r for r in train_reps if _match(r)]
        infer_reps = [r for r in infer_reps if _match(r)]
        logger.info(f"Filter '{args.filter}': training {before_t}→{len(train_reps)}, "
                     f"inference {before_i}→{len(infer_reps)}")

    if not train_reps and not infer_reps:
        logger.error("No profiling JSON files found"
                      + (f" matching filter '{args.filter}'" if args.filter else "")
                      + ".")
        sys.exit(1)

    logger.info(f"Loaded {len(train_reps)} training + {len(infer_reps)} inference report(s)")

    # ── generate per-run reports ───────────────────────────────────
    if not args.cumulative_only:
        logger.info("Generating per-run reports…")
        for r in train_reps:
            _generate_per_run_training(r, output_dir)
        for r in infer_reps:
            _generate_per_run_inference(r, output_dir)

    # ── generate cumulative reports ────────────────────────────────
    logger.info("Generating cumulative reports…")
    _generate_cumulative(train_reps, infer_reps, output_dir)

    # ── generate lightweight Markdown ──────────────────────────────
    logger.info("Generating Markdown report…")
    md_path = os.path.join(output_dir, "PROFILING_REPORT.md")
    md = _build_markdown(train_reps, infer_reps, output_dir, args.filter)
    os.makedirs(output_dir, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(md)

    logger.info(f"✅  All reports saved to: {output_dir}/")
    logger.info(f"    Training runs : {len(train_reps)}")
    logger.info(f"    Inference runs: {len(infer_reps)}")
    logger.info(f"    Markdown      : {md_path}")
    logger.info(f"    Cumulative    : {output_dir}/cumulative/")
    if not args.cumulative_only:
        logger.info(f"    Per-run       : {output_dir}/per_run/")


def _build_markdown(train_reps: List[Dict], infer_reps: List[Dict],
                     output_dir: str, filter_str: str = None) -> str:
    """Build the combined Markdown report with embedded image links."""
    md = []
    md.append("# LLM Training & Inference — Profiling Report\n")
    md.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # ── inventory ──────────────────────────────────────────────────
    md.append("## Report Inventory\n")
    if filter_str:
        md.append(f"Filter: `{filter_str}`\n")
    if train_reps:
        md.append(f"**Training runs ({len(train_reps)}):**\n")
        for r in train_reps:
            model_name = r.get("config", {}).get("model_name", "?")
            ds_type = r.get("dataset", {}).get("dataset_type", "?")
            md.append(f"- `{r.get('run_name','?')}` — {r.get('mode','?')} "
                      f"— model: `{model_name}` — dataset: {ds_type}")
        md.append("")
    if infer_reps:
        md.append(f"**Inference runs ({len(infer_reps)}):**\n")
        for r in infer_reps:
            md.append(f"- `{r.get('model_name','?')}` — device: {r.get('device','?')}")
        md.append("")

    md.append("## Table of Contents\n")
    md.append("1. [Training Profiling](#1--training-profiling)")
    md.append("2. [Inference Profiling](#2--inference-profiling)")
    md.append("3. [GPU vs CPU Comparison](#3--gpu-vs-cpu-comparison)")
    md.append("4. [Cumulative Comparison](#4--cumulative-comparison)\n")
    md.append("---\n")

    # ── training section ───────────────────────────────────────────
    md.append(_section_training(train_reps, output_dir))
    md.append("---\n")

    # ── inference section ──────────────────────────────────────────
    md.append(_section_inference(infer_reps, output_dir))

    # ── GPU vs CPU ─────────────────────────────────────────────────
    gvc = _section_gpu_vs_cpu(infer_reps)
    if gvc:
        md.append("---\n")
        md.append(gvc)

    # ── cumulative comparison images ───────────────────────────────
    md.append("---\n")
    md.append("## 4 · Cumulative Comparison\n")
    md.append("### Comparison plots\n")
    cum_dir = os.path.join(output_dir, "cumulative")
    for png_name, caption in [
        ("loss_comparison.png", "Training Loss Comparison"),
        ("training_throughput.png", "Training Throughput Comparison"),
        ("memory_comparison.png", "Peak GPU Memory Comparison"),
        ("inference_comparison.png", "Inference Throughput Comparison"),
        ("gpu_vs_cpu.png", "GPU vs CPU Inference"),
    ]:
        png_path = os.path.join(cum_dir, png_name)
        if os.path.exists(png_path):
            # Use relative path from the Markdown file location
            rel = os.path.relpath(png_path, output_dir)
            md.append(f"#### {caption}\n")
            md.append(f"![{caption}]({rel})\n")

    md.append("\n### 📁 Full Data\n")
    md.append(f"- **Cumulative Excel:** `{os.path.join('cumulative', 'cumulative_report.xlsx')}`")
    md.append(f"- **Per-run reports:** `per_run/<run_name>/`  (each has `.xlsx` + `.png` files)\n")

    return "\n".join(md)


if __name__ == "__main__":
    main()
