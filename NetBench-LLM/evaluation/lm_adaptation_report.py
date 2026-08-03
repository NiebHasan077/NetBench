#!/usr/bin/env python3
"""
Language Modeling Adaptation Report

Measures three metrics to assess how well continual pre-training (CPT)
adapted a model to the HPN research corpus while preserving general ability:

  1. HPN validation loss / perplexity   — cross-entropy on the processed
       research-corpus validation split (domain adaptation quality)
  2. General LM loss / perplexity       — cross-entropy on WikiText-2 test
       set (general language-modeling capability)
  3. Forgetting gap = GeneralLossAfter − GeneralLossBefore
       Positive  → CPT degraded general capability ("catastrophic forgetting")
       Negative  → CPT also improved general capability (rare but possible)
       Zero      → General capability was perfectly preserved

Usage
-----
# Full evaluation (all samples):
python evaluation/lm_adaptation_report.py \\
    --base_model_path    models/base/Qwen3.5-2B-Base \\
    --trained_model_path models/pretrained/Qwen3.5-2B-trained-new \\
    --hpn_data_dir       data/processed/qwen3.5-2b

# Fast debug run (limit to 20 batches each):
python evaluation/lm_adaptation_report.py \\
    --base_model_path    models/base/Qwen3.5-2B-Base \\
    --trained_model_path models/pretrained/Qwen3.5-2B-trained-new \\
    --hpn_data_dir       data/processed/qwen3.5-2b \\
    --max_hpn_batches 20 --max_general_batches 20

# Low-VRAM machine (4-bit quantization):
python evaluation/lm_adaptation_report.py \\
    --base_model_path    models/base/Qwen3.5-2B-Base \\
    --trained_model_path models/pretrained/Qwen3.5-2B-trained-new \\
    --quantize_4bit
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import detect_model_family, get_gemma_token_type_key
from utils.setup_token import setup_hf_auth


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(
    model_path: str,
    quantize_4bit: bool,
) -> tuple:
    """Load model and tokenizer. Returns (model, tokenizer, model_family)."""
    print(f"  Loading tokenizer from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_family = detect_model_family(model_path)
    print(f"  Model family: {model_family}")
    print(f"  Loading model from {model_path} …")

    load_kwargs: dict = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    print(f"  Model loaded on device(s): {set(str(p.device) for p in model.parameters())}")
    return model, tokenizer, model_family


def _unload_model(model) -> None:
    """Free GPU memory occupied by the model."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── loss computation ──────────────────────────────────────────────────────────

def _compute_loss_on_dataset(
    model,
    dataset: Dataset,
    batch_size: int,
    max_batches: Optional[int],
    model_family: str,
    label: str = "dataset",
) -> dict:
    """
    Compute the mean cross-entropy loss (and perplexity) on an HF Dataset.

    The dataset must have 'input_ids', 'attention_mask', and 'labels' columns.
    Positions with label == -100 are excluded from the mean (standard CLM masking).

    Parameters
    ----------
    model        : loaded HuggingFace causal-LM model (eval mode)
    dataset      : HF Dataset with input_ids / attention_mask / labels
    batch_size   : number of sequences per forward pass
    max_batches  : if set, stop after this many batches
    model_family : 'qwen' | 'gemma' | 'llama' — governs token_type_ids injection
    label        : human-readable name for progress printout

    Returns
    -------
    dict with keys: loss, perplexity, batches, tokens
    """
    needs_tti = (model_family == "gemma")
    if needs_tti:
        tt_key = get_gemma_token_type_key(model)
        print(f"    Gemma model — injecting {tt_key} (all zeros)")

    # Determine the first device for tensor placement when device_map="auto"
    device = next(model.parameters()).device

    n_samples = len(dataset)
    if max_batches is not None:
        n_samples = min(n_samples, max_batches * batch_size)

    total_ce = 0.0
    total_tokens = 0
    n_batches = 0

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        raw = dataset[batch_start:batch_end]

        input_ids = torch.tensor(raw["input_ids"], dtype=torch.long).to(device)
        attention_mask = torch.tensor(raw["attention_mask"], dtype=torch.long).to(device)
        labels = torch.tensor(raw["labels"], dtype=torch.long).to(device)

        model_inputs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if needs_tti:
            model_inputs[tt_key] = torch.zeros_like(input_ids)

        with torch.no_grad():
            out = model(**model_inputs)

        # out.loss = mean CE over non-(-100) tokens in this batch (already normalised).
        # Weight by active-token count for a globally-consistent mean.
        active = (labels != -100).sum().item()
        if active == 0:
            continue  # skip all-padding batches

        total_ce += out.loss.item() * active
        total_tokens += active
        n_batches += 1

        if n_batches % 10 == 0:
            running_ppl = (
                math.exp(total_ce / total_tokens) if total_tokens > 0 else float("nan")
            )
            print(
                f"    [{label}] batch {n_batches:>4} | "
                f"running ppl = {running_ppl:8.3f}",
                end="\r",
            )

    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "batches": 0, "tokens": 0}

    mean_loss = total_ce / total_tokens
    ppl = math.exp(mean_loss)
    print(
        f"    [{label}] {n_batches} batches | {total_tokens:,} tokens | "
        f"loss = {mean_loss:.5f} | ppl = {ppl:.3f}          "
    )
    return {
        "loss": round(mean_loss, 6),
        "perplexity": round(ppl, 4),
        "batches": n_batches,
        "tokens": total_tokens,
    }


# ── WikiText-2 general corpus ─────────────────────────────────────────────────

def _build_wikitext2_dataset(tokenizer, seq_len: int) -> Dataset:
    """
    Download WikiText-2 test split, tokenise it with `tokenizer`, and pack
    the token stream into non-overlapping `seq_len`-token blocks.

    Mirrors the CPT token-packing approach so the evaluation setup is
    consistent with how the model was trained.
    """
    from datasets import load_dataset

    print("  Downloading WikiText-2 test split …")
    wt2 = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        trust_remote_code=False,
    )

    # Concatenate all non-empty lines with double-newline paragraph breaks
    all_text = "\n\n".join(t for t in wt2["text"] if t.strip())

    print("  Tokenising WikiText-2 …")
    token_ids = tokenizer(
        all_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    n_blocks = len(token_ids) // seq_len
    if n_blocks == 0:
        raise ValueError(
            f"WikiText-2 produced {len(token_ids)} tokens — too short for "
            f"seq_len={seq_len}. Use a smaller --seq_len."
        )

    blocks_input = []
    for i in range(n_blocks):
        chunk = token_ids[i * seq_len : (i + 1) * seq_len]
        blocks_input.append(chunk)

    ds = Dataset.from_dict({
        "input_ids":      blocks_input,
        "attention_mask": [[1] * seq_len] * n_blocks,
        "labels":         blocks_input,   # CLM: predict every token
    })
    print(f"  WikiText-2 dataset: {len(ds)} blocks × {seq_len} tokens")
    return ds


# ── HPN data-dir auto-detection ───────────────────────────────────────────────

def _auto_hpn_data_dir(trained_model_path: str) -> Optional[str]:
    """
    Heuristically find a matching directory under data/processed/ based on
    the trained model's directory name.

    Example: 'models/pretrained/Qwen3.5-2B-trained-new'
             → tries to match 'data/processed/qwen3.5-2b'
    """
    model_name = Path(trained_model_path).name.lower()
    data_processed = Path("data/processed")
    if not data_processed.exists():
        return None

    best: Optional[str] = None
    best_score = 0

    for d in sorted(data_processed.iterdir()):
        if not (d.is_dir() and (d / "validation").exists()):
            continue
        dn = d.name.lower()
        # Score by how many characters of dn appear at the start of model_name
        score = sum(
            1 for a, b in zip(model_name, dn) if a == b
        )
        if score > best_score:
            best_score = score
            best = str(d)

    if best:
        print(f"  Auto-detected HPN data dir: {best}")
    return best


# ── report helpers ────────────────────────────────────────────────────────────

def _print_report(results: dict, base_name: str, trained_name: str) -> None:
    b = results["base"]
    t = results["trained"]
    gap = results["forgetting_gap"]

    print()
    print("=" * 74)
    print("  LANGUAGE MODELING ADAPTATION REPORT")
    print("=" * 74)
    print(f"  Base model    : {base_name}")
    print(f"  Trained model : {trained_name}")
    print("-" * 74)
    print(f"  {'Metric':<38} {'Before CPT':>12} {'After CPT':>12}")
    print("-" * 74)

    def fmt_val(v):
        return f"{v:12.5f}" if isinstance(v, float) and not math.isnan(v) else f"{'N/A':>12}"

    def fmt_ppl(v):
        return f"{v:12.3f}" if isinstance(v, float) and not math.isnan(v) else f"{'N/A':>12}"

    print(f"  {'HPN Domain Loss':<38} {fmt_val(b['hpn']['loss'])} {fmt_val(t['hpn']['loss'])}")
    print(f"  {'HPN Domain Perplexity':<38} {fmt_ppl(b['hpn']['perplexity'])} {fmt_ppl(t['hpn']['perplexity'])}")
    print(f"  {'General LM Loss (WikiText-2)':<38} {fmt_val(b['general']['loss'])} {fmt_val(t['general']['loss'])}")
    print(f"  {'General LM Perplexity (WikiText-2)':<38} {fmt_ppl(b['general']['perplexity'])} {fmt_ppl(t['general']['perplexity'])}")
    print("-" * 74)

    if isinstance(gap, float) and not math.isnan(gap):
        direction = "degraded" if gap > 0.001 else ("improved" if gap < -0.001 else "unchanged")
        print(f"  {'Forgetting Gap  (General: After − Before)':<38} {gap:+12.5f}  [{direction}]")
    else:
        print(f"  {'Forgetting Gap  (General: After − Before)':<38} {'N/A':>12}")

    print("=" * 74)
    print()


def _save_excel(results: dict, base_name: str, trained_name: str, output_dir: str, ts: str) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Adaptation Report"

    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True)
    headers = ["Metric", "Before CPT", "After CPT", "Delta (After − Before)"]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=h)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center")

    b = results["base"]
    t = results["trained"]

    def _delta(after, before):
        try:
            return round(after - before, 6)
        except Exception:
            return None

    rows = [
        ("HPN Domain Loss",
         b["hpn"]["loss"],       t["hpn"]["loss"],
         _delta(t["hpn"]["loss"],       b["hpn"]["loss"])),
        ("HPN Domain Perplexity",
         b["hpn"]["perplexity"], t["hpn"]["perplexity"],
         _delta(t["hpn"]["perplexity"], b["hpn"]["perplexity"])),
        ("General LM Loss (WikiText-2)",
         b["general"]["loss"],   t["general"]["loss"],
         _delta(t["general"]["loss"],   b["general"]["loss"])),
        ("General LM Perplexity (WikiText-2)",
         b["general"]["perplexity"], t["general"]["perplexity"],
         _delta(t["general"]["perplexity"], b["general"]["perplexity"])),
        ("Forgetting Gap  (General: After − Before)",
         None, None, results["forgetting_gap"]),
    ]

    for row_i, (metric, before, after, delta) in enumerate(rows, 2):
        ws.cell(row=row_i, column=1, value=metric)
        if before is not None:
            ws.cell(row=row_i, column=2, value=before)
        if after is not None:
            ws.cell(row=row_i, column=3, value=after)
        if delta is not None:
            c = ws.cell(row=row_i, column=4, value=delta)
            # Colour delta cells: red = worse, green = better
            # For loss/perplexity/gap: positive delta is bad
            if isinstance(delta, float):
                if delta > 0:
                    c.font = Font(color="CC0000", bold=True)
                elif delta < 0:
                    c.font = Font(color="007700", bold=True)

    # Metadata block
    meta_row = len(rows) + 3
    for label, value in [
        ("Base model",    base_name),
        ("Trained model", trained_name),
        ("Generated at",  results["timestamp"]),
    ]:
        ws.cell(row=meta_row, column=1, value=label).font = Font(bold=True)
        ws.cell(row=meta_row, column=2, value=value)
        meta_row += 1

    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 18
    ws.column_dimensions["D"].width = 24

    trained_name_safe = trained_name.replace("/", "_").replace("\\", "_")
    xlsx_path = Path(output_dir) / f"adaptation_report_{trained_name_safe}_{ts}.xlsx"
    wb.save(str(xlsx_path))
    print(f"Excel report saved  : {xlsx_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Language Modeling Adaptation Report — measure CPT quality and forgetting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--base_model_path", required=True,
        help="Path to the base model (before CPT). E.g. models/base/Qwen3.5-2B-Base",
    )
    p.add_argument(
        "--trained_model_path", required=True,
        help="Path to the trained model (after CPT). E.g. models/pretrained/Qwen3.5-2B-trained-new",
    )
    p.add_argument(
        "--hpn_data_dir", default=None,
        help=(
            "Path to the processed HPN dataset directory (contains train/ and validation/ "
            "subdirs). If omitted, auto-detected from data/processed/ based on trained model name."
        ),
    )
    p.add_argument(
        "--output_dir", default="outputs/evaluations/adaptation_reports",
        help="Directory to write JSON (and optionally Excel) report.",
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Sequences per forward pass. Reduce if you get OOM errors. Default: 4",
    )
    p.add_argument(
        "--seq_len", type=int, default=2048,
        help="Token block length for WikiText-2 packing. Should match CPT seq_len. Default: 2048",
    )
    p.add_argument(
        "--max_hpn_batches", type=int, default=None,
        help="Stop HPN evaluation after N batches (None = all). Useful for quick smoke tests.",
    )
    p.add_argument(
        "--max_general_batches", type=int, default=None,
        help="Stop WikiText-2 evaluation after N batches (None = all).",
    )
    p.add_argument(
        "--quantize_4bit", action="store_true",
        help="Load both models in 4-bit (requires bitsandbytes). Reduces VRAM by ~4×.",
    )
    p.add_argument(
        "--hf_token_file", default="hf_token.txt",
        help="HuggingFace auth token file path. Default: hf_token.txt",
    )
    p.add_argument(
        "--no_excel", action="store_true",
        help="Skip Excel output (JSON only).",
    )
    return p.parse_args()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # HF auth (needed to download WikiText-2 on first run)
    try:
        setup_hf_auth(args.hf_token_file)
    except Exception:
        pass  # Offline / no token required for public WikiText-2

    # Resolve HPN data directory
    hpn_data_dir = args.hpn_data_dir
    if hpn_data_dir is None:
        hpn_data_dir = _auto_hpn_data_dir(args.trained_model_path)
    if hpn_data_dir is None:
        print(
            "ERROR: Could not auto-detect --hpn_data_dir. "
            "Please supply it explicitly with --hpn_data_dir <path>."
        )
        sys.exit(1)

    hpn_val_dir = Path(hpn_data_dir) / "validation"
    if not hpn_val_dir.exists():
        print(f"ERROR: HPN validation split not found at {hpn_val_dir}")
        sys.exit(1)

    # Print run config
    print()
    print("Language Modeling Adaptation Report")
    print("=" * 60)
    print(f"  Base model          : {args.base_model_path}")
    print(f"  Trained model       : {args.trained_model_path}")
    print(f"  HPN validation dir  : {hpn_val_dir}")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  Sequence length     : {args.seq_len}")
    if args.max_hpn_batches:
        print(f"  Max HPN batches     : {args.max_hpn_batches}")
    if args.max_general_batches:
        print(f"  Max general batches : {args.max_general_batches}")
    if args.quantize_4bit:
        print("  Quantization        : 4-bit (bitsandbytes)")
    print()

    # Load HPN validation set once (shared between both model evaluations)
    print("Loading HPN validation dataset …")
    hpn_val_ds = load_from_disk(str(hpn_val_dir))
    print(f"  {len(hpn_val_ds)} samples × {len(hpn_val_ds[0]['input_ids'])} tokens")

    results: dict = {
        "timestamp":           datetime.now().isoformat(),
        "base_model_path":     str(args.base_model_path),
        "trained_model_path":  str(args.trained_model_path),
        "hpn_data_dir":        str(hpn_data_dir),
        "config": {
            "batch_size":           args.batch_size,
            "seq_len":              args.seq_len,
            "max_hpn_batches":      args.max_hpn_batches,
            "max_general_batches":  args.max_general_batches,
            "quantize_4bit":        args.quantize_4bit,
        },
        "base":          {},
        "trained":       {},
        "forgetting_gap": None,
    }

    # ── PHASE 1 — Base model ─────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("PHASE 1 — Base model")
    print(sep)

    model, tokenizer, model_family = _load_model_and_tokenizer(
        args.base_model_path, args.quantize_4bit
    )

    # WikiText-2: tokenise with the base model's tokenizer
    print("\n[1/2] Building WikiText-2 general evaluation set …")
    general_ds_base = _build_wikitext2_dataset(tokenizer, args.seq_len)

    print("\n[2/2] Evaluating base model …")
    print("  — HPN domain —")
    base_hpn = _compute_loss_on_dataset(
        model, hpn_val_ds, args.batch_size,
        args.max_hpn_batches, model_family, "HPN-base"
    )
    print("  — General (WikiText-2) —")
    base_general = _compute_loss_on_dataset(
        model, general_ds_base, args.batch_size,
        args.max_general_batches, model_family, "WT2-base"
    )

    results["base"] = {"hpn": base_hpn, "general": base_general}
    _unload_model(model)
    del general_ds_base

    # ── PHASE 2 — Trained model ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("PHASE 2 — Trained model")
    print(sep)

    model, tokenizer, model_family = _load_model_and_tokenizer(
        args.trained_model_path, args.quantize_4bit
    )

    # Re-tokenise WikiText-2 with the trained model's tokenizer (usually same)
    print("\n[1/2] Building WikiText-2 general evaluation set …")
    general_ds_trained = _build_wikitext2_dataset(tokenizer, args.seq_len)

    print("\n[2/2] Evaluating trained model …")
    print("  — HPN domain —")
    trained_hpn = _compute_loss_on_dataset(
        model, hpn_val_ds, args.batch_size,
        args.max_hpn_batches, model_family, "HPN-trained"
    )
    print("  — General (WikiText-2) —")
    trained_general = _compute_loss_on_dataset(
        model, general_ds_trained, args.batch_size,
        args.max_general_batches, model_family, "WT2-trained"
    )

    results["trained"] = {"hpn": trained_hpn, "general": trained_general}
    _unload_model(model)
    del general_ds_trained

    # ── Forgetting gap ────────────────────────────────────────────────────────
    try:
        forgetting_gap = round(
            results["trained"]["general"]["loss"] - results["base"]["general"]["loss"],
            6,
        )
    except (TypeError, KeyError):
        forgetting_gap = float("nan")
    results["forgetting_gap"] = forgetting_gap

    # ── Print report ──────────────────────────────────────────────────────────
    base_name    = Path(args.base_model_path).name
    trained_name = Path(args.trained_model_path).name
    _print_report(results, base_name, trained_name)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trained_name_safe = trained_name.replace("/", "_").replace("\\", "_")
    json_path = Path(args.output_dir) / f"adaptation_report_{trained_name_safe}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON report saved   : {json_path}")

    # ── Save Excel (optional) ─────────────────────────────────────────────────
    if not args.no_excel:
        try:
            _save_excel(results, base_name, trained_name, args.output_dir, ts)
        except ImportError:
            print("openpyxl not available — skipping Excel output (use --no_excel to suppress).")
        except Exception as e:
            print(f"Excel generation failed: {e}")

    print()
    return results


if __name__ == "__main__":
    main()
