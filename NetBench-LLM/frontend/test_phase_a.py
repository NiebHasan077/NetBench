#!/usr/bin/env python3
"""
Phase A validation test.

Tests model_manager and job_runner without any Gradio UI.
Run from project root:
    python frontend/test_phase_a.py
"""

import sys
import time
from pathlib import Path

# Make sure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.core.model_manager import manager, ModelType
from frontend.core.job_runner import JobRunner, JobStatus


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: list_available_models
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1 — list_available_models()")
print("=" * 60)

models = manager.list_available_models()
if not models:
    print("  [WARN] No models found under models/. Skipping load test.")
else:
    for m in models:
        print(f"  {m.display_name}")
        print(f"    path     : {m.path}")
        print(f"    type     : {m.model_type.value}  |  is_instruct: {m.model_type.is_instruct}")
        print(f"    category : {m.category}")
        print()

print(f"  Total models found: {len(models)}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: VRAM usage (no model loaded)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 2 — get_vram_usage() (no model loaded)")
print("=" * 60)

vram = manager.get_vram_usage()
if not vram:
    print("  [WARN] No CUDA GPUs found — VRAM stats unavailable.")
else:
    for v in vram:
        d = v.to_dict()
        for k, val in d.items():
            print(f"  {k:<16}: {val}")
        print()
print()


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Load the smallest available model
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 3 — load_model() + VRAM delta")
print("=" * 60)

# Pick the first 1B model of type BASE (smallest footprint)
target = None
for m in models:
    if "1b" in m.nickname.lower() and m.model_type == ModelType.BASE:
        target = m
        break
if target is None and models:
    target = models[0]  # Fall back to any model

if target is None:
    print("  [SKIP] No models available to load.")
else:
    print(f"  Loading: {target.display_name}")
    try:
        loaded = manager.load_model(target.nickname)
        vram_after = manager.get_vram_usage()
        print(f"  ✅ Loaded successfully")
        print(f"  VRAM consumed on load: {loaded.vram_bytes_on_load / 1024**3:.2f} GB")
        for v in vram_after:
            print(f"  GPU {v.gpu_index}: {v.used_gb:.2f} / {v.total_gb:.2f} GB used")
        print()

        # Test double-load guard
        print("  Testing double-load guard…")
        try:
            manager.load_model(target.nickname)
            print("  [WARN] Double-load should have returned the cached object (OK)")
        except Exception as e:
            print(f"  [FAIL] Unexpected error on double-load: {e}")

        # Unload
        print(f"  Unloading {target.nickname}…")
        manager.unload_model(target.nickname)
        vram_freed = manager.get_vram_usage()
        for v in vram_freed:
            print(f"  GPU {v.gpu_index}: {v.used_gb:.2f} / {v.total_gb:.2f} GB after unload")
        print("  ✅ Unloaded successfully")

    except Exception as e:
        print(f"  [ERROR] {e}")

print()


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Over-limit guard
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 4 — over-limit guard (max_loaded=1)")
print("=" * 60)

if len(models) >= 2:
    first  = models[0].nickname
    second = models[1].nickname
    try:
        manager.load_model(first, max_loaded=1)
        print(f"  Loaded {first}")
        try:
            manager.load_model(second, max_loaded=1)
            print(f"  [FAIL] Should have raised RuntimeError for max_loaded=1")
        except RuntimeError as e:
            print(f"  ✅ Correctly blocked: {str(e)[:80]}…")
        finally:
            manager.unload_all()
    except Exception as e:
        print(f"  [ERROR] {e}")
else:
    print("  [SKIP] Need at least 2 models to test limit guard.")

print()


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: JobRunner — fast dummy job
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 5 — JobRunner (fast dummy job)")
print("=" * 60)

def dummy_job(steps: int = 5):
    for i in range(steps):
        print(f"  Step {i + 1}/{steps}")
        time.sleep(0.3)
    print("  Dummy job finished.")
    return {"status": "ok", "steps": steps}

runner = JobRunner(name="test")
runner.submit(dummy_job, steps=5)

collected = []
while runner.is_running():
    lines = runner.read_logs()
    collected.extend(lines)
    time.sleep(0.2)

# Drain any remaining lines
collected.extend(runner.read_logs())

print("  Captured log:")
for line in collected:
    print(f"    {line}", end="")

result = runner.get_result()
print(f"\n  Status     : {runner.get_status().value}")
print(f"  Return val : {result.return_value if result else 'N/A'}")
print(f"  Error      : {result.error_message if result and result.error_message else 'none'}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: JobRunner — cancel
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 6 — JobRunner (cancel)")
print("=" * 60)

def cancellable_job(steps: int = 20, _cancel_event=None):
    for i in range(steps):
        if _cancel_event and _cancel_event.is_set():
            print("  [cancellable_job] Received cancel signal, exiting.")
            return
        print(f"  Step {i + 1}/{steps}")
        time.sleep(0.2)

runner2 = JobRunner(name="cancellable")
runner2.submit(cancellable_job, steps=20)
time.sleep(0.5)   # Let it run a couple of steps
runner2.cancel()
runner2._thread.join(timeout=3.0)   # Wait for the thread to exit

all_lines = runner2.read_logs()
print("  Captured log:")
for line in all_lines:
    print(f"    {line}", end="")

print(f"\n  Final status: {runner2.get_status().value}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: JobRunner — error propagation
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 7 — JobRunner (error propagation)")
print("=" * 60)

def failing_job():
    print("  About to raise…")
    raise ValueError("Intentional test error")

runner3 = JobRunner(name="failing")
runner3.submit(failing_job)
runner3._thread.join(timeout=5.0)

lines = runner3.read_logs()
print("  Captured log:")
for line in lines:
    print(f"    {line}", end="")

result3 = runner3.get_result()
print(f"\n  Status : {runner3.get_status().value}")
print(f"  Error  : {result3.error_message if result3 else 'N/A'}")
print()


print("=" * 60)
print("Phase A validation complete.")
print("=" * 60)
