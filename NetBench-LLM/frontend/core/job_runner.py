"""
frontend/core/job_runner.py
───────────────────────────
Runs long-running Python callables (benchmark phases, profiling) in a
background thread and makes their output available line-by-line through
a queue.  Designed for use with Gradio's gr.Timer polling pattern.

Architecture
────────────
  - The callable is executed inside a daemon Thread.
  - sys.stdout and sys.stderr inside the thread are redirected to a
    _QueueWriter that puts each line into a collections.deque (capped at
    MAX_LOG_LINES) AND a queue.Queue that Gradio polls.
  - A threading.Event (cancel_event) is passed as a keyword argument
    `_cancel_event` to the callable when it accepts it, so well-behaved
    long jobs can check and exit early.
  - One JobRunner instance should be created per logical job slot
    (e.g. one for benchmark, one for profiling).

State machine
────────────
  IDLE → RUNNING → DONE
                 → ERROR
  DONE/ERROR → IDLE  (after reset())

Public API
────────────
  submit(fn, *args, **kwargs) → None   start the job
  read_logs() → list[str]              drain queued lines (call from gr.Timer)
  get_status() → JobStatus
  get_full_log() → str                 all lines since last reset
  cancel() → None                      request cancellation
  reset() → None                       clear state; allows re-use

Usage example (Gradio)
────────────────────────
    runner = JobRunner()

    def on_run_click():
        runner.submit(run_benchmark, model_path="...", output_dir="...")

    def poll_logs(current_log):
        new_lines = runner.read_logs()
        status    = runner.get_status()
        return current_log + "".join(new_lines), status.value

    with gr.Blocks() as demo:
        log_box = gr.Textbox(lines=12)
        status_txt = gr.Textbox()
        gr.Timer(1.0).tick(poll_logs, inputs=log_box, outputs=[log_box, status_txt])
"""

from __future__ import annotations

import inspect
import io
import queue
import sys
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Job state enum
# ──────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    IDLE    = "IDLE"
    RUNNING = "RUNNING"
    DONE    = "DONE"
    ERROR   = "ERROR"


# ──────────────────────────────────────────────────────────────────────────────
# Internal: thread-local stdout/stderr redirector
# ──────────────────────────────────────────────────────────────────────────────

MAX_LOG_LINES = 2_000   # Maximum lines kept in the rolling deque

# ──────────────────────────────────────────────────────────────────────────────
# Thread-local stdout/stderr proxy
# ──────────────────────────────────────────────────────────────────────────────

class _ThreadLocalProxy(io.TextIOBase):
    """
    Installed once as sys.stdout/sys.stderr at module import time.

    Routes writes to a per-thread override (_QueueWriter) when one is set,
    otherwise passes through to the original stream.

    This ensures that only the JobRunner worker thread's prints are captured
    into the log queue; the main Gradio thread continues writing to the real
    terminal unchanged.
    """

    def __init__(self, original: io.TextIOBase) -> None:
        self._original = original
        self._local    = threading.local()

    def _writer(self):
        return getattr(self._local, "override", None)

    def write(self, text: str) -> int:
        w = self._writer()
        if w is not None:
            return w.write(text)
        return self._original.write(text)

    def flush(self) -> None:
        w = self._writer()
        if w is not None:
            w.flush()
        else:
            self._original.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return getattr(self._original, "isatty", lambda: False)()

    def set_override(self, writer) -> None:
        """Called by the worker thread before a job starts."""
        self._local.override = writer

    def clear_override(self) -> None:
        """Called by the worker thread after a job finishes."""
        self._local.override = None


# Install the proxies once at module load time.
# Subsequent assignments to sys.stdout/sys.stderr are thread-safe.
_stdout_proxy = _ThreadLocalProxy(sys.stdout)
_stderr_proxy = _ThreadLocalProxy(sys.stderr)
sys.stdout = _stdout_proxy  # type: ignore[assignment]
sys.stderr = _stderr_proxy  # type: ignore[assignment]


class _QueueWriter(io.TextIOBase):
    """
    A write()-compatible object that:
      1. Puts each line into a queue.Queue (for Gradio polling).
      2. Appends to a deque (for get_full_log()).
      3. Also writes through to the real stdout (so VS Code terminal shows output).
    """

    def __init__(
        self,
        line_queue: queue.Queue,
        log_deque: Deque[str],
        real_stream: io.TextIOBase,
    ) -> None:
        self._queue = line_queue
        self._deque = log_deque
        self._real  = real_stream
        self._buf   = ""

    def write(self, text: str) -> int:
        self._buf += text
        # Flush complete lines
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            full_line = line + "\n"
            self._queue.put(full_line)
            self._deque.append(full_line)
            try:
                self._real.write(full_line)
                self._real.flush()
            except Exception:
                pass
        return len(text)

    def flush(self) -> None:
        # Flush any remaining incomplete line
        if self._buf:
            self._queue.put(self._buf)
            self._deque.append(self._buf)
            try:
                self._real.write(self._buf)
                self._real.flush()
            except Exception:
                pass
            self._buf = ""


# ──────────────────────────────────────────────────────────────────────────────
# JobRunner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JobResult:
    status: JobStatus
    return_value: Any = None
    error_message: str = ""
    traceback_str: str = ""


class JobRunner:
    """
    Run one long-running callable at a time in a background thread,
    capturing its stdout/stderr for live display in Gradio.

    Parameters
    ----------
    name : optional label for this runner (used in log headers)
    """

    def __init__(self, name: str = "job") -> None:
        self.name = name
        self._status       = JobStatus.IDLE
        self._thread: Optional[threading.Thread]   = None
        self._cancel_event = threading.Event()
        self._line_queue: queue.Queue[str]          = queue.Queue()
        self._log_deque: Deque[str]                 = deque(maxlen=MAX_LOG_LINES)
        self._result: Optional[JobResult]           = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Start `fn(*args, **kwargs)` in a background thread.

        If the callable's signature accepts a `_cancel_event` parameter, the
        runner's threading.Event is injected so the job can poll for
        cancellation.

        Raises
        ------
        RuntimeError : Another job is already RUNNING.
        """
        with self._lock:
            if self._status == JobStatus.RUNNING:
                raise RuntimeError(
                    f"JobRunner '{self.name}' is already running. "
                    f"Wait for it to finish or call cancel() first."
                )
            self._cancel_event.clear()
            self._status  = JobStatus.RUNNING
            self._result  = None
            # Clear the queue and deque for this new run
            while not self._line_queue.empty():
                try:
                    self._line_queue.get_nowait()
                except queue.Empty:
                    break
            self._log_deque.clear()

        # Inject cancel event if the callable accepts it
        try:
            sig = inspect.signature(fn)
            if "_cancel_event" in sig.parameters:
                kwargs["_cancel_event"] = self._cancel_event
        except (ValueError, TypeError):
            pass

        self._thread = threading.Thread(
            target=self._run,
            args=(fn, args, kwargs),
            daemon=True,
            name=f"JobRunner-{self.name}",
        )
        self._thread.start()

    def read_logs(self) -> list[str]:
        """
        Drain all currently queued log lines.

        Returns a list of strings (each ending with \\n).
        Call this from a gr.Timer callback to update a Textbox.
        """
        lines: list[str] = []
        while True:
            try:
                lines.append(self._line_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def get_status(self) -> JobStatus:
        """Return the current job status (thread-safe)."""
        with self._lock:
            return self._status

    def get_full_log(self) -> str:
        """Return the entire captured log as a single string."""
        return "".join(self._log_deque)

    def get_result(self) -> Optional[JobResult]:
        """Return the JobResult (only available after DONE or ERROR)."""
        return self._result

    def cancel(self) -> None:
        """
        Request cancellation.

        Sets the cancel_event; the job function must poll it to exit.
        If the job ignores it, termination is not guaranteed.
        """
        self._cancel_event.set()
        with self._lock:
            # We mark cancellation in the log but don't force-kill the thread
            self._line_queue.put("[JobRunner] Cancellation requested.\n")

    def is_running(self) -> bool:
        with self._lock:
            return self._status == JobStatus.RUNNING

    def is_done(self) -> bool:
        with self._lock:
            return self._status in (JobStatus.DONE, JobStatus.ERROR)

    def reset(self) -> None:
        """
        Reset to IDLE so the runner can accept a new job.
        Safe to call after DONE or ERROR.

        Raises
        ------
        RuntimeError : Job is still RUNNING (call cancel() first and wait).
        """
        with self._lock:
            if self._status == JobStatus.RUNNING:
                raise RuntimeError(
                    f"JobRunner '{self.name}' is still running. "
                    f"Call cancel() and wait, then reset()."
                )
            self._status = JobStatus.IDLE
            self._result = None
            self._cancel_event.clear()
            while not self._line_queue.empty():
                try:
                    self._line_queue.get_nowait()
                except queue.Empty:
                    break
            self._log_deque.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, fn: Callable, args: tuple, kwargs: dict) -> None:
        """Worker: runs inside the background thread."""
        # Capture this thread's output via the thread-local proxy.
        # The main Gradio thread is unaffected.
        writer = _QueueWriter(self._line_queue, self._log_deque, _stdout_proxy._original)
        _stdout_proxy.set_override(writer)
        _stderr_proxy.set_override(writer)

        return_value = None
        try:
            print(f"[JobRunner] Starting: {fn.__name__}")
            return_value = fn(*args, **kwargs)
            if self._cancel_event.is_set():
                print("[JobRunner] Job was cancelled.")
                with self._lock:
                    self._status = JobStatus.ERROR
                self._result = JobResult(
                    status=JobStatus.ERROR,
                    error_message="Cancelled by user.",
                )
            else:
                print(f"[JobRunner] Finished: {fn.__name__}")
                with self._lock:
                    self._status = JobStatus.DONE
                self._result = JobResult(
                    status=JobStatus.DONE,
                    return_value=return_value,
                )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n[JobRunner] ERROR in {fn.__name__}:\n{tb}")
            with self._lock:
                self._status = JobStatus.ERROR
            self._result = JobResult(
                status=JobStatus.ERROR,
                error_message=str(exc),
                traceback_str=tb,
            )
        finally:
            # Flush any partial buffered line and release the thread-local override.
            writer.flush()
            _stdout_proxy.clear_override()
            _stderr_proxy.clear_override()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory — one runner per logical job slot
# ──────────────────────────────────────────────────────────────────────────────

def make_runner(name: str) -> JobRunner:
    """Create a named JobRunner instance."""
    return JobRunner(name=name)
