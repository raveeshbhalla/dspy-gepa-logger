"""Unified GEPA tracker combining all v2 hooks.

GEPATracker provides a single interface to:
1. Capture GEPA state via stop_callbacks (GEPAStateLogger)
2. Capture all LM calls with context tags (DSPyLMLogger)
3. Capture evaluation details via wrapped metric (LoggedMetric)
4. Optionally track reflection/proposal phases (LoggedInstructionProposer)

This is the v2.2 architecture that uses public hooks instead of monkey-patching.

Usage:
    from dspy_gepa_logger import GEPATracker

    tracker = GEPATracker()

    gepa = GEPA(
        metric=tracker.wrap_metric(my_metric),
        gepa_kwargs={'stop_callbacks': [tracker.get_stop_callback()]},
    )

    # Configure DSPy with LM logging
    import dspy
    dspy.configure(callbacks=tracker.get_dspy_callbacks())

    # Run optimization
    result = gepa.compile(student, trainset=train, valset=val)

    # Access captured data
    print(tracker.get_summary())
    print(tracker.get_candidate_diff(0, 5))
"""

import atexit
import io
import json
import logging
import queue
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, TextIO

from .context import clear_ctx, get_ctx, set_ctx
from .state_logger import GEPAStateLogger, IterationDelta, IterationMetadata
from .lm_logger import DSPyLMLogger, LMCall
from .logged_metric import LoggedMetric, EvaluationRecord
from .logged_proposer import (
    LoggedInstructionProposer,
    LoggedSelector,
    ReflectionCall,
    ProposalCall,
)


logger = logging.getLogger(__name__)

# Global capture state to prevent double-wrapping stdout/stderr
# when multiple GEPATracker instances exist in the same process
_global_capture_lock = threading.Lock()
_global_stdout_capture: "StreamCapture | None" = None
_global_stderr_capture: "StreamCapture | None" = None
_global_capture_owner: "GEPATracker | None" = None


class LoggedLM:
    """Wrapper that sets phase context before LM calls.

    This enables proper phase attribution for reflection/proposal LM calls
    without requiring users to manually wrap the proposer.

    Usage:
        # Wrap reflection_lm to tag all its calls as 'reflection' phase
        logged_lm = LoggedLM(reflection_lm, phase="reflection")
        gepa = GEPA(reflection_lm=logged_lm, ...)

    How it works:
        1. Before each call: sets phase in context
        2. LM call executes (DSPyLMLogger will capture with phase tag)
        3. After call: clears phase from context
    """

    def __init__(self, base_lm: Any, phase: str):
        """Initialize the logged LM wrapper.

        Args:
            base_lm: The LM instance to wrap
            phase: The phase to tag calls with (e.g., "reflection", "proposal")
        """
        self._base_lm = base_lm
        self._phase = phase

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the LM call with phase context set."""
        prev_phase = get_ctx().get("phase")
        set_ctx(phase=self._phase)
        try:
            return self._base_lm(*args, **kwargs)
        finally:
            set_ctx(phase=prev_phase)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base LM.

        This allows the wrapper to be a drop-in replacement for the base LM.
        """
        return getattr(self._base_lm, name)


class LogPushWorker:
    """Background worker that pushes log entries from a queue.

    Uses a single worker thread with a bounded queue to avoid
    unbounded thread creation and ensure logs are delivered.
    """

    def __init__(
        self,
        push_callback: Callable[[str, str, float, int | None, str | None], None],
        max_queue_size: int = 1000,
    ):
        """Initialize the log push worker.

        Args:
            push_callback: Function to call with (log_type, content, timestamp, iteration, phase)
            max_queue_size: Max entries to queue before dropping (default: 1000)
        """
        self._push_callback = push_callback
        self._queue: queue.Queue[tuple[str, str, float, int | None, str | None] | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._dropped_count: int = 0
        self._push_error_count: int = 0

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker thread and flush remaining logs.

        Args:
            timeout: Max seconds to wait for queue to drain
        """
        if not self._running:
            return
        self._running = False
        # Signal worker to stop
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        # Wait for worker to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def push(
        self,
        log_type: str,
        content: str,
        timestamp: float,
        iteration: int | None,
        phase: str | None,
    ) -> None:
        """Queue a log entry for pushing.

        Non-blocking; drops if queue is full and increments dropped_count.
        """
        if not self._running:
            return
        try:
            self._queue.put_nowait((log_type, content, timestamp, iteration, phase))
        except queue.Full:
            self._dropped_count += 1

    def _worker_loop(self) -> None:
        """Worker thread loop that processes the queue."""
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                log_type, content, timestamp, iteration, phase = item
                try:
                    self._push_callback(log_type, content, timestamp, iteration, phase)
                except Exception:
                    self._push_error_count += 1
            except queue.Empty:
                continue

    @property
    def dropped_count(self) -> int:
        """Number of log entries dropped due to full queue."""
        return self._dropped_count

    @property
    def push_error_count(self) -> int:
        """Number of push errors encountered."""
        return self._push_error_count


class StreamCapture(io.TextIOBase):
    """Captures stdout/stderr and pushes to server while preserving original output.

    This class wraps the original stream and:
    1. Passes all writes through to the original stream
    2. Buffers content and periodically pushes to the server via a worker queue
    3. Is thread-safe for concurrent writes
    """

    def __init__(
        self,
        original_stream: TextIO,
        stream_type: str,  # "stdout" or "stderr"
        log_worker: LogPushWorker,
        buffer_size: int = 256,
        flush_interval: float = 0.5,
    ):
        """Initialize the stream capture.

        Args:
            original_stream: The original sys.stdout or sys.stderr
            stream_type: "stdout" or "stderr" for log type
            log_worker: LogPushWorker instance to use for async pushes
            buffer_size: Max characters to buffer before flushing
            flush_interval: Max seconds between flushes
        """
        super().__init__()
        self._original = original_stream
        self._stream_type = stream_type
        self._log_worker = log_worker
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval

        self._buffer: list[str] = []
        self._buffer_len = 0
        self._last_flush = time.time()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        """Write to both original stream and buffer."""
        if not s:
            return 0

        # Always write to original stream
        result = self._original.write(s)
        # Normalize return value (some streams return None)
        written = len(s) if result is None else result

        # Buffer for server push
        with self._lock:
            self._buffer.append(s)
            self._buffer_len += len(s)

            # Check if we should flush
            now = time.time()
            should_flush = (
                self._buffer_len >= self._buffer_size or
                (now - self._last_flush) >= self._flush_interval
            )

            if should_flush and self._buffer:
                content = "".join(self._buffer)
                self._buffer = []
                self._buffer_len = 0
                self._last_flush = now

                # Get current context for iteration/phase
                ctx = get_ctx()
                iteration = ctx.get("iteration")
                phase = ctx.get("phase")

                # Queue for async push (non-blocking)
                self._log_worker.push(self._stream_type, content, now, iteration, phase)

        return written

    def flush(self) -> None:
        """Flush both original stream and any buffered content."""
        self._original.flush()

        with self._lock:
            if self._buffer:
                content = "".join(self._buffer)
                self._buffer = []
                self._buffer_len = 0
                now = time.time()
                self._last_flush = now

                ctx = get_ctx()
                iteration = ctx.get("iteration")
                phase = ctx.get("phase")

                self._log_worker.push(self._stream_type, content, now, iteration, phase)

    def fileno(self) -> int:
        """Return the file descriptor of the original stream."""
        return self._original.fileno()

    @property
    def encoding(self) -> str:
        """Return the encoding of the original stream."""
        return self._original.encoding

    def isatty(self) -> bool:
        """Return whether the original stream is a TTY."""
        return self._original.isatty()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


@dataclass
class CandidateDiff:
    """Diff between two candidates showing prompts and evaluation changes.

    Useful for understanding what changed and why.
    """

    from_idx: int
    to_idx: int
    prompt_changes: dict[str, tuple[str, str]]  # key -> (old, new)
    evaluation_changes: list[dict[str, Any]]  # Per-example changes
    lineage: list[int]  # Full lineage from to_idx to seed


class GEPATracker:
    """Unified tracker that combines all GEPA logging hooks.

    This class integrates:
    - GEPAStateLogger: Captures iteration state incrementally via stop_callbacks
    - DSPyLMLogger: Captures all LM calls with context tags
    - LoggedMetric: Captures evaluation details (score, feedback, prediction)
    - LoggedInstructionProposer: Optionally tracks reflection/proposal phases

    The tracker automatically manages context propagation between components.

    Attributes:
        state_logger: GEPAStateLogger instance for state capture
        lm_logger: DSPyLMLogger instance for LM call capture
        metric_logger: LoggedMetric wrapper (created via wrap_metric())
        proposer_logger: Optional LoggedInstructionProposer (created via wrap_proposer())
    """

    def __init__(
        self,
        capture_lm_calls: bool = True,
        dspy_version: str | None = None,
        gepa_version: str | None = None,
        server_url: str | None = None,
        project_name: str = "Default",
        auto_capture_stdout: bool = False,
    ):
        """Initialize the tracker.

        Args:
            capture_lm_calls: Whether to capture LM calls (default: True)
            dspy_version: DSPy version (for compatibility info)
            gepa_version: GEPA version (for compatibility info)
            server_url: Optional URL of GEPA Logger web server for real-time updates
            project_name: Project name for organizing runs (default: "Default")
            auto_capture_stdout: Whether to automatically start stdout/stderr capture
                when server_url is set (default: False)
        """
        self._start_time: float | None = None
        self._capture_lm_calls = capture_lm_calls

        # Core components
        self.state_logger = GEPAStateLogger(
            dspy_version=dspy_version,
            gepa_version=gepa_version,
        )
        self.lm_logger = DSPyLMLogger(
            on_call_complete=self._on_lm_call_complete if server_url else None
        ) if capture_lm_calls else None
        self.metric_logger: LoggedMetric | None = None
        self.proposer_logger: LoggedInstructionProposer | None = None
        self.selector_logger: LoggedSelector | None = None

        # Wrapped objects (returned to user)
        self._wrapped_metric: Callable[..., Any] | None = None
        self._wrapped_proposer: Any | None = None
        self._wrapped_selector: Any | None = None

        # Valset example IDs for filtering comparisons
        self._valset_example_ids: set[str] | None = None

        # Server integration (optional)
        self._server_client: Any | None = None
        self._server_url = server_url
        self._project_name = project_name
        self._last_pushed_iteration = -1
        self._last_pushed_delta_count = 0  # Track which deltas we've pushed candidates from
        self._last_pushed_eval_count = 0
        self._last_pushed_lm_count = 0
        self._pending_seed_prompt: dict[str, str] | None = None  # For start_run retries

        # Stream capture for stdout/stderr
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None
        self._stdout_capture: StreamCapture | None = None
        self._stderr_capture: StreamCapture | None = None
        self._log_worker: LogPushWorker | None = None
        self._capture_streams = False
        self._atexit_registered = False
        self._auto_capture_stdout = auto_capture_stdout

        if server_url:
            self._init_server_client()
            # Auto-start stdout capture if requested
            if auto_capture_stdout:
                self.start_log_capture()

    def set_valset(self, valset: list[Any]) -> None:
        """Set the validation set for comparison filtering.

        Call this before running GEPA optimization to ensure performance
        comparisons only include validation examples (not training examples).

        Args:
            valset: List of validation examples (dspy.Example or similar)

        Example:
            tracker.set_valset(val_data)
            result = gepa.compile(student, trainset=train_data, valset=val_data)
        """
        from .logged_metric import _deterministic_example_id

        self._valset_example_ids = {
            _deterministic_example_id(example) for example in valset
        }

    # ==================== Server Integration ====================

    def _init_server_client(self) -> None:
        """Initialize the server client and start a new run.

        We start the run immediately (even without seed_prompt) so that
        log capture can begin pushing logs to the server. The seed_prompt
        and other metadata will be updated on the first iteration callback.
        """
        try:
            from ..server.client import ServerClient

            self._server_client = ServerClient(
                server_url=self._server_url,
                project_name=self._project_name,
            )
            logger.info(f"Server client initialized for {self._server_url}")

            # Start the run immediately so logs can be pushed before optimization starts.
            # We'll update seed_prompt and other metadata on the first iteration.
            run_id = self._server_client.start_run(
                config={
                    "dspy_version": self.state_logger.versions.get("dspy"),
                    "gepa_version": self.state_logger.versions.get("gepa"),
                    "capture_lm_calls": self._capture_lm_calls,
                },
            )
            if run_id:
                logger.info(f"Server run started: {run_id}")
            else:
                logger.warning("Failed to start server run - logs may not be pushed")
        except ImportError:
            logger.warning("Could not import ServerClient - server integration disabled")
            self._server_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize server client: {e}")
            self._server_client = None

    def _push_log_to_server(
        self,
        log_type: str,
        content: str,
        timestamp: float,
        iteration: int | None,
        phase: str | None,
    ) -> None:
        """Push a log entry to the server (called from StreamCapture)."""
        if self._server_client is None or not self._server_client.run_id:
            return

        try:
            self._server_client.push_logs([{
                "logType": log_type,
                "content": content,
                "timestamp": timestamp,
                "iteration": iteration,
                "phase": phase,
            }])
        except Exception as e:
            # Don't log this error to avoid infinite recursion with stdout capture
            pass

    def _cleanup_log_capture(self) -> None:
        """Cleanup handler for atexit to ensure streams are restored."""
        if self._capture_streams:
            self.stop_log_capture()

    def start_log_capture(self) -> None:
        """Start capturing stdout/stderr and pushing to server.

        Call this before running GEPA optimization to capture all console output.
        Must call stop_log_capture() when done to restore original streams.

        If another tracker is already capturing, this call is a no-op to prevent
        double-wrapping of streams.

        Prefer using the context manager log_capture() instead:
            with tracker.log_capture():
                result = gepa.compile(student, trainset=train, valset=val)

        Example (manual):
            tracker.start_log_capture()
            try:
                result = gepa.compile(student, trainset=train, valset=val)
            finally:
                tracker.stop_log_capture()
        """
        global _global_stdout_capture, _global_stderr_capture, _global_capture_owner

        if self._capture_streams:
            return  # Already capturing (this tracker)

        if self._server_client is None:
            logger.warning("Cannot start log capture - no server client configured")
            return

        with _global_capture_lock:
            # Check if another tracker is already capturing
            if _global_capture_owner is not None and _global_capture_owner is not self:
                logger.debug(
                    "Another tracker is already capturing stdout/stderr, skipping"
                )
                return

            # Also check if sys.stdout is already a StreamCapture (defensive)
            if isinstance(sys.stdout, StreamCapture):
                logger.debug("stdout is already wrapped by StreamCapture, skipping")
                return

            # Register atexit handler to restore streams on unexpected exit
            if not self._atexit_registered:
                atexit.register(self._cleanup_log_capture)
                self._atexit_registered = True

            # Create log worker for async push
            self._log_worker = LogPushWorker(push_callback=self._push_log_to_server)
            self._log_worker.start()

            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr

            self._stdout_capture = StreamCapture(
                original_stream=self._original_stdout,
                stream_type="stdout",
                log_worker=self._log_worker,
            )
            self._stderr_capture = StreamCapture(
                original_stream=self._original_stderr,
                stream_type="stderr",
                log_worker=self._log_worker,
            )

            sys.stdout = self._stdout_capture  # type: ignore
            sys.stderr = self._stderr_capture  # type: ignore
            self._capture_streams = True

            # Update global state
            _global_stdout_capture = self._stdout_capture
            _global_stderr_capture = self._stderr_capture
            _global_capture_owner = self

        logger.info("Started capturing stdout/stderr for server streaming")

    def stop_log_capture(self) -> None:
        """Stop capturing stdout/stderr and restore original streams.

        Safe to call even if capture wasn't started.

        Only restores streams if this tracker owns them (i.e., sys.stdout is still
        the StreamCapture we installed). If something else has replaced sys.stdout,
        we don't clobber it.
        """
        global _global_stdout_capture, _global_stderr_capture, _global_capture_owner

        if not self._capture_streams:
            return

        with _global_capture_lock:
            # Flush any remaining buffered content
            if self._stdout_capture:
                self._stdout_capture.flush()
            if self._stderr_capture:
                self._stderr_capture.flush()

            # Only restore if we still own the streams (haven't been replaced externally)
            if self._original_stdout and sys.stdout is self._stdout_capture:
                sys.stdout = self._original_stdout
            elif self._original_stdout and sys.stdout is not self._stdout_capture:
                logger.debug(
                    "stdout was replaced externally, not restoring original"
                )

            if self._original_stderr and sys.stderr is self._stderr_capture:
                sys.stderr = self._original_stderr
            elif self._original_stderr and sys.stderr is not self._stderr_capture:
                logger.debug(
                    "stderr was replaced externally, not restoring original"
                )

            # Stop the log worker (waits for queue to drain)
            if self._log_worker:
                # Log dropped/error counts before stopping
                if self._log_worker.dropped_count > 0:
                    logger.warning(
                        f"LogPushWorker dropped {self._log_worker.dropped_count} log entries"
                    )
                if self._log_worker.push_error_count > 0:
                    logger.warning(
                        f"LogPushWorker encountered {self._log_worker.push_error_count} push errors"
                    )
                self._log_worker.stop()
                self._log_worker = None

            # Clear global state if we were the owner
            if _global_capture_owner is self:
                _global_stdout_capture = None
                _global_stderr_capture = None
                _global_capture_owner = None

            self._stdout_capture = None
            self._stderr_capture = None
            self._original_stdout = None
            self._original_stderr = None
            self._capture_streams = False

        logger.info("Stopped capturing stdout/stderr")

    @contextmanager
    def log_capture(self) -> Generator[None, None, None]:
        """Context manager for capturing stdout/stderr.

        Automatically starts and stops log capture, ensuring streams
        are restored even if an exception occurs.

        Example:
            with tracker.log_capture():
                result = gepa.compile(student, trainset=train, valset=val)
        """
        self.start_log_capture()
        try:
            yield
        finally:
            self.stop_log_capture()

    def _on_lm_call_complete(self, lm_call: LMCall) -> None:
        """Called when an LM call completes (for real-time streaming).

        This pushes each LM call to the server immediately, enabling
        real-time log viewing in the web UI.
        """
        if self._server_client is None or not self._server_client.run_id:
            return

        try:
            # Push as a log entry for the logs tab
            log_content = json.dumps({
                "call_id": lm_call.call_id,
                "model": lm_call.model,
                "duration_ms": lm_call.duration_ms,
                "iteration": lm_call.iteration,
                "phase": lm_call.phase,
                "candidate_idx": lm_call.candidate_idx,
                "inputs_preview": str(lm_call.inputs)[:500] if lm_call.inputs else None,
                "outputs_preview": str(lm_call.outputs)[:500] if lm_call.outputs else None,
            })

            self._server_client.push_logs([{
                "logType": "lm_call",
                "content": log_content,
                "timestamp": time.time(),
                "iteration": lm_call.iteration,
                "phase": lm_call.phase,
            }])
        except Exception:
            pass  # Don't let streaming errors affect normal operation

    def _update_server_run(self, seed_prompt: dict[str, str] | None = None) -> None:
        """Update the run with seed_prompt and valset (called on first iteration).

        The run is created in _init_server_client, but we update it here with
        seed_prompt and valset_example_ids once they're available.
        """
        if self._server_client is None or self._server_client.run_id is None:
            return

        try:
            # Convert valset example IDs to list for JSON serialization
            valset_ids = list(self._valset_example_ids) if self._valset_example_ids else None

            success = self._server_client.update_run(
                seed_prompt=seed_prompt,
                valset_example_ids=valset_ids,
            )
            if success:
                logger.info(f"Updated run {self._server_client.run_id} with seed_prompt")
        except Exception as e:
            logger.warning(f"Failed to update server run: {e}")

    def _retry_start_server_run(self) -> None:
        """Retry starting the server run if it failed during init.

        Called from _push_to_server if the run_id is None.
        """
        if self._server_client is None:
            return

        try:
            run_id = self._server_client.start_run(
                config={
                    "dspy_version": self.state_logger.versions.get("dspy"),
                    "gepa_version": self.state_logger.versions.get("gepa"),
                    "capture_lm_calls": self._capture_lm_calls,
                },
            )
            if run_id:
                logger.info(f"Server run started on retry: {run_id}")
                # If we have a pending seed_prompt, update the run with it
                if self._pending_seed_prompt:
                    self._update_server_run(self._pending_seed_prompt)
        except Exception as e:
            logger.warning(f"Failed to start server run on retry: {e}")

    def _push_to_server(self) -> None:
        """Push pending data to server (called after each iteration).

        Always attempts to push if a server client exists, even if previously
        disconnected. This allows recovery after transient network issues since
        the client will restore is_connected on successful requests.

        If start_run failed previously (run_id is None), we retry it here.
        This ensures data is eventually pushed once the server becomes available.
        """
        if self._server_client is None:
            return

        try:
            # Retry start_run if it previously failed
            # This handles cases where the server was down during init
            if self._server_client.run_id is None:
                logger.info("Retrying start_run (server was unavailable previously)")
                self._retry_start_server_run()
                # If still no run_id, skip pushing this time (will retry next iteration)
                if self._server_client.run_id is None:
                    return

            self._push_iterations_to_server()
            self._push_candidates_to_server()
            self._push_evaluations_to_server()
            self._push_lm_calls_to_server()
        except Exception as e:
            logger.warning(f"Failed to push data to server: {e}")

    def _push_iterations_to_server(self) -> None:
        """Push new iterations to server.

        Note: We iterate all metadata and filter by iteration number rather than
        slicing by index, because iteration numbers may not be zero-based or
        contiguous. This ensures we don't skip or drop iterations.
        """
        if self._server_client is None:
            return

        # Filter metadata by iteration number, not list index
        # This handles non-zero-based or non-contiguous iteration numbers
        for meta in self.state_logger.metadata:
            if meta.iteration <= self._last_pushed_iteration:
                continue  # Already pushed this iteration
            # Find the corresponding delta for pareto data
            delta = None
            for d in self.state_logger.deltas:
                if d.iteration == meta.iteration:
                    delta = d
                    break

            pareto_frontier = None
            pareto_programs = None
            child_candidate_idxs = None
            parent_candidate_idx = None
            if delta:
                # Convert pareto data to serializable format
                pareto_frontier = {k: v[0] for k, v in delta.pareto_additions.items()}
                pareto_programs = {
                    k: min(v[1]) if v[1] else None
                    for k, v in delta.pareto_additions.items()
                }
                # Get new candidate indices from this delta
                if delta.new_candidates:
                    child_candidate_idxs = [idx for idx, _ in delta.new_candidates]
                # Get parent from lineage (if any new candidates were created)
                if delta.new_lineage:
                    for idx, lineage in delta.new_lineage:
                        if lineage:
                            parent_candidate_idx = lineage[0]
                            break

            # Get reflection/proposal data from proposer logger if available
            reflection_input = None
            reflection_output = None
            proposed_changes = None
            if self.proposer_logger:
                # Get reflection LM calls for this iteration
                reflection_lm_calls = self._get_reflection_lm_calls(meta.iteration)
                if reflection_lm_calls:
                    # Use the first reflection call's data
                    first_call = reflection_lm_calls[0]
                    reflection_input = json.dumps(first_call.inputs) if first_call.inputs else None
                    reflection_output = json.dumps(first_call.outputs) if first_call.outputs else None

                # Get proposal data for this iteration
                proposal_calls = self.proposer_logger.get_proposal_calls_for_iteration(meta.iteration)
                if proposal_calls:
                    # Collect all proposed changes
                    proposed_changes = []
                    for pc in proposal_calls:
                        proposed_changes.extend(pc.proposals)
                        # If parent_candidate_idx wasn't set from lineage, get it from proposal
                        if parent_candidate_idx is None:
                            parent_candidate_idx = pc.candidate_idx

            success = self._server_client.push_iteration(
                iteration_number=meta.iteration,
                timestamp=meta.timestamp,
                total_evals=meta.total_evals,
                num_candidates=meta.num_candidates,
                pareto_size=meta.pareto_size,
                pareto_frontier=pareto_frontier,
                pareto_programs=pareto_programs,
                reflection_input=reflection_input,
                reflection_output=reflection_output,
                proposed_changes=proposed_changes,
                parent_candidate_idx=parent_candidate_idx,
                child_candidate_idxs=child_candidate_idxs,
            )
            # Only update tracking if push succeeded to allow retry on failure
            if success:
                self._last_pushed_iteration = meta.iteration
            else:
                # Stop processing further iterations if one fails
                break

    def _get_reflection_lm_calls(self, iteration: int) -> list[LMCall]:
        """Get LM calls tagged as reflection for a specific iteration."""
        if self.lm_logger is None:
            return []
        return [
            call for call in self.lm_logger.calls
            if call.iteration == iteration and call.phase == "reflection"
        ]

    def _push_candidates_to_server(self) -> None:
        """Push new candidates to server (incremental - only unpushed deltas)."""
        if self._server_client is None:
            return

        # Only process deltas we haven't pushed yet
        new_deltas = self.state_logger.deltas[self._last_pushed_delta_count:]
        if not new_deltas:
            return

        # Collect candidates from new deltas only
        candidates_to_push = []
        for delta in new_deltas:
            for idx, content in delta.new_candidates:
                # Find parent from lineage
                parent_idx = None
                for lin_idx, lineage in delta.new_lineage:
                    if lin_idx == idx and lineage:
                        parent_idx = lineage[0]
                        break

                candidates_to_push.append((
                    idx,
                    content,
                    parent_idx,
                    delta.iteration,
                ))

        if candidates_to_push:
            success = self._server_client.push_candidates(candidates_to_push)
            # Only update tracking if push succeeded to allow retry on failure
            if success:
                self._last_pushed_delta_count = len(self.state_logger.deltas)

    def _push_evaluations_to_server(self) -> None:
        """Push new evaluations to server immediately.

        Pushes all new evaluations since the last push. This ensures real-time
        dashboard updates and prevents data loss on short runs or crashes.
        """
        if self._server_client is None or self.metric_logger is None:
            return

        evals = self.metric_logger.evaluations
        new_evals = evals[self._last_pushed_eval_count:]

        if new_evals:
            success = self._server_client.push_evaluations(new_evals)
            # Only update tracking if push succeeded to allow retry on failure
            if success:
                self._last_pushed_eval_count = len(evals)

    def _push_lm_calls_to_server(self) -> None:
        """Push new LM calls to server immediately.

        Pushes all new LM calls since the last push. This ensures real-time
        dashboard updates and prevents data loss on short runs or crashes.
        """
        if self._server_client is None or self.lm_logger is None:
            return

        calls = self.lm_logger.calls
        new_calls = calls[self._last_pushed_lm_count:]

        if new_calls:
            success = self._server_client.push_lm_calls(new_calls)
            # Only update tracking if push succeeded to allow retry on failure
            if success:
                self._last_pushed_lm_count = len(calls)

    def finalize(
        self,
        status: str = "COMPLETED",
        best_candidate_idx: int | None = None,
    ) -> None:
        """Finalize the run and push all remaining data to server.

        Call this after optimization completes to ensure all data is pushed
        to the server and the run is marked as complete.

        Args:
            status: "COMPLETED" or "FAILED"
            best_candidate_idx: Index of the best candidate (auto-detected if None)
        """
        if self._server_client is None:
            return

        try:
            # Retry start_run if it previously failed (server was down during optimization)
            # This ensures we can still push all data if the server comes back before finalize
            if self._server_client.run_id is None:
                logger.info("Retrying start_run in finalize (server was unavailable previously)")
                self._retry_start_server_run()
                if self._server_client.run_id is None:
                    logger.warning("Could not start server run - data will not be pushed")
                    return
            # Push any remaining evaluations
            if self.metric_logger:
                remaining_evals = self.metric_logger.evaluations[
                    self._last_pushed_eval_count :
                ]
                if remaining_evals:
                    self._server_client.push_evaluations(remaining_evals)

            # Push any remaining LM calls
            if self.lm_logger:
                remaining_lm = self.lm_logger.calls[self._last_pushed_lm_count :]
                if remaining_lm:
                    self._server_client.push_lm_calls(remaining_lm)

            # Push any remaining iterations
            self._push_iterations_to_server()
            self._push_candidates_to_server()

            # Determine best candidate if not provided
            if best_candidate_idx is None:
                report = self.get_optimization_report()
                best_candidate_idx = report.get("optimized_idx", 0)

            # Get best prompt and scores
            best_prompt = None
            if best_candidate_idx < len(self.state_logger.final_candidates):
                best_prompt = self.state_logger.final_candidates[best_candidate_idx]

            # Calculate scores
            seed_score = None
            best_score = None
            if self.metric_logger:
                seed_idx = self.state_logger.seed_candidate_idx or 0
                seed_evals = self.metric_logger.get_evaluations_for_candidate(seed_idx)
                best_evals = self.metric_logger.get_evaluations_for_candidate(
                    best_candidate_idx
                )
                if seed_evals:
                    seed_score = sum(e.score for e in seed_evals) / len(seed_evals)
                if best_evals:
                    best_score = sum(e.score for e in best_evals) / len(best_evals)

            # Complete the run
            self._server_client.complete_run(
                status=status,
                best_prompt=best_prompt,
                best_candidate_idx=best_candidate_idx,
                best_score=best_score,
                seed_score=seed_score,
            )
            logger.info(f"Run finalized with status: {status}")
        except Exception as e:
            logger.warning(f"Failed to finalize server run: {e}")
        finally:
            # Stop log capture if it was auto-started
            if self._auto_capture_stdout and self._capture_streams:
                self.stop_log_capture()

    @property
    def server_run_id(self) -> str | None:
        """Get the server run ID if connected."""
        if self._server_client:
            return self._server_client.run_id
        return None

    # ==================== Metric/Proposer Wrapping ====================

    def wrap_metric(
        self,
        metric: Callable[..., Any],
        capture_prediction: bool = True,
        max_prediction_preview: int = 200,
        failure_score: float = 0.0,
    ) -> LoggedMetric:
        """Wrap a metric function to capture evaluation details.

        The wrapped metric will:
        1. Set phase="eval" in context before calling the metric
        2. Capture score, feedback, and prediction for each call
        3. Restore the previous phase after the call
        4. Handle exceptions gracefully by returning failure_score

        Args:
            metric: The metric function to wrap
            capture_prediction: Whether to capture predictions (default: True)
            max_prediction_preview: Max length for prediction preview
            failure_score: Score to return when the metric throws an exception

        Returns:
            LoggedMetric wrapper that can be used in place of the original metric
        """
        self.metric_logger = LoggedMetric(
            metric_fn=metric,
            capture_prediction=capture_prediction,
            max_prediction_preview=max_prediction_preview,
            failure_score=failure_score,
        )
        self._wrapped_metric = self.metric_logger
        return self.metric_logger

    def wrap_proposer(self, proposer: Any) -> LoggedInstructionProposer:
        """Wrap an instruction proposer to track reflection/proposal phases.

        The wrapped proposer will:
        1. Set phase="reflection" during reflection analysis
        2. Set phase="proposal" during proposal generation
        3. Record all reflection and proposal calls

        Args:
            proposer: The instruction proposer to wrap

        Returns:
            LoggedInstructionProposer wrapper
        """
        self.proposer_logger = LoggedInstructionProposer(proposer)
        self._wrapped_proposer = self.proposer_logger
        return self.proposer_logger

    def wrap_selector(self, selector: Any) -> LoggedSelector:
        """Wrap a selector to track selection phases.

        Args:
            selector: The selector to wrap

        Returns:
            LoggedSelector wrapper
        """
        self.selector_logger = LoggedSelector(selector)
        self._wrapped_selector = self.selector_logger
        return self.selector_logger

    def get_stop_callback(self) -> Callable[[Any], bool]:
        """Get the stop_callback for use in gepa_kwargs.

        Returns a callback that:
        1. Calls the GEPAStateLogger to capture state
        2. Pushes updates to the server (if connected)
        3. Returns False (never stops optimization)

        Usage:
            gepa = GEPA(
                metric=logged_metric,
                gepa_kwargs={'stop_callbacks': [tracker.get_stop_callback()]},
            )

        Returns:
            Callable that wraps GEPAStateLogger with server push
        """

        def stop_callback_with_server_push(state: Any) -> bool:
            # First iteration - update the run with seed_prompt and valset
            is_first_iteration = len(self.state_logger.deltas) == 0
            if is_first_iteration and self._server_client is not None:
                # Extract seed prompt from state if available
                candidates = getattr(state, "program_candidates", [])
                seed_prompt = candidates[0] if candidates else None
                # Store seed prompt for potential retries
                self._pending_seed_prompt = seed_prompt
                # Update the run with seed_prompt and valset (run was created in _init_server_client)
                self._update_server_run(seed_prompt)

            # Call the original state logger
            result = self.state_logger(state)

            # Push to server after each iteration
            self._push_to_server()

            return result

        return stop_callback_with_server_push

    def get_dspy_callbacks(self) -> list[DSPyLMLogger]:
        """Get DSPy callbacks for LM call capture.

        Returns list of callbacks to configure with dspy.configure().

        Usage:
            import dspy
            dspy.configure(callbacks=tracker.get_dspy_callbacks())

        Returns:
            List containing DSPyLMLogger (or empty if capture_lm_calls=False)
        """
        if self.lm_logger is not None:
            return [self.lm_logger]
        return []

    # ==================== Query Methods ====================

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of the optimization run.

        Returns:
            Dict with:
            - state_summary: From GEPAStateLogger
            - lm_summary: From DSPyLMLogger (if enabled)
            - metric_summary: From LoggedMetric (if wrapped)
            - proposer_summary: From LoggedInstructionProposer (if wrapped)
        """
        summary: dict[str, Any] = {
            "state": self.state_logger.get_summary(),
        }

        if self.lm_logger is not None:
            summary["lm_calls"] = self.lm_logger.get_summary()

        if self.metric_logger is not None:
            summary["evaluations"] = {
                "total_evaluations": len(self.metric_logger.evaluations),
                "unique_examples": len(
                    set(e.example_id for e in self.metric_logger.evaluations)
                ),
                "unique_candidates": len(
                    set(
                        e.candidate_idx
                        for e in self.metric_logger.evaluations
                        if e.candidate_idx is not None
                    )
                ),
            }

        if self.proposer_logger is not None:
            summary["proposer"] = {
                "total_reflections": len(self.proposer_logger.reflection_calls),
                "total_proposals": len(self.proposer_logger.proposal_calls),
            }

        return summary

    def get_candidate_diff(
        self,
        from_idx: int,
        to_idx: int,
    ) -> CandidateDiff:
        """Get diff between two candidates.

        Shows what changed in prompts and evaluations.

        Args:
            from_idx: Source candidate index
            to_idx: Target candidate index

        Returns:
            CandidateDiff with prompt changes, evaluation changes, and lineage
        """
        candidates = self.state_logger.get_all_candidates()

        # Get prompts
        from_prompt = candidates[from_idx] if from_idx < len(candidates) else {}
        to_prompt = candidates[to_idx] if to_idx < len(candidates) else {}

        # Compute prompt changes
        all_keys = set(from_prompt.keys()) | set(to_prompt.keys())
        prompt_changes: dict[str, tuple[str, str]] = {}
        for key in all_keys:
            old_val = from_prompt.get(key, "")
            new_val = to_prompt.get(key, "")
            if old_val != new_val:
                prompt_changes[key] = (old_val, new_val)

        # Compute evaluation changes if metric logger is available
        evaluation_changes: list[dict[str, Any]] = []
        if self.metric_logger is not None:
            from_evals = self.metric_logger.get_evaluations_for_candidate(from_idx)
            to_evals = self.metric_logger.get_evaluations_for_candidate(to_idx)

            # Group by example_id
            from_by_example = {e.example_id: e for e in from_evals}
            to_by_example = {e.example_id: e for e in to_evals}

            all_examples = set(from_by_example.keys()) | set(to_by_example.keys())
            for example_id in sorted(all_examples):
                from_eval = from_by_example.get(example_id)
                to_eval = to_by_example.get(example_id)

                change: dict[str, Any] = {"example_id": example_id}
                if from_eval and to_eval:
                    change["from_score"] = from_eval.score
                    change["to_score"] = to_eval.score
                    change["score_delta"] = to_eval.score - from_eval.score
                    if from_eval.feedback != to_eval.feedback:
                        change["feedback_changed"] = True
                        change["from_feedback"] = from_eval.feedback
                        change["to_feedback"] = to_eval.feedback
                elif from_eval:
                    change["from_score"] = from_eval.score
                    change["to_score"] = None
                    change["removed"] = True
                elif to_eval:
                    change["from_score"] = None
                    change["to_score"] = to_eval.score
                    change["added"] = True

                evaluation_changes.append(change)

        # Get lineage
        lineage = self.state_logger.get_lineage(to_idx)

        return CandidateDiff(
            from_idx=from_idx,
            to_idx=to_idx,
            prompt_changes=prompt_changes,
            evaluation_changes=evaluation_changes,
            lineage=lineage,
        )

    def get_lm_calls_for_iteration(self, iteration: int) -> list[LMCall]:
        """Get all LM calls for a specific iteration.

        Args:
            iteration: The iteration number

        Returns:
            List of LMCall records
        """
        if self.lm_logger is None:
            return []
        return self.lm_logger.get_calls_for_iteration(iteration)

    def get_lm_calls_for_phase(self, phase: str) -> list[LMCall]:
        """Get all LM calls for a specific phase.

        Args:
            phase: The phase ('reflection', 'proposal', 'eval', etc.)

        Returns:
            List of LMCall records
        """
        if self.lm_logger is None:
            return []
        return self.lm_logger.get_calls_for_phase(phase)

    def get_evaluations_for_example(self, example_id: str) -> list[EvaluationRecord]:
        """Get all evaluations for a specific example.

        Args:
            example_id: The example ID

        Returns:
            List of EvaluationRecord
        """
        if self.metric_logger is None:
            return []
        return self.metric_logger.get_evaluations_for_example(example_id)

    def get_evaluations_for_candidate(
        self, candidate_idx: int
    ) -> list[EvaluationRecord]:
        """Get all evaluations for a specific candidate.

        Args:
            candidate_idx: The candidate index

        Returns:
            List of EvaluationRecord
        """
        if self.metric_logger is None:
            return []
        return self.metric_logger.get_evaluations_for_candidate(candidate_idx)

    def compute_lift(
        self,
        baseline_candidate_idx: int,
        candidate_idx: int,
    ) -> dict[str, Any]:
        """Compute score lift from baseline to candidate.

        Compares scores across all examples evaluated by both candidates.

        Args:
            baseline_candidate_idx: Baseline candidate index
            candidate_idx: Candidate to compare

        Returns:
            Dict with lift statistics: mean_lift, total_examples, etc.
        """
        if self.metric_logger is None:
            return {"error": "No metric logger available"}

        # Get evaluations for both candidates
        baseline_evals = self.metric_logger.get_evaluations_for_candidate(
            baseline_candidate_idx
        )
        candidate_evals = self.metric_logger.get_evaluations_for_candidate(candidate_idx)

        # Build lookup by example_id
        baseline_by_example = {e.example_id: e for e in baseline_evals}
        candidate_by_example = {e.example_id: e for e in candidate_evals}

        # Find common examples
        common_examples = set(baseline_by_example.keys()) & set(
            candidate_by_example.keys()
        )

        if not common_examples:
            return {
                "mean_lift": 0.0,
                "total_examples": 0,
                "improved": 0,
                "regressed": 0,
                "unchanged": 0,
            }

        # Compute lift per example
        lifts = []
        improved = 0
        regressed = 0
        unchanged = 0

        for ex_id in common_examples:
            baseline_score = baseline_by_example[ex_id].score
            candidate_score = candidate_by_example[ex_id].score
            lift = candidate_score - baseline_score
            lifts.append(lift)

            if lift > 0:
                improved += 1
            elif lift < 0:
                regressed += 1
            else:
                unchanged += 1

        return {
            "mean_lift": sum(lifts) / len(lifts),
            "total_examples": len(common_examples),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        }

    def get_regressions(
        self,
        baseline_candidate_idx: int,
        candidate_idx: int,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Find examples where candidate regressed vs baseline.

        Args:
            baseline_candidate_idx: Baseline candidate index
            candidate_idx: Candidate to compare
            threshold: Minimum score drop to count as regression

        Returns:
            List of regression details
        """
        if self.metric_logger is None:
            return []

        # Get evaluations for both candidates
        baseline_evals = self.metric_logger.get_evaluations_for_candidate(
            baseline_candidate_idx
        )
        candidate_evals = self.metric_logger.get_evaluations_for_candidate(candidate_idx)

        # Build lookup by example_id
        baseline_by_example = {e.example_id: e for e in baseline_evals}
        candidate_by_example = {e.example_id: e for e in candidate_evals}

        # Find regressions
        regressions = []
        for ex_id in baseline_by_example:
            if ex_id not in candidate_by_example:
                continue

            baseline_score = baseline_by_example[ex_id].score
            candidate_score = candidate_by_example[ex_id].score
            delta = candidate_score - baseline_score

            if delta < -threshold:  # Score dropped more than threshold
                regressions.append(
                    {
                        "example_id": ex_id,
                        "baseline_score": baseline_score,
                        "baseline_feedback": baseline_by_example[ex_id].feedback,
                        "candidate_score": candidate_score,
                        "candidate_feedback": candidate_by_example[ex_id].feedback,
                        "delta": delta,
                    }
                )

        # Sort by delta (worst first)
        return sorted(regressions, key=lambda x: x["delta"])

    def get_pareto_evolution(self) -> list[dict[str, tuple[float, set[int]]]]:
        """Get Pareto frontier state at each iteration.

        Returns:
            List of dicts: data_id -> (score, set of program indices)
        """
        return self.state_logger.get_pareto_evolution()

    def get_lineage(self, candidate_idx: int) -> list[int]:
        """Trace a candidate back to its ancestors.

        Args:
            candidate_idx: The candidate to trace

        Returns:
            List of candidate indices from candidate to seed
        """
        return self.state_logger.get_lineage(candidate_idx)

    def get_all_candidates(self) -> list[dict[str, str]]:
        """Get all candidate prompts.

        Returns:
            List of candidate prompt dicts
        """
        return self.state_logger.get_all_candidates()

    # ==================== Properties ====================

    @property
    def seed_candidate(self) -> dict[str, str] | None:
        """Get the seed candidate prompt."""
        return self.state_logger.seed_candidate

    @property
    def seed_candidate_idx(self) -> int | None:
        """Get the seed candidate index."""
        return self.state_logger.seed_candidate_idx

    @property
    def final_candidates(self) -> list[dict[str, str]]:
        """Get final candidate prompts."""
        return self.state_logger.final_candidates

    @property
    def final_pareto(self) -> dict[str, float]:
        """Get final Pareto frontier scores."""
        return self.state_logger.final_pareto

    def get_best_candidate_idx(self) -> int:
        """Determine which candidate GEPA selected as best.

        Uses Pareto frontier data to identify the most frequently appearing
        candidate across all data points. Falls back to seed if no data.

        Note: This is an approximation - GEPA selects based on aggregate score,
        not Pareto frequency. For accurate comparison, use get_evaluation_comparison()
        which compares seed vs Pareto-optimal evaluation per example.

        Returns:
            The index of the best/selected candidate
        """
        pareto_programs = self.state_logger.final_pareto_programs
        if not pareto_programs:
            return self.seed_candidate_idx or 0

        # Count how often each candidate appears on the Pareto frontier
        candidate_counts: dict[int, int] = {}
        for prog_set in pareto_programs.values():
            if prog_set:
                # Use min() for deterministic selection when multiple candidates tie
                best = min(prog_set)
                candidate_counts[best] = candidate_counts.get(best, 0) + 1

        if not candidate_counts:
            return self.seed_candidate_idx or 0

        # Return most frequent candidate (or lowest index in case of tie)
        best_candidate = max(
            candidate_counts.keys(),
            key=lambda x: (candidate_counts[x], -x)  # Higher count, then lower index
        )
        return best_candidate

    @property
    def iterations(self) -> list[IterationDelta]:
        """Get all iteration deltas."""
        return self.state_logger.deltas

    @property
    def metadata(self) -> list[IterationMetadata]:
        """Get all iteration metadata."""
        return self.state_logger.metadata

    @property
    def lm_calls(self) -> list[LMCall]:
        """Get all LM calls."""
        if self.lm_logger is None:
            return []
        return self.lm_logger.calls

    @property
    def evaluations(self) -> list[EvaluationRecord]:
        """Get all evaluation records."""
        if self.metric_logger is None:
            return []
        return self.metric_logger.evaluations

    def get_evaluation_comparison(
        self,
        baseline_idx: int | None = None,
        optimized_idx: int | None = None,
    ) -> dict[str, Any]:
        """Get evaluation comparison between baseline and optimized candidates.

        Groups examples into improvements, regressions, and same categories.
        Each entry includes example inputs, baseline/optimized outputs, and scores.

        IMPORTANT: This compares seed (baseline) vs the SELECTED candidate's
        evaluations, not just the last evaluated candidate. Uses Pareto frontier
        data to determine which candidate GEPA actually selected as best.

        Since GEPA doesn't expose candidate_idx through public hooks, we use
        timestamp ordering to map evaluations to candidates:
        - Evaluations are sorted by timestamp
        - Evaluation[0] = candidate 0 (seed)
        - Evaluation[N] = candidate N
        - We compare eval[0] (seed) vs eval[selected_idx] (selected candidate)

        Args:
            baseline_idx: Baseline candidate index (default: seed)
            optimized_idx: Optimized candidate index (default: SELECTED best candidate)

        Returns:
            Dict with:
            - improvements: List of examples that improved
            - regressions: List of examples that regressed
            - same: List of examples with no change
            - summary: Overall stats (counts, avg scores)
        """
        if self.metric_logger is None:
            return {
                "improvements": [],
                "regressions": [],
                "same": [],
                "summary": {"error": "No evaluation data available"},
            }

        if baseline_idx is None:
            baseline_idx = self.seed_candidate_idx or 0
        if optimized_idx is None:
            # Use SELECTED best candidate, not just last by index
            optimized_idx = self.get_best_candidate_idx()

        # Try candidate_idx-based comparison first
        baseline_evals = self.metric_logger.get_evaluations_for_candidate(baseline_idx)
        optimized_evals = self.metric_logger.get_evaluations_for_candidate(optimized_idx)

        # Fallback to timestamp-based comparison if candidate_idx not available
        # (GEPA doesn't expose candidate_idx through public hooks)
        if not baseline_evals or not optimized_evals:
            all_evals = self.metric_logger.evaluations
            if not all_evals:
                return {
                    "improvements": [],
                    "regressions": [],
                    "same": [],
                    "summary": {"error": "No evaluation data available"},
                }

            # Filter to valset examples only if valset was set
            if self._valset_example_ids:
                all_evals = [
                    e for e in all_evals if e.example_id in self._valset_example_ids
                ]

            # GEPA evaluation flow:
            # - Candidates are evaluated in order: 0 (seed), 1, 2, 3, etc.
            # - Each candidate gets evaluated on the valset
            # - Evaluations sorted by timestamp correspond to candidate indices
            #
            # We need to compare seed (candidate 0) vs the SELECTED candidate.
            # The selected candidate is determined from Pareto frontier data,
            # NOT just the last candidate evaluated.

            # Group by example_id and sort by timestamp
            evals_by_example: dict[str, list] = {}
            for e in all_evals:
                if e.example_id not in evals_by_example:
                    evals_by_example[e.example_id] = []
                evals_by_example[e.example_id].append(e)

            baseline_evals = []
            optimized_evals = []

            for example_id, evals in evals_by_example.items():
                # Sort by timestamp to get chronological order (candidate 0, 1, 2, ...)
                sorted_evals = sorted(evals, key=lambda x: x.timestamp)

                # Baseline = candidate 0 (seed) = first evaluation
                if len(sorted_evals) >= 1:
                    baseline_evals.append(sorted_evals[0])

                # Optimized = BEST evaluation for this example (Pareto-optimal)
                # This represents the actual improvement achieved, regardless of
                # which candidate GEPA ultimately selected.
                #
                # Why best instead of specific candidate:
                # - GEPA selects based on aggregate score, not per-example
                # - We can't reliably map evaluations to candidates (candidate_idx=None)
                # - The Pareto frontier IS the set of best per-example scores
                # - Comparing seed vs best shows the actual optimization gain
                if len(sorted_evals) >= 1:
                    best_eval = max(sorted_evals, key=lambda x: x.score)
                    optimized_evals.append(best_eval)

        # Build lookup by example_id
        baseline_by_example = {e.example_id: e for e in baseline_evals}
        optimized_by_example = {e.example_id: e for e in optimized_evals}

        improvements = []
        regressions = []
        same = []

        # Find common examples
        all_examples = set(baseline_by_example.keys()) | set(optimized_by_example.keys())

        for example_id in sorted(all_examples):
            baseline_eval = baseline_by_example.get(example_id)
            optimized_eval = optimized_by_example.get(example_id)

            if not baseline_eval or not optimized_eval:
                continue  # Skip examples not evaluated by both

            delta = optimized_eval.score - baseline_eval.score

            entry = {
                "example_id": example_id,
                "inputs": baseline_eval.example_inputs or optimized_eval.example_inputs,
                "baseline_score": baseline_eval.score,
                "baseline_output": baseline_eval.prediction_preview,
                "baseline_feedback": baseline_eval.feedback,
                "optimized_score": optimized_eval.score,
                "optimized_output": optimized_eval.prediction_preview,
                "optimized_feedback": optimized_eval.feedback,
                "delta": delta,
            }

            if delta > 0.01:  # Small threshold to avoid float comparison issues
                improvements.append(entry)
            elif delta < -0.01:
                regressions.append(entry)
            else:
                same.append(entry)

        # Sort by delta magnitude
        improvements.sort(key=lambda x: x["delta"], reverse=True)
        regressions.sort(key=lambda x: x["delta"])

        # Compute summary stats
        total = len(improvements) + len(regressions) + len(same)
        baseline_avg = (
            sum(e["baseline_score"] for e in improvements + regressions + same) / total
            if total > 0
            else 0
        )
        optimized_avg = (
            sum(e["optimized_score"] for e in improvements + regressions + same) / total
            if total > 0
            else 0
        )

        return {
            "improvements": improvements,
            "regressions": regressions,
            "same": same,
            "summary": {
                "total_examples": total,
                "num_improvements": len(improvements),
                "num_regressions": len(regressions),
                "num_same": len(same),
                "baseline_avg_score": baseline_avg,
                "optimized_avg_score": optimized_avg,
                "avg_lift": optimized_avg - baseline_avg,
                "baseline_idx": baseline_idx,
                "optimized_idx": optimized_idx,
            },
        }

    # ==================== Visualization ====================

    def print_prompt_diff(
        self,
        from_idx: int | None = None,
        to_idx: int | None = None,
        show_full: bool = False,
        max_width: int = 80,
    ) -> None:
        """Print a formatted diff between original and optimized prompts.

        Args:
            from_idx: Source candidate index (default: seed)
            to_idx: Target candidate index (default: best/last)
            show_full: Show full prompt text (default: truncated)
            max_width: Max width for each column
        """
        if from_idx is None:
            from_idx = self.seed_candidate_idx or 0
        if to_idx is None:
            to_idx = len(self.final_candidates) - 1 if self.final_candidates else 0

        if from_idx == to_idx:
            print("No changes - same candidate")
            return

        diff = self.get_candidate_diff(from_idx, to_idx)
        lineage_str = " → ".join(str(i) for i in reversed(diff.lineage))

        print("\n" + "=" * 70)
        print("PROMPT COMPARISON")
        print("=" * 70)
        print(f"From: Candidate {from_idx} (seed)" if from_idx == self.seed_candidate_idx else f"From: Candidate {from_idx}")
        print(f"To:   Candidate {to_idx} (optimized)")
        print(f"Lineage: {lineage_str}")
        print("=" * 70)

        if not diff.prompt_changes:
            print("\nNo prompt changes detected.")
            return

        for key, (old_val, new_val) in diff.prompt_changes.items():
            print(f"\n📝 {key}")
            print("-" * 70)

            # Format old value
            print("\n❌ ORIGINAL:")
            self._print_wrapped(old_val, max_width, show_full)

            # Format new value
            print("\n✅ OPTIMIZED:")
            self._print_wrapped(new_val, max_width, show_full)

        print("\n" + "=" * 70)

    def _print_wrapped(self, text: str, max_width: int, show_full: bool) -> None:
        """Print text with word wrapping."""
        if not text:
            print("   (empty)")
            return

        # Truncate if needed
        if not show_full and len(text) > 500:
            text = text[:500] + "...\n   [truncated - use show_full=True to see all]"

        # Word wrap
        import textwrap
        wrapped = textwrap.fill(text, width=max_width, initial_indent="   ", subsequent_indent="   ")
        print(wrapped)

    def print_summary(self) -> None:
        """Print a formatted summary of the optimization run."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("GEPA OPTIMIZATION SUMMARY")
        print("=" * 60)

        # State summary
        state = summary.get("state", {})
        print(f"\n📊 State:")
        print(f"   Iterations:     {state.get('total_iterations', 0)}")
        print(f"   Candidates:     {state.get('total_candidates', 0)}")
        print(f"   Evaluations:    {state.get('total_evaluations', 0)}")
        duration = state.get('duration_seconds')
        if duration:
            print(f"   Duration:       {duration:.1f}s")

        # LM calls
        lm = summary.get("lm_calls", {})
        if lm:
            print(f"\n🤖 LM Calls:")
            print(f"   Total:          {lm.get('total_calls', 0)}")
            print(f"   Duration:       {lm.get('total_duration_ms', 0):.0f}ms")
            phases = lm.get('calls_by_phase', {})
            if phases:
                phase_str = ", ".join(f"{k}: {v}" for k, v in phases.items())
                print(f"   By Phase:       {phase_str}")

        # Evaluations
        evals = summary.get("evaluations", {})
        if evals:
            print(f"\n📋 Evaluations:")
            print(f"   Total:          {evals.get('total_evaluations', 0)}")
            print(f"   Unique Examples: {evals.get('unique_examples', 0)}")

        # Candidates
        if self.final_candidates:
            print(f"\n🎯 Candidates:")
            print(f"   Seed (idx {self.seed_candidate_idx}):")
            if self.seed_candidate:
                for key, val in self.seed_candidate.items():
                    truncated = val[:60] + "..." if len(val) > 60 else val
                    print(f"      {key}: {truncated}")

            best_idx = len(self.final_candidates) - 1
            if best_idx != self.seed_candidate_idx:
                print(f"   Best (idx {best_idx}):")
                for key, val in self.final_candidates[best_idx].items():
                    truncated = val[:60] + "..." if len(val) > 60 else val
                    print(f"      {key}: {truncated}")

        print("\n" + "=" * 60)

    def get_optimization_report(self) -> dict[str, Any]:
        """Get a structured optimization report.

        Returns:
            Dict with:
            - summary: Overall statistics
            - seed_prompt: Original prompt(s)
            - optimized_prompt: Best prompt(s)
            - prompt_changes: What changed
            - lineage: Evolution path
        """
        seed_idx = self.seed_candidate_idx or 0
        best_idx = len(self.final_candidates) - 1 if self.final_candidates else 0

        diff = self.get_candidate_diff(seed_idx, best_idx)

        return {
            "summary": self.get_summary(),
            "seed_prompt": self.seed_candidate,
            "seed_idx": seed_idx,
            "optimized_prompt": self.final_candidates[best_idx] if self.final_candidates else None,
            "optimized_idx": best_idx,
            "prompt_changes": diff.prompt_changes,
            "lineage": diff.lineage,
            "total_candidates": len(self.final_candidates),
        }

    def export_html(
        self,
        output_path: str | None = None,
        title: str = "GEPA Optimization Report",
    ) -> str:
        """Generate an HTML report of the optimization run.

        Works in both CLI and Jupyter notebooks:
        - If output_path is provided, writes to file and returns the path
        - If output_path is None, returns HTML string (for notebook display)

        For Jupyter notebooks, use:
            from IPython.display import HTML, display
            display(HTML(tracker.export_html()))

        Args:
            output_path: Optional path to write HTML file
            title: Report title

        Returns:
            HTML string if output_path is None, otherwise the output path
        """
        import html
        from datetime import datetime

        report = self.get_optimization_report()
        summary = report["summary"]
        state = summary.get("state", {})
        lm_summary = summary.get("lm_calls", {})
        eval_summary = summary.get("evaluations", {})

        # Build candidate cards
        candidates_html = ""
        for idx, candidate in enumerate(self.final_candidates):
            is_seed = idx == report["seed_idx"]
            is_best = idx == report["optimized_idx"]
            badge = ""
            if is_seed:
                badge = '<span class="badge badge-seed">SEED</span>'
            if is_best:
                badge += '<span class="badge badge-best">BEST</span>'

            prompts_html = ""
            for key, val in candidate.items():
                escaped_val = html.escape(val).replace("\n", "<br>")
                prompts_html += f"""
                <div class="prompt-section">
                    <div class="prompt-key">{html.escape(key)}</div>
                    <div class="prompt-value">{escaped_val}</div>
                </div>
                """

            candidates_html += f"""
            <div class="candidate-card" id="candidate-{idx}">
                <div class="candidate-header">
                    <span class="candidate-idx">Candidate {idx}</span>
                    {badge}
                </div>
                <div class="candidate-body">
                    {prompts_html}
                </div>
            </div>
            """

        # Build lineage visualization
        lineage = report.get("lineage", [])
        lineage_html = " → ".join(
            f'<a href="#candidate-{i}" class="lineage-link">{i}</a>'
            for i in reversed(lineage)
        )

        # Build prompt diff
        diff_html = ""
        for key, (old_val, new_val) in report.get("prompt_changes", {}).items():
            escaped_old = html.escape(old_val).replace("\n", "<br>")
            escaped_new = html.escape(new_val).replace("\n", "<br>")
            diff_html += f"""
            <div class="diff-section">
                <h4>{html.escape(key)}</h4>
                <div class="diff-container">
                    <div class="diff-panel diff-old">
                        <div class="diff-label">❌ Original (Candidate {report['seed_idx']})</div>
                        <div class="diff-content">{escaped_old}</div>
                    </div>
                    <div class="diff-panel diff-new">
                        <div class="diff-label">✅ Optimized (Candidate {report['optimized_idx']})</div>
                        <div class="diff-content">{escaped_new}</div>
                    </div>
                </div>
            </div>
            """

        # Phase breakdown
        phases = lm_summary.get("calls_by_phase", {})
        phase_rows = "".join(
            f"<tr><td>{html.escape(str(phase))}</td><td>{count}</td></tr>"
            for phase, count in phases.items()
        )

        # Get evaluation comparison
        eval_comparison = self.get_evaluation_comparison()
        eval_comp_summary = eval_comparison.get("summary", {})

        # Store all entries for modal access
        all_eval_entries: list[dict] = []

        def build_eval_table(entries: list, category: str) -> str:
            if not entries:
                return f'<p class="empty-message">No {category} examples</p>'

            # Get input field names from first entry
            input_fields = list(entries[0].get("inputs", {}).keys()) if entries else []

            header_cells = "".join(f"<th>{html.escape(f)}</th>" for f in input_fields)
            rows = ""
            for i, entry in enumerate(entries):
                # Store entry for modal access (with global index)
                global_idx = len(all_eval_entries)
                all_eval_entries.append(entry)

                inputs = entry.get("inputs", {})
                input_cells = "".join(
                    f'<td class="input-cell">{html.escape(str(inputs.get(f, "")))[:100]}</td>'
                    for f in input_fields
                )

                delta_class = "delta-positive" if entry["delta"] > 0 else "delta-negative" if entry["delta"] < 0 else ""
                delta_str = f'+{entry["delta"]:.2f}' if entry["delta"] > 0 else f'{entry["delta"]:.2f}'

                rows += f"""
                <tr onclick="openModal({global_idx})" title="Click for details">
                    <td class="row-num">{i + 1}</td>
                    {input_cells}
                    <td class="score-cell">{entry["baseline_score"]:.2f}</td>
                    <td class="output-cell">{html.escape(str(entry.get("baseline_output", ""))[:150])}</td>
                    <td class="score-cell">{entry["optimized_score"]:.2f}</td>
                    <td class="output-cell">{html.escape(str(entry.get("optimized_output", ""))[:150])}</td>
                    <td class="delta-cell {delta_class}">{delta_str}</td>
                </tr>
                """

            return f"""
            <table class="eval-table">
                <thead>
                    <tr>
                        <th>#</th>
                        {header_cells}
                        <th>Base Score</th>
                        <th>Baseline Output</th>
                        <th>Opt Score</th>
                        <th>Optimized Output</th>
                        <th>Delta</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            """

        improvements_table = build_eval_table(eval_comparison.get("improvements", []), "improvement")
        regressions_table = build_eval_table(eval_comparison.get("regressions", []), "regression")
        same_table = build_eval_table(eval_comparison.get("same", []), "unchanged")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
            --accent-orange: #d29922;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }}

        h2 {{
            font-size: 1.5rem;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        h3 {{
            font-size: 1.25rem;
            margin: 1.5rem 0 1rem;
            color: var(--text-primary);
        }}

        h4 {{
            font-size: 1rem;
            margin: 1rem 0 0.5rem;
            color: var(--accent-blue);
        }}

        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-bottom: 2rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}

        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--accent-blue);
        }}

        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        .lineage-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 1.25rem;
            text-align: center;
        }}

        .lineage-link {{
            color: var(--accent-purple);
            text-decoration: none;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: background 0.2s;
        }}

        .lineage-link:hover {{
            background: var(--bg-tertiary);
        }}

        .diff-section {{
            margin: 1.5rem 0;
        }}

        .diff-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}

        .diff-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}

        .diff-label {{
            padding: 0.75rem 1rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
        }}

        .diff-old .diff-label {{
            background: rgba(248, 81, 73, 0.1);
            color: var(--accent-red);
        }}

        .diff-new .diff-label {{
            background: rgba(63, 185, 80, 0.1);
            color: var(--accent-green);
        }}

        .diff-content {{
            padding: 1rem;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
        }}

        .candidate-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
        }}

        .candidate-header {{
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .candidate-idx {{
            font-weight: 600;
        }}

        .badge {{
            font-size: 0.75rem;
            padding: 0.125rem 0.5rem;
            border-radius: 12px;
            font-weight: 600;
        }}

        .badge-seed {{
            background: var(--accent-orange);
            color: var(--bg-primary);
        }}

        .badge-best {{
            background: var(--accent-green);
            color: var(--bg-primary);
        }}

        .candidate-body {{
            padding: 1rem;
        }}

        .prompt-section {{
            margin: 0.5rem 0;
        }}

        .prompt-key {{
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 0.25rem;
        }}

        .prompt-value {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.875rem;
            background: var(--bg-tertiary);
            padding: 0.75rem;
            border-radius: 6px;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}

        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--bg-tertiary);
            font-weight: 600;
        }}

        tr:hover {{
            background: var(--bg-secondary);
        }}

        .section {{
            margin: 2rem 0;
        }}

        /* Tabs */
        .tabs {{
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }}

        .tab {{
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}

        .tab:hover {{
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }}

        .tab.active {{
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
        }}

        .tab-count {{
            margin-left: 0.5rem;
            padding: 0.125rem 0.5rem;
            border-radius: 10px;
            font-size: 0.75rem;
        }}

        .tab-count.green {{
            background: rgba(63, 185, 80, 0.2);
            color: var(--accent-green);
        }}

        .tab-count.red {{
            background: rgba(248, 81, 73, 0.2);
            color: var(--accent-red);
        }}

        .tab-count.gray {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* Evaluation table */
        .eval-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}

        .eval-table th {{
            background: var(--bg-tertiary);
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
            white-space: nowrap;
        }}

        .eval-table td {{
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            vertical-align: top;
        }}

        .eval-table tr:hover {{
            background: var(--bg-secondary);
        }}

        .row-num {{
            color: var(--text-secondary);
            font-weight: 500;
            width: 40px;
        }}

        .input-cell {{
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .output-cell {{
            max-width: 250px;
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .score-cell {{
            font-weight: 600;
            width: 80px;
            text-align: center;
        }}

        .delta-cell {{
            font-weight: 600;
            width: 70px;
            text-align: center;
        }}

        .delta-positive {{
            color: var(--accent-green);
        }}

        .delta-negative {{
            color: var(--accent-red);
        }}

        .empty-message {{
            color: var(--text-secondary);
            padding: 2rem;
            text-align: center;
        }}

        .score-summary {{
            display: flex;
            gap: 2rem;
            margin-bottom: 1rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
        }}

        .score-summary-item {{
            text-align: center;
        }}

        .score-summary-value {{
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .score-summary-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        @media (max-width: 768px) {{
            .diff-container {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}

        /* Modal styles */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            overflow-y: auto;
            padding: 2rem;
        }}

        .modal-overlay.active {{
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }}

        .modal {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            max-width: 900px;
            width: 100%;
            margin: 2rem auto;
            position: relative;
            animation: modalSlideIn 0.2s ease-out;
        }}

        @keyframes modalSlideIn {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            border-radius: 12px 12px 0 0;
        }}

        .modal-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .modal-close {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: all 0.2s;
        }}

        .modal-close:hover {{
            background: var(--bg-primary);
            color: var(--text-primary);
        }}

        .modal-body {{
            padding: 1.5rem;
        }}

        .modal-section {{
            margin-bottom: 1.5rem;
        }}

        .modal-section:last-child {{
            margin-bottom: 0;
        }}

        .modal-section-title {{
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}

        .modal-score-row {{
            display: flex;
            gap: 2rem;
            margin-bottom: 1rem;
        }}

        .modal-score-item {{
            flex: 1;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            text-align: center;
        }}

        .modal-score-value {{
            font-size: 1.75rem;
            font-weight: 700;
        }}

        .modal-score-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .modal-content-box {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
        }}

        .modal-feedback {{
            background: var(--bg-tertiary);
            border-left: 3px solid var(--accent-blue);
            padding: 0.75rem 1rem;
            font-style: italic;
            color: var(--text-secondary);
            border-radius: 0 8px 8px 0;
        }}

        .modal-inputs {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}

        .modal-input-item {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }}

        .modal-input-key {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }}

        .modal-input-value {{
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 0.875rem;
            word-break: break-word;
        }}

        .modal-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}

        .modal-comparison-panel {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
        }}

        .modal-comparison-header {{
            padding: 0.75rem 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }}

        .modal-comparison-header.baseline {{
            background: rgba(248, 81, 73, 0.1);
            color: var(--accent-red);
        }}

        .modal-comparison-header.optimized {{
            background: rgba(63, 185, 80, 0.1);
            color: var(--accent-green);
        }}

        .modal-comparison-content {{
            padding: 1rem;
        }}

        .modal-comparison-output {{
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-break: break-word;
            margin-bottom: 1rem;
            max-height: 200px;
            overflow-y: auto;
        }}

        .modal-comparison-feedback {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-style: italic;
            border-top: 1px solid var(--border-color);
            padding-top: 0.75rem;
        }}

        .eval-table tbody tr {{
            cursor: pointer;
            transition: background 0.15s;
        }}

        .eval-table tbody tr:hover {{
            background: var(--bg-tertiary) !important;
        }}

        @media (max-width: 768px) {{
            .modal-comparison {{
                grid-template-columns: 1fr;
            }}

            .modal-score-row {{
                flex-wrap: wrap;
            }}

            .modal-score-item {{
                min-width: 120px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 {html.escape(title)}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

        <div class="section">
            <h2>📊 Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{state.get('total_iterations', 0)}</div>
                    <div class="stat-label">Iterations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{state.get('total_candidates', 0)}</div>
                    <div class="stat-label">Candidates</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lm_summary.get('total_calls', 0)}</div>
                    <div class="stat-label">LM Calls</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{eval_summary.get('total_evaluations', 0)}</div>
                    <div class="stat-label">Evaluations</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🧬 Lineage</h2>
            <div class="lineage-box">
                {lineage_html if lineage_html else "No lineage data"}
            </div>
        </div>

        <div class="section">
            <h2>🔄 Prompt Changes</h2>
            {diff_html if diff_html else "<p>No prompt changes detected.</p>"}
        </div>

        <div class="section">
            <h2>📈 Performance Comparison</h2>
            <div class="score-summary">
                <div class="score-summary-item">
                    <div class="score-summary-value">{eval_comp_summary.get('baseline_avg_score', 0):.2f}</div>
                    <div class="score-summary-label">Baseline Avg</div>
                </div>
                <div class="score-summary-item">
                    <div class="score-summary-value" style="color: var(--accent-green);">{eval_comp_summary.get('optimized_avg_score', 0):.2f}</div>
                    <div class="score-summary-label">Optimized Avg</div>
                </div>
                <div class="score-summary-item">
                    <div class="score-summary-value" style="color: {'var(--accent-green)' if eval_comp_summary.get('avg_lift', 0) >= 0 else 'var(--accent-red)'};">{'+' if eval_comp_summary.get('avg_lift', 0) >= 0 else ''}{eval_comp_summary.get('avg_lift', 0):.2f}</div>
                    <div class="score-summary-label">Avg Lift</div>
                </div>
                <div class="score-summary-item">
                    <div class="score-summary-value">{eval_comp_summary.get('total_examples', 0)}</div>
                    <div class="score-summary-label">Examples</div>
                </div>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="showTab('improvements')">
                    Improvements <span class="tab-count green">{eval_comp_summary.get('num_improvements', 0)}</span>
                </button>
                <button class="tab" onclick="showTab('regressions')">
                    Regressions <span class="tab-count red">{eval_comp_summary.get('num_regressions', 0)}</span>
                </button>
                <button class="tab" onclick="showTab('same')">
                    Same <span class="tab-count gray">{eval_comp_summary.get('num_same', 0)}</span>
                </button>
            </div>

            <div id="improvements" class="tab-content active">
                {improvements_table}
            </div>
            <div id="regressions" class="tab-content">
                {regressions_table}
            </div>
            <div id="same" class="tab-content">
                {same_table}
            </div>
        </div>

        {"<div class='section'><h2>📞 LM Calls by Phase</h2><table><thead><tr><th>Phase</th><th>Count</th></tr></thead><tbody>" + phase_rows + "</tbody></table></div>" if phase_rows else ""}

        <div class="section">
            <h2>📝 All Candidates</h2>
            {candidates_html}
        </div>
    </div>

    <!-- Modal for evaluation details -->
    <div id="evalModal" class="modal-overlay" onclick="if(event.target === this) closeModal()">
        <div class="modal">
            <div class="modal-header">
                <span class="modal-title">Evaluation Details</span>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-section">
                    <div class="modal-section-title">Scores</div>
                    <div class="modal-score-row">
                        <div class="modal-score-item">
                            <div class="modal-score-value" id="modal-baseline-score">-</div>
                            <div class="modal-score-label">Baseline</div>
                        </div>
                        <div class="modal-score-item">
                            <div class="modal-score-value" id="modal-optimized-score" style="color: var(--accent-green);">-</div>
                            <div class="modal-score-label">Optimized</div>
                        </div>
                        <div class="modal-score-item">
                            <div class="modal-score-value" id="modal-delta">-</div>
                            <div class="modal-score-label">Delta</div>
                        </div>
                    </div>
                </div>

                <div class="modal-section">
                    <div class="modal-section-title">Inputs</div>
                    <div class="modal-inputs" id="modal-inputs"></div>
                </div>

                <div class="modal-section">
                    <div class="modal-section-title">Outputs & Feedback</div>
                    <div class="modal-comparison">
                        <div class="modal-comparison-panel">
                            <div class="modal-comparison-header baseline">Baseline</div>
                            <div class="modal-comparison-content">
                                <div class="modal-comparison-output" id="modal-baseline-output"></div>
                                <div class="modal-comparison-feedback" id="modal-baseline-feedback"></div>
                            </div>
                        </div>
                        <div class="modal-comparison-panel">
                            <div class="modal-comparison-header optimized">Optimized</div>
                            <div class="modal-comparison-content">
                                <div class="modal-comparison-output" id="modal-optimized-output"></div>
                                <div class="modal-comparison-feedback" id="modal-optimized-feedback"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="modal-section">
                    <div class="modal-section-title">Example ID</div>
                    <div class="modal-content-box" id="modal-example-id" style="max-height: none; font-size: 0.75rem; color: var(--text-secondary);"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Evaluation data for modal
        const evalData = {json.dumps(all_eval_entries, default=str)};

        function showTab(tabId) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            // Deactivate all tabs
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            // Show selected tab content
            document.getElementById(tabId).classList.add('active');
            // Activate clicked tab
            event.target.closest('.tab').classList.add('active');
        }}

        function openModal(index) {{
            const entry = evalData[index];
            if (!entry) return;

            // Update scores
            document.getElementById('modal-baseline-score').textContent = entry.baseline_score.toFixed(2);
            document.getElementById('modal-optimized-score').textContent = entry.optimized_score.toFixed(2);

            const delta = entry.delta;
            const deltaEl = document.getElementById('modal-delta');
            deltaEl.textContent = (delta >= 0 ? '+' : '') + delta.toFixed(2);
            deltaEl.style.color = delta > 0 ? 'var(--accent-green)' : delta < 0 ? 'var(--accent-red)' : 'var(--text-primary)';

            // Update inputs
            const inputsContainer = document.getElementById('modal-inputs');
            inputsContainer.innerHTML = '';
            const inputs = entry.inputs || {{}};
            for (const [key, value] of Object.entries(inputs)) {{
                const item = document.createElement('div');
                item.className = 'modal-input-item';
                item.innerHTML = `
                    <div class="modal-input-key">${{escapeHtml(key)}}</div>
                    <div class="modal-input-value">${{escapeHtml(String(value))}}</div>
                `;
                inputsContainer.appendChild(item);
            }}

            // Update outputs
            document.getElementById('modal-baseline-output').textContent = entry.baseline_output || '(no output)';
            document.getElementById('modal-optimized-output').textContent = entry.optimized_output || '(no output)';

            // Update feedback
            document.getElementById('modal-baseline-feedback').textContent = entry.baseline_feedback || '(no feedback)';
            document.getElementById('modal-optimized-feedback').textContent = entry.optimized_feedback || '(no feedback)';

            // Update example ID
            document.getElementById('modal-example-id').textContent = entry.example_id || '-';

            // Show modal
            document.getElementById('evalModal').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}

        function closeModal() {{
            document.getElementById('evalModal').classList.remove('active');
            document.body.style.overflow = '';
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>"""

        if output_path:
            with open(output_path, "w") as f:
                f.write(html_content)
            return output_path

        return html_content

    # ==================== Lifecycle ====================

    def clear(self) -> None:
        """Clear all captured data."""
        self._start_time = None
        self.state_logger.clear()
        if self.lm_logger is not None:
            self.lm_logger.clear()
        if self.metric_logger is not None:
            self.metric_logger.clear()
        if self.proposer_logger is not None:
            self.proposer_logger.clear()
        if self.selector_logger is not None:
            self.selector_logger.clear()
        clear_ctx()

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        total_iters = summary.get("state", {}).get("total_iterations", 0)
        total_candidates = summary.get("state", {}).get("total_candidates", 0)
        total_lm = summary.get("lm_calls", {}).get("total_calls", 0)
        total_evals = summary.get("evaluations", {}).get("total_evaluations", 0)

        return (
            f"GEPATracker(iterations={total_iters}, candidates={total_candidates}, "
            f"lm_calls={total_lm}, evaluations={total_evals})"
        )
