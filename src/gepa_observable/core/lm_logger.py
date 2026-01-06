"""Captures all LM calls via DSPy callbacks with context tagging.

DSPyLMLogger is a DSPy callback that captures all LM calls and tags them with
the current context (iteration, phase, candidate_idx) from the shared contextvar.

This is more robust than timestamp-based correlation because:
1. Works with num_threads > 1 (each thread has its own context)
2. No timing drift issues
3. Direct tagging instead of inference
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dspy.utils.callback import BaseCallback

from .context import get_ctx


@dataclass
class LMCall:
    """Record of a single LM call with context tags.

    Context tags are captured from the contextvar at call start time,
    so they reflect the state at the moment the call was made.
    """

    call_id: str
    start_time: float  # Relative to logger start
    end_time: float | None = None
    duration_ms: float = 0.0
    model: str = ""

    # Request data
    inputs: dict[str, Any] = field(default_factory=dict)

    # Response data
    outputs: dict[str, Any] = field(default_factory=dict)

    # Context tags (from contextvar, NOT timestamps!)
    iteration: int | None = None
    phase: str | None = None  # 'reflection', 'proposal', 'eval', 'validation'
    candidate_idx: int | None = None

    # Optional additional context
    component_name: str | None = None  # Which DSPy component made the call


class DSPyLMLogger(BaseCallback):
    """DSPy callback that captures all LM calls WITH context tags.

    Inherits from BaseCallback to be properly recognized by DSPy's callback system.

    Instead of correlating by timestamp (fragile with threads),
    we read the current context set by the state logger / metric wrapper.

    Usage:
        lm_logger = DSPyLMLogger()

        # Register with DSPy
        import dspy
        dspy.configure(callbacks=[lm_logger])

        # Or pass to LM directly
        lm = dspy.LM("openai/gpt-4o-mini", callbacks=[lm_logger])

        # After optimization:
        for call in lm_logger.calls:
            print(f"Iteration {call.iteration}, phase={call.phase}: {call.model}")
    """

    def __init__(
        self,
        on_call_complete: "Callable[[LMCall], None] | None" = None,
    ):
        """Initialize the LM logger.

        Args:
            on_call_complete: Optional callback called when each LM call completes.
                             Used for real-time streaming to server.
        """
        super().__init__()  # Initialize BaseCallback
        self.calls: list[LMCall] = []
        self._pending: dict[str, LMCall] = {}
        self._start_time: float | None = None
        self._call_counter: int = 0
        self._on_call_complete = on_call_complete

    def on_lm_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        """Called by DSPy when an LM call starts.

        Args:
            call_id: Unique identifier for this call
            instance: The LM instance making the call
            inputs: Input parameters for the call
        """
        if self._start_time is None:
            self._start_time = time.time()

        # Read current context (set by state logger, proposer, or metric wrapper)
        ctx = get_ctx()

        # Extract model name safely
        model_name = ""
        if hasattr(instance, "model"):
            model_name = str(instance.model)
        elif hasattr(instance, "__class__"):
            model_name = instance.__class__.__name__

        call = LMCall(
            call_id=call_id,
            start_time=time.time() - self._start_time,
            model=model_name,
            inputs=inputs,
            # Tag with context!
            iteration=ctx.get("iteration"),
            phase=ctx.get("phase"),
            candidate_idx=ctx.get("candidate_idx"),
            component_name=ctx.get("component_name"),
        )
        self._pending[call_id] = call
        self._call_counter += 1

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Called by DSPy when an LM call completes.

        Note: DSPy 3.0+ does NOT pass instance to on_lm_end (only on_lm_start has it).

        Args:
            call_id: Unique identifier for this call
            outputs: Output from the call (may be None if exception)
            exception: Optional exception if the call failed
        """
        if call_id not in self._pending:
            return

        call = self._pending.pop(call_id)
        current_time = time.time()
        call.end_time = current_time - (self._start_time or current_time)
        call.duration_ms = (call.end_time - call.start_time) * 1000

        if exception is not None:
            call.outputs = {
                "error": str(exception),
                "error_type": type(exception).__name__,
            }
        else:
            call.outputs = outputs or {}

        self.calls.append(call)

        # Call the callback for real-time streaming
        if self._on_call_complete is not None:
            try:
                self._on_call_complete(call)
            except Exception:
                pass  # Don't let callback errors affect normal operation

    def on_lm_error(
        self, call_id: str, instance: Any, error: Exception
    ) -> None:
        """Called by DSPy when an LM call fails (legacy method).

        Args:
            call_id: Unique identifier for this call
            instance: The LM instance that made the call
            error: The exception that was raised
        """
        if call_id not in self._pending:
            return

        call = self._pending.pop(call_id)
        current_time = time.time()
        call.end_time = current_time - (self._start_time or current_time)
        call.duration_ms = (call.end_time - call.start_time) * 1000
        call.outputs = {"error": str(error), "error_type": type(error).__name__}
        self.calls.append(call)

        # Call the callback for real-time streaming
        if self._on_call_complete is not None:
            try:
                self._on_call_complete(call)
            except Exception:
                pass  # Don't let callback errors affect normal operation

    # Query methods

    def get_calls_for_iteration(self, iteration: int) -> list[LMCall]:
        """Get all LM calls for a specific iteration.

        Uses context tags (not timestamps!) for filtering.

        Args:
            iteration: The iteration number to filter by

        Returns:
            List of LM calls for that iteration
        """
        return [c for c in self.calls if c.iteration == iteration]

    def get_calls_for_phase(self, phase: str) -> list[LMCall]:
        """Get all LM calls for a specific phase.

        Args:
            phase: The phase to filter by ('reflection', 'proposal', 'eval', etc.)

        Returns:
            List of LM calls for that phase
        """
        return [c for c in self.calls if c.phase == phase]

    def get_calls_for_candidate(self, candidate_idx: int) -> list[LMCall]:
        """Get all LM calls related to a specific candidate.

        Args:
            candidate_idx: The candidate index to filter by

        Returns:
            List of LM calls for that candidate
        """
        return [c for c in self.calls if c.candidate_idx == candidate_idx]

    def get_reflection_calls(self) -> list[LMCall]:
        """Get all reflection LM calls."""
        return self.get_calls_for_phase("reflection")

    def get_proposal_calls(self) -> list[LMCall]:
        """Get all proposal generation LM calls."""
        return self.get_calls_for_phase("proposal")

    def get_eval_calls(self) -> list[LMCall]:
        """Get all evaluation LM calls.

        Includes calls with phase 'eval', 'validation', or 'minibatch'.
        """
        return [
            c for c in self.calls
            if c.phase in ("eval", "validation", "minibatch")
        ]

    def get_untagged_calls(self) -> list[LMCall]:
        """Get LM calls that weren't tagged with a phase.

        Useful for debugging context propagation issues.
        """
        return [c for c in self.calls if c.phase is None]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all LM calls.

        Returns:
            Dict with total calls, calls by phase, duration stats, etc.
        """
        total = len(self.calls)
        if total == 0:
            return {
                "total_calls": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "calls_by_phase": {},
                "calls_by_iteration": {},
                "pending_calls": len(self._pending),
            }

        # Group by phase
        calls_by_phase: dict[str, int] = {}
        for call in self.calls:
            phase = call.phase or "untagged"
            calls_by_phase[phase] = calls_by_phase.get(phase, 0) + 1

        # Group by iteration
        calls_by_iteration: dict[int, int] = {}
        for call in self.calls:
            if call.iteration is not None:
                calls_by_iteration[call.iteration] = (
                    calls_by_iteration.get(call.iteration, 0) + 1
                )

        # Duration stats
        total_duration = sum(c.duration_ms for c in self.calls)
        avg_duration = total_duration / total

        return {
            "total_calls": total,
            "total_duration_ms": total_duration,
            "avg_duration_ms": avg_duration,
            "calls_by_phase": calls_by_phase,
            "calls_by_iteration": calls_by_iteration,
            "pending_calls": len(self._pending),
        }

    def clear(self) -> None:
        """Clear all recorded calls and reset state."""
        self.calls = []
        self._pending = {}
        self._start_time = None
        self._call_counter = 0

    def __len__(self) -> int:
        """Return number of completed LM calls."""
        return len(self.calls)

    def __bool__(self) -> bool:
        """Return True since logger is always valid, regardless of call count."""
        return True
