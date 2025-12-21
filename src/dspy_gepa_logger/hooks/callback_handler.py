"""DSPy callback handler for capturing LM calls during GEPA optimization."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dspy_gepa_logger.core.tracker import GEPARunTracker

try:
    from dspy.utils.callback import BaseCallback
except ImportError:
    # Define a stub if dspy is not installed
    class BaseCallback:
        """Stub BaseCallback for when dspy is not installed."""

        def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
            pass

        def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
            pass


class GEPALoggingCallback(BaseCallback):
    """DSPy callback that captures LM calls during GEPA optimization.

    This callback integrates with DSPy's callback system to capture:
    - All LM calls made during evaluation
    - Reflection LM calls (marked specially via context)
    - Timing and token usage information

    Usage:
        tracker = GEPARunTracker(storage)
        callback = GEPALoggingCallback(tracker)
        dspy.configure(callbacks=[callback])
    """

    def __init__(self, tracker: "GEPARunTracker"):
        """Initialize the callback.

        Args:
            tracker: The GEPA run tracker to send LM call data to
        """
        self.tracker = tracker

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Handle LM call start.

        Args:
            call_id: Unique identifier for the call
            instance: The LM instance
            inputs: Call inputs
        """
        model = getattr(instance, "model", "unknown")
        self.tracker.start_lm_call(call_id, model, inputs)

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Handle LM call end.

        Args:
            call_id: Unique identifier for the call
            outputs: Call outputs (None if exception)
            exception: Exception if call failed
        """
        self.tracker.end_lm_call(call_id, outputs, exception)
