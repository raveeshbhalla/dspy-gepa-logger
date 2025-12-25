"""Public API for GEPA Logger v2.2.

This module provides user-friendly functions for setting up GEPA logging
using the public hooks architecture.

Example usage:

    from dspy_gepa_logger import create_logged_gepa, configure_dspy_logging

    # Create GEPA with logging
    gepa, tracker, logged_metric = create_logged_gepa(
        metric=my_metric,
        num_candidates=3,
        num_iterations=5,
        # Any other GEPA kwargs...
    )

    # Configure DSPy with LM logging
    configure_dspy_logging(tracker)

    # Run optimization
    result = gepa.compile(student, trainset=train, valset=val)

    # Access captured data
    print(tracker.get_summary())
    for eval in tracker.evaluations:
        print(f"{eval.example_id}: {eval.score}")
"""

import logging
from typing import Any, Callable

from .core.tracker_v2 import GEPATracker, LoggedLM
from .core.logged_metric import LoggedMetric
from .core.logged_proposer import LoggedInstructionProposer, LoggedSelector

logger = logging.getLogger(__name__)


def create_logged_gepa(
    metric: Callable[..., Any],
    *,
    capture_lm_calls: bool = True,
    capture_prediction: bool = True,
    capture_stdout: bool = True,
    failure_score: float = 0.0,
    wrap_proposer: bool = True,
    base_proposer: Any | None = None,
    wrap_selector: bool = False,
    base_selector: Any | None = None,
    gepa_kwargs: dict[str, Any] | None = None,
    server_url: str | None = None,
    project_name: str = "Default",
    **kwargs: Any,
) -> tuple[Any, GEPATracker, LoggedMetric]:
    """Create a GEPA optimizer with logging configured.

    This is the main entry point for using GEPA with comprehensive logging.
    It returns a configured GEPA instance along with the tracker and logged metric.

    Args:
        metric: The metric function for evaluation
        capture_lm_calls: Whether to capture LM calls (default: True)
        capture_prediction: Whether to capture predictions (default: True)
        capture_stdout: Whether to capture stdout/stderr when server_url is set (default: True)
        failure_score: Score to return when metric throws exception (default: 0.0).
            This is synchronized with GEPA's failure_score parameter.
        wrap_proposer: Whether to wrap the instruction proposer (default: True).
            Enables capturing reflection/proposal LM calls with proper phase tags.
        base_proposer: Custom instruction proposer to wrap (if wrap_proposer=True)
        wrap_selector: Whether to wrap the selector (default: False)
        base_selector: Custom selector to wrap (if wrap_selector=True)
        gepa_kwargs: Additional kwargs for GEPA's gepa_kwargs parameter
        server_url: Optional URL of GEPA Logger web server for real-time updates.
            If provided, run data will be pushed to the server for visualization.
        project_name: Project name for organizing runs (default: "Default").
            Only used when server_url is provided.
        **kwargs: Additional kwargs passed to GEPA constructor

    Returns:
        Tuple of (gepa, tracker, logged_metric):
        - gepa: Configured GEPA optimizer instance
        - tracker: GEPATracker with all logging hooks
        - logged_metric: The wrapped metric (same as passed to GEPA)

    Example (in-memory only):
        gepa, tracker, logged_metric = create_logged_gepa(
            metric=my_metric,
            num_candidates=3,
            num_iterations=5,
        )

        configure_dspy_logging(tracker)
        result = gepa.compile(student, trainset=train, valset=val)
        tracker.export_html("report.html")

    Example (with server):
        gepa, tracker, logged_metric = create_logged_gepa(
            metric=my_metric,
            server_url="http://localhost:3000",
            project_name="My Experiment",
        )

        configure_dspy_logging(tracker)
        result = gepa.compile(student, trainset=train, valset=val)
        tracker.finalize()  # Push remaining data and mark complete
    """
    try:
        from dspy.teleprompt import GEPA
    except ImportError:
        raise ImportError(
            "DSPy is required to use create_logged_gepa. "
            "Install it with: pip install dspy"
        )

    # Extract failure_score from kwargs if provided (for GEPA sync)
    # If not in kwargs, use the explicit parameter
    gepa_failure_score = kwargs.get("failure_score", failure_score)

    # Create tracker with optional server integration
    tracker = GEPATracker(
        capture_lm_calls=capture_lm_calls,
        server_url=server_url,
        project_name=project_name,
        auto_capture_stdout=capture_stdout and server_url is not None,
    )

    # Wrap metric with synchronized failure_score
    logged_metric = tracker.wrap_metric(
        metric,
        capture_prediction=capture_prediction,
        failure_score=gepa_failure_score,
    )

    # Ensure GEPA uses the same failure_score
    if "failure_score" not in kwargs:
        kwargs["failure_score"] = gepa_failure_score

    # Build gepa_kwargs with stop_callback
    final_gepa_kwargs = dict(gepa_kwargs or {})

    # Merge stop_callbacks without double-nesting
    existing_stop_callbacks = final_gepa_kwargs.get("stop_callbacks", [])
    if not isinstance(existing_stop_callbacks, list):
        existing_stop_callbacks = [existing_stop_callbacks]
    final_gepa_kwargs["stop_callbacks"] = existing_stop_callbacks + [
        tracker.get_stop_callback()
    ]

    # Handle proposer wrapping
    # Note: We only wrap if a base_proposer is provided, since GEPA's internal
    # default proposer is not easily accessible in newer versions
    if wrap_proposer and base_proposer is not None:
        kwargs["instruction_proposer"] = tracker.wrap_proposer(base_proposer)
    elif wrap_proposer and base_proposer is None:
        logger.debug(
            "wrap_proposer=True but no base_proposer provided; proposer wrapping skipped. "
            "To wrap the proposer, pass base_proposer=YourProposer()."
        )

    # Handle selector wrapping
    # Note: We only wrap if a base_selector is provided
    if wrap_selector and base_selector is not None:
        kwargs["selector"] = tracker.wrap_selector(base_selector)

    # Wrap reflection_lm to tag its calls with phase="reflection"
    # This enables proper phase attribution without requiring proposer wrapping
    if "reflection_lm" in kwargs and kwargs["reflection_lm"] is not None:
        kwargs["reflection_lm"] = LoggedLM(
            kwargs["reflection_lm"],
            phase="reflection",
        )

    # Create GEPA with merged kwargs
    gepa = GEPA(
        metric=logged_metric,
        gepa_kwargs=final_gepa_kwargs,
        **kwargs,
    )

    return gepa, tracker, logged_metric


def configure_dspy_logging(tracker: GEPATracker) -> None:
    """Configure DSPy to use the tracker's LM callbacks.

    This registers the tracker's LM logger with DSPy so all LM calls
    are captured with context tags (iteration, phase, candidate_idx).

    Must be called before running GEPA optimization.

    Args:
        tracker: The GEPATracker instance

    Example:
        tracker = GEPATracker()
        configure_dspy_logging(tracker)

        # Now all LM calls will be captured
        gepa.compile(...)
    """
    try:
        import dspy
    except ImportError:
        raise ImportError(
            "DSPy is required to use configure_dspy_logging. "
            "Install it with: pip install dspy"
        )

    callbacks = tracker.get_dspy_callbacks()
    if callbacks:
        existing = dspy.settings.get("callbacks", [])
        dspy.configure(callbacks=existing + callbacks)


def create_tracker(
    capture_lm_calls: bool = True,
    dspy_version: str | None = None,
    gepa_version: str | None = None,
) -> GEPATracker:
    """Create a standalone GEPATracker.

    Use this when you want more control over how the tracker is configured.

    Args:
        capture_lm_calls: Whether to capture LM calls (default: True)
        dspy_version: DSPy version (for compatibility info)
        gepa_version: GEPA version (for compatibility info)

    Returns:
        GEPATracker instance

    Example:
        tracker = create_tracker()

        # Manually wrap components
        logged_metric = tracker.wrap_metric(my_metric)

        # Set up GEPA manually
        gepa = GEPA(
            metric=logged_metric,
            gepa_kwargs={'stop_callbacks': [tracker.get_stop_callback()]},
        )

        configure_dspy_logging(tracker)
    """
    return GEPATracker(
        capture_lm_calls=capture_lm_calls,
        dspy_version=dspy_version,
        gepa_version=gepa_version,
    )


def wrap_metric(
    metric: Callable[..., Any],
    capture_prediction: bool = True,
    max_prediction_preview: int = 200,
    failure_score: float = 0.0,
) -> LoggedMetric:
    """Create a standalone LoggedMetric wrapper.

    Use this when you want to wrap a metric without using the full tracker.

    Args:
        metric: The metric function to wrap
        capture_prediction: Whether to capture predictions (default: True)
        max_prediction_preview: Max length for prediction preview
        failure_score: Score to return when metric throws exception (default: 0.0)

    Returns:
        LoggedMetric wrapper

    Example:
        logged_metric = wrap_metric(my_metric)
        result = logged_metric(example, prediction)
        print(f"Score: {logged_metric.evaluations[-1].score}")
    """
    return LoggedMetric(
        metric_fn=metric,
        capture_prediction=capture_prediction,
        max_prediction_preview=max_prediction_preview,
        failure_score=failure_score,
    )


def wrap_proposer(proposer: Any) -> LoggedInstructionProposer:
    """Create a standalone LoggedInstructionProposer wrapper.

    Use this when you want to wrap a proposer without using the full tracker.

    Args:
        proposer: The instruction proposer to wrap

    Returns:
        LoggedInstructionProposer wrapper
    """
    return LoggedInstructionProposer(proposer)


def wrap_selector(selector: Any) -> LoggedSelector:
    """Create a standalone LoggedSelector wrapper.

    Use this when you want to wrap a selector without using the full tracker.

    Args:
        selector: The selector to wrap

    Returns:
        LoggedSelector wrapper
    """
    return LoggedSelector(selector)
