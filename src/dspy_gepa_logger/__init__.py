"""GEPA Run Tracking Library for DSPy.

This library provides comprehensive logging and visualization infrastructure
for DSPy GEPA optimization runs.

v2.2 usage (recommended - public hooks, no monkey-patching):
    from dspy_gepa_logger import create_logged_gepa, configure_dspy_logging

    gepa, tracker, logged_metric = create_logged_gepa(metric=my_metric)
    configure_dspy_logging(tracker)

    result = gepa.compile(student=MyProgram(), trainset=train, valset=val)
    print(tracker.get_summary())

v1 usage (legacy):
    from dspy_gepa_logger import track_gepa_run

    tracker = track_gepa_run(log_dir="./gepa_logs")

    with tracker.track():
        optimized = gepa.compile(student=MyProgram(), trainset=train, valset=val)

    # Export results
    tracker.export_summary_json("summary.json")
"""

# v1 (legacy) exports
from dspy_gepa_logger.core.tracker import GEPARunTracker
from dspy_gepa_logger.core.config import TrackerConfig
from dspy_gepa_logger.storage import JSONLStorageBackend, MemoryStorageBackend
from dspy_gepa_logger.export import VisualizationExporter, DataFrameExporter
from dspy_gepa_logger.hooks.gepa_adapter import create_instrumented_gepa, cleanup_instrumented_gepa

# v2.2 exports (public hooks architecture)
from dspy_gepa_logger.api import (
    create_logged_gepa,
    configure_dspy_logging,
    create_tracker,
    wrap_metric,
    wrap_proposer,
    wrap_selector,
)
from dspy_gepa_logger.core.tracker_v2 import GEPATracker, CandidateDiff
from dspy_gepa_logger.core.logged_metric import LoggedMetric, EvaluationRecord
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger, LMCall
from dspy_gepa_logger.core.state_logger import GEPAStateLogger, IterationDelta, IterationMetadata
from dspy_gepa_logger.core.logged_proposer import (
    LoggedInstructionProposer,
    LoggedSelector,
    ReflectionCall,
    ProposalCall,
)
from dspy_gepa_logger.core.context import set_ctx, get_ctx, clear_ctx, with_ctx

__version__ = "0.1.0"

__all__ = [
    # v1 (legacy)
    "GEPARunTracker",
    "TrackerConfig",
    "JSONLStorageBackend",
    "MemoryStorageBackend",
    "VisualizationExporter",
    "DataFrameExporter",
    "track_gepa_run",
    "create_instrumented_gepa",
    "cleanup_instrumented_gepa",
    # v2.2 API functions
    "create_logged_gepa",
    "configure_dspy_logging",
    "create_tracker",
    "wrap_metric",
    "wrap_proposer",
    "wrap_selector",
    # v2.2 tracker
    "GEPATracker",
    "CandidateDiff",
    # v2.2 components
    "LoggedMetric",
    "EvaluationRecord",
    "DSPyLMLogger",
    "LMCall",
    "GEPAStateLogger",
    "IterationDelta",
    "IterationMetadata",
    "LoggedInstructionProposer",
    "LoggedSelector",
    "ReflectionCall",
    "ProposalCall",
    # v2.2 context
    "set_ctx",
    "get_ctx",
    "clear_ctx",
    "with_ctx",
]


def track_gepa_run(
    log_dir: str = "./gepa_logs",
    capture_traces: bool = True,
    capture_lm_calls: bool = True,
    save_traces_separately: bool = False,
    config: TrackerConfig | None = None,
) -> GEPARunTracker:
    """Create a configured tracker for GEPA runs.

    This is the main entry point for the library. Creates a tracker
    with sensible defaults that can be used with a context manager.

    Args:
        log_dir: Directory to save run logs
        capture_traces: Whether to capture full execution traces
        capture_lm_calls: Whether to capture LM call details
        save_traces_separately: Whether to save traces in separate files
        config: Custom tracker configuration (overrides other args)

    Returns:
        Configured GEPARunTracker instance

    Example:
        >>> tracker = track_gepa_run(log_dir="./my_logs")
        >>> with tracker.track():
        ...     optimized = gepa.compile(student=program, trainset=train, valset=val)
        >>> print(f"Best score: {tracker.get_run().best_aggregate_score}")
    """
    if config is None:
        config = TrackerConfig(
            capture_traces=capture_traces,
            capture_lm_calls=capture_lm_calls,
        )

    storage = JSONLStorageBackend(
        base_dir=log_dir,
        save_traces_separately=save_traces_separately,
    )

    return GEPARunTracker(storage=storage, config=config)
