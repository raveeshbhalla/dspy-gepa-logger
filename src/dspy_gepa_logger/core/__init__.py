"""Core tracking infrastructure."""

from pathlib import Path
from dspy_gepa_logger.core.config import TrackerConfig
from dspy_gepa_logger.core.tracker import GEPARunTracker
from dspy_gepa_logger.storage.sqlite_adapter import SQLiteStorageAdapter

# v2.2 components (public hooks architecture)
from dspy_gepa_logger.core.context import set_ctx, get_ctx, clear_ctx, with_ctx
from dspy_gepa_logger.core.logged_metric import LoggedMetric, EvaluationRecord
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger, LMCall
from dspy_gepa_logger.core.state_logger import GEPAStateLogger, IterationDelta, IterationMetadata
from dspy_gepa_logger.core.logged_proposer import (
    LoggedInstructionProposer,
    LoggedSelector,
    ReflectionCall,
    ProposalCall,
)
from dspy_gepa_logger.core.tracker_v2 import GEPATracker, CandidateDiff


def create_sqlite_tracker(db_path: str | Path = "./gepa_runs.db") -> GEPARunTracker:
    """Create a GEPA tracker with SQLite storage.

    Args:
        db_path: Path to SQLite database file

    Returns:
        GEPARunTracker configured with SQLite storage

    Example:
        tracker = create_sqlite_tracker("./runs.db")
        with tracker.track():
            optimized = gepa.compile(student=program, trainset=train, valset=val)
    """
    storage = SQLiteStorageAdapter(db_path)
    return GEPARunTracker(storage=storage)


__all__ = [
    # v1 (legacy)
    "TrackerConfig",
    "GEPARunTracker",
    "create_sqlite_tracker",
    # v2.2 context
    "set_ctx",
    "get_ctx",
    "clear_ctx",
    "with_ctx",
    # v2.2 metric
    "LoggedMetric",
    "EvaluationRecord",
    # v2.2 LM logger
    "DSPyLMLogger",
    "LMCall",
    # v2.2 state logger
    "GEPAStateLogger",
    "IterationDelta",
    "IterationMetadata",
    # v2.2 proposer
    "LoggedInstructionProposer",
    "LoggedSelector",
    "ReflectionCall",
    "ProposalCall",
    # v2.2 unified tracker
    "GEPATracker",
    "CandidateDiff",
]
