"""Core tracking infrastructure."""

from pathlib import Path
from dspy_gepa_logger.core.config import TrackerConfig
from dspy_gepa_logger.core.tracker import GEPARunTracker
from dspy_gepa_logger.storage.sqlite_adapter import SQLiteStorageAdapter

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

__all__ = ["TrackerConfig", "GEPARunTracker", "create_sqlite_tracker"]
