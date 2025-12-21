"""Pareto frontier data models for GEPA logging."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParetoTaskRecord:
    """Represents a single task in a pareto frontier snapshot."""

    task_id: Optional[int] = None  # Set after database insertion
    snapshot_id: Optional[int] = None  # Parent snapshot
    example_id: int = 0  # Task/example index
    dominant_program_id: Optional[int] = None  # Best program for this task
    dominant_score: Optional[float] = None  # Score on this task

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "example_id": self.example_id,
            "dominant_program_id": self.dominant_program_id,
            "dominant_score": self.dominant_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParetoTaskRecord":
        """Create from dictionary representation."""
        return cls(
            task_id=data.get("task_id"),
            snapshot_id=data.get("snapshot_id"),
            example_id=data.get("example_id", 0),
            dominant_program_id=data.get("dominant_program_id"),
            dominant_score=data.get("dominant_score"),
        )


@dataclass
class ParetoSnapshotRecord:
    """Represents a snapshot of the pareto frontier at a point in time.

    Captures the state of the pareto frontier either at:
    - Initial (seed program)
    - After each iteration
    - Final (end of optimization)
    """

    snapshot_id: Optional[int] = None  # Set after database insertion
    run_id: str = ""
    iteration_id: Optional[int] = None  # NULL for initial/final snapshots
    snapshot_type: str = ""  # 'initial', 'iteration', 'final'
    best_program_id: Optional[int] = None  # Overall best program
    best_score: Optional[float] = None  # Overall best score
    timestamp: Optional[str] = None  # ISO timestamp
    tasks: list[ParetoTaskRecord] = field(default_factory=list)  # Per-task breakdown

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "iteration_id": self.iteration_id,
            "snapshot_type": self.snapshot_type,
            "best_program_id": self.best_program_id,
            "best_score": self.best_score,
            "timestamp": self.timestamp,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParetoSnapshotRecord":
        """Create from dictionary representation."""
        return cls(
            snapshot_id=data.get("snapshot_id"),
            run_id=data.get("run_id", ""),
            iteration_id=data.get("iteration_id"),
            snapshot_type=data.get("snapshot_type", ""),
            best_program_id=data.get("best_program_id"),
            best_score=data.get("best_score"),
            timestamp=data.get("timestamp"),
            tasks=[ParetoTaskRecord.from_dict(task) for task in data.get("tasks", [])],
        )
