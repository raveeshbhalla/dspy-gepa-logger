"""Run-level data models for complete GEPA optimization runs."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dspy_gepa_logger.models.iteration import IterationRecord
from dspy_gepa_logger.models.candidate import CandidateRecord


@dataclass
class GEPARunConfig:
    """Configuration for a GEPA optimization run."""

    # Budget settings
    auto: str | None = None  # "light", "medium", "heavy"
    max_full_evals: int | None = None
    max_metric_calls: int | None = None

    # Reflection settings
    reflection_minibatch_size: int | None = None
    candidate_selection_strategy: str = "pareto"
    skip_perfect_score: bool = True
    add_format_failure_as_feedback: bool = False

    # Module settings
    enable_tool_optimization: bool = False

    # Other
    num_threads: int | None = None
    failure_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "auto": self.auto,
            "max_full_evals": self.max_full_evals,
            "max_metric_calls": self.max_metric_calls,
            "reflection_minibatch_size": self.reflection_minibatch_size,
            "candidate_selection_strategy": self.candidate_selection_strategy,
            "skip_perfect_score": self.skip_perfect_score,
            "add_format_failure_as_feedback": self.add_format_failure_as_feedback,
            "enable_tool_optimization": self.enable_tool_optimization,
            "num_threads": self.num_threads,
            "failure_score": self.failure_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GEPARunConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GEPARunRecord:
    """Complete record of a GEPA optimization run.

    This is the top-level container that holds all data for a single
    optimization run, including configuration, iterations, candidates,
    and final results.
    """

    # Identity
    run_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Configuration
    config: GEPARunConfig = field(default_factory=GEPARunConfig)

    # Seed candidate
    seed_candidate: dict[str, str] = field(default_factory=dict)
    seed_val_scores: dict[int, float] = field(default_factory=dict)
    seed_aggregate_score: float = 0.0

    # Iterations (populated during run)
    iterations: list[IterationRecord] = field(default_factory=list)

    # All candidates discovered
    candidates: list[CandidateRecord] = field(default_factory=list)

    # Final Pareto frontier: val_id -> set of candidate indices
    final_pareto_frontier: dict[int, set[int]] = field(default_factory=dict)

    # Summary metrics
    total_iterations: int = 0
    total_metric_calls: int = 0
    best_candidate_idx: int = 0
    best_aggregate_score: float = 0.0

    # Status
    status: str = "running"  # "running", "completed", "failed", "stopped"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self.config.to_dict(),
            "seed_candidate": self.seed_candidate,
            "seed_val_scores": {str(k): v for k, v in self.seed_val_scores.items()},
            "seed_aggregate_score": self.seed_aggregate_score,
            "iterations": [it.to_dict() for it in self.iterations],
            "candidates": [c.to_dict() for c in self.candidates],
            "final_pareto_frontier": {str(k): list(v) for k, v in self.final_pareto_frontier.items()},
            "total_iterations": self.total_iterations,
            "total_metric_calls": self.total_metric_calls,
            "best_candidate_idx": self.best_candidate_idx,
            "best_aggregate_score": self.best_aggregate_score,
            "status": self.status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GEPARunRecord":
        """Create from dictionary."""
        data = data.copy()

        # Parse timestamps
        if isinstance(data.get("started_at"), str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if isinstance(data.get("completed_at"), str):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])

        # Parse config
        if "config" in data:
            data["config"] = GEPARunConfig.from_dict(data["config"])

        # Parse seed_val_scores
        if "seed_val_scores" in data:
            data["seed_val_scores"] = {int(k): v for k, v in data["seed_val_scores"].items()}

        # Parse iterations
        if "iterations" in data:
            data["iterations"] = [IterationRecord.from_dict(it) for it in data["iterations"]]

        # Parse candidates
        if "candidates" in data:
            data["candidates"] = [CandidateRecord.from_dict(c) for c in data["candidates"]]

        # Parse pareto frontier
        if "final_pareto_frontier" in data:
            data["final_pareto_frontier"] = {int(k): set(v) for k, v in data["final_pareto_frontier"].items()}

        return cls(**data)

    def add_iteration(self, iteration: IterationRecord) -> None:
        """Add an iteration to the run."""
        self.iterations.append(iteration)
        self.total_iterations = len(self.iterations)

    def add_candidate(self, candidate: CandidateRecord) -> None:
        """Add a candidate to the run."""
        self.candidates.append(candidate)
        # Update best if this candidate is better
        if candidate.val_aggregate_score > self.best_aggregate_score:
            self.best_aggregate_score = candidate.val_aggregate_score
            self.best_candidate_idx = candidate.candidate_idx

    @property
    def improvement(self) -> float:
        """Calculate total improvement over seed."""
        return self.best_aggregate_score - self.seed_aggregate_score

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total run duration in seconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def accepted_iterations(self) -> list[IterationRecord]:
        """Get all iterations where the candidate was accepted."""
        return [it for it in self.iterations if it.accepted]

    @property
    def acceptance_rate(self) -> float:
        """Calculate the acceptance rate of proposed candidates."""
        if not self.iterations:
            return 0.0
        return len(self.accepted_iterations) / len(self.iterations)
