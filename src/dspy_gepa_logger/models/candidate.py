"""Candidate and Pareto frontier data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    """Complete record of a candidate program in the optimization.

    Tracks the prompt configuration, lineage, and performance of each candidate.
    """

    candidate_idx: int

    # Prompt configuration
    instructions: dict[str, str] = field(default_factory=dict)  # component_name -> instruction

    # Lineage
    parent_indices: list[int] = field(default_factory=list)
    creation_iteration: int = 0
    creation_type: str = "seed"  # "seed", "reflective_mutation", "merge"

    # Validation set scores
    val_subscores: dict[int, float] = field(default_factory=dict)  # val_id -> score
    val_aggregate_score: float = 0.0

    # Budget tracking
    metric_calls_at_discovery: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate_idx": self.candidate_idx,
            "instructions": self.instructions,
            "parent_indices": self.parent_indices,
            "creation_iteration": self.creation_iteration,
            "creation_type": self.creation_type,
            "val_subscores": {str(k): v for k, v in self.val_subscores.items()},  # JSON keys must be strings
            "val_aggregate_score": self.val_aggregate_score,
            "metric_calls_at_discovery": self.metric_calls_at_discovery,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRecord":
        """Create from dictionary."""
        data = data.copy()
        # Convert string keys back to integers
        if "val_subscores" in data:
            data["val_subscores"] = {int(k): v for k, v in data["val_subscores"].items()}
        return cls(**data)


@dataclass
class ParetoFrontierUpdate:
    """Record of Pareto frontier changes after an iteration.

    Captures the state of the frontier before and after accepting a new candidate.
    """

    iteration_number: int

    # Before state: val_id -> set of candidate indices that are best for that instance
    frontier_before: dict[int, set[int]] = field(default_factory=dict)
    frontier_scores_before: dict[int, float] = field(default_factory=dict)  # val_id -> best score

    # After state
    frontier_after: dict[int, set[int]] = field(default_factory=dict)
    frontier_scores_after: dict[int, float] = field(default_factory=dict)

    # Changes
    new_best_instances: list[int] = field(default_factory=list)  # val_ids where new candidate is now best
    ties_added: list[int] = field(default_factory=list)  # val_ids where new candidate tied existing best

    # Aggregate metrics
    pareto_aggregate_score_before: float = 0.0
    pareto_aggregate_score_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration_number": self.iteration_number,
            "frontier_before": {str(k): list(v) for k, v in self.frontier_before.items()},
            "frontier_scores_before": {str(k): v for k, v in self.frontier_scores_before.items()},
            "frontier_after": {str(k): list(v) for k, v in self.frontier_after.items()},
            "frontier_scores_after": {str(k): v for k, v in self.frontier_scores_after.items()},
            "new_best_instances": self.new_best_instances,
            "ties_added": self.ties_added,
            "pareto_aggregate_score_before": self.pareto_aggregate_score_before,
            "pareto_aggregate_score_after": self.pareto_aggregate_score_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParetoFrontierUpdate":
        """Create from dictionary."""
        data = data.copy()
        # Convert string keys back to integers and lists back to sets
        if "frontier_before" in data:
            data["frontier_before"] = {int(k): set(v) for k, v in data["frontier_before"].items()}
        if "frontier_scores_before" in data:
            data["frontier_scores_before"] = {int(k): v for k, v in data["frontier_scores_before"].items()}
        if "frontier_after" in data:
            data["frontier_after"] = {int(k): set(v) for k, v in data["frontier_after"].items()}
        if "frontier_scores_after" in data:
            data["frontier_scores_after"] = {int(k): v for k, v in data["frontier_scores_after"].items()}
        return cls(**data)
