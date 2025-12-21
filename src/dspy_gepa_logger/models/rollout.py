"""Rollout data model for GEPA logging."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RolloutRecord:
    """Represents a single program execution (rollout) on one example.

    Rollouts reference programs by ID rather than embedding the full program,
    allowing for normalized storage and efficient deduplication.
    """

    rollout_id: Optional[int] = None  # Set after database insertion
    iteration_id: Optional[int] = None  # Parent iteration
    program_id: Optional[int] = None  # Reference to program
    rollout_type: str = ""  # 'parent_minibatch', 'candidate_minibatch', 'validation'
    example_id: int = 0  # Task/example index
    input_data: Any = None  # Input to the program
    output_data: Any = None  # Program output
    score: Optional[float] = None  # Evaluation score
    feedback: Optional[str] = None  # Evaluation feedback

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "rollout_id": self.rollout_id,
            "iteration_id": self.iteration_id,
            "program_id": self.program_id,
            "rollout_type": self.rollout_type,
            "example_id": self.example_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "score": self.score,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RolloutRecord":
        """Create from dictionary representation."""
        return cls(
            rollout_id=data.get("rollout_id"),
            iteration_id=data.get("iteration_id"),
            program_id=data.get("program_id"),
            rollout_type=data.get("rollout_type", ""),
            example_id=data.get("example_id", 0),
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            score=data.get("score"),
            feedback=data.get("feedback"),
        )
