"""Evaluation-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dspy_gepa_logger.models.trace import TraceRecord


@dataclass
class ExampleEvaluation:
    """Evaluation result for a single example within a minibatch.

    Contains the input, prediction, score, and feedback for one example.
    """

    example_id: int

    # Input data (serialized Example fields)
    inputs: dict[str, Any] = field(default_factory=dict)

    # Prediction
    prediction: dict[str, Any] = field(default_factory=dict)
    prediction_success: bool = True

    # Scoring
    score: float = 0.0

    # Feedback from metric
    feedback: str = ""
    feedback_components: dict[str, Any] | None = None  # If multi-metric breakdown

    # Trace (if captured)
    trace: TraceRecord | None = None

    # Error info (if prediction failed)
    error_message: str | None = None
    raw_completion: str | None = None  # For failed parses

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "example_id": self.example_id,
            "inputs": self.inputs,
            "prediction": self.prediction,
            "prediction_success": self.prediction_success,
            "score": self.score,
            "feedback": self.feedback,
            "feedback_components": self.feedback_components,
            "error_message": self.error_message,
            "raw_completion": self.raw_completion,
        }
        if self.trace is not None:
            result["trace"] = self.trace.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExampleEvaluation":
        """Create from dictionary."""
        data = data.copy()
        if "trace" in data and data["trace"] is not None:
            data["trace"] = TraceRecord.from_dict(data["trace"])
        return cls(**data)


@dataclass
class MinibatchEvaluationRecord:
    """Record of evaluating a candidate on a minibatch.

    Groups all example evaluations for a single candidate/minibatch combination.
    """

    iteration_number: int
    candidate_idx: int
    is_parent: bool  # True for parent evaluation, False for new candidate

    # Individual example evaluations
    examples: list[ExampleEvaluation] = field(default_factory=list)

    # Aggregate score
    aggregate_score: float = 0.0

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration_number": self.iteration_number,
            "candidate_idx": self.candidate_idx,
            "is_parent": self.is_parent,
            "examples": [ex.to_dict() for ex in self.examples],
            "aggregate_score": self.aggregate_score,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MinibatchEvaluationRecord":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["examples"] = [ExampleEvaluation.from_dict(ex) for ex in data.get("examples", [])]
        return cls(**data)
