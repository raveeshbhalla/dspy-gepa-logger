"""Iteration data model - the core record for a single GEPA optimization step."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dspy_gepa_logger.models.trace import TraceRecord
from dspy_gepa_logger.models.reflection import ReflectionRecord
from dspy_gepa_logger.models.candidate import ParetoFrontierUpdate


@dataclass
class IterationRecord:
    """Complete record of a single GEPA iteration.

    This is the main data structure that captures everything that happens
    in one iteration of the GEPA optimization loop:
    1. Parent selection
    2. Minibatch evaluation of parent
    3. Reflection and proposal
    4. Minibatch evaluation of candidate
    5. Acceptance decision
    6. Validation set evaluation (if accepted)
    7. Pareto frontier update
    """

    # Identity
    run_id: str
    iteration_number: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Parent selection
    parent_candidate_idx: int = 0
    parent_prompt: dict[str, str] = field(default_factory=dict)  # component_name -> instruction
    parent_val_score: float = 0.0
    selection_strategy: str = "pareto"  # "pareto", "current_best", etc.

    # Minibatch evaluation (parent - before reflection)
    minibatch_ids: list[int] = field(default_factory=list)
    minibatch_inputs: list[dict[str, Any]] = field(default_factory=list)
    minibatch_outputs: list[dict[str, Any]] = field(default_factory=list)
    minibatch_scores: list[float] = field(default_factory=list)
    minibatch_feedback: list[str] = field(default_factory=list)
    minibatch_traces: list[TraceRecord] | None = None

    # Reflection
    reflection: ReflectionRecord | None = None

    # New candidate evaluation on same minibatch
    new_candidate_prompt: dict[str, str] | None = None
    new_candidate_minibatch_outputs: list[dict[str, Any]] | None = None
    new_candidate_minibatch_scores: list[float] | None = None
    new_candidate_minibatch_feedback: list[str] | None = None

    # Acceptance decision
    accepted: bool = False
    acceptance_reason: str | None = None  # "score_improved", "rejected", etc.

    # Validation set evaluation (if accepted)
    val_scores: dict[int, float] | None = None  # val_id -> score
    val_outputs: dict[int, Any] | None = None  # val_id -> model output
    val_inputs: dict[int, Any] | None = None  # val_id -> input data
    val_aggregate_score: float | None = None
    new_candidate_idx: int | None = None

    # Pareto frontier update
    pareto_update: ParetoFrontierUpdate | None = None

    # LM calls made during this iteration
    lm_calls: list[Any] | None = None  # List of LMCallRecord

    # Timing
    duration_ms: float | None = None

    # Metadata
    iteration_type: str = "reflective_mutation"  # or "merge"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "run_id": self.run_id,
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp.isoformat(),
            "parent_candidate_idx": self.parent_candidate_idx,
            "parent_prompt": self.parent_prompt,
            "parent_val_score": self.parent_val_score,
            "selection_strategy": self.selection_strategy,
            "minibatch_ids": self.minibatch_ids,
            "minibatch_inputs": self.minibatch_inputs,
            "minibatch_outputs": self.minibatch_outputs,
            "minibatch_scores": self.minibatch_scores,
            "minibatch_feedback": self.minibatch_feedback,
            "new_candidate_prompt": self.new_candidate_prompt,
            "new_candidate_minibatch_outputs": self.new_candidate_minibatch_outputs,
            "new_candidate_minibatch_scores": self.new_candidate_minibatch_scores,
            "new_candidate_minibatch_feedback": self.new_candidate_minibatch_feedback,
            "accepted": self.accepted,
            "acceptance_reason": self.acceptance_reason,
            "val_aggregate_score": self.val_aggregate_score,
            "new_candidate_idx": self.new_candidate_idx,
            "duration_ms": self.duration_ms,
            "iteration_type": self.iteration_type,
            "metadata": self.metadata,
        }

        # Handle optional complex fields
        if self.minibatch_traces is not None:
            result["minibatch_traces"] = [t.to_dict() for t in self.minibatch_traces]

        if self.reflection is not None:
            result["reflection"] = self.reflection.to_dict()

        if self.val_scores is not None:
            result["val_scores"] = {str(k): v for k, v in self.val_scores.items()}

        if self.val_outputs is not None:
            result["val_outputs"] = {str(k): v for k, v in self.val_outputs.items()}

        if self.val_inputs is not None:
            result["val_inputs"] = {str(k): v for k, v in self.val_inputs.items()}

        if self.pareto_update is not None:
            result["pareto_update"] = self.pareto_update.to_dict()

        if self.lm_calls is not None:
            result["lm_calls"] = [call.to_dict() if hasattr(call, 'to_dict') else call for call in self.lm_calls]

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IterationRecord":
        """Create from dictionary."""
        data = data.copy()

        # Parse timestamp
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        # Parse complex fields
        if "minibatch_traces" in data and data["minibatch_traces"] is not None:
            data["minibatch_traces"] = [TraceRecord.from_dict(t) for t in data["minibatch_traces"]]

        if "reflection" in data and data["reflection"] is not None:
            data["reflection"] = ReflectionRecord.from_dict(data["reflection"])

        if "val_scores" in data and data["val_scores"] is not None:
            data["val_scores"] = {int(k): v for k, v in data["val_scores"].items()}

        if "val_outputs" in data and data["val_outputs"] is not None:
            data["val_outputs"] = {int(k): v for k, v in data["val_outputs"].items()}

        if "val_inputs" in data and data["val_inputs"] is not None:
            data["val_inputs"] = {int(k): v for k, v in data["val_inputs"].items()}

        if "pareto_update" in data and data["pareto_update"] is not None:
            data["pareto_update"] = ParetoFrontierUpdate.from_dict(data["pareto_update"])

        if "lm_calls" in data and data["lm_calls"] is not None:
            from dspy_gepa_logger.models.trace import LMCallRecord
            data["lm_calls"] = [LMCallRecord.from_dict(call) if isinstance(call, dict) else call for call in data["lm_calls"]]

        return cls(**data)

    @property
    def parent_minibatch_aggregate(self) -> float:
        """Calculate aggregate score for parent on minibatch."""
        if not self.minibatch_scores:
            return 0.0
        return sum(self.minibatch_scores) / len(self.minibatch_scores)

    @property
    def candidate_minibatch_aggregate(self) -> float | None:
        """Calculate aggregate score for new candidate on minibatch."""
        if not self.new_candidate_minibatch_scores:
            return None
        return sum(self.new_candidate_minibatch_scores) / len(self.new_candidate_minibatch_scores)

    @property
    def minibatch_improvement(self) -> float | None:
        """Calculate improvement on minibatch (candidate - parent)."""
        candidate_agg = self.candidate_minibatch_aggregate
        if candidate_agg is None:
            return None
        return candidate_agg - self.parent_minibatch_aggregate
