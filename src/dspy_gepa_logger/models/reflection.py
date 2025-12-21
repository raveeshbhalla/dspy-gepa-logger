"""Reflection-related data models."""

from dataclasses import dataclass, field
from typing import Any

from dspy_gepa_logger.models.trace import LMCallRecord


@dataclass
class ReflectionRecord:
    """Record of the reflection/proposal step in a GEPA iteration.

    Captures everything about how the reflective LM analyzed performance
    and proposed new instructions.
    """

    iteration_number: int

    # Components being updated
    components_to_update: list[str] = field(default_factory=list)

    # Reflective dataset sent to LM (per component)
    # Structure: {component_name: [{Inputs, Generated_Outputs, Feedback}, ...]}
    reflective_datasets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # LM call details for reflection
    lm_calls: list[LMCallRecord] = field(default_factory=list)

    # Proposed changes
    proposed_instructions: dict[str, str] = field(default_factory=dict)  # component_name -> new instruction

    # Timing
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration_number": self.iteration_number,
            "components_to_update": self.components_to_update,
            "reflective_datasets": self.reflective_datasets,
            "lm_calls": [lm.to_dict() for lm in self.lm_calls],
            "proposed_instructions": self.proposed_instructions,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflectionRecord":
        """Create from dictionary."""
        data = data.copy()
        data["lm_calls"] = [LMCallRecord.from_dict(lm) for lm in data.get("lm_calls", [])]
        return cls(**data)
