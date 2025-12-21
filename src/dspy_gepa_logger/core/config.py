"""Configuration for GEPA run tracking."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    """Configuration options for the GEPA run tracker.

    Controls what data is captured and how it is stored.
    """

    # Capture settings
    capture_traces: bool = True  # Capture full execution traces
    capture_lm_calls: bool = True  # Capture LM call details via callbacks
    capture_full_inputs: bool = True  # Capture full input data (vs. hash only)
    capture_full_outputs: bool = True  # Capture full output data

    # Trace detail settings
    max_trace_depth: int = 100  # Max predictor calls to capture per trace
    max_lm_calls_per_trace: int = 50  # Max LM calls to capture per trace

    # Sampling (for performance with large datasets)
    sample_rate: float = 1.0  # 1.0 = capture all, 0.1 = capture 10%

    # Storage settings
    async_writes: bool = True  # Write to storage asynchronously
    buffer_size: int = 100  # Number of records to buffer before flushing

    # Content truncation (to manage storage size)
    max_input_length: int | None = None  # Truncate inputs longer than this
    max_output_length: int | None = None  # Truncate outputs longer than this
    max_feedback_length: int | None = None  # Truncate feedback longer than this

    # Metadata
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "capture_traces": self.capture_traces,
            "capture_lm_calls": self.capture_lm_calls,
            "capture_full_inputs": self.capture_full_inputs,
            "capture_full_outputs": self.capture_full_outputs,
            "max_trace_depth": self.max_trace_depth,
            "max_lm_calls_per_trace": self.max_lm_calls_per_trace,
            "sample_rate": self.sample_rate,
            "async_writes": self.async_writes,
            "buffer_size": self.buffer_size,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "max_feedback_length": self.max_feedback_length,
            "extra_metadata": self.extra_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackerConfig":
        """Create from dictionary."""
        return cls(**data)
