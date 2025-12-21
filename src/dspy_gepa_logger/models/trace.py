"""Trace-related data models for capturing execution details."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LMCallRecord:
    """Record of a single LM (language model) call.

    Captures the full request/response of an LM invocation,
    including whether it was part of a reflection step.

    MLflow-style comprehensive tracking of LM calls including:
    - Full request/response details
    - Token usage and costs
    - Latency metrics
    - Model parameters
    - Error handling
    """

    call_id: str
    model: str

    # Request details
    messages: list[dict[str, Any]] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop_sequences: list[str] | None = None

    # Full request/response for debugging
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None

    # Response
    response_text: str = ""
    finish_reason: str | None = None

    # Token usage (comprehensive)
    usage: dict[str, int] | None = None  # e.g., {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    input_tokens: int | None = None  # Normalized field
    output_tokens: int | None = None  # Normalized field
    total_tokens: int | None = None  # Normalized field

    # Cost tracking (if available)
    estimated_cost_usd: float | None = None
    input_cost_per_1k: float | None = None
    output_cost_per_1k: float | None = None

    # Timing (MLflow-style)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Latency breakdown
    time_to_first_token_ms: float | None = None  # For streaming

    # Error handling
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0

    # Context (DSPy-specific)
    is_reflection: bool = False
    component_name: str | None = None  # Which component this reflection is for
    iteration_number: int | None = None  # Which GEPA iteration

    # Additional metadata
    tags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "call_id": self.call_id,
            "model": self.model,
            # Request
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences,
            "raw_request": self.raw_request,
            "raw_response": self.raw_response,
            # Response
            "response_text": self.response_text,
            "finish_reason": self.finish_reason,
            # Usage
            "usage": self.usage,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            # Cost
            "estimated_cost_usd": self.estimated_cost_usd,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            # Timing
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            # Error handling
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            # Context
            "is_reflection": self.is_reflection,
            "component_name": self.component_name,
            "iteration_number": self.iteration_number,
            # Metadata
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LMCallRecord":
        """Create from dictionary."""
        data = data.copy()
        # Parse datetime fields
        for field in ["timestamp", "start_time", "end_time"]:
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)

    def calculate_cost(self, pricing: dict[str, dict[str, float]] | None = None) -> float:
        """Calculate estimated cost for this LM call.

        Args:
            pricing: Optional pricing dictionary with format:
                {
                    "model_name": {
                        "input_per_1k": 0.001,
                        "output_per_1k": 0.002
                    }
                }
                If not provided, uses default OpenAI pricing.

        Returns:
            Estimated cost in USD
        """
        if not self.input_tokens or not self.output_tokens:
            return 0.0

        # Default pricing (OpenAI as of late 2024)
        default_pricing = {
            "gpt-4": {"input_per_1k": 0.03, "output_per_1k": 0.06},
            "gpt-4-turbo": {"input_per_1k": 0.01, "output_per_1k": 0.03},
            "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.00060},
            "gpt-3.5-turbo": {"input_per_1k": 0.0015, "output_per_1k": 0.002},
            "claude-3-opus": {"input_per_1k": 0.015, "output_per_1k": 0.075},
            "claude-3-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015},
            "claude-3-haiku": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
        }

        pricing = pricing or default_pricing

        # Find matching model in pricing
        model_pricing = None
        for model_name, prices in pricing.items():
            if model_name in self.model.lower():
                model_pricing = prices
                break

        if not model_pricing:
            return 0.0

        # Calculate cost
        input_cost = (self.input_tokens / 1000.0) * model_pricing["input_per_1k"]
        output_cost = (self.output_tokens / 1000.0) * model_pricing["output_per_1k"]
        total_cost = input_cost + output_cost

        # Store for later reference
        self.input_cost_per_1k = model_pricing["input_per_1k"]
        self.output_cost_per_1k = model_pricing["output_per_1k"]
        self.estimated_cost_usd = total_cost

        return total_cost


@dataclass
class PredictorCallRecord:
    """Record of a single predictor invocation within a trace.

    Captures the inputs and outputs of a DSPy predictor call.
    """

    predictor_name: str
    signature_name: str

    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    success: bool = True
    error_message: str | None = None

    # Timing
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "predictor_name": self.predictor_name,
            "signature_name": self.signature_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "success": self.success,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictorCallRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TraceRecord:
    """DSPy execution trace for a single example.

    Contains the full execution history including all predictor
    and LM calls made during processing.
    """

    example_id: int

    # Predictor invocations in order
    predictor_calls: list[PredictorCallRecord] = field(default_factory=list)

    # LM calls captured via callback system
    lm_calls: list[LMCallRecord] = field(default_factory=list)

    # Total duration
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "example_id": self.example_id,
            "predictor_calls": [pc.to_dict() for pc in self.predictor_calls],
            "lm_calls": [lm.to_dict() for lm in self.lm_calls],
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRecord":
        """Create from dictionary."""
        return cls(
            example_id=data["example_id"],
            predictor_calls=[PredictorCallRecord.from_dict(pc) for pc in data.get("predictor_calls", [])],
            lm_calls=[LMCallRecord.from_dict(lm) for lm in data.get("lm_calls", [])],
            duration_ms=data.get("duration_ms", 0.0),
        )
