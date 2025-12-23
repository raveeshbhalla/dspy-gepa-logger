"""Metric wrapper for evaluation capture.

LoggedMetric wraps user metrics to capture:
- Score and feedback for each evaluation
- Prediction data (serialized as ref + preview)
- Example ID (deterministic SHA256 hash)
- Context (iteration, phase, candidate_idx)

This is required because GEPA state only stores scores, not feedback strings.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .context import get_ctx, set_ctx


def _deterministic_example_id(example: Any) -> str:
    """Generate a stable, deterministic example ID.

    IMPORTANT: Python's hash() is salted per process - NOT stable across runs.
    We use SHA256 of canonical JSON representation instead.

    Priority:
    1. Use explicit .id attribute if present
    2. Use toDict() method if available
    3. Use __dict__ for objects
    4. Fall back to string representation

    Args:
        example: The example object to generate ID for

    Returns:
        16-character hex string (first 16 chars of SHA256)
    """
    # Try explicit ID first
    if hasattr(example, "id") and example.id is not None:
        return str(example.id)

    # Build canonical representation
    if hasattr(example, "toDict") and callable(example.toDict):
        data = example.toDict()
    elif hasattr(example, "__dict__"):
        # Filter out private attributes
        data = {k: v for k, v in example.__dict__.items() if not k.startswith("_")}
    else:
        data = str(example)

    # Create deterministic hash
    try:
        canonical = json.dumps(data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fall back to string representation if JSON fails
        canonical = str(data)

    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _serialize_prediction(
    prediction: Any, max_preview_len: int = 200
) -> tuple[str | None, str]:
    """Serialize prediction to ref + preview.

    Stores predictions as:
    - ref: Full JSON representation (for reconstruction)
    - preview: Short string for display

    Args:
        prediction: The prediction object
        max_preview_len: Maximum length for preview string

    Returns:
        Tuple of (prediction_ref, prediction_preview)
        - prediction_ref: JSON string (or None if not serializable)
        - prediction_preview: Short string for display
    """
    # Build preview
    preview_str = str(prediction)[:max_preview_len]
    if len(str(prediction)) > max_preview_len:
        preview_str += "..."

    # Try to serialize to JSON
    try:
        if hasattr(prediction, "toDict") and callable(prediction.toDict):
            ref = json.dumps(prediction.toDict(), default=str)
        elif hasattr(prediction, "__dict__"):
            ref = json.dumps(
                {k: v for k, v in prediction.__dict__.items() if not k.startswith("_")},
                default=str,
            )
        else:
            ref = json.dumps(str(prediction))
        return ref, preview_str
    except (TypeError, ValueError):
        return None, preview_str


def _serialize_example(example: Any) -> dict[str, Any]:
    """Serialize example inputs to a dict for display.

    Args:
        example: The example object

    Returns:
        Dict of input field names to values
    """
    if hasattr(example, "inputs") and callable(example.inputs):
        # DSPy Example with .inputs() method
        input_keys = example.inputs()
        return {k: getattr(example, k, None) for k in input_keys}
    elif hasattr(example, "toDict") and callable(example.toDict):
        return example.toDict()
    elif hasattr(example, "__dict__"):
        return {k: v for k, v in example.__dict__.items() if not k.startswith("_")}
    else:
        return {"value": str(example)}


@dataclass
class EvaluationRecord:
    """Single metric evaluation record.

    Captures everything needed to analyze "why did performance change?"
    """

    eval_id: str
    example_id: str  # Deterministic hash, stable across runs
    candidate_idx: int | None  # Which program candidate (from context)
    iteration: int | None  # Which GEPA iteration (from context)
    phase: str  # 'eval', 'validation', 'minibatch'
    score: float
    feedback: str | None  # The "why" - critical for UX

    # Example inputs for display
    example_inputs: dict[str, Any] = field(default_factory=dict)

    # Prediction stored as ref + preview (not raw object!)
    prediction_ref: str | None = None  # JSON blob for full reconstruction
    prediction_preview: str = ""  # Short string for display

    # Timestamp for debugging
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class LoggedMetric:
    """Wraps a metric function to capture evaluations.

    This is a PUBLIC API hook - no monkey-patching required.
    Just wrap your metric before passing to GEPA.

    CRITICAL: This class SETS context (phase="eval") before calling the metric,
    so DSPy LM callbacks will be correctly tagged.

    Usage:
        def my_metric(example, prediction, trace=None):
            is_correct = prediction.answer == example.answer
            feedback = f"Expected {example.answer}, got {prediction.answer}"
            return (1.0 if is_correct else 0.0, feedback)

        logged_metric = LoggedMetric(my_metric)

        # Pass logged_metric to GEPA instead of my_metric
        gepa = GEPA(metric=logged_metric, ...)

        # After optimization, access evaluations:
        for eval in logged_metric.evaluations:
            print(f"{eval.example_id}: {eval.score} - {eval.feedback}")
    """

    def __init__(
        self,
        metric_fn: Callable[..., Any],
        *,
        capture_prediction: bool = True,
        max_prediction_preview: int = 200,
        failure_score: float = 0.0,
    ):
        """Initialize the logged metric wrapper.

        Args:
            metric_fn: The metric function to wrap
            capture_prediction: Whether to capture predictions
            max_prediction_preview: Max length for prediction preview
            failure_score: Score to return when the metric throws an exception
        """
        self.metric_fn = metric_fn
        self.capture_prediction = capture_prediction
        self.max_prediction_preview = max_prediction_preview
        self.failure_score = failure_score
        self.evaluations: list[EvaluationRecord] = []

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Evaluate and log the result.

        Supports both DSPy standard metrics (3 args) and GEPA metrics (5 args).

        If the metric throws an exception (e.g., when accessing attributes that
        don't exist on a FailedPrediction), this wrapper catches it, logs the
        failure, and returns the configured failure_score.

        Args:
            example: The example being evaluated
            prediction: The model's prediction
            trace: Optional execution trace
            pred_name: Optional prediction name (GEPA uses this)
            pred_trace: Optional prediction trace (GEPA uses this)
            **kwargs: Additional arguments for the metric

        Returns:
            The original metric result (unchanged), or failure_score on exception
        """
        import logging
        import traceback

        logger = logging.getLogger(__name__)

        # Get current context (iteration, candidate_idx set by state logger or proposer)
        ctx = get_ctx()

        # SET phase="eval" BEFORE calling metric (so any LM calls inside are tagged)
        previous_phase = ctx.get("phase")
        set_ctx(phase="eval")

        # Generate deterministic example ID
        try:
            example_id = _deterministic_example_id(example)
        except Exception:
            # Fallback to object id if serialization fails
            example_id = f"example_{id(example)}"

        result = None
        score = self.failure_score
        feedback = None
        metric_error = None

        try:
            # Call actual metric with all args it might need
            # Try to pass GEPA args if the metric accepts them
            import inspect
            sig = inspect.signature(self.metric_fn)
            params = list(sig.parameters.keys())

            if "pred_name" in params or "pred_trace" in params:
                # GEPA-style metric with 5 args
                result = self.metric_fn(
                    example, prediction, trace, pred_name, pred_trace, **kwargs
                )
            else:
                # Standard DSPy metric with 3 args
                result = self.metric_fn(example, prediction, trace=trace, **kwargs)

            # Extract score and feedback from various return types
            if isinstance(result, tuple) and len(result) == 2:
                # Tuple format: (score, feedback)
                score, feedback = result
            elif hasattr(result, "score"):
                # dspy.Prediction or object with .score attribute
                score = getattr(result, "score", 0.0)
                feedback = getattr(result, "feedback", None)
            else:
                # Plain score (float/int)
                score, feedback = result, None

        except Exception as e:
            # Log the exception with traceback for debugging
            metric_error = str(e)
            tb_str = traceback.format_exc()
            logger.warning(
                f"Metric evaluation failed for example {example_id}: {e}\n"
                f"Using failure_score={self.failure_score}"
            )
            logger.debug(f"Full traceback:\n{tb_str}")

            # Set failure values
            score = self.failure_score
            feedback = f"Metric error: {metric_error}"
            result = self.failure_score  # Return simple score for GEPA compatibility

        # Serialize prediction (ref + preview, not raw object)
        if self.capture_prediction:
            try:
                pred_ref, pred_preview = _serialize_prediction(
                    prediction, self.max_prediction_preview
                )
            except Exception:
                pred_ref, pred_preview = None, str(prediction)[:200]
        else:
            pred_ref, pred_preview = None, ""

        # Serialize example inputs for display
        try:
            example_inputs = _serialize_example(example)
        except Exception:
            example_inputs = {"_error": "Failed to serialize example inputs"}

        # Record evaluation (including failures)
        record = EvaluationRecord(
            eval_id=str(uuid.uuid4()),
            example_id=example_id,
            candidate_idx=ctx.get("candidate_idx"),
            iteration=ctx.get("iteration"),
            phase="eval",
            score=float(score) if score is not None else 0.0,
            feedback=str(feedback) if feedback else None,
            example_inputs=example_inputs,
            prediction_ref=pred_ref,
            prediction_preview=pred_preview,
        )
        self.evaluations.append(record)

        # Restore previous phase if it was set
        if previous_phase is not None:
            set_ctx(phase=previous_phase)

        return result

    def get_evaluations_for_example(self, example_id: str) -> list[EvaluationRecord]:
        """Get all evaluations for a specific example.

        Useful for delta analysis (comparing same example across candidates).

        Args:
            example_id: The example ID to filter by

        Returns:
            List of evaluation records for that example
        """
        return [e for e in self.evaluations if e.example_id == example_id]

    def get_evaluations_for_candidate(
        self, candidate_idx: int
    ) -> list[EvaluationRecord]:
        """Get all evaluations for a specific candidate.

        Args:
            candidate_idx: The candidate index to filter by

        Returns:
            List of evaluation records for that candidate
        """
        return [e for e in self.evaluations if e.candidate_idx == candidate_idx]

    def get_evaluations_for_iteration(self, iteration: int) -> list[EvaluationRecord]:
        """Get all evaluations for a specific iteration.

        Args:
            iteration: The iteration number to filter by

        Returns:
            List of evaluation records for that iteration
        """
        return [e for e in self.evaluations if e.iteration == iteration]

    def compute_lift(
        self, example_id: str, before_candidate: int, after_candidate: int
    ) -> dict[str, Any]:
        """Compute score lift for an example between two candidates.

        Args:
            example_id: The example to analyze
            before_candidate: The "before" candidate index
            after_candidate: The "after" candidate index

        Returns:
            Dict with 'lift', 'before', and 'after' keys
        """
        before = [
            e
            for e in self.evaluations
            if e.example_id == example_id and e.candidate_idx == before_candidate
        ]
        after = [
            e
            for e in self.evaluations
            if e.example_id == example_id and e.candidate_idx == after_candidate
        ]

        if not before or not after:
            return {"lift": None, "before": None, "after": None}

        return {
            "lift": after[0].score - before[0].score,
            "before": before[0],
            "after": after[0],
        }

    def get_regressions(self, seed_candidate: int = 0) -> list[dict[str, Any]]:
        """Find examples where optimized candidates scored worse than seed.

        Args:
            seed_candidate: The seed/baseline candidate index

        Returns:
            List of regression dicts, sorted by delta (worst first)
        """
        regressions = []
        example_ids = set(e.example_id for e in self.evaluations)

        for ex_id in example_ids:
            seed_evals = [
                e
                for e in self.evaluations
                if e.example_id == ex_id and e.candidate_idx == seed_candidate
            ]
            if not seed_evals:
                continue

            seed_score = seed_evals[0].score

            # Find any candidate that scored worse
            for e in self.evaluations:
                if e.example_id == ex_id and e.candidate_idx != seed_candidate:
                    if e.score < seed_score:
                        regressions.append(
                            {
                                "example_id": ex_id,
                                "seed_score": seed_score,
                                "seed_feedback": seed_evals[0].feedback,
                                "regressed_candidate": e.candidate_idx,
                                "regressed_score": e.score,
                                "regressed_feedback": e.feedback,
                                "delta": e.score - seed_score,
                            }
                        )

        return sorted(regressions, key=lambda x: x["delta"])

    def get_improvements(self, seed_candidate: int = 0) -> list[dict[str, Any]]:
        """Find examples where optimized candidates scored better than seed.

        Args:
            seed_candidate: The seed/baseline candidate index

        Returns:
            List of improvement dicts, sorted by delta (best first)
        """
        improvements = []
        example_ids = set(e.example_id for e in self.evaluations)

        for ex_id in example_ids:
            seed_evals = [
                e
                for e in self.evaluations
                if e.example_id == ex_id and e.candidate_idx == seed_candidate
            ]
            if not seed_evals:
                continue

            seed_score = seed_evals[0].score

            # Find any candidate that scored better
            for e in self.evaluations:
                if e.example_id == ex_id and e.candidate_idx != seed_candidate:
                    if e.score > seed_score:
                        improvements.append(
                            {
                                "example_id": ex_id,
                                "seed_score": seed_score,
                                "seed_feedback": seed_evals[0].feedback,
                                "improved_candidate": e.candidate_idx,
                                "improved_score": e.score,
                                "improved_feedback": e.feedback,
                                "delta": e.score - seed_score,
                            }
                        )

        return sorted(improvements, key=lambda x: x["delta"], reverse=True)

    def clear(self) -> None:
        """Clear all recorded evaluations."""
        self.evaluations = []

    def __len__(self) -> int:
        """Return number of recorded evaluations."""
        return len(self.evaluations)
