"""Tests for LoggedMetric wrapper.

Tests cover:
1. Basic metric wrapping (returns same result)
2. Score and feedback capture
3. Deterministic example IDs
4. Prediction serialization
5. Context integration (phase is set to "eval")
6. Query methods (get_evaluations_for_*, compute_lift, get_regressions)
"""

import json
import pytest
from dataclasses import dataclass
from typing import Any

from dspy_gepa_logger.core.context import clear_ctx, set_ctx, get_ctx
from dspy_gepa_logger.core.logged_metric import (
    LoggedMetric,
    EvaluationRecord,
    _deterministic_example_id,
    _serialize_prediction,
)


# Test fixtures


@dataclass
class MockExample:
    """Mock DSPy-like Example."""

    question: str
    answer: str
    id: str | None = None

    def toDict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer}


@dataclass
class MockPrediction:
    """Mock DSPy-like Prediction."""

    answer: str

    def toDict(self) -> dict[str, Any]:
        return {"answer": self.answer}


def simple_metric(example, prediction, trace=None):
    """Simple metric that returns (score, feedback) tuple."""
    is_correct = prediction.answer == example.answer
    score = 1.0 if is_correct else 0.0
    feedback = f"Expected '{example.answer}', got '{prediction.answer}'"
    return score, feedback


def score_only_metric(example, prediction, trace=None):
    """Metric that returns only a score, no feedback."""
    return 1.0 if prediction.answer == example.answer else 0.0


class TestDeterministicExampleId:
    """Test _deterministic_example_id function."""

    def test_explicit_id(self):
        """Should use explicit id if present."""
        example = MockExample(question="q", answer="a", id="explicit-123")
        assert _deterministic_example_id(example) == "explicit-123"

    def test_to_dict_hash(self):
        """Should hash toDict() if no explicit id."""
        example = MockExample(question="What is 2+2?", answer="4")
        id1 = _deterministic_example_id(example)

        # Same content should produce same hash
        example2 = MockExample(question="What is 2+2?", answer="4")
        id2 = _deterministic_example_id(example2)

        assert id1 == id2
        assert len(id1) == 16  # First 16 chars of SHA256

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        example1 = MockExample(question="What is 2+2?", answer="4")
        example2 = MockExample(question="What is 3+3?", answer="6")

        id1 = _deterministic_example_id(example1)
        id2 = _deterministic_example_id(example2)

        assert id1 != id2

    def test_dict_example(self):
        """Should work with plain dicts."""
        example = {"question": "test", "answer": "yes"}
        id1 = _deterministic_example_id(example)
        assert len(id1) == 16

    def test_string_fallback(self):
        """Should fall back to str() for non-serializable objects."""
        example = "plain string example"
        id1 = _deterministic_example_id(example)
        assert len(id1) == 16

    def test_deterministic_across_calls(self):
        """Same input should always produce same ID."""
        example = MockExample(question="q", answer="a")
        ids = [_deterministic_example_id(example) for _ in range(100)]
        assert len(set(ids)) == 1  # All identical


class TestSerializePrediction:
    """Test _serialize_prediction function."""

    def test_to_dict_prediction(self):
        """Should serialize toDict() predictions."""
        pred = MockPrediction(answer="42")
        ref, preview = _serialize_prediction(pred)

        assert ref is not None
        assert json.loads(ref) == {"answer": "42"}
        assert "42" in preview

    def test_dict_prediction(self):
        """Should serialize dict predictions."""
        pred = {"answer": "42", "confidence": 0.9}
        ref, preview = _serialize_prediction(pred)

        assert ref is not None
        # Plain dicts fall through to str()
        assert "42" in preview

    def test_preview_truncation(self):
        """Should truncate long previews."""
        pred = MockPrediction(answer="a" * 500)
        ref, preview = _serialize_prediction(pred, max_preview_len=50)

        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")

    def test_non_serializable(self):
        """Should handle non-serializable predictions gracefully."""

        class NonSerializable:
            def __str__(self):
                return "non-serializable-object"

        pred = NonSerializable()
        ref, preview = _serialize_prediction(pred)

        # Should have preview even if ref fails
        assert "non-serializable-object" in preview


class TestLoggedMetricBasic:
    """Test basic LoggedMetric functionality."""

    def setup_method(self):
        clear_ctx()

    def test_returns_original_result(self):
        """Wrapped metric should return same result as original."""
        logged = LoggedMetric(simple_metric)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        result = logged(example, pred)
        expected = simple_metric(example, pred)

        assert result == expected

    def test_captures_score_and_feedback(self):
        """Should capture score and feedback from tuple result."""
        logged = LoggedMetric(simple_metric)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        assert len(logged.evaluations) == 1
        record = logged.evaluations[0]
        assert record.score == 1.0
        assert "Expected '4', got '4'" in record.feedback

    def test_captures_score_only_metric(self):
        """Should handle metrics that return only score."""
        logged = LoggedMetric(score_only_metric)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        assert len(logged.evaluations) == 1
        record = logged.evaluations[0]
        assert record.score == 1.0
        assert record.feedback is None

    def test_captures_prediction(self):
        """Should capture prediction ref and preview."""
        logged = LoggedMetric(simple_metric, capture_prediction=True)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        record = logged.evaluations[0]
        assert record.prediction_ref is not None
        assert "4" in record.prediction_preview
        assert json.loads(record.prediction_ref) == {"answer": "4"}

    def test_skip_prediction_capture(self):
        """Should not capture prediction when disabled."""
        logged = LoggedMetric(simple_metric, capture_prediction=False)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        record = logged.evaluations[0]
        assert record.prediction_ref is None
        assert record.prediction_preview == ""

    def test_clear_evaluations(self):
        """Should be able to clear evaluations."""
        logged = LoggedMetric(simple_metric)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)
        assert len(logged) == 1

        logged.clear()
        assert len(logged) == 0


class TestLoggedMetricContext:
    """Test context integration."""

    def setup_method(self):
        clear_ctx()

    def test_sets_phase_to_eval(self):
        """Should set phase to 'eval' during metric execution."""
        phases_during_eval = []

        def metric_that_checks_context(example, prediction, trace=None):
            phases_during_eval.append(get_ctx().get("phase"))
            return 1.0, "ok"

        logged = LoggedMetric(metric_that_checks_context)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        assert phases_during_eval == ["eval"]

    def test_captures_iteration_from_context(self):
        """Should capture iteration from context."""
        set_ctx(iteration=5)

        logged = LoggedMetric(simple_metric)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        record = logged.evaluations[0]
        assert record.iteration == 5

    def test_captures_candidate_idx_from_context(self):
        """Should capture candidate_idx from context."""
        set_ctx(iteration=3, candidate_idx=7)

        logged = LoggedMetric(simple_metric)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        record = logged.evaluations[0]
        assert record.candidate_idx == 7
        assert record.iteration == 3

    def test_restores_previous_phase(self):
        """Should restore previous phase after evaluation."""
        set_ctx(phase="reflection")

        logged = LoggedMetric(simple_metric)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        # Phase should be restored
        assert get_ctx().get("phase") == "reflection"


class TestLoggedMetricQueries:
    """Test query methods."""

    def setup_method(self):
        clear_ctx()
        self.logged = LoggedMetric(simple_metric)
        self._add_test_evaluations()

    def _add_test_evaluations(self):
        """Add test evaluations with different contexts."""
        examples = [
            MockExample(question="q1", answer="a1", id="ex1"),
            MockExample(question="q2", answer="a2", id="ex2"),
            MockExample(question="q3", answer="a3", id="ex3"),
        ]

        # Candidate 0 (seed) evaluations
        set_ctx(iteration=0, candidate_idx=0)
        self.logged(examples[0], MockPrediction(answer="a1"))  # Correct
        self.logged(examples[1], MockPrediction(answer="wrong"))  # Wrong
        self.logged(examples[2], MockPrediction(answer="a3"))  # Correct

        # Candidate 1 evaluations
        set_ctx(iteration=1, candidate_idx=1)
        self.logged(examples[0], MockPrediction(answer="a1"))  # Correct
        self.logged(examples[1], MockPrediction(answer="a2"))  # Now correct (improvement)
        self.logged(examples[2], MockPrediction(answer="wrong"))  # Now wrong (regression)

        # Candidate 2 evaluations
        set_ctx(iteration=2, candidate_idx=2)
        self.logged(examples[0], MockPrediction(answer="a1"))  # Correct
        self.logged(examples[1], MockPrediction(answer="a2"))  # Correct
        self.logged(examples[2], MockPrediction(answer="a3"))  # Correct

    def test_get_evaluations_for_example(self):
        """Should filter by example ID."""
        evals = self.logged.get_evaluations_for_example("ex1")
        assert len(evals) == 3  # One per candidate
        assert all(e.example_id == "ex1" for e in evals)

    def test_get_evaluations_for_candidate(self):
        """Should filter by candidate index."""
        evals = self.logged.get_evaluations_for_candidate(0)
        assert len(evals) == 3  # One per example
        assert all(e.candidate_idx == 0 for e in evals)

    def test_get_evaluations_for_iteration(self):
        """Should filter by iteration."""
        evals = self.logged.get_evaluations_for_iteration(1)
        assert len(evals) == 3  # One per example
        assert all(e.iteration == 1 for e in evals)

    def test_compute_lift(self):
        """Should compute lift between candidates."""
        # ex2: candidate 0 got it wrong (0.0), candidate 1 got it right (1.0)
        lift = self.logged.compute_lift("ex2", before_candidate=0, after_candidate=1)

        assert lift["lift"] == 1.0  # 1.0 - 0.0
        assert lift["before"].score == 0.0
        assert lift["after"].score == 1.0

    def test_compute_lift_missing_candidate(self):
        """Should handle missing candidates gracefully."""
        lift = self.logged.compute_lift("ex1", before_candidate=0, after_candidate=99)

        assert lift["lift"] is None
        assert lift["before"] is None
        assert lift["after"] is None

    def test_get_regressions(self):
        """Should find examples where candidates regressed."""
        regressions = self.logged.get_regressions(seed_candidate=0)

        # ex3: candidate 0 was correct (1.0), candidate 1 was wrong (0.0)
        assert len(regressions) >= 1
        ex3_regression = [r for r in regressions if r["example_id"] == "ex3"]
        assert len(ex3_regression) == 1
        assert ex3_regression[0]["delta"] == -1.0

    def test_get_improvements(self):
        """Should find examples where candidates improved."""
        improvements = self.logged.get_improvements(seed_candidate=0)

        # ex2: candidate 0 was wrong (0.0), candidates 1 and 2 were correct (1.0)
        assert len(improvements) >= 1
        ex2_improvements = [r for r in improvements if r["example_id"] == "ex2"]
        assert len(ex2_improvements) >= 1
        assert ex2_improvements[0]["delta"] == 1.0


class TestEvaluationRecord:
    """Test EvaluationRecord dataclass."""

    def test_required_fields(self):
        """Should have all required fields."""
        record = EvaluationRecord(
            eval_id="test-id",
            example_id="ex-123",
            candidate_idx=0,
            iteration=1,
            phase="eval",
            score=0.85,
            feedback="Good job",
        )

        assert record.eval_id == "test-id"
        assert record.example_id == "ex-123"
        assert record.candidate_idx == 0
        assert record.iteration == 1
        assert record.phase == "eval"
        assert record.score == 0.85
        assert record.feedback == "Good job"

    def test_optional_fields_defaults(self):
        """Optional fields should have defaults."""
        record = EvaluationRecord(
            eval_id="test-id",
            example_id="ex-123",
            candidate_idx=None,
            iteration=None,
            phase="eval",
            score=0.5,
            feedback=None,
        )

        assert record.prediction_ref is None
        assert record.prediction_preview == ""
        assert record.timestamp > 0


class TestLoggedMetricEdgeCases:
    """Test edge cases."""

    def setup_method(self):
        clear_ctx()

    def test_multiple_metric_instances(self):
        """Multiple LoggedMetric instances should be independent."""
        logged1 = LoggedMetric(simple_metric)
        logged2 = LoggedMetric(simple_metric)

        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged1(example, pred)

        assert len(logged1.evaluations) == 1
        assert len(logged2.evaluations) == 0

    def test_none_score(self):
        """Should handle None score gracefully."""

        def none_metric(example, prediction, trace=None):
            return None, "no score"

        logged = LoggedMetric(none_metric)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        record = logged.evaluations[0]
        assert record.score == 0.0  # Converted to 0.0

    def test_metric_with_kwargs(self):
        """Should pass through kwargs to metric."""
        received_kwargs = {}

        def kwarg_metric(example, prediction, trace=None, **kwargs):
            received_kwargs.update(kwargs)
            return 1.0, "ok"

        logged = LoggedMetric(kwarg_metric)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred, custom_kwarg="value")

        assert received_kwargs.get("custom_kwarg") == "value"


class TestLoggedMetricExceptionHandling:
    """Test exception handling in LoggedMetric."""

    def setup_method(self):
        clear_ctx()

    def test_metric_exception_returns_failure_score(self):
        """When metric throws, should return failure_score."""

        def failing_metric(example, prediction, trace=None):
            raise ValueError("Metric failed!")

        logged = LoggedMetric(failing_metric, failure_score=0.0)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        result = logged(example, pred)

        assert result == 0.0

    def test_metric_exception_records_evaluation(self):
        """When metric throws, should still record an evaluation."""

        def failing_metric(example, prediction, trace=None):
            raise ValueError("Metric failed!")

        logged = LoggedMetric(failing_metric, failure_score=0.0)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        assert len(logged.evaluations) == 1
        record = logged.evaluations[0]
        assert record.score == 0.0
        assert "Metric error" in record.feedback
        assert "Metric failed!" in record.feedback

    def test_custom_failure_score(self):
        """Should use custom failure_score when configured."""

        def failing_metric(example, prediction, trace=None):
            raise RuntimeError("Something went wrong")

        logged = LoggedMetric(failing_metric, failure_score=-1.0)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        result = logged(example, pred)

        assert result == -1.0
        assert logged.evaluations[0].score == -1.0

    def test_attribute_error_on_prediction(self):
        """Should handle AttributeError when accessing prediction attributes."""

        def metric_accessing_missing_attr(example, prediction, trace=None):
            # This simulates accessing an attribute that doesn't exist
            return prediction.nonexistent_field

        logged = LoggedMetric(metric_accessing_missing_attr, failure_score=0.0)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")  # Doesn't have nonexistent_field

        result = logged(example, pred)

        assert result == 0.0
        record = logged.evaluations[0]
        assert record.score == 0.0
        assert "Metric error" in record.feedback

    def test_exception_doesnt_break_context_restoration(self):
        """Context should be restored even when metric throws."""
        set_ctx(phase="reflection")

        def failing_metric(example, prediction, trace=None):
            raise ValueError("Boom!")

        logged = LoggedMetric(failing_metric, failure_score=0.0)
        example = MockExample(question="q", answer="4")
        pred = MockPrediction(answer="4")

        logged(example, pred)

        # Phase should be restored to previous value
        assert get_ctx().get("phase") == "reflection"

    def test_serialization_error_handled(self):
        """Should handle errors during prediction serialization."""

        class BadPrediction:
            """Prediction that fails to serialize."""

            def toDict(self):
                raise ValueError("Cannot serialize")

            def __str__(self):
                return "BadPrediction()"

        logged = LoggedMetric(simple_metric, failure_score=0.0)
        example = MockExample(question="q", answer="4")
        pred = BadPrediction()

        # Should not raise - serialization errors are caught
        logged(example, pred)

        record = logged.evaluations[0]
        # Fallback preview should be used
        assert "BadPrediction" in record.prediction_preview

    def test_example_serialization_error_handled(self):
        """Should handle errors during example input serialization."""

        class BadExample:
            """Example that fails to serialize."""

            question = "test"
            answer = "test"

            def inputs(self):
                raise ValueError("Cannot serialize inputs")

            def toDict(self):
                raise ValueError("Cannot serialize")

        logged = LoggedMetric(simple_metric, failure_score=0.0)
        example = BadExample()
        pred = MockPrediction(answer="4")

        # Should not raise
        logged(example, pred)

        record = logged.evaluations[0]
        assert "_error" in record.example_inputs
