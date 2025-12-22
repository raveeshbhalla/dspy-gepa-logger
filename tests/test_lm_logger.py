"""Tests for DSPyLMLogger.

Tests cover:
1. Basic LM call capture (start/end)
2. Context tag capture (iteration, phase, candidate_idx)
3. Error handling
4. Query methods
5. Summary statistics
"""

import pytest
import time
from unittest.mock import Mock

from dspy_gepa_logger.core.context import clear_ctx, set_ctx
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger, LMCall


class MockLMInstance:
    """Mock LM instance for testing."""

    def __init__(self, model: str = "test-model"):
        self.model = model


class TestLMCallDataclass:
    """Test LMCall dataclass."""

    def test_required_fields(self):
        """Should have required fields."""
        call = LMCall(
            call_id="test-123",
            start_time=1.0,
        )
        assert call.call_id == "test-123"
        assert call.start_time == 1.0

    def test_default_values(self):
        """Should have sensible defaults."""
        call = LMCall(call_id="test", start_time=0.0)
        assert call.end_time is None
        assert call.duration_ms == 0.0
        assert call.model == ""
        assert call.inputs == {}
        assert call.outputs == {}
        assert call.iteration is None
        assert call.phase is None
        assert call.candidate_idx is None


class TestBasicCapture:
    """Test basic LM call capture."""

    def setup_method(self):
        clear_ctx()
        self.logger = DSPyLMLogger()

    def test_capture_single_call(self):
        """Should capture a single LM call."""
        instance = MockLMInstance()

        self.logger.on_lm_start(
            call_id="call-1",
            instance=instance,
            inputs={"messages": [{"role": "user", "content": "Hello"}]},
        )

        self.logger.on_lm_end(
            call_id="call-1",
            outputs={"text": "Hi there!"},
        )

        assert len(self.logger) == 1
        call = self.logger.calls[0]
        assert call.call_id == "call-1"
        assert call.model == "test-model"
        assert call.inputs["messages"][0]["content"] == "Hello"
        assert call.outputs["text"] == "Hi there!"

    def test_capture_duration(self):
        """Should capture call duration."""
        instance = MockLMInstance()

        self.logger.on_lm_start("call-1", instance, {})
        time.sleep(0.05)  # 50ms
        self.logger.on_lm_end("call-1", {})

        call = self.logger.calls[0]
        assert call.duration_ms >= 40  # Allow some slack
        assert call.end_time is not None
        assert call.end_time > call.start_time

    def test_capture_multiple_calls(self):
        """Should capture multiple sequential calls."""
        instance = MockLMInstance()

        for i in range(5):
            self.logger.on_lm_start(f"call-{i}", instance, {"index": i})
            self.logger.on_lm_end(f"call-{i}", {"result": i * 2})

        assert len(self.logger) == 5
        for i, call in enumerate(self.logger.calls):
            assert call.call_id == f"call-{i}"
            assert call.inputs["index"] == i
            assert call.outputs["result"] == i * 2

    def test_handle_unknown_end(self):
        """Should ignore end call for unknown call_id."""
        instance = MockLMInstance()

        # End without start
        self.logger.on_lm_end("unknown-call", {"result": "test"})

        assert len(self.logger) == 0

    def test_model_name_extraction(self):
        """Should extract model name from various instance types."""
        # With model attribute
        instance1 = MockLMInstance("gpt-4")
        self.logger.on_lm_start("call-1", instance1, {})
        self.logger.on_lm_end("call-1", {})
        assert self.logger.calls[0].model == "gpt-4"

        # Without model attribute (uses class name)
        instance2 = Mock()
        del instance2.model  # Remove model attribute
        self.logger.on_lm_start("call-2", instance2, {})
        self.logger.on_lm_end("call-2", {})
        assert "Mock" in self.logger.calls[1].model


class TestContextCapture:
    """Test context tag capture."""

    def setup_method(self):
        clear_ctx()
        self.logger = DSPyLMLogger()
        self.instance = MockLMInstance()

    def test_captures_iteration(self):
        """Should capture iteration from context."""
        set_ctx(iteration=5)

        self.logger.on_lm_start("call-1", self.instance, {})
        self.logger.on_lm_end("call-1", {})

        assert self.logger.calls[0].iteration == 5

    def test_captures_phase(self):
        """Should capture phase from context."""
        set_ctx(phase="reflection")

        self.logger.on_lm_start("call-1", self.instance, {})
        self.logger.on_lm_end("call-1", {})

        assert self.logger.calls[0].phase == "reflection"

    def test_captures_candidate_idx(self):
        """Should capture candidate_idx from context."""
        set_ctx(candidate_idx=3)

        self.logger.on_lm_start("call-1", self.instance, {})
        self.logger.on_lm_end("call-1", {})

        assert self.logger.calls[0].candidate_idx == 3

    def test_captures_full_context(self):
        """Should capture all context fields."""
        set_ctx(iteration=2, phase="eval", candidate_idx=7)

        self.logger.on_lm_start("call-1", self.instance, {})
        self.logger.on_lm_end("call-1", {})

        call = self.logger.calls[0]
        assert call.iteration == 2
        assert call.phase == "eval"
        assert call.candidate_idx == 7

    def test_context_at_start_time(self):
        """Should capture context at call start, not end."""
        set_ctx(iteration=1, phase="reflection")

        self.logger.on_lm_start("call-1", self.instance, {})

        # Change context before end
        set_ctx(iteration=2, phase="proposal")

        self.logger.on_lm_end("call-1", {})

        # Should have context from start time
        call = self.logger.calls[0]
        assert call.iteration == 1
        assert call.phase == "reflection"

    def test_no_context(self):
        """Should handle calls with no context set."""
        self.logger.on_lm_start("call-1", self.instance, {})
        self.logger.on_lm_end("call-1", {})

        call = self.logger.calls[0]
        assert call.iteration is None
        assert call.phase is None
        assert call.candidate_idx is None


class TestErrorHandling:
    """Test error handling for failed LM calls."""

    def setup_method(self):
        clear_ctx()
        self.logger = DSPyLMLogger()
        self.instance = MockLMInstance()

    def test_capture_error(self):
        """Should capture error information."""
        self.logger.on_lm_start("call-1", self.instance, {})
        self.logger.on_lm_error(
            "call-1",
            self.instance,
            ValueError("API rate limit exceeded"),
        )

        assert len(self.logger) == 1
        call = self.logger.calls[0]
        assert "API rate limit exceeded" in call.outputs["error"]
        assert call.outputs["error_type"] == "ValueError"

    def test_error_captures_duration(self):
        """Should capture duration even on error."""
        self.logger.on_lm_start("call-1", self.instance, {})
        time.sleep(0.02)
        self.logger.on_lm_error("call-1", self.instance, Exception("error"))

        call = self.logger.calls[0]
        assert call.duration_ms >= 15
        assert call.end_time is not None


class TestQueryMethods:
    """Test query/filter methods."""

    def setup_method(self):
        clear_ctx()
        self.logger = DSPyLMLogger()
        self.instance = MockLMInstance()
        self._add_test_calls()

    def _add_test_calls(self):
        """Add test calls with various contexts."""
        # Iteration 0
        set_ctx(iteration=0, phase="eval", candidate_idx=0)
        self.logger.on_lm_start("call-0-eval", self.instance, {})
        self.logger.on_lm_end("call-0-eval", {})

        # Iteration 1 - reflection
        set_ctx(iteration=1, phase="reflection", candidate_idx=0)
        self.logger.on_lm_start("call-1-refl", self.instance, {})
        self.logger.on_lm_end("call-1-refl", {})

        # Iteration 1 - proposal
        set_ctx(iteration=1, phase="proposal", candidate_idx=0)
        self.logger.on_lm_start("call-1-prop", self.instance, {})
        self.logger.on_lm_end("call-1-prop", {})

        # Iteration 1 - eval
        set_ctx(iteration=1, phase="eval", candidate_idx=1)
        self.logger.on_lm_start("call-1-eval", self.instance, {})
        self.logger.on_lm_end("call-1-eval", {})

        # Iteration 2
        set_ctx(iteration=2, phase="validation", candidate_idx=2)
        self.logger.on_lm_start("call-2-val", self.instance, {})
        self.logger.on_lm_end("call-2-val", {})

        # Untagged call
        clear_ctx()
        self.logger.on_lm_start("call-untagged", self.instance, {})
        self.logger.on_lm_end("call-untagged", {})

    def test_get_calls_for_iteration(self):
        """Should filter by iteration."""
        calls = self.logger.get_calls_for_iteration(1)
        assert len(calls) == 3
        assert all(c.iteration == 1 for c in calls)

    def test_get_calls_for_phase(self):
        """Should filter by phase."""
        calls = self.logger.get_calls_for_phase("reflection")
        assert len(calls) == 1
        assert calls[0].call_id == "call-1-refl"

    def test_get_calls_for_candidate(self):
        """Should filter by candidate."""
        calls = self.logger.get_calls_for_candidate(0)
        assert len(calls) == 3  # iter 0 eval, iter 1 refl, iter 1 prop

    def test_get_reflection_calls(self):
        """Should return reflection calls."""
        calls = self.logger.get_reflection_calls()
        assert len(calls) == 1
        assert calls[0].phase == "reflection"

    def test_get_proposal_calls(self):
        """Should return proposal calls."""
        calls = self.logger.get_proposal_calls()
        assert len(calls) == 1
        assert calls[0].phase == "proposal"

    def test_get_eval_calls(self):
        """Should return eval/validation/minibatch calls."""
        calls = self.logger.get_eval_calls()
        assert len(calls) == 3
        assert all(c.phase in ("eval", "validation") for c in calls)

    def test_get_untagged_calls(self):
        """Should return untagged calls."""
        calls = self.logger.get_untagged_calls()
        assert len(calls) == 1
        assert calls[0].call_id == "call-untagged"


class TestSummary:
    """Test summary statistics."""

    def setup_method(self):
        clear_ctx()
        self.logger = DSPyLMLogger()

    def test_empty_summary(self):
        """Should handle empty logger."""
        summary = self.logger.get_summary()
        assert summary["total_calls"] == 0
        assert summary["total_duration_ms"] == 0

    def test_summary_with_calls(self):
        """Should compute summary stats."""
        instance = MockLMInstance()

        # Add calls with different contexts
        set_ctx(iteration=0, phase="eval")
        self.logger.on_lm_start("call-1", instance, {})
        self.logger.on_lm_end("call-1", {})

        set_ctx(iteration=0, phase="eval")
        self.logger.on_lm_start("call-2", instance, {})
        self.logger.on_lm_end("call-2", {})

        set_ctx(iteration=1, phase="reflection")
        self.logger.on_lm_start("call-3", instance, {})
        self.logger.on_lm_end("call-3", {})

        summary = self.logger.get_summary()
        assert summary["total_calls"] == 3
        assert summary["calls_by_phase"]["eval"] == 2
        assert summary["calls_by_phase"]["reflection"] == 1
        assert summary["calls_by_iteration"][0] == 2
        assert summary["calls_by_iteration"][1] == 1


class TestClearAndLen:
    """Test utility methods."""

    def setup_method(self):
        clear_ctx()
        self.logger = DSPyLMLogger()

    def test_len(self):
        """Should return number of calls."""
        instance = MockLMInstance()

        assert len(self.logger) == 0

        self.logger.on_lm_start("call-1", instance, {})
        self.logger.on_lm_end("call-1", {})

        assert len(self.logger) == 1

    def test_clear(self):
        """Should clear all state."""
        instance = MockLMInstance()

        self.logger.on_lm_start("call-1", instance, {})
        self.logger.on_lm_end("call-1", {})

        assert len(self.logger) == 1

        self.logger.clear()

        assert len(self.logger) == 0
        assert self.logger._pending == {}
        assert self.logger._start_time is None


class TestConcurrentCalls:
    """Test handling of concurrent/overlapping calls."""

    def setup_method(self):
        clear_ctx()
        self.logger = DSPyLMLogger()

    def test_overlapping_calls(self):
        """Should handle overlapping calls correctly."""
        instance = MockLMInstance()

        # Start multiple calls
        self.logger.on_lm_start("call-1", instance, {"index": 1})
        self.logger.on_lm_start("call-2", instance, {"index": 2})
        self.logger.on_lm_start("call-3", instance, {"index": 3})

        # End in different order
        self.logger.on_lm_end("call-2", {"result": 2})
        self.logger.on_lm_end("call-1", {"result": 1})
        self.logger.on_lm_end("call-3", {"result": 3})

        assert len(self.logger) == 3

        # Verify each call has correct inputs/outputs
        by_id = {c.call_id: c for c in self.logger.calls}
        assert by_id["call-1"].inputs["index"] == 1
        assert by_id["call-1"].outputs["result"] == 1
        assert by_id["call-2"].inputs["index"] == 2
        assert by_id["call-2"].outputs["result"] == 2

    def test_pending_calls(self):
        """Should track pending calls in summary."""
        instance = MockLMInstance()

        self.logger.on_lm_start("call-1", instance, {})
        self.logger.on_lm_start("call-2", instance, {})

        summary = self.logger.get_summary()
        assert summary["pending_calls"] == 2

        self.logger.on_lm_end("call-1", {})
        summary = self.logger.get_summary()
        assert summary["pending_calls"] == 1
