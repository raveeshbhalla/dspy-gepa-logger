"""Tests for GEPATracker (v2).

Tests cover:
1. Component integration
2. wrap_metric() functionality
3. wrap_proposer() functionality
4. get_stop_callback() and get_dspy_callbacks()
5. Query methods (get_summary, get_candidate_diff, etc.)
6. Context flow between components
7. Properties
"""

import pytest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock, MagicMock

from dspy_gepa_logger.core.context import clear_ctx, get_ctx, set_ctx
from dspy_gepa_logger.core.tracker_v2 import GEPATracker, CandidateDiff, LoggedLM
from dspy_gepa_logger.core.state_logger import GEPAStateLogger
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger
from dspy_gepa_logger.core.logged_metric import LoggedMetric
from dspy_gepa_logger.core.logged_proposer import LoggedInstructionProposer


@dataclass
class MockGEPAState:
    """Mock GEPAState for testing."""

    i: int = 0
    total_num_evals: int = 0
    program_candidates: list[dict[str, str]] = field(default_factory=list)
    parent_program_for_candidate: list[list[int | None]] = field(default_factory=list)
    pareto_front_valset: dict[str, float] = field(default_factory=dict)
    program_at_pareto_front_valset: dict[str, set[int]] = field(default_factory=dict)


class MockExample:
    """Mock DSPy Example for testing."""

    def __init__(self, id_: str, **fields):
        self.id = id_
        self._fields = fields
        for k, v in fields.items():
            setattr(self, k, v)

    def toDict(self):
        return {"id": self.id, **self._fields}


class MockPrediction:
    """Mock prediction for testing."""

    def __init__(self, answer: str):
        self.answer = answer

    def toDict(self):
        return {"answer": self.answer}


class TestTrackerInitialization:
    """Test GEPATracker initialization."""

    def setup_method(self):
        clear_ctx()

    def test_creates_state_logger(self):
        """Should create GEPAStateLogger."""
        tracker = GEPATracker()
        assert tracker.state_logger is not None
        assert isinstance(tracker.state_logger, GEPAStateLogger)

    def test_creates_lm_logger_by_default(self):
        """Should create DSPyLMLogger by default."""
        tracker = GEPATracker()
        assert tracker.lm_logger is not None
        assert isinstance(tracker.lm_logger, DSPyLMLogger)

    def test_no_lm_logger_if_disabled(self):
        """Should not create LM logger if capture_lm_calls=False."""
        tracker = GEPATracker(capture_lm_calls=False)
        assert tracker.lm_logger is None

    def test_no_metric_logger_initially(self):
        """Should not have metric logger until wrap_metric called."""
        tracker = GEPATracker()
        assert tracker.metric_logger is None

    def test_no_proposer_logger_initially(self):
        """Should not have proposer logger until wrap_proposer called."""
        tracker = GEPATracker()
        assert tracker.proposer_logger is None


class TestWrapMetric:
    """Test wrap_metric() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()

    def test_returns_logged_metric(self):
        """Should return LoggedMetric instance."""

        def my_metric(example, prediction, trace=None):
            return 1.0

        wrapped = self.tracker.wrap_metric(my_metric)
        assert isinstance(wrapped, LoggedMetric)

    def test_stores_metric_logger(self):
        """Should store metric logger in tracker."""

        def my_metric(example, prediction, trace=None):
            return 1.0

        wrapped = self.tracker.wrap_metric(my_metric)
        assert self.tracker.metric_logger is wrapped

    def test_wrapped_metric_captures_evaluations(self):
        """Wrapped metric should capture evaluation records."""

        def my_metric(example, prediction, trace=None):
            return 0.8

        wrapped = self.tracker.wrap_metric(my_metric)

        example = MockExample("ex1", question="What is 2+2?")
        prediction = MockPrediction("4")
        set_ctx(iteration=1, candidate_idx=0)

        result = wrapped(example, prediction)

        assert result == 0.8
        assert len(self.tracker.metric_logger.evaluations) == 1
        eval_record = self.tracker.metric_logger.evaluations[0]
        assert eval_record.score == 0.8
        assert eval_record.iteration == 1
        assert eval_record.candidate_idx == 0


class TestWrapProposer:
    """Test wrap_proposer() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()

    def test_returns_logged_proposer(self):
        """Should return LoggedInstructionProposer instance."""

        class MockProposer:
            def propose(self, **kwargs):
                return []

        wrapped = self.tracker.wrap_proposer(MockProposer())
        assert isinstance(wrapped, LoggedInstructionProposer)

    def test_stores_proposer_logger(self):
        """Should store proposer logger in tracker."""

        class MockProposer:
            def propose(self, **kwargs):
                return []

        wrapped = self.tracker.wrap_proposer(MockProposer())
        assert self.tracker.proposer_logger is wrapped


class TestGetStopCallback:
    """Test get_stop_callback() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()

    def test_returns_state_logger(self):
        """Should return the GEPAStateLogger."""
        callback = self.tracker.get_stop_callback()
        assert callback is self.tracker.state_logger

    def test_callback_returns_false(self):
        """Callback should return False (never stops)."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(i=0)
        result = callback(state)
        assert result is False

    def test_callback_captures_state(self):
        """Callback should capture iteration state."""
        callback = self.tracker.get_stop_callback()

        state = MockGEPAState(
            i=0,
            program_candidates=[{"instructions": "seed"}],
            parent_program_for_candidate=[[None]],
        )
        callback(state)

        assert len(self.tracker.state_logger.deltas) == 1
        assert self.tracker.seed_candidate == {"instructions": "seed"}


class TestGetDspyCallbacks:
    """Test get_dspy_callbacks() functionality."""

    def setup_method(self):
        clear_ctx()

    def test_returns_list_with_lm_logger(self):
        """Should return list containing LM logger."""
        tracker = GEPATracker()
        callbacks = tracker.get_dspy_callbacks()
        assert len(callbacks) == 1
        assert callbacks[0] is tracker.lm_logger

    def test_returns_empty_if_lm_disabled(self):
        """Should return empty list if LM capture disabled."""
        tracker = GEPATracker(capture_lm_calls=False)
        callbacks = tracker.get_dspy_callbacks()
        assert callbacks == []


class TestGetSummary:
    """Test get_summary() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()
        self._run_simulation()

    def _run_simulation(self):
        """Simulate a GEPA run."""
        # Wrap a metric
        def my_metric(example, prediction, trace=None):
            return 0.8

        wrapped = self.tracker.wrap_metric(my_metric)

        # Simulate state iterations
        callback = self.tracker.get_stop_callback()
        for i in range(3):
            set_ctx(iteration=i)
            state = MockGEPAState(
                i=i,
                total_num_evals=(i + 1) * 10,
                program_candidates=[{"inst": f"cand_{j}"} for j in range(i + 1)],
                parent_program_for_candidate=[[None]] + [[j - 1] for j in range(1, i + 1)],
            )
            callback(state)

        # Simulate some evaluations
        for i in range(5):
            set_ctx(iteration=i % 3, candidate_idx=i % 2)
            example = MockExample(f"ex{i}", question=f"Q{i}")
            prediction = MockPrediction(f"A{i}")
            wrapped(example, prediction)

    def test_includes_state_summary(self):
        """Should include state logger summary."""
        summary = self.tracker.get_summary()
        assert "state" in summary
        assert summary["state"]["total_iterations"] == 3

    def test_includes_lm_summary(self):
        """Should include LM logger summary."""
        summary = self.tracker.get_summary()
        assert "lm_calls" in summary

    def test_includes_evaluation_summary(self):
        """Should include evaluation summary."""
        summary = self.tracker.get_summary()
        assert "evaluations" in summary
        assert summary["evaluations"]["total_evaluations"] == 5


class TestGetCandidateDiff:
    """Test get_candidate_diff() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()
        self._run_simulation()

    def _run_simulation(self):
        """Simulate a GEPA run with candidates."""

        def my_metric(example, prediction, trace=None):
            return 0.8

        wrapped = self.tracker.wrap_metric(my_metric)
        callback = self.tracker.get_stop_callback()

        # Iteration 0: seed
        state0 = MockGEPAState(
            i=0,
            program_candidates=[{"instructions": "seed prompt"}],
            parent_program_for_candidate=[[None]],
        )
        callback(state0)

        # Iteration 1: child with changed prompt
        state1 = MockGEPAState(
            i=1,
            program_candidates=[
                {"instructions": "seed prompt"},
                {"instructions": "improved prompt"},
            ],
            parent_program_for_candidate=[[None], [0]],
        )
        callback(state1)

        # Simulate evaluations for both candidates
        for cand_idx in [0, 1]:
            set_ctx(iteration=cand_idx, candidate_idx=cand_idx)
            example = MockExample("ex1", question="Q1")
            prediction = MockPrediction(f"A{cand_idx}")
            wrapped(example, prediction)

    def test_returns_candidate_diff(self):
        """Should return CandidateDiff instance."""
        diff = self.tracker.get_candidate_diff(0, 1)
        assert isinstance(diff, CandidateDiff)

    def test_captures_prompt_changes(self):
        """Should capture prompt differences."""
        diff = self.tracker.get_candidate_diff(0, 1)
        assert "instructions" in diff.prompt_changes
        old, new = diff.prompt_changes["instructions"]
        assert old == "seed prompt"
        assert new == "improved prompt"

    def test_captures_lineage(self):
        """Should capture candidate lineage."""
        diff = self.tracker.get_candidate_diff(0, 1)
        assert diff.lineage == [1, 0]

    def test_captures_evaluation_changes(self):
        """Should capture evaluation changes."""
        diff = self.tracker.get_candidate_diff(0, 1)
        assert len(diff.evaluation_changes) > 0


class TestLMCallQueries:
    """Test LM call query methods."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()

    def test_get_lm_calls_for_iteration_empty(self):
        """Should return empty list if no calls."""
        calls = self.tracker.get_lm_calls_for_iteration(0)
        assert calls == []

    def test_get_lm_calls_for_phase_empty(self):
        """Should return empty list if no calls."""
        calls = self.tracker.get_lm_calls_for_phase("eval")
        assert calls == []

    def test_get_lm_calls_disabled_returns_empty(self):
        """Should return empty if LM capture disabled."""
        tracker = GEPATracker(capture_lm_calls=False)
        calls = tracker.get_lm_calls_for_iteration(0)
        assert calls == []


class TestEvaluationQueries:
    """Test evaluation query methods."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()

        def my_metric(example, prediction, trace=None):
            return 0.8

        self.wrapped = self.tracker.wrap_metric(my_metric)

    def test_get_evaluations_for_example(self):
        """Should return evaluations for example."""
        set_ctx(iteration=0, candidate_idx=0)
        example = MockExample("ex1", question="Q1")
        prediction = MockPrediction("A1")
        self.wrapped(example, prediction)

        evals = self.tracker.get_evaluations_for_example("ex1")
        assert len(evals) == 1
        assert evals[0].example_id == "ex1"

    def test_get_evaluations_for_candidate(self):
        """Should return evaluations for candidate."""
        set_ctx(iteration=0, candidate_idx=5)
        example = MockExample("ex1", question="Q1")
        prediction = MockPrediction("A1")
        self.wrapped(example, prediction)

        evals = self.tracker.get_evaluations_for_candidate(5)
        assert len(evals) == 1
        assert evals[0].candidate_idx == 5

    def test_evaluations_empty_without_metric(self):
        """Should return empty if no metric wrapped."""
        tracker = GEPATracker()
        evals = tracker.get_evaluations_for_example("ex1")
        assert evals == []


class TestComputeLift:
    """Test compute_lift() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()
        self._run_simulation()

    def _run_simulation(self):
        """Simulate evaluations for two candidates."""
        scores = {0: [0.5, 0.6, 0.4], 1: [0.7, 0.8, 0.6]}

        def my_metric(example, prediction, trace=None):
            cand = get_ctx().get("candidate_idx", 0)
            ex_idx = int(example.id.replace("ex", ""))
            return scores[cand][ex_idx]

        wrapped = self.tracker.wrap_metric(my_metric)

        for cand_idx in [0, 1]:
            for i in range(3):
                set_ctx(iteration=cand_idx, candidate_idx=cand_idx)
                example = MockExample(f"ex{i}", question=f"Q{i}")
                prediction = MockPrediction(f"A{i}")
                wrapped(example, prediction)

    def test_returns_lift_stats(self):
        """Should return lift statistics."""
        lift = self.tracker.compute_lift(0, 1)
        assert "mean_lift" in lift
        assert lift["mean_lift"] > 0  # 1 improved vs 0

    def test_no_metric_returns_error(self):
        """Should return error if no metric wrapped."""
        tracker = GEPATracker()
        lift = tracker.compute_lift(0, 1)
        assert "error" in lift


class TestGetRegressions:
    """Test get_regressions() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()
        self._run_simulation()

    def _run_simulation(self):
        """Simulate evaluations with regressions."""
        # Candidate 1 regresses on ex2
        scores = {0: [0.5, 0.6, 0.8], 1: [0.7, 0.8, 0.3]}

        def my_metric(example, prediction, trace=None):
            cand = get_ctx().get("candidate_idx", 0)
            ex_idx = int(example.id.replace("ex", ""))
            return scores[cand][ex_idx]

        wrapped = self.tracker.wrap_metric(my_metric)

        for cand_idx in [0, 1]:
            for i in range(3):
                set_ctx(iteration=cand_idx, candidate_idx=cand_idx)
                example = MockExample(f"ex{i}", question=f"Q{i}")
                prediction = MockPrediction(f"A{i}")
                wrapped(example, prediction)

    def test_finds_regressions(self):
        """Should find examples where candidate regressed."""
        regressions = self.tracker.get_regressions(0, 1)
        # ex2: 0.8 -> 0.3 is a regression
        assert len(regressions) == 1
        assert regressions[0]["example_id"] == "ex2"

    def test_no_metric_returns_empty(self):
        """Should return empty if no metric wrapped."""
        tracker = GEPATracker()
        regressions = tracker.get_regressions(0, 1)
        assert regressions == []


class TestParetoAndLineage:
    """Test Pareto and lineage methods."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()
        self._run_simulation()

    def _run_simulation(self):
        """Simulate iterations with Pareto evolution."""
        callback = self.tracker.get_stop_callback()

        # Build lineage: 0 -> 1 -> 2
        states = [
            MockGEPAState(
                i=0,
                program_candidates=[{"inst": "seed"}],
                parent_program_for_candidate=[[None]],
                pareto_front_valset={"ex1": 0.5},
                program_at_pareto_front_valset={"ex1": {0}},
            ),
            MockGEPAState(
                i=1,
                program_candidates=[{"inst": "seed"}, {"inst": "gen1"}],
                parent_program_for_candidate=[[None], [0]],
                pareto_front_valset={"ex1": 0.7},
                program_at_pareto_front_valset={"ex1": {1}},
            ),
            MockGEPAState(
                i=2,
                program_candidates=[{"inst": "seed"}, {"inst": "gen1"}, {"inst": "gen2"}],
                parent_program_for_candidate=[[None], [0], [1]],
                pareto_front_valset={"ex1": 0.9},
                program_at_pareto_front_valset={"ex1": {2}},
            ),
        ]

        for state in states:
            callback(state)

    def test_get_pareto_evolution(self):
        """Should return Pareto frontier at each iteration."""
        evolution = self.tracker.get_pareto_evolution()
        assert len(evolution) == 3
        assert evolution[0]["ex1"][0] == 0.5
        assert evolution[1]["ex1"][0] == 0.7
        assert evolution[2]["ex1"][0] == 0.9

    def test_get_lineage(self):
        """Should trace lineage back to seed."""
        lineage = self.tracker.get_lineage(2)
        assert lineage == [2, 1, 0]

    def test_get_all_candidates(self):
        """Should return all candidate prompts."""
        candidates = self.tracker.get_all_candidates()
        assert len(candidates) == 3


class TestProperties:
    """Test tracker properties."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()

    def test_seed_candidate_none_initially(self):
        """Should be None before any state captured."""
        assert self.tracker.seed_candidate is None

    def test_seed_candidate_after_state(self):
        """Should return seed after state captured."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(
            i=0,
            program_candidates=[{"instructions": "seed"}],
            parent_program_for_candidate=[[None]],
        )
        callback(state)

        assert self.tracker.seed_candidate == {"instructions": "seed"}
        assert self.tracker.seed_candidate_idx == 0

    def test_final_candidates(self):
        """Should return final candidate list."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(
            i=0,
            program_candidates=[{"inst": "a"}, {"inst": "b"}],
            parent_program_for_candidate=[[None], [0]],
        )
        callback(state)

        assert len(self.tracker.final_candidates) == 2

    def test_final_pareto(self):
        """Should return final Pareto scores."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(
            i=0,
            pareto_front_valset={"ex1": 0.9},
        )
        callback(state)

        assert self.tracker.final_pareto == {"ex1": 0.9}

    def test_iterations_property(self):
        """Should return iteration deltas."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(i=0)
        callback(state)

        assert len(self.tracker.iterations) == 1

    def test_metadata_property(self):
        """Should return iteration metadata."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(i=0)
        callback(state)

        assert len(self.tracker.metadata) == 1

    def test_lm_calls_property(self):
        """Should return LM calls list."""
        assert self.tracker.lm_calls == []

    def test_evaluations_property(self):
        """Should return evaluations list."""
        assert self.tracker.evaluations == []


class TestClear:
    """Test clear() functionality."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()
        self._populate()

    def _populate(self):
        """Add some data to tracker."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(
            i=0,
            program_candidates=[{"inst": "test"}],
            parent_program_for_candidate=[[None]],
        )
        callback(state)

        def my_metric(example, prediction, trace=None):
            return 1.0

        wrapped = self.tracker.wrap_metric(my_metric)
        set_ctx(iteration=0, candidate_idx=0)
        wrapped(MockExample("ex1"), MockPrediction("A1"))

    def test_clears_state_logger(self):
        """Should clear state logger."""
        assert len(self.tracker.iterations) > 0
        self.tracker.clear()
        assert len(self.tracker.iterations) == 0

    def test_clears_metric_logger(self):
        """Should clear metric logger."""
        assert len(self.tracker.evaluations) > 0
        self.tracker.clear()
        assert len(self.tracker.evaluations) == 0

    def test_clears_context(self):
        """Should clear context."""
        set_ctx(iteration=5)
        self.tracker.clear()
        assert get_ctx().get("iteration") is None


class TestRepr:
    """Test string representation."""

    def setup_method(self):
        clear_ctx()

    def test_repr_empty(self):
        """Should show zeros when empty."""
        tracker = GEPATracker()
        repr_str = repr(tracker)
        assert "GEPATracker" in repr_str
        assert "iterations=0" in repr_str

    def test_repr_with_data(self):
        """Should show counts when populated."""
        tracker = GEPATracker()
        callback = tracker.get_stop_callback()

        for i in range(3):
            state = MockGEPAState(
                i=i,
                program_candidates=[{"inst": f"c{j}"} for j in range(i + 1)],
                parent_program_for_candidate=[[None]] + [[j - 1] for j in range(1, i + 1)],
            )
            callback(state)

        repr_str = repr(tracker)
        assert "iterations=3" in repr_str


class TestContextFlow:
    """Test context flows correctly between components."""

    def setup_method(self):
        clear_ctx()
        self.tracker = GEPATracker()

    def test_state_logger_sets_iteration(self):
        """State logger should set iteration in context."""
        callback = self.tracker.get_stop_callback()
        state = MockGEPAState(i=5)
        callback(state)

        ctx = get_ctx()
        assert ctx.get("iteration") == 5

    def test_metric_sets_phase(self):
        """Metric wrapper should set phase during evaluation."""
        phases_seen = []

        def my_metric(example, prediction, trace=None):
            phases_seen.append(get_ctx().get("phase"))
            return 1.0

        wrapped = self.tracker.wrap_metric(my_metric)
        wrapped(MockExample("ex1"), MockPrediction("A1"))

        assert phases_seen == ["eval"]

    def test_proposer_sets_phases(self):
        """Proposer wrapper should set reflection/proposal phases."""
        phases_seen = []

        class MockProposer:
            def propose(self, **kwargs):
                phases_seen.append(get_ctx().get("phase"))
                return []

        wrapped = self.tracker.wrap_proposer(MockProposer())
        set_ctx(iteration=0)
        wrapped.propose(
            current_instructions={},
            candidate_idx=0,
            reflection_data=None,
        )

        # Should have seen "proposal" phase during propose
        assert "proposal" in phases_seen


class TestLoggedLMContextRestoration:
    """Test LoggedLM properly saves and restores phase context."""

    def setup_method(self):
        clear_ctx()

    def test_restores_previous_phase(self):
        """LoggedLM should restore previous phase after call."""
        call_count = 0

        class MockLM:
            def __call__(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                return "response"

        # Set initial phase
        set_ctx(phase="eval")
        assert get_ctx().get("phase") == "eval"

        # Create LoggedLM with different phase
        logged_lm = LoggedLM(MockLM(), phase="reflection")

        # Call should temporarily set phase to "reflection"
        logged_lm("test input")

        # After call, phase should be restored to "eval"
        assert get_ctx().get("phase") == "eval"
        assert call_count == 1

    def test_restores_none_phase(self):
        """LoggedLM should restore None phase after call."""
        class MockLM:
            def __call__(self, *args, **kwargs):
                return "response"

        # No initial phase set
        assert get_ctx().get("phase") is None

        logged_lm = LoggedLM(MockLM(), phase="proposal")
        logged_lm("test input")

        # Phase should be cleared (restored to None)
        assert get_ctx().get("phase") is None

    def test_phase_is_set_during_call(self):
        """LoggedLM should set phase during the call."""
        captured_phase = None

        class MockLM:
            def __call__(self, *args, **kwargs):
                nonlocal captured_phase
                captured_phase = get_ctx().get("phase")
                return "response"

        set_ctx(phase="eval")
        logged_lm = LoggedLM(MockLM(), phase="reflection")
        logged_lm("test input")

        # During call, phase should have been "reflection"
        assert captured_phase == "reflection"
        # After call, restored to "eval"
        assert get_ctx().get("phase") == "eval"

    def test_restores_on_exception(self):
        """LoggedLM should restore phase even on exception."""
        class MockLM:
            def __call__(self, *args, **kwargs):
                raise ValueError("test error")

        set_ctx(phase="eval")
        logged_lm = LoggedLM(MockLM(), phase="reflection")

        with pytest.raises(ValueError):
            logged_lm("test input")

        # Phase should still be restored
        assert get_ctx().get("phase") == "eval"

    def test_nested_logged_lm_calls(self):
        """Nested LoggedLM calls should properly restore phases."""
        phases_during_calls = []

        class OuterMockLM:
            def __init__(self, inner_lm):
                self.inner_lm = inner_lm

            def __call__(self, *args, **kwargs):
                phases_during_calls.append(("outer_before", get_ctx().get("phase")))
                self.inner_lm("inner call")
                phases_during_calls.append(("outer_after", get_ctx().get("phase")))
                return "outer response"

        class InnerMockLM:
            def __call__(self, *args, **kwargs):
                phases_during_calls.append(("inner", get_ctx().get("phase")))
                return "inner response"

        inner_logged = LoggedLM(InnerMockLM(), phase="inner_phase")
        outer_logged = LoggedLM(OuterMockLM(inner_logged), phase="outer_phase")

        set_ctx(phase="initial")
        outer_logged("test")

        # Verify phase transitions
        assert phases_during_calls == [
            ("outer_before", "outer_phase"),
            ("inner", "inner_phase"),
            ("outer_after", "outer_phase"),  # Restored after inner call
        ]
        # Final state should be restored to initial
        assert get_ctx().get("phase") == "initial"

    def test_delegates_attributes_to_base_lm(self):
        """LoggedLM should delegate attribute access to base LM."""
        class MockLM:
            model_name = "test-model"
            temperature = 0.7

            def __call__(self, *args, **kwargs):
                return "response"

        logged_lm = LoggedLM(MockLM(), phase="test")

        assert logged_lm.model_name == "test-model"
        assert logged_lm.temperature == 0.7


class TestLogPushWorker:
    """Test LogPushWorker functionality."""

    def test_tracks_dropped_logs_when_queue_full(self):
        """Should count dropped logs when queue is full."""
        from dspy_gepa_logger.core.tracker_v2 import LogPushWorker
        import time

        pushed = []

        def slow_push(log_type, content, timestamp, iteration, phase):
            # Simulate slow push to cause queue backup
            time.sleep(0.05)
            pushed.append((log_type, content))

        worker = LogPushWorker(push_callback=slow_push, max_queue_size=2)
        worker.start()

        # Rapidly push more than queue can hold
        for i in range(10):
            worker.push("stdout", f"msg{i}", time.time(), None, None)

        worker.stop(timeout=2.0)

        # Should have dropped some entries
        assert worker.dropped_count > 0
        # Some should have been pushed
        assert len(pushed) > 0

    def test_tracks_push_errors(self):
        """Should count push errors."""
        from dspy_gepa_logger.core.tracker_v2 import LogPushWorker
        import time

        error_count = 0

        def failing_push(log_type, content, timestamp, iteration, phase):
            nonlocal error_count
            error_count += 1
            raise RuntimeError("Push failed")

        worker = LogPushWorker(push_callback=failing_push, max_queue_size=10)
        worker.start()

        # Push some entries
        for i in range(3):
            worker.push("stdout", f"msg{i}", time.time(), None, None)

        # Give worker time to process
        time.sleep(0.2)
        worker.stop(timeout=1.0)

        # All pushes should have failed
        assert worker.push_error_count == error_count
        assert worker.push_error_count > 0

    def test_properties_accessible(self):
        """Should expose dropped_count and push_error_count properties."""
        from dspy_gepa_logger.core.tracker_v2 import LogPushWorker

        worker = LogPushWorker(push_callback=lambda *args: None)

        # Initially zero
        assert worker.dropped_count == 0
        assert worker.push_error_count == 0


class TestStreamCaptureGlobalGuard:
    """Test stdout/stderr capture with global double-wrap protection."""

    def setup_method(self):
        """Reset global capture state before each test."""
        import sys
        import dspy_gepa_logger.core.tracker_v2 as tracker_module

        # Clear global state
        with tracker_module._global_capture_lock:
            tracker_module._global_stdout_capture = None
            tracker_module._global_stderr_capture = None
            tracker_module._global_capture_owner = None

        clear_ctx()

    def test_single_tracker_captures_and_restores(self):
        """Single tracker should capture and restore stdout correctly."""
        import sys
        from unittest.mock import Mock, patch

        original_stdout = sys.stdout

        # Create tracker with mocked server client
        tracker = GEPATracker()
        tracker._server_client = Mock()
        tracker._server_client.run_id = "test-run"

        tracker.start_log_capture()
        assert tracker._capture_streams is True
        assert sys.stdout is not original_stdout

        tracker.stop_log_capture()
        assert tracker._capture_streams is False
        assert sys.stdout is original_stdout

    def test_second_tracker_skips_capture(self):
        """Second tracker should not double-wrap stdout."""
        import sys
        from unittest.mock import Mock
        import dspy_gepa_logger.core.tracker_v2 as tracker_module

        original_stdout = sys.stdout

        # First tracker captures
        tracker1 = GEPATracker()
        tracker1._server_client = Mock()
        tracker1._server_client.run_id = "run1"
        tracker1.start_log_capture()

        wrapped_stdout = sys.stdout
        assert wrapped_stdout is not original_stdout

        # Second tracker should skip (not double-wrap)
        tracker2 = GEPATracker()
        tracker2._server_client = Mock()
        tracker2._server_client.run_id = "run2"
        tracker2.start_log_capture()

        # stdout should still be the first wrapper, not double-wrapped
        assert sys.stdout is wrapped_stdout
        assert tracker2._capture_streams is False  # tracker2 didn't capture

        # First tracker stops - should restore
        tracker1.stop_log_capture()
        assert sys.stdout is original_stdout

    def test_second_tracker_stop_does_not_restore(self):
        """Second tracker's stop should not affect stdout if it didn't capture."""
        import sys
        from unittest.mock import Mock

        original_stdout = sys.stdout

        # First tracker captures
        tracker1 = GEPATracker()
        tracker1._server_client = Mock()
        tracker1._server_client.run_id = "run1"
        tracker1.start_log_capture()

        wrapped_stdout = sys.stdout

        # Second tracker tries but skips
        tracker2 = GEPATracker()
        tracker2._server_client = Mock()
        tracker2._server_client.run_id = "run2"
        tracker2.start_log_capture()

        # Second tracker stops - should be no-op
        tracker2.stop_log_capture()
        assert sys.stdout is wrapped_stdout  # Still wrapped by tracker1

        # First tracker stops
        tracker1.stop_log_capture()
        assert sys.stdout is original_stdout


class TestStreamCaptureExternalChanges:
    """Test that stop_log_capture respects external stream changes."""

    def setup_method(self):
        """Reset global capture state before each test."""
        import dspy_gepa_logger.core.tracker_v2 as tracker_module

        with tracker_module._global_capture_lock:
            tracker_module._global_stdout_capture = None
            tracker_module._global_stderr_capture = None
            tracker_module._global_capture_owner = None

        clear_ctx()

    def test_does_not_clobber_external_stdout_change(self):
        """stop_log_capture should not restore if stdout was externally replaced."""
        import sys
        import io
        from unittest.mock import Mock

        original_stdout = sys.stdout

        # Tracker starts capture
        tracker = GEPATracker()
        tracker._server_client = Mock()
        tracker._server_client.run_id = "test"
        tracker.start_log_capture()

        our_capture = sys.stdout

        # External code replaces stdout
        external_stream = io.StringIO()
        sys.stdout = external_stream

        # Tracker stops - should NOT restore to original because
        # current stdout is not our capture
        tracker.stop_log_capture()

        # External stream should still be in place
        assert sys.stdout is external_stream

        # Cleanup
        sys.stdout = original_stdout

    def test_restores_when_still_owning_stream(self):
        """stop_log_capture should restore if we still own stdout."""
        import sys
        from unittest.mock import Mock

        original_stdout = sys.stdout

        tracker = GEPATracker()
        tracker._server_client = Mock()
        tracker._server_client.run_id = "test"
        tracker.start_log_capture()

        # No external changes - we still own it
        tracker.stop_log_capture()

        # Should be restored
        assert sys.stdout is original_stdout
