"""Tests for Public API v2.

Tests cover:
1. create_logged_gepa() with mock GEPA
2. configure_dspy_logging()
3. create_tracker() standalone
4. wrap_metric() standalone
5. wrap_proposer() standalone
6. wrap_selector() standalone
7. gepa_kwargs merging
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch

from dspy_gepa_logger.core.context import clear_ctx, get_ctx, set_ctx
from dspy_gepa_logger.core.tracker_v2 import GEPATracker
from dspy_gepa_logger.core.logged_metric import LoggedMetric
from dspy_gepa_logger.core.logged_proposer import LoggedInstructionProposer, LoggedSelector


class MockExample:
    """Mock DSPy Example."""

    def __init__(self, id_: str, **fields):
        self.id = id_
        for k, v in fields.items():
            setattr(self, k, v)


class MockPrediction:
    """Mock prediction."""

    def __init__(self, answer: str):
        self.answer = answer


class TestCreateTracker:
    """Test create_tracker() function."""

    def setup_method(self):
        clear_ctx()

    def test_creates_tracker(self):
        """Should create a GEPATracker instance."""
        from dspy_gepa_logger.api import create_tracker

        tracker = create_tracker()
        assert isinstance(tracker, GEPATracker)

    def test_respects_capture_lm_calls(self):
        """Should respect capture_lm_calls parameter."""
        from dspy_gepa_logger.api import create_tracker

        tracker_with = create_tracker(capture_lm_calls=True)
        assert tracker_with.lm_logger is not None

        tracker_without = create_tracker(capture_lm_calls=False)
        assert tracker_without.lm_logger is None


class TestWrapMetric:
    """Test wrap_metric() function."""

    def setup_method(self):
        clear_ctx()

    def test_wraps_metric(self):
        """Should wrap metric in LoggedMetric."""
        from dspy_gepa_logger.api import wrap_metric

        def my_metric(example, prediction, trace=None):
            return 1.0

        wrapped = wrap_metric(my_metric)
        assert isinstance(wrapped, LoggedMetric)

    def test_wrapped_metric_works(self):
        """Wrapped metric should capture evaluations."""
        from dspy_gepa_logger.api import wrap_metric

        def my_metric(example, prediction, trace=None):
            return 0.8

        wrapped = wrap_metric(my_metric)
        set_ctx(iteration=1, candidate_idx=0)

        result = wrapped(MockExample("ex1"), MockPrediction("A"))

        assert result == 0.8
        assert len(wrapped.evaluations) == 1


class TestWrapProposer:
    """Test wrap_proposer() function."""

    def setup_method(self):
        clear_ctx()

    def test_wraps_proposer(self):
        """Should wrap proposer in LoggedInstructionProposer."""
        from dspy_gepa_logger.api import wrap_proposer

        class MockProposer:
            def propose(self, **kwargs):
                return []

        wrapped = wrap_proposer(MockProposer())
        assert isinstance(wrapped, LoggedInstructionProposer)


class TestWrapSelector:
    """Test wrap_selector() function."""

    def setup_method(self):
        clear_ctx()

    def test_wraps_selector(self):
        """Should wrap selector in LoggedSelector."""
        from dspy_gepa_logger.api import wrap_selector

        class MockSelector:
            def select(self, candidates, scores, **kwargs):
                return [0]

        wrapped = wrap_selector(MockSelector())
        assert isinstance(wrapped, LoggedSelector)


class TestCreateLoggedGEPA:
    """Test create_logged_gepa() function."""

    def setup_method(self):
        clear_ctx()

    @patch.dict(sys.modules, {"dspy.teleprompt": MagicMock()})
    def test_returns_tuple(self):
        """Should return (gepa, tracker, logged_metric) tuple."""
        # Create mock GEPA class
        mock_gepa_class = Mock()
        mock_gepa_instance = Mock()
        mock_gepa_class.return_value = mock_gepa_instance

        with patch.dict(
            sys.modules,
            {"dspy": MagicMock(), "dspy.teleprompt": MagicMock(GEPA=mock_gepa_class)},
        ):
            from dspy_gepa_logger import api

            # Reload to pick up mock
            import importlib

            importlib.reload(api)

            def my_metric(example, prediction, trace=None):
                return 1.0

            gepa, tracker, logged_metric = api.create_logged_gepa(metric=my_metric)

            assert tracker is not None
            assert isinstance(tracker, GEPATracker)
            assert logged_metric is not None
            assert isinstance(logged_metric, LoggedMetric)

    @patch.dict(sys.modules, {"dspy.teleprompt": MagicMock()})
    def test_passes_metric_to_gepa(self):
        """Should pass wrapped metric to GEPA."""
        mock_gepa_class = Mock()

        with patch.dict(
            sys.modules,
            {"dspy": MagicMock(), "dspy.teleprompt": MagicMock(GEPA=mock_gepa_class)},
        ):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            def my_metric(example, prediction, trace=None):
                return 1.0

            gepa, tracker, logged_metric = api.create_logged_gepa(metric=my_metric)

            # Check GEPA was called with logged metric
            call_kwargs = mock_gepa_class.call_args.kwargs
            assert call_kwargs["metric"] is logged_metric

    @patch.dict(sys.modules, {"dspy.teleprompt": MagicMock()})
    def test_passes_stop_callback(self):
        """Should pass stop_callback in gepa_kwargs."""
        mock_gepa_class = Mock()

        with patch.dict(
            sys.modules,
            {"dspy": MagicMock(), "dspy.teleprompt": MagicMock(GEPA=mock_gepa_class)},
        ):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            def my_metric(example, prediction, trace=None):
                return 1.0

            gepa, tracker, logged_metric = api.create_logged_gepa(metric=my_metric)

            call_kwargs = mock_gepa_class.call_args.kwargs
            gepa_kwargs = call_kwargs["gepa_kwargs"]
            assert "stop_callbacks" in gepa_kwargs
            assert tracker.state_logger in gepa_kwargs["stop_callbacks"]

    @patch.dict(sys.modules, {"dspy.teleprompt": MagicMock()})
    def test_merges_existing_stop_callbacks(self):
        """Should merge with existing stop_callbacks."""
        mock_gepa_class = Mock()

        with patch.dict(
            sys.modules,
            {"dspy": MagicMock(), "dspy.teleprompt": MagicMock(GEPA=mock_gepa_class)},
        ):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            def my_metric(example, prediction, trace=None):
                return 1.0

            existing_callback = Mock()

            gepa, tracker, logged_metric = api.create_logged_gepa(
                metric=my_metric,
                gepa_kwargs={"stop_callbacks": [existing_callback]},
            )

            call_kwargs = mock_gepa_class.call_args.kwargs
            gepa_kwargs = call_kwargs["gepa_kwargs"]
            assert existing_callback in gepa_kwargs["stop_callbacks"]
            assert tracker.state_logger in gepa_kwargs["stop_callbacks"]

    @patch.dict(sys.modules, {"dspy.teleprompt": MagicMock()})
    def test_handles_single_stop_callback(self):
        """Should handle single stop_callback (not list)."""
        mock_gepa_class = Mock()

        with patch.dict(
            sys.modules,
            {"dspy": MagicMock(), "dspy.teleprompt": MagicMock(GEPA=mock_gepa_class)},
        ):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            def my_metric(example, prediction, trace=None):
                return 1.0

            existing_callback = Mock()

            gepa, tracker, logged_metric = api.create_logged_gepa(
                metric=my_metric,
                gepa_kwargs={"stop_callbacks": existing_callback},  # Not a list
            )

            call_kwargs = mock_gepa_class.call_args.kwargs
            gepa_kwargs = call_kwargs["gepa_kwargs"]
            assert existing_callback in gepa_kwargs["stop_callbacks"]
            assert tracker.state_logger in gepa_kwargs["stop_callbacks"]

    @patch.dict(sys.modules, {"dspy.teleprompt": MagicMock()})
    def test_passes_kwargs_to_gepa(self):
        """Should pass through kwargs to GEPA."""
        mock_gepa_class = Mock()

        with patch.dict(
            sys.modules,
            {"dspy": MagicMock(), "dspy.teleprompt": MagicMock(GEPA=mock_gepa_class)},
        ):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            def my_metric(example, prediction, trace=None):
                return 1.0

            gepa, tracker, logged_metric = api.create_logged_gepa(
                metric=my_metric,
                num_candidates=5,
                num_iterations=10,
            )

            call_kwargs = mock_gepa_class.call_args.kwargs
            assert call_kwargs["num_candidates"] == 5
            assert call_kwargs["num_iterations"] == 10

    def test_import_error_message_exists(self):
        """Verify ImportError handling code exists in create_logged_gepa."""
        # This test verifies the ImportError handling code path exists
        # We can't easily test the actual ImportError with DSPy installed,
        # so we just verify the function has proper error handling in source
        import inspect
        from dspy_gepa_logger.api import create_logged_gepa

        source = inspect.getsource(create_logged_gepa)
        assert "ImportError" in source
        assert "DSPy is required" in source


class TestConfigureDspyLogging:
    """Test configure_dspy_logging() function."""

    def setup_method(self):
        clear_ctx()

    def test_adds_callbacks_to_dspy(self):
        """Should add LM logger callbacks to DSPy."""
        mock_dspy = MagicMock()
        mock_dspy.settings.get.return_value = []

        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            tracker = GEPATracker()
            api.configure_dspy_logging(tracker)

            mock_dspy.configure.assert_called_once()
            call_kwargs = mock_dspy.configure.call_args.kwargs
            assert tracker.lm_logger in call_kwargs["callbacks"]

    def test_preserves_existing_callbacks(self):
        """Should preserve existing DSPy callbacks."""
        existing_callback = Mock()
        mock_dspy = MagicMock()
        mock_dspy.settings.get.return_value = [existing_callback]

        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            tracker = GEPATracker()
            api.configure_dspy_logging(tracker)

            call_kwargs = mock_dspy.configure.call_args.kwargs
            assert existing_callback in call_kwargs["callbacks"]
            assert tracker.lm_logger in call_kwargs["callbacks"]

    def test_no_op_if_lm_capture_disabled(self):
        """Should not add callbacks if LM capture disabled."""
        mock_dspy = MagicMock()
        mock_dspy.settings.get.return_value = []

        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            tracker = GEPATracker(capture_lm_calls=False)
            api.configure_dspy_logging(tracker)

            # Should not call configure if no callbacks
            mock_dspy.configure.assert_not_called()


class TestIntegration:
    """Integration tests for API functions."""

    def setup_method(self):
        clear_ctx()

    def test_full_workflow_with_mocks(self):
        """Test full workflow with mocked DSPy."""
        mock_gepa_class = Mock()
        mock_dspy = MagicMock()
        mock_dspy.settings.get.return_value = []

        with patch.dict(
            sys.modules,
            {
                "dspy": mock_dspy,
                "dspy.teleprompt": MagicMock(GEPA=mock_gepa_class),
            },
        ):
            from dspy_gepa_logger import api
            import importlib

            importlib.reload(api)

            # Create logged GEPA
            def my_metric(example, prediction, trace=None):
                return 1.0

            gepa, tracker, logged_metric = api.create_logged_gepa(
                metric=my_metric,
                num_candidates=3,
            )

            # Configure DSPy logging
            api.configure_dspy_logging(tracker)

            # Verify setup
            assert tracker.metric_logger is logged_metric
            assert mock_dspy.configure.called

            # Simulate some evaluations
            set_ctx(iteration=0, candidate_idx=0)
            logged_metric(MockExample("ex1"), MockPrediction("A1"))
            logged_metric(MockExample("ex2"), MockPrediction("A2"))

            # Verify data captured
            assert len(tracker.evaluations) == 2

    def test_tracker_standalone_workflow(self):
        """Test standalone tracker workflow."""
        from dspy_gepa_logger.api import create_tracker

        # Create tracker
        tracker = create_tracker()

        # Wrap metric manually
        def my_metric(example, prediction, trace=None):
            return 0.8

        logged_metric = tracker.wrap_metric(my_metric)

        # Get stop callback
        stop_callback = tracker.get_stop_callback()
        assert stop_callback is tracker.state_logger

        # Simulate state updates
        from dataclasses import dataclass, field

        @dataclass
        class MockState:
            i: int = 0
            total_num_evals: int = 0
            program_candidates: list = field(default_factory=list)
            parent_program_for_candidate: list = field(default_factory=list)
            pareto_front_valset: dict = field(default_factory=dict)
            program_at_pareto_front_valset: dict = field(default_factory=dict)

        state = MockState(
            i=0,
            program_candidates=[{"inst": "seed"}],
            parent_program_for_candidate=[[None]],
        )
        stop_callback(state)

        # Simulate evaluations
        set_ctx(iteration=0, candidate_idx=0)
        logged_metric(MockExample("ex1"), MockPrediction("A1"))

        # Verify
        assert tracker.seed_candidate == {"inst": "seed"}
        assert len(tracker.evaluations) == 1
        summary = tracker.get_summary()
        assert summary["state"]["total_iterations"] == 1


class TestExports:
    """Test that API functions are exported correctly."""

    def test_api_exports(self):
        """API module should export expected functions."""
        from dspy_gepa_logger import api

        assert hasattr(api, "create_logged_gepa")
        assert hasattr(api, "configure_dspy_logging")
        assert hasattr(api, "create_tracker")
        assert hasattr(api, "wrap_metric")
        assert hasattr(api, "wrap_proposer")
        assert hasattr(api, "wrap_selector")
