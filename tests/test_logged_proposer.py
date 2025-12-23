"""Tests for LoggedInstructionProposer and LoggedSelector.

Tests cover:
1. Phase context setting during reflection/proposal
2. Recording of reflection and proposal calls
3. Attribute delegation to base proposer
4. Query methods
5. LoggedSelector
"""

import pytest
from unittest.mock import Mock, MagicMock

from dspy_gepa_logger.core.context import clear_ctx, get_ctx, set_ctx
from dspy_gepa_logger.core.logged_proposer import (
    LoggedInstructionProposer,
    LoggedSelector,
    ReflectionCall,
    ProposalCall,
)


class MockProposer:
    """Mock instruction proposer for testing."""

    def __init__(self, proposals: list[dict[str, str]] | None = None):
        self.proposals = proposals or [{"instructions": "new prompt"}]
        self.call_count = 0
        self.phases_during_call: list[str | None] = []
        self.some_attribute = "test_value"

    def propose(
        self,
        current_instructions: dict[str, str],
        candidate_idx: int,
        reflection_data: any,
        **kwargs,
    ) -> list[dict[str, str]]:
        self.call_count += 1
        # Record what phase was set when we were called
        self.phases_during_call.append(get_ctx().get("phase"))
        return self.proposals


class MockSelector:
    """Mock selector for testing."""

    def __init__(self, selected: list[int] | None = None):
        self.selected = selected or [0]
        self.call_count = 0

    def select(
        self, candidates: list, scores: list[float], **kwargs
    ) -> list[int]:
        self.call_count += 1
        return self.selected


class TestReflectionCallDataclass:
    """Test ReflectionCall dataclass."""

    def test_required_fields(self):
        """Should have required fields."""
        call = ReflectionCall(
            iteration=1,
            candidate_idx=2,
            current_instructions={"inst": "test"},
        )
        assert call.iteration == 1
        assert call.candidate_idx == 2
        assert call.current_instructions == {"inst": "test"}
        assert call.timestamp > 0


class TestProposalCallDataclass:
    """Test ProposalCall dataclass."""

    def test_required_fields(self):
        """Should have required fields."""
        call = ProposalCall(
            iteration=1,
            candidate_idx=2,
            num_proposals=3,
            proposals=[{"inst": "a"}, {"inst": "b"}, {"inst": "c"}],
        )
        assert call.iteration == 1
        assert call.num_proposals == 3
        assert len(call.proposals) == 3


class TestPhaseContextSetting:
    """Test that phase context is set correctly."""

    def setup_method(self):
        clear_ctx()
        set_ctx(iteration=5)
        self.base_proposer = MockProposer()
        self.logged = LoggedInstructionProposer(self.base_proposer)

    def test_sets_phase_to_proposal_during_call(self):
        """Phase should be 'proposal' when base proposer is called."""
        self.logged.propose(
            current_instructions={"inst": "current"},
            candidate_idx=3,
            reflection_data={"data": "test"},
        )

        # Base proposer should have been called during "proposal" phase
        assert self.base_proposer.phases_during_call == ["proposal"]

    def test_sets_candidate_idx_in_context(self):
        """Should set candidate_idx in context."""
        candidate_idxs_seen = []

        class CapturingProposer:
            def propose(self, **kwargs):
                candidate_idxs_seen.append(get_ctx().get("candidate_idx"))
                return []

        logged = LoggedInstructionProposer(CapturingProposer())
        logged.propose(
            current_instructions={},
            candidate_idx=7,
            reflection_data=None,
        )

        assert candidate_idxs_seen == [7]

    def test_clears_phase_after_call(self):
        """Phase should be cleared after propose() returns."""
        self.logged.propose(
            current_instructions={},
            candidate_idx=0,
            reflection_data=None,
        )

        # Phase should be None after call
        assert get_ctx().get("phase") is None

    def test_preserves_iteration_context(self):
        """Should preserve existing iteration context."""
        self.logged.propose(
            current_instructions={},
            candidate_idx=0,
            reflection_data=None,
        )

        # Iteration should still be set
        assert get_ctx().get("iteration") == 5


class TestReflectionRecording:
    """Test recording of reflection calls."""

    def setup_method(self):
        clear_ctx()
        set_ctx(iteration=3)
        self.base_proposer = MockProposer()
        self.logged = LoggedInstructionProposer(self.base_proposer)

    def test_records_reflection_call(self):
        """Should record reflection call with inputs."""
        self.logged.propose(
            current_instructions={"inst": "original"},
            candidate_idx=2,
            reflection_data={"feedback": "needs improvement"},
        )

        assert len(self.logged.reflection_calls) == 1
        call = self.logged.reflection_calls[0]
        assert call.iteration == 3
        assert call.candidate_idx == 2
        assert call.current_instructions == {"inst": "original"}

    def test_records_multiple_reflection_calls(self):
        """Should record multiple reflection calls."""
        for i in range(3):
            set_ctx(iteration=i)
            self.logged.propose(
                current_instructions={"inst": f"prompt_{i}"},
                candidate_idx=i,
                reflection_data=None,
            )

        assert len(self.logged.reflection_calls) == 3


class TestProposalRecording:
    """Test recording of proposal calls."""

    def setup_method(self):
        clear_ctx()
        set_ctx(iteration=2)

    def test_records_proposal_call(self):
        """Should record proposal call with outputs."""
        proposals = [{"inst": "new1"}, {"inst": "new2"}]
        base_proposer = MockProposer(proposals=proposals)
        logged = LoggedInstructionProposer(base_proposer)

        result = logged.propose(
            current_instructions={},
            candidate_idx=1,
            reflection_data=None,
        )

        assert len(logged.proposal_calls) == 1
        call = logged.proposal_calls[0]
        assert call.iteration == 2
        assert call.candidate_idx == 1
        assert call.num_proposals == 2
        assert call.proposals == proposals
        assert result == proposals

    def test_handles_single_proposal(self):
        """Should handle single proposal (not list)."""

        class SingleProposer:
            def propose(self, **kwargs):
                return {"inst": "single"}  # Not a list

        logged = LoggedInstructionProposer(SingleProposer())
        result = logged.propose({}, 0, None)

        assert len(logged.proposal_calls) == 1
        assert logged.proposal_calls[0].num_proposals == 1

    def test_handles_empty_proposals(self):
        """Should handle empty proposals."""

        class EmptyProposer:
            def propose(self, **kwargs):
                return []

        logged = LoggedInstructionProposer(EmptyProposer())
        result = logged.propose({}, 0, None)

        assert len(logged.proposal_calls) == 1
        assert logged.proposal_calls[0].num_proposals == 0


class TestAttributeDelegation:
    """Test delegation of attributes to base proposer."""

    def setup_method(self):
        clear_ctx()
        self.base_proposer = MockProposer()
        self.logged = LoggedInstructionProposer(self.base_proposer)

    def test_delegates_unknown_attribute(self):
        """Should delegate unknown attributes to base proposer."""
        assert self.logged.some_attribute == "test_value"

    def test_callable_interface(self):
        """Should support being called directly."""
        result = self.logged(
            current_instructions={"inst": "test"},
            candidate_idx=0,
            reflection_data=None,
        )

        assert result == [{"instructions": "new prompt"}]
        assert self.base_proposer.call_count == 1


class TestQueryMethods:
    """Test query methods."""

    def setup_method(self):
        clear_ctx()
        self.base_proposer = MockProposer()
        self.logged = LoggedInstructionProposer(self.base_proposer)
        self._add_test_calls()

    def _add_test_calls(self):
        """Add test calls."""
        for i in range(3):
            set_ctx(iteration=i)
            self.logged.propose(
                current_instructions={"inst": f"iter_{i}"},
                candidate_idx=i * 2,
                reflection_data=None,
            )

    def test_get_reflection_calls_for_iteration(self):
        """Should filter reflection calls by iteration."""
        calls = self.logged.get_reflection_calls_for_iteration(1)
        assert len(calls) == 1
        assert calls[0].iteration == 1

    def test_get_proposal_calls_for_iteration(self):
        """Should filter proposal calls by iteration."""
        calls = self.logged.get_proposal_calls_for_iteration(2)
        assert len(calls) == 1
        assert calls[0].iteration == 2

    def test_get_proposals_for_candidate(self):
        """Should filter proposals by candidate."""
        calls = self.logged.get_proposals_for_candidate(2)
        assert len(calls) == 1
        assert calls[0].candidate_idx == 2

    def test_clear(self):
        """Should clear all recorded calls."""
        assert len(self.logged.reflection_calls) > 0
        assert len(self.logged.proposal_calls) > 0

        self.logged.clear()

        assert len(self.logged.reflection_calls) == 0
        assert len(self.logged.proposal_calls) == 0


class TestLoggedSelector:
    """Test LoggedSelector."""

    def setup_method(self):
        clear_ctx()
        set_ctx(iteration=5)
        self.base_selector = MockSelector(selected=[1, 3])
        self.logged = LoggedSelector(self.base_selector)

    def test_sets_phase_to_selection(self):
        """Should set phase to 'selection' during select."""
        phases_seen = []

        class CapturingSelector:
            def select(self, candidates, scores, **kwargs):
                phases_seen.append(get_ctx().get("phase"))
                return [0]

        logged = LoggedSelector(CapturingSelector())
        logged.select([1, 2, 3], [0.5, 0.6, 0.7])

        assert phases_seen == ["selection"]

    def test_clears_phase_after_select(self):
        """Should clear phase after select."""
        self.logged.select([1, 2], [0.5, 0.6])
        assert get_ctx().get("phase") is None

    def test_records_selection_call(self):
        """Should record selection call."""
        self.logged.select([1, 2, 3], [0.5, 0.6, 0.7])

        assert len(self.logged.selection_calls) == 1
        call = self.logged.selection_calls[0]
        assert call["num_candidates"] == 3
        assert call["scores"] == [0.5, 0.6, 0.7]
        assert call["selected"] == [1, 3]
        assert call["iteration"] == 5

    def test_returns_base_result(self):
        """Should return result from base selector."""
        result = self.logged.select([1, 2], [0.5, 0.6])
        assert result == [1, 3]

    def test_callable_interface(self):
        """Should support being called directly."""
        result = self.logged([1, 2], [0.5, 0.6])
        assert result == [1, 3]

    def test_delegates_attributes(self):
        """Should delegate unknown attributes."""
        self.base_selector.custom_attr = "custom"
        assert self.logged.custom_attr == "custom"

    def test_clear(self):
        """Should clear selection calls."""
        self.logged.select([1], [0.5])
        assert len(self.logged.selection_calls) == 1

        self.logged.clear()
        assert len(self.logged.selection_calls) == 0
