"""Tests for GEPAStateLogger.

Tests cover:
1. Basic state capture
2. Incremental snapshots (diffs only)
3. Seed candidate detection by lineage
4. Pareto frontier tracking with full sets
5. Context setting
6. Query methods (lineage, Pareto evolution)
"""

import pytest
from dataclasses import dataclass, field
from typing import Any

from dspy_gepa_logger.core.context import clear_ctx, get_ctx
from dspy_gepa_logger.core.state_logger import (
    GEPAStateLogger,
    IterationDelta,
    IterationMetadata,
)


@dataclass
class MockGEPAState:
    """Mock GEPAState for testing."""

    i: int = 0  # Iteration number
    total_num_evals: int = 0
    num_full_ds_evals: int = 0

    program_candidates: list[dict[str, str]] = field(default_factory=list)
    parent_program_for_candidate: list[list[int | None]] = field(default_factory=list)
    prog_candidate_val_subscores: list[dict[str, float]] = field(default_factory=list)

    pareto_front_valset: dict[str, float] = field(default_factory=dict)
    program_at_pareto_front_valset: dict[str, set[int]] = field(default_factory=dict)

    full_program_trace: list[dict[str, Any]] = field(default_factory=list)
    best_outputs_valset: dict[str, Any] | None = None


class TestIterationDelta:
    """Test IterationDelta dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        delta = IterationDelta(iteration=0, timestamp=1.0, total_evals=10)
        assert delta.new_candidates == []
        assert delta.new_lineage == []
        assert delta.pareto_additions == {}
        assert delta.pareto_removals == set()


class TestBasicCapture:
    """Test basic state capture."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()

    def test_returns_false(self):
        """Should always return False (never stops)."""
        state = MockGEPAState(i=0)
        result = self.logger(state)
        assert result is False

    def test_captures_iteration_metadata(self):
        """Should capture iteration metadata."""
        state = MockGEPAState(i=5, total_num_evals=100)
        self.logger(state)

        assert len(self.logger.metadata) == 1
        meta = self.logger.metadata[0]
        assert meta.iteration == 5
        assert meta.total_evals == 100

    def test_sets_context_iteration(self):
        """Should set iteration in context."""
        state = MockGEPAState(i=3)
        self.logger(state)

        ctx = get_ctx()
        assert ctx.get("iteration") == 3

    def test_multiple_iterations(self):
        """Should capture multiple iterations."""
        for i in range(5):
            state = MockGEPAState(i=i, total_num_evals=i * 10)
            self.logger(state)

        assert len(self.logger.metadata) == 5
        assert len(self.logger.deltas) == 5


class TestIncrementalSnapshots:
    """Test incremental (diff-only) capture."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()

    def test_captures_only_new_candidates(self):
        """Should only capture NEW candidates, not all."""
        # Iteration 0: 1 candidate
        state0 = MockGEPAState(
            i=0,
            program_candidates=[{"instructions": "seed prompt"}],
            parent_program_for_candidate=[[None]],
        )
        self.logger(state0)

        # Iteration 1: 2 candidates (1 new)
        state1 = MockGEPAState(
            i=1,
            program_candidates=[
                {"instructions": "seed prompt"},
                {"instructions": "new prompt"},
            ],
            parent_program_for_candidate=[[None], [0]],
        )
        self.logger(state1)

        # Check deltas
        assert len(self.logger.deltas[0].new_candidates) == 1
        assert self.logger.deltas[0].new_candidates[0][1]["instructions"] == "seed prompt"

        assert len(self.logger.deltas[1].new_candidates) == 1
        assert self.logger.deltas[1].new_candidates[0][1]["instructions"] == "new prompt"

    def test_captures_pareto_diffs(self):
        """Should capture Pareto additions and removals."""
        # Iteration 0: Initial Pareto
        state0 = MockGEPAState(
            i=0,
            pareto_front_valset={"ex1": 0.8, "ex2": 0.7},
            program_at_pareto_front_valset={"ex1": {0}, "ex2": {0}},
        )
        self.logger(state0)

        # Iteration 1: ex1 improved, ex3 added, ex2 removed
        state1 = MockGEPAState(
            i=1,
            pareto_front_valset={"ex1": 0.9, "ex3": 0.6},
            program_at_pareto_front_valset={"ex1": {1}, "ex3": {1}},
        )
        self.logger(state1)

        # Check delta 1 additions (ex1 changed, ex3 new)
        delta1 = self.logger.deltas[1]
        assert "ex1" in delta1.pareto_additions
        assert delta1.pareto_additions["ex1"][0] == 0.9  # New score
        assert "ex3" in delta1.pareto_additions

        # ex2 was removed
        assert "ex2" in delta1.pareto_removals


class TestSeedDetection:
    """Test seed candidate detection by lineage."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()

    def test_detects_seed_by_none_parent(self):
        """Should detect seed by parent being None."""
        state = MockGEPAState(
            i=0,
            program_candidates=[
                {"instructions": "seed"},
                {"instructions": "child"},
            ],
            parent_program_for_candidate=[[None], [0]],
        )
        self.logger(state)

        assert self.logger.seed_candidate == {"instructions": "seed"}
        assert self.logger.seed_candidate_idx == 0

    def test_seed_not_necessarily_first(self):
        """Seed can be at any position (detected by None parent)."""
        state = MockGEPAState(
            i=0,
            program_candidates=[
                {"instructions": "child1"},
                {"instructions": "seed"},
                {"instructions": "child2"},
            ],
            # Second candidate is the seed (None parent)
            parent_program_for_candidate=[[1], [None], [1]],
        )
        self.logger(state)

        assert self.logger.seed_candidate == {"instructions": "seed"}
        assert self.logger.seed_candidate_idx == 1

    def test_fallback_to_first_candidate(self):
        """Should fall back to first candidate if no None parent found."""
        state = MockGEPAState(
            i=0,
            program_candidates=[{"instructions": "first"}],
            parent_program_for_candidate=[[]],  # Empty parent list
        )
        self.logger(state)

        assert self.logger.seed_candidate == {"instructions": "first"}
        assert self.logger.seed_candidate_idx == 0


class TestParetoTracking:
    """Test Pareto frontier tracking with full sets."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()

    def test_stores_full_program_sets(self):
        """Should store full sets of programs at Pareto."""
        state = MockGEPAState(
            i=0,
            pareto_front_valset={"ex1": 0.9},
            # Multiple programs tied on Pareto for ex1
            program_at_pareto_front_valset={"ex1": {0, 2, 5}},
        )
        self.logger(state)

        # Should store the full set
        assert "ex1" in self.logger.deltas[0].pareto_additions
        score, prog_set = self.logger.deltas[0].pareto_additions["ex1"]
        assert prog_set == {0, 2, 5}

    def test_get_pareto_best_deterministic(self):
        """get_pareto_best_candidate should return min() for determinism."""
        state = MockGEPAState(
            i=0,
            pareto_front_valset={"ex1": 0.9},
            program_at_pareto_front_valset={"ex1": {5, 2, 10}},
        )
        self.logger(state)

        # Should return lowest index (2)
        best = self.logger.get_pareto_best_candidate("ex1")
        assert best == 2

    def test_handles_non_set_pareto_programs(self):
        """Should handle cases where pareto_programs is not a set."""
        state = MockGEPAState(
            i=0,
            pareto_front_valset={"ex1": 0.9},
            # Single int instead of set
            program_at_pareto_front_valset={"ex1": 3},
        )
        self.logger(state)

        # Should convert to set
        best = self.logger.get_pareto_best_candidate("ex1")
        assert best == 3


class TestLineageTracking:
    """Test lineage reconstruction."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()

    def test_get_lineage(self):
        """Should trace lineage back to seed."""
        # Build up state over iterations
        state0 = MockGEPAState(
            i=0,
            program_candidates=[{"inst": "seed"}],
            parent_program_for_candidate=[[None]],
        )
        self.logger(state0)

        state1 = MockGEPAState(
            i=1,
            program_candidates=[
                {"inst": "seed"},
                {"inst": "gen1"},
            ],
            parent_program_for_candidate=[[None], [0]],
        )
        self.logger(state1)

        state2 = MockGEPAState(
            i=2,
            program_candidates=[
                {"inst": "seed"},
                {"inst": "gen1"},
                {"inst": "gen2"},
            ],
            parent_program_for_candidate=[[None], [0], [1]],
        )
        self.logger(state2)

        # Lineage of candidate 2: 2 -> 1 -> 0
        lineage = self.logger.get_lineage(2)
        assert lineage == [2, 1, 0]

    def test_get_lineage_seed(self):
        """Seed's lineage should just be itself."""
        state = MockGEPAState(
            i=0,
            program_candidates=[{"inst": "seed"}],
            parent_program_for_candidate=[[None]],
        )
        self.logger(state)

        lineage = self.logger.get_lineage(0)
        assert lineage == [0]


class TestParetoEvolution:
    """Test Pareto frontier evolution reconstruction."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()

    def test_pareto_evolution(self):
        """Should reconstruct Pareto state at each iteration."""
        # Iteration 0
        state0 = MockGEPAState(
            i=0,
            pareto_front_valset={"ex1": 0.5, "ex2": 0.6},
            program_at_pareto_front_valset={"ex1": {0}, "ex2": {0}},
        )
        self.logger(state0)

        # Iteration 1: ex1 improved
        state1 = MockGEPAState(
            i=1,
            pareto_front_valset={"ex1": 0.8, "ex2": 0.6},
            program_at_pareto_front_valset={"ex1": {1}, "ex2": {0}},
        )
        self.logger(state1)

        # Iteration 2: ex3 added
        state2 = MockGEPAState(
            i=2,
            pareto_front_valset={"ex1": 0.8, "ex2": 0.6, "ex3": 0.7},
            program_at_pareto_front_valset={"ex1": {1}, "ex2": {0}, "ex3": {2}},
        )
        self.logger(state2)

        evolution = self.logger.get_pareto_evolution()

        # Check each iteration's Pareto state
        assert len(evolution) == 3

        # Iteration 0
        assert "ex1" in evolution[0]
        assert evolution[0]["ex1"][0] == 0.5

        # Iteration 1 (ex1 improved)
        assert evolution[1]["ex1"][0] == 0.8

        # Iteration 2 (ex3 added)
        assert "ex3" in evolution[2]


class TestQueryMethods:
    """Test additional query methods."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()
        self._run_simulation()

    def _run_simulation(self):
        """Simulate a few iterations."""
        for i in range(3):
            candidates = [{"inst": f"cand_{j}"} for j in range(i + 1)]
            lineage = [[None]] + [[j - 1] for j in range(1, i + 1)]

            state = MockGEPAState(
                i=i,
                total_num_evals=(i + 1) * 10,
                program_candidates=candidates,
                parent_program_for_candidate=lineage,
            )
            self.logger(state)

    def test_get_all_candidates(self):
        """Should reconstruct all candidates."""
        candidates = self.logger.get_all_candidates()
        # Should have 3 candidates (added one per iteration)
        assert len(candidates) >= 1

    def test_get_candidates_added_in_iteration(self):
        """Should return candidates added in specific iteration."""
        added = self.logger.get_candidates_added_in_iteration(1)
        assert len(added) == 1  # One candidate added in iteration 1

    def test_get_summary(self):
        """Should return summary stats."""
        summary = self.logger.get_summary()
        assert summary["total_iterations"] == 3
        assert summary["total_evaluations"] == 30
        assert summary["seed_candidate_idx"] == 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        clear_ctx()
        self.logger = GEPAStateLogger()

    def test_handles_missing_attributes(self):
        """Should handle state objects missing attributes."""

        class MinimalState:
            i = 0

        state = MinimalState()
        # Should not raise
        result = self.logger(state)
        assert result is False

    def test_handles_empty_state(self):
        """Should handle empty state."""
        state = MockGEPAState(i=0)
        result = self.logger(state)
        assert result is False

    def test_clear(self):
        """Should clear all state."""
        state = MockGEPAState(
            i=0,
            program_candidates=[{"inst": "test"}],
            parent_program_for_candidate=[[None]],
        )
        self.logger(state)

        assert len(self.logger.deltas) == 1
        assert self.logger.seed_candidate is not None

        self.logger.clear()

        assert len(self.logger.deltas) == 0
        assert self.logger.seed_candidate is None
        assert self.logger.final_candidates == []


class TestVersionDetection:
    """Test version detection."""

    def test_explicit_versions(self):
        """Should use explicitly provided versions."""
        logger = GEPAStateLogger(dspy_version="2.5.0", gepa_version="1.0.0")
        assert logger.versions["dspy"] == "2.5.0"
        assert logger.versions["gepa"] == "1.0.0"

    def test_auto_detect_dspy(self):
        """Should auto-detect DSPy version."""
        logger = GEPAStateLogger()
        # Should have some version (either detected or "unknown"/"not-installed")
        assert "dspy" in logger.versions
