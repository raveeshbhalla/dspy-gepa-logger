"""Captures GEPA state incrementally via stop_callbacks.

GEPAStateLogger is a callable that implements the stop_callbacks interface.
It captures incremental changes (deltas) instead of deep-copying the full state
each iteration, avoiding O(iterations * state_size) memory blowup.

Key design decisions:
1. Incremental snapshots - only store new candidates, new lineage, Pareto diffs
2. Seed detection by lineage (parent is None), not by iteration==0
3. Store full Pareto program sets, use min() for deterministic selection
4. Version guards for GEPA field compatibility
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .context import set_ctx, get_ctx, clear_ctx


logger = logging.getLogger(__name__)


@dataclass
class IterationDelta:
    """Incremental changes from one iteration to the next.

    Only stores what's NEW or CHANGED, not the full state.
    """

    iteration: int
    timestamp: float  # Relative to start time
    total_evals: int

    # Only NEW candidates added this iteration (not all candidates)
    new_candidates: list[tuple[int, dict[str, str]]] = field(default_factory=list)

    # Only NEW parent links (not full lineage)
    new_lineage: list[tuple[int, list[int | None]]] = field(default_factory=list)

    # Pareto frontier changes - store FULL sets for additions
    pareto_additions: dict[str, tuple[float, set[int]]] = field(default_factory=dict)
    pareto_removals: set[str] = field(default_factory=set)


@dataclass
class IterationMetadata:
    """Lightweight metadata per iteration (always stored)."""

    iteration: int
    timestamp: float
    total_evals: int
    num_candidates: int
    pareto_size: int


class GEPAStateLogger:
    """Callback for stop_callbacks that captures state INCREMENTALLY.

    Avoids O(iterations * state_size) memory by only storing diffs.

    NOTE: This is NOT a "stopper" - it always returns False.
    We just use stop_callbacks as the iteration hook.

    Usage:
        state_logger = GEPAStateLogger()

        gepa = GEPA(
            metric=my_metric,
            gepa_kwargs={'stop_callbacks': [state_logger]}
        )

        result = gepa.compile(student, trainset=train, valset=val)

        # Access captured data
        print(f"Seed candidate: {state_logger.seed_candidate}")
        print(f"Final candidates: {state_logger.final_candidates}")
        print(f"Lineage: {state_logger.get_lineage(candidate_idx=5)}")
    """

    def __init__(
        self,
        dspy_version: str | None = None,
        gepa_version: str | None = None,
    ):
        """Initialize the state logger.

        Args:
            dspy_version: DSPy version (for compatibility checking)
            gepa_version: GEPA version (for compatibility checking)
        """
        self.deltas: list[IterationDelta] = []
        self.metadata: list[IterationMetadata] = []
        self._start_time: float | None = None

        # Track previous state for diffing
        self._prev_num_candidates: int = 0
        self._prev_pareto: dict[str, float] = {}

        # Version info for compatibility
        self.versions: dict[str, str] = {}
        self._detect_versions(dspy_version, gepa_version)

        # Seed candidate - detected by lineage, NOT by iteration==0
        self.seed_candidate: dict[str, str] | None = None
        self.seed_candidate_idx: int | None = None

        # Final state (populated at end)
        self.final_candidates: list[dict[str, str]] = []
        self.final_lineage: list[list[int | None]] = []
        self.final_pareto: dict[str, float] = {}
        self.final_pareto_programs: dict[str, set[int]] = {}  # Store full sets!

    def _detect_versions(
        self, dspy_version: str | None, gepa_version: str | None
    ) -> None:
        """Detect DSPy and GEPA versions."""
        if dspy_version:
            self.versions["dspy"] = dspy_version
        else:
            try:
                import dspy

                self.versions["dspy"] = getattr(dspy, "__version__", "unknown")
            except ImportError:
                self.versions["dspy"] = "not-installed"

        self.versions["gepa"] = gepa_version or "unknown"

    def __call__(self, state: Any) -> bool:
        """Called by GEPA each iteration. Returns False to continue.

        Args:
            state: GEPAState object from GEPA

        Returns:
            False (never stops optimization - we're just logging)
        """
        if self._start_time is None:
            self._start_time = time.time()
            clear_ctx()  # Start fresh

        try:
            self._record_iteration(state)
        except Exception as e:
            # Don't crash GEPA if logging fails
            iteration = self._safe_get(state, "i", "?")
            logger.warning(f"State logging failed at iteration {iteration}: {e}")

        # Update context for downstream hooks (LoggedMetric, LM callbacks)
        set_ctx(iteration=self._safe_get(state, "i", 0))

        return False  # Never stop - we're just logging

    def _record_iteration(self, state: Any) -> None:
        """Record incremental changes from this iteration."""
        iteration = self._safe_get(state, "i", 0)
        timestamp = time.time() - (self._start_time or time.time())
        total_evals = self._safe_get(state, "total_num_evals", 0)

        # Get current state
        candidates = self._safe_get(state, "program_candidates", [])
        lineage = self._safe_get(state, "parent_program_for_candidate", [])
        # Use safe conversion for dict-like objects that may not iterate correctly
        pareto_raw = self._safe_get(state, "pareto_front_valset", {})
        pareto = self._safe_dict_convert(pareto_raw, {})
        pareto_programs_raw = self._safe_get(state, "program_at_pareto_front_valset", {})
        pareto_programs = self._safe_dict_convert(pareto_programs_raw, {})

        # Detect seed candidate by lineage (parent is None), NOT by iteration==0
        if self.seed_candidate is None and candidates and lineage:
            for idx, parents in enumerate(lineage):
                if not parents or (len(parents) > 0 and parents[0] is None):
                    self.seed_candidate = self._copy_dict(candidates[idx])
                    self.seed_candidate_idx = idx
                    break

            # Fallback: first candidate if no parentless candidate found
            if self.seed_candidate is None and candidates:
                self.seed_candidate = self._copy_dict(candidates[0])
                self.seed_candidate_idx = 0

        # Compute NEW candidates (indices >= prev count)
        new_candidates: list[tuple[int, dict[str, str]]] = []
        for idx in range(self._prev_num_candidates, len(candidates)):
            new_candidates.append((idx, self._copy_dict(candidates[idx])))

        # Compute NEW lineage entries
        new_lineage: list[tuple[int, list[int | None]]] = []
        for idx in range(self._prev_num_candidates, len(lineage)):
            new_lineage.append((idx, list(lineage[idx])))

        # Compute Pareto frontier changes - store FULL sets
        pareto_additions: dict[str, tuple[float, set[int]]] = {}
        pareto_removals: set[str] = set()

        for data_id, score in pareto.items():
            data_id_str = str(data_id)
            if (
                data_id_str not in self._prev_pareto
                or self._prev_pareto[data_id_str] != score
            ):
                # New or improved - store the FULL set of program indices
                prog_set = pareto_programs.get(data_id, set())
                if isinstance(prog_set, set):
                    pareto_additions[data_id_str] = (score, set(prog_set))
                elif prog_set is not None:
                    pareto_additions[data_id_str] = (score, {prog_set})
                else:
                    pareto_additions[data_id_str] = (score, set())

        for data_id_str in self._prev_pareto:
            # Check both string and original key types
            if data_id_str not in pareto and data_id_str not in [str(k) for k in pareto]:
                pareto_removals.add(data_id_str)

        # Store delta
        delta = IterationDelta(
            iteration=iteration,
            timestamp=timestamp,
            total_evals=total_evals,
            new_candidates=new_candidates,
            new_lineage=new_lineage,
            pareto_additions=pareto_additions,
            pareto_removals=pareto_removals,
        )
        self.deltas.append(delta)

        # Store lightweight metadata
        meta = IterationMetadata(
            iteration=iteration,
            timestamp=timestamp,
            total_evals=total_evals,
            num_candidates=len(candidates),
            pareto_size=len(pareto),
        )
        self.metadata.append(meta)

        # Update tracking for next iteration
        self._prev_num_candidates = len(candidates)
        self._prev_pareto = {str(k): v for k, v in pareto.items()}

        # Keep final state reference
        self.final_candidates = [self._copy_dict(c) for c in candidates]
        self.final_lineage = [list(p) for p in lineage]
        self.final_pareto = {str(k): v for k, v in pareto.items()}
        self.final_pareto_programs = {
            str(k): set(v) if isinstance(v, set) else {v} if v is not None else set()
            for k, v in pareto_programs.items()
        }

    def _safe_get(self, state: Any, attr: str, default: Any) -> Any:
        """Safely get attribute with fallback (version compatibility)."""
        try:
            return getattr(state, attr, default)
        except Exception:
            return default

    def _copy_dict(self, d: Any) -> dict[str, str]:
        """Safely copy a dict-like object."""
        if isinstance(d, dict):
            return dict(d)
        elif hasattr(d, "items"):
            return dict(d.items())
        else:
            return {}

    def _safe_dict_convert(self, d: Any, default: dict | None = None) -> dict:
        """Safely convert any dict-like object to a regular dict.

        Handles cases where dict() fails because the object iterates over
        keys only instead of (key, value) pairs.
        """
        if default is None:
            default = {}
        if d is None:
            return default
        if isinstance(d, dict):
            return dict(d)
        # Use .items() for dict-like objects that don't iterate correctly
        if hasattr(d, "items"):
            try:
                return dict(d.items())
            except (TypeError, ValueError):
                pass
        # Last resort: try direct conversion
        try:
            return dict(d)
        except (TypeError, ValueError):
            return default

    # Query methods

    def get_all_candidates(self) -> list[dict[str, str]]:
        """Reconstruct all candidates from deltas.

        Returns:
            List of all candidate prompts
        """
        candidates: list[dict[str, str]] = []
        for delta in self.deltas:
            for idx, content in delta.new_candidates:
                while len(candidates) <= idx:
                    candidates.append({})
                candidates[idx] = content
        return candidates

    def get_lineage(self, candidate_idx: int) -> list[int]:
        """Trace a candidate back to its ancestors.

        Args:
            candidate_idx: The candidate to trace

        Returns:
            List of candidate indices from candidate to seed
            Example: [5, 3, 1, 0] means 5 came from 3, which came from 1, which came from seed (0)
        """
        if not self.final_lineage or candidate_idx >= len(self.final_lineage):
            return [candidate_idx]

        lineage = [candidate_idx]
        current = candidate_idx

        while current is not None and current < len(self.final_lineage):
            parents = self.final_lineage[current]
            if parents and len(parents) > 0 and parents[0] is not None:
                lineage.append(parents[0])
                current = parents[0]
            else:
                break

        return lineage

    def get_pareto_evolution(self) -> list[dict[str, tuple[float, set[int]]]]:
        """Get Pareto frontier state at each iteration.

        Returns:
            List of dicts: data_id -> (score, set of program indices)
        """
        pareto_states: list[dict[str, tuple[float, set[int]]]] = []
        current_pareto: dict[str, tuple[float, set[int]]] = {}

        for delta in self.deltas:
            # Apply removals
            for data_id in delta.pareto_removals:
                current_pareto.pop(data_id, None)

            # Apply additions
            for data_id, (score, prog_set) in delta.pareto_additions.items():
                current_pareto[data_id] = (score, set(prog_set))

            pareto_states.append(dict(current_pareto))

        return pareto_states

    def get_pareto_best_candidate(self, data_id: str) -> int | None:
        """Get deterministic "best" candidate for a Pareto data point.

        Uses min() for deterministic selection when multiple candidates tie.

        Args:
            data_id: The data point ID

        Returns:
            The lowest candidate index on the Pareto frontier for this data point
        """
        prog_set = self.final_pareto_programs.get(str(data_id))
        if prog_set:
            return min(prog_set)  # Deterministic: lowest index
        return None

    def get_candidates_added_in_iteration(
        self, iteration: int
    ) -> list[tuple[int, dict[str, str]]]:
        """Get candidates added in a specific iteration.

        Args:
            iteration: The iteration number

        Returns:
            List of (candidate_idx, content) tuples
        """
        for delta in self.deltas:
            if delta.iteration == iteration:
                return delta.new_candidates
        return []

    def get_summary(self) -> dict[str, Any]:
        """Get summary of the optimization run.

        Returns:
            Dict with iteration count, candidate count, etc.
        """
        if not self.metadata:
            return {
                "total_iterations": 0,
                "total_evaluations": 0,
                "total_candidates": 0,
                "final_pareto_size": 0,
            }

        last_meta = self.metadata[-1]
        return {
            "total_iterations": last_meta.iteration + 1,
            "total_evaluations": last_meta.total_evals,
            "total_candidates": len(self.final_candidates),
            "final_pareto_size": len(self.final_pareto),
            "duration_seconds": last_meta.timestamp,
            "seed_candidate_idx": self.seed_candidate_idx,
            "versions": self.versions,
        }

    def clear(self) -> None:
        """Clear all recorded state."""
        self.deltas = []
        self.metadata = []
        self._start_time = None
        self._prev_num_candidates = 0
        self._prev_pareto = {}
        self.seed_candidate = None
        self.seed_candidate_idx = None
        self.final_candidates = []
        self.final_lineage = []
        self.final_pareto = {}
        self.final_pareto_programs = {}
