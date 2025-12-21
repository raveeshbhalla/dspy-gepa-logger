"""In-memory storage backend for testing."""

from typing import Iterator

from dspy_gepa_logger.models.run import GEPARunRecord
from dspy_gepa_logger.models.iteration import IterationRecord
from dspy_gepa_logger.models.candidate import CandidateRecord


class MemoryStorageBackend:
    """In-memory storage backend for testing and development.

    All data is stored in dictionaries and lost when the object is destroyed.
    """

    def __init__(self):
        """Initialize empty storage."""
        self._runs: dict[str, GEPARunRecord] = {}
        self._iterations: dict[str, list[IterationRecord]] = {}
        self._candidates: dict[str, list[CandidateRecord]] = {}

    def save_run_start(self, run: GEPARunRecord) -> None:
        """Persist initial run metadata when a run begins."""
        self._runs[run.run_id] = run
        self._iterations[run.run_id] = []
        self._candidates[run.run_id] = []

    def save_run_end(self, run: GEPARunRecord) -> None:
        """Persist final run state when a run completes."""
        self._runs[run.run_id] = run

    def save_iteration(self, run_id: str, iteration: IterationRecord) -> None:
        """Persist a single iteration record."""
        if run_id not in self._iterations:
            self._iterations[run_id] = []
        self._iterations[run_id].append(iteration)

    def save_candidate(self, run_id: str, candidate: CandidateRecord) -> None:
        """Persist a candidate record."""
        if run_id not in self._candidates:
            self._candidates[run_id] = []
        self._candidates[run_id].append(candidate)

    def load_run(self, run_id: str) -> GEPARunRecord | None:
        """Load a complete run by ID."""
        if run_id not in self._runs:
            return None

        run = self._runs[run_id]
        run.iterations = list(self.load_iterations(run_id))
        run.candidates = list(self.load_candidates(run_id))
        return run

    def load_run_metadata(self, run_id: str) -> GEPARunRecord | None:
        """Load just the run metadata (without iterations/candidates)."""
        return self._runs.get(run_id)

    def load_iterations(self, run_id: str) -> Iterator[IterationRecord]:
        """Stream iterations for a run."""
        for iteration in self._iterations.get(run_id, []):
            yield iteration

    def load_candidates(self, run_id: str) -> Iterator[CandidateRecord]:
        """Stream candidates for a run."""
        for candidate in self._candidates.get(run_id, []):
            yield candidate

    def list_runs(self) -> list[str]:
        """List all available run IDs."""
        return sorted(self._runs.keys())

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its data."""
        if run_id not in self._runs:
            return False
        del self._runs[run_id]
        self._iterations.pop(run_id, None)
        self._candidates.pop(run_id, None)
        return True

    def clear(self) -> None:
        """Clear all stored data."""
        self._runs.clear()
        self._iterations.clear()
        self._candidates.clear()
