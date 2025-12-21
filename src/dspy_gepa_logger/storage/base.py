"""Base protocol for storage backends."""

from typing import Iterator, Protocol, runtime_checkable

from dspy_gepa_logger.models.run import GEPARunRecord
from dspy_gepa_logger.models.iteration import IterationRecord
from dspy_gepa_logger.models.candidate import CandidateRecord


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the interface for GEPA run storage backends.

    Implementations must provide methods for:
    - Saving run metadata at start/end
    - Saving iterations incrementally
    - Saving candidates
    - Loading runs and their components
    - Listing available runs
    """

    def save_run_start(self, run: GEPARunRecord) -> None:
        """Persist initial run metadata when a run begins.

        Args:
            run: The run record with initial state (config, seed candidate, etc.)
        """
        ...

    def save_run_end(self, run: GEPARunRecord) -> None:
        """Persist final run state when a run completes.

        Args:
            run: The run record with final state (status, summary metrics, etc.)
        """
        ...

    def save_iteration(self, run_id: str, iteration: IterationRecord) -> None:
        """Persist a single iteration record.

        Called after each iteration completes. Should append to existing data.

        Args:
            run_id: The ID of the run this iteration belongs to
            iteration: The iteration record to save
        """
        ...

    def save_candidate(self, run_id: str, candidate: CandidateRecord) -> None:
        """Persist a candidate record.

        Called when a new candidate is accepted.

        Args:
            run_id: The ID of the run this candidate belongs to
            candidate: The candidate record to save
        """
        ...

    def load_run(self, run_id: str) -> GEPARunRecord | None:
        """Load a complete run by ID.

        Args:
            run_id: The ID of the run to load

        Returns:
            The complete run record, or None if not found
        """
        ...

    def load_run_metadata(self, run_id: str) -> GEPARunRecord | None:
        """Load just the run metadata (without iterations/candidates).

        Useful for displaying run summaries without loading all data.

        Args:
            run_id: The ID of the run to load

        Returns:
            The run record with just metadata, or None if not found
        """
        ...

    def load_iterations(self, run_id: str) -> Iterator[IterationRecord]:
        """Stream iterations for a run.

        Args:
            run_id: The ID of the run

        Yields:
            Iteration records in order
        """
        ...

    def load_candidates(self, run_id: str) -> Iterator[CandidateRecord]:
        """Stream candidates for a run.

        Args:
            run_id: The ID of the run

        Yields:
            Candidate records
        """
        ...

    def list_runs(self) -> list[str]:
        """List all available run IDs.

        Returns:
            List of run IDs
        """
        ...

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its data.

        Args:
            run_id: The ID of the run to delete

        Returns:
            True if deleted, False if not found
        """
        ...
