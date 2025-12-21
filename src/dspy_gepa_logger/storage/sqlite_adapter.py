"""Adapter to make SQLiteStorage compatible with StorageBackend protocol.

This adapter bridges the SQLite normalized schema with the flat
data models used by GEPARunTracker.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Iterator

from dspy_gepa_logger.storage.sqlite_backend import SQLiteStorage
from dspy_gepa_logger.models.run import GEPARunRecord
from dspy_gepa_logger.models.iteration import IterationRecord
from dspy_gepa_logger.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)


class SQLiteStorageAdapter:
    """Adapter that makes SQLiteStorage compatible with StorageBackend protocol.

    This allows using SQLite with the existing GEPARunTracker for full instrumentation.
    """

    def __init__(self, db_path: str | Path):
        """Initialize the adapter.

        Args:
            db_path: Path to SQLite database
        """
        self.storage = SQLiteStorage(db_path)
        self._run_metadata = {}  # Cache run metadata
        self._candidate_idx_to_program_id = {}  # Map GEPA candidate indices to program IDs per run

    def save_run_start(self, run: GEPARunRecord) -> None:
        """Save initial run metadata.

        Args:
            run: Run record with initial state
        """
        self._run_metadata[run.run_id] = run

        # Create run in database
        self.storage.create_run(
            run_id=run.run_id,
            config=run.config.__dict__ if hasattr(run, 'config') else {},
        )

        # Store seed candidate if present (it's a dict of instructions)
        if run.seed_candidate:
            self._get_or_create_program(
                run_id=run.run_id,
                instructions=run.seed_candidate,
                signature=None,
            )

    def save_run_end(self, run: GEPARunRecord) -> None:
        """Save final run state.

        Args:
            run: Run record with final state
        """
        self._run_metadata[run.run_id] = run

        # Update run status and metrics
        self.storage.update_run_status(
            run_id=run.run_id,
            status=run.status,
            error_message=run.error_message if hasattr(run, 'error_message') else None,
        )

        # Calculate metrics from iterations and run data
        total_iterations = len(run.iterations) if hasattr(run, 'iterations') else run.total_iterations
        accepted_count = sum(1 for it in run.iterations if it.accepted) if hasattr(run, 'iterations') else 0
        seed_score = run.seed_aggregate_score if hasattr(run, 'seed_aggregate_score') else 0.0

        # Find best aggregate score and corresponding program from all iterations
        final_score = seed_score
        best_program_id = None

        # Get candidate idx to program_id mapping for this run
        idx_to_program = self._candidate_idx_to_program_id.get(run.run_id, {})

        if hasattr(run, 'iterations'):
            for it in run.iterations:
                if it.val_aggregate_score is not None and it.val_aggregate_score > final_score:
                    final_score = it.val_aggregate_score
                    # The new_candidate_idx is the index of the accepted candidate
                    if it.new_candidate_idx is not None and it.new_candidate_idx in idx_to_program:
                        best_program_id = idx_to_program[it.new_candidate_idx]

        # Use best_aggregate_score if explicitly set
        if hasattr(run, 'best_aggregate_score') and run.best_aggregate_score > 0:
            final_score = run.best_aggregate_score

        self.storage.update_run_metrics(
            run_id=run.run_id,
            total_iterations=total_iterations,
            accepted_count=accepted_count,
            seed_score=seed_score,
            final_score=final_score,
            optimized_program_id=best_program_id,
        )

    def save_iteration(self, run_id: str, iteration: IterationRecord) -> None:
        """Save an iteration with all its data.

        Args:
            run_id: Run ID
            iteration: Iteration record
        """
        # Get or create parent program
        parent_program_id = None
        if iteration.parent_prompt:
            parent_program_id = self._get_or_create_program(
                run_id=run_id,
                instructions=iteration.parent_prompt,
                signature=None,
            )
            # Track mapping for Pareto frontier lookups
            if run_id not in self._candidate_idx_to_program_id:
                self._candidate_idx_to_program_id[run_id] = {}
            self._candidate_idx_to_program_id[run_id][iteration.parent_candidate_idx] = parent_program_id

        # Get or create candidate program (if not rejected)
        candidate_program_id = None
        if iteration.new_candidate_prompt:
            candidate_program_id = self._get_or_create_program(
                run_id=run_id,
                instructions=iteration.new_candidate_prompt,
                signature=None,
            )
            # Track mapping for Pareto frontier lookups
            if iteration.new_candidate_idx is not None:
                if run_id not in self._candidate_idx_to_program_id:
                    self._candidate_idx_to_program_id[run_id] = {}
                self._candidate_idx_to_program_id[run_id][iteration.new_candidate_idx] = candidate_program_id

        # Calculate aggregate scores
        parent_minibatch_score = (
            sum(iteration.minibatch_scores) / len(iteration.minibatch_scores)
            if iteration.minibatch_scores else 0.0
        )
        candidate_minibatch_score = (
            sum(iteration.new_candidate_minibatch_scores) / len(iteration.new_candidate_minibatch_scores)
            if iteration.new_candidate_minibatch_scores else None
        )

        # Create iteration record
        iteration_id = self.storage.create_iteration(
            run_id=run_id,
            iteration_number=iteration.iteration_number,
            iteration_type=iteration.iteration_type,
            parent_program_id=parent_program_id,
            candidate_program_id=candidate_program_id,
            parent_minibatch_score=parent_minibatch_score,
            candidate_minibatch_score=candidate_minibatch_score,
            parent_val_score=iteration.parent_val_score,
            val_aggregate_score=iteration.val_aggregate_score,
            accepted=iteration.accepted,
            acceptance_reason=iteration.acceptance_reason,
        )

        # For iteration 0 (baseline), set original_program_id to the candidate
        # This is the evaluated baseline program that has validation rollouts
        if iteration.iteration_number == 0 and candidate_program_id is not None:
            self.storage.update_run_metrics(
                run_id=run_id,
                original_program_id=candidate_program_id,
            )
            logger.info(f"Set original_program_id to {candidate_program_id} for baseline iteration")

        # Store parent minibatch rollouts
        if iteration.minibatch_ids:
            for i, example_id in enumerate(iteration.minibatch_ids):
                self.storage.create_rollout(
                    iteration_id=iteration_id,
                    program_id=parent_program_id,
                    rollout_type="parent_minibatch",
                    example_id=example_id,
                    input_data=iteration.minibatch_inputs[i] if i < len(iteration.minibatch_inputs) else None,
                    output_data=iteration.minibatch_outputs[i] if i < len(iteration.minibatch_outputs) else None,
                    score=iteration.minibatch_scores[i] if i < len(iteration.minibatch_scores) else None,
                    feedback=iteration.minibatch_feedback[i] if i < len(iteration.minibatch_feedback) else None,
                )

        # Store candidate minibatch rollouts
        if iteration.new_candidate_minibatch_outputs:
            for i, example_id in enumerate(iteration.minibatch_ids):
                candidate_feedback = None
                if hasattr(iteration, 'new_candidate_minibatch_feedback') and iteration.new_candidate_minibatch_feedback and i < len(iteration.new_candidate_minibatch_feedback):
                    candidate_feedback = iteration.new_candidate_minibatch_feedback[i]

                self.storage.create_rollout(
                    iteration_id=iteration_id,
                    program_id=candidate_program_id,
                    rollout_type="candidate_minibatch",
                    example_id=example_id,
                    input_data=iteration.minibatch_inputs[i] if i < len(iteration.minibatch_inputs) else None,
                    output_data=iteration.new_candidate_minibatch_outputs[i] if i < len(iteration.new_candidate_minibatch_outputs) else None,
                    score=iteration.new_candidate_minibatch_scores[i] if iteration.new_candidate_minibatch_scores and i < len(iteration.new_candidate_minibatch_scores) else None,
                    feedback=candidate_feedback,
                )

        # Store validation rollouts if present
        if iteration.val_scores:
            logger.info(f"Storing {len(iteration.val_scores)} validation rollouts for iteration {iteration.iteration_number}")
            # Use candidate_program_id if available, otherwise parent (for baseline iteration)
            validation_program_id = candidate_program_id if candidate_program_id is not None else parent_program_id

            # Get outputs and inputs if available
            val_outputs = getattr(iteration, 'val_outputs', None) or {}
            val_inputs = getattr(iteration, 'val_inputs', None) or {}

            for example_id, score in iteration.val_scores.items():
                # Get output and input for this example, defaulting to empty dict
                output_data = val_outputs.get(example_id, {})
                input_data = val_inputs.get(example_id, {})

                self.storage.create_rollout(
                    iteration_id=iteration_id,
                    program_id=validation_program_id,
                    rollout_type="validation",
                    example_id=example_id,
                    input_data=input_data,
                    output_data=output_data,
                    score=score,
                    feedback=None,
                )

            logger.info(f"Stored {len(iteration.val_scores)} validation rollouts with outputs: {len(val_outputs) > 0}")
        else:
            logger.warning(f"No validation scores to store for iteration {iteration.iteration_number}")

        # Store reflection if present
        if iteration.reflection:
            reflection = iteration.reflection
            self.storage.create_reflection(
                iteration_id=iteration_id,
                components=reflection.components_to_update,
                reflective_dataset=reflection.reflective_datasets if hasattr(reflection, 'reflective_datasets') else None,
                proposed_instructions=reflection.proposed_instructions,
            )

        # Store LM calls if present
        if hasattr(iteration, 'lm_calls') and iteration.lm_calls:
            for lm_call in iteration.lm_calls:
                self.storage.create_lm_call(
                    run_id=run_id,
                    iteration_id=iteration_id,
                    call_type=lm_call.call_type if hasattr(lm_call, 'call_type') else "unknown",
                    predictor_name=lm_call.predictor_name if hasattr(lm_call, 'predictor_name') else None,
                    signature_name=lm_call.signature_name if hasattr(lm_call, 'signature_name') else None,
                    model=lm_call.model if hasattr(lm_call, 'model') else None,
                    input_tokens=lm_call.input_tokens if hasattr(lm_call, 'input_tokens') else None,
                    output_tokens=lm_call.output_tokens if hasattr(lm_call, 'output_tokens') else None,
                    total_tokens=lm_call.total_tokens if hasattr(lm_call, 'total_tokens') else None,
                    latency_ms=lm_call.latency_ms if hasattr(lm_call, 'latency_ms') else None,
                )

        # Store Pareto frontier update if present
        if iteration.pareto_update:
            logger.info(f"Storing Pareto update for iteration {iteration.iteration_number}")
            pareto = iteration.pareto_update

            # Find best overall program (program that appears in most task frontiers)
            program_frequency = {}
            for task_programs in pareto.frontier_after.values():
                for prog_idx in task_programs:
                    program_frequency[prog_idx] = program_frequency.get(prog_idx, 0) + 1

            best_overall_idx = max(program_frequency, key=program_frequency.get) if program_frequency else None
            best_overall_score = pareto.pareto_aggregate_score_after
            logger.info(f"Pareto: {len(pareto.frontier_after)} tasks, best_score={best_overall_score}, best_idx={best_overall_idx}")

            # Convert program index to program ID using our mapping
            best_program_id = None
            if best_overall_idx is not None and run_id in self._candidate_idx_to_program_id:
                best_program_id = self._candidate_idx_to_program_id[run_id].get(best_overall_idx)
                if best_program_id:
                    logger.info(f"Mapped candidate idx {best_overall_idx} to program_id {best_program_id}")
                else:
                    logger.warning(f"No program_id mapping found for candidate idx {best_overall_idx}")

            # Create snapshot
            snapshot_id = self.storage.create_pareto_snapshot(
                run_id=run_id,
                iteration_id=iteration_id,
                snapshot_type="iteration",
                best_program_id=best_program_id,
                best_score=best_overall_score,
                timestamp=iteration.timestamp.isoformat() if hasattr(iteration, 'timestamp') else None,
            )

            # Create task-level Pareto data
            for task_id, task_programs in pareto.frontier_after.items():
                # For each task, record the dominant programs and score
                task_score = pareto.frontier_scores_after.get(task_id, 0.0)

                # Map all program indices to program IDs
                dominant_program_ids = []
                if task_programs and run_id in self._candidate_idx_to_program_id:
                    for prog_idx in task_programs:
                        prog_id = self._candidate_idx_to_program_id[run_id].get(prog_idx)
                        if prog_id is not None:
                            dominant_program_ids.append(prog_id)

                # Store the dominant programs for this task
                self.storage.create_pareto_task(
                    snapshot_id=snapshot_id,
                    example_id=task_id,
                    dominant_program_ids=dominant_program_ids if dominant_program_ids else None,
                    dominant_score=task_score,
                )
        else:
            logger.warning(f"No Pareto update to store for iteration {iteration.iteration_number}")

    def save_candidate(self, run_id: str, candidate: CandidateRecord) -> None:
        """Save a candidate record.

        Args:
            run_id: Run ID
            candidate: Candidate record
        """
        self._save_candidate_internal(run_id, candidate, is_seed=False)

    def _save_candidate_internal(self, run_id: str, candidate: CandidateRecord, is_seed: bool) -> int:
        """Internal method to save a candidate.

        Args:
            run_id: Run ID
            candidate: Candidate record
            is_seed: Whether this is the seed candidate

        Returns:
            Program ID
        """
        return self._get_or_create_program(
            run_id=run_id,
            instructions=candidate.program_instructions,
            signature=None,
        )

    def _get_or_create_program(self, run_id: str, instructions: dict, signature: str | None) -> int:
        """Get existing program ID or create new one.

        Programs are deduplicated by instruction hash.

        Args:
            run_id: Run ID
            instructions: Program instructions dict
            signature: Optional signature name

        Returns:
            Program ID
        """
        # Compute hash of instructions
        instructions_json = json.dumps(instructions, sort_keys=True)
        instruction_hash = hashlib.sha256(instructions_json.encode()).hexdigest()[:16]

        # Check if program already exists
        conn = self.storage._get_connection()
        cursor = conn.execute(
            "SELECT program_id FROM programs WHERE run_id = ? AND instruction_hash = ?",
            (run_id, instruction_hash)
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        # Create new program (instruction_hash computed internally)
        return self.storage.create_program(
            run_id=run_id,
            signature=signature,
            instructions=instructions,
        )

    def load_run(self, run_id: str) -> GEPARunRecord | None:
        """Load complete run with all iterations.

        Args:
            run_id: Run ID

        Returns:
            Complete run record or None
        """
        # Check cache first
        if run_id in self._run_metadata:
            return self._run_metadata[run_id]

        # Load from database (basic metadata only)
        return self.load_run_metadata(run_id)

    def load_run_metadata(self, run_id: str) -> GEPARunRecord | None:
        """Load run metadata without iterations.

        Args:
            run_id: Run ID

        Returns:
            Run metadata or None
        """
        run_data = self.storage.get_run(run_id)
        if not run_data:
            return None

        # Convert to GEPARunRecord (simplified)
        # Note: Full reconstruction would require loading all iterations/candidates
        # For now, return cached version if available
        if run_id in self._run_metadata:
            return self._run_metadata[run_id]

        # Return None if not in cache (requires full reconstruction)
        return None

    def load_iterations(self, run_id: str) -> Iterator[IterationRecord]:
        """Load iterations for a run.

        Args:
            run_id: Run ID

        Yields:
            Iteration records
        """
        # This would require reconstructing IterationRecord from normalized data
        # For now, not implemented
        return iter([])

    def load_candidates(self, run_id: str) -> Iterator[CandidateRecord]:
        """Load candidates for a run.

        Args:
            run_id: Run ID

        Yields:
            Candidate records
        """
        # This would require reconstructing CandidateRecord from programs table
        # For now, not implemented
        return iter([])

    def list_runs(self) -> list[str]:
        """List all run IDs.

        Returns:
            List of run IDs
        """
        return self.storage.list_runs()

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its data.

        Args:
            run_id: Run ID

        Returns:
            True if deleted, False if not found
        """
        # SQLite will cascade delete due to foreign keys
        conn = self.storage._get_connection()
        cursor = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()

        # Remove from cache
        if run_id in self._run_metadata:
            del self._run_metadata[run_id]

        return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connections."""
        self.storage.close()
