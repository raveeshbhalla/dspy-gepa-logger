"""JSON export functionality for GEPA runs.

Exports complete run data from SQLite to human-readable JSON format
following the schema from detailed_errors.md.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from dspy_gepa_logger.storage.sqlite_backend import SQLiteStorage

logger = logging.getLogger(__name__)


class JSONExporter:
    """Exports GEPA run data from SQLite to JSON format.

    Denormalizes the relational data for human readability.
    """

    def __init__(self, storage: SQLiteStorage):
        """Initialize the exporter.

        Args:
            storage: SQLite storage backend
        """
        self.storage = storage

    def export_run(
        self,
        run_id: str,
        output_path: Optional[str | Path] = None,
        pretty: bool = True,
        include_lm_calls: bool = False,
    ) -> dict[str, Any]:
        """Export a complete run to JSON format.

        Args:
            run_id: Run identifier
            output_path: Optional path to write JSON file
            pretty: Whether to pretty-print JSON
            include_lm_calls: Whether to include LM call traces (can be large)

        Returns:
            Complete run dictionary

        Raises:
            ValueError: If run not found
        """
        # Get run metadata
        run = self.storage.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        # Build complete run structure
        run_data = {
            "run_id": run["run_id"],
            "started_at": run["started_at"],
            "completed_at": run["completed_at"],
            "status": run["status"],
            "error_message": run["error_message"],
            "config": json.loads(run["config_json"]) if run["config_json"] else {},
            "metrics": {
                "total_iterations": run["total_iterations"],
                "accepted_count": run["accepted_count"],
                "seed_score": run["seed_score"],
                "final_score": run["final_score"],
                "improvement": run["final_score"] - run["seed_score"],
            },
            "iterations": [],
            "pareto_history": [],
        }

        # Get iterations
        for iteration in self.storage.get_iterations(run_id):
            iteration_data = self._export_iteration(iteration, include_lm_calls)
            run_data["iterations"].append(iteration_data)

        # Get pareto snapshots
        for snapshot in self.storage.get_pareto_snapshots(run_id):
            snapshot_data = self._export_pareto_snapshot(snapshot)
            run_data["pareto_history"].append(snapshot_data)

        # Write to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                if pretty:
                    json.dump(run_data, f, indent=2, default=str)
                else:
                    json.dump(run_data, f, default=str)
            logger.info(f"Exported run {run_id} to {output_path}")

        return run_data

    def _export_iteration(
        self,
        iteration: dict[str, Any],
        include_lm_calls: bool,
    ) -> dict[str, Any]:
        """Export a single iteration with all related data.

        Args:
            iteration: Iteration record from database
            include_lm_calls: Whether to include LM calls

        Returns:
            Complete iteration dictionary
        """
        iteration_id = iteration["iteration_id"]

        # Get programs
        parent_program = None
        if iteration["parent_program_id"]:
            parent_program = self.storage.get_program(iteration["parent_program_id"])

        candidate_program = None
        if iteration["candidate_program_id"]:
            candidate_program = self.storage.get_program(iteration["candidate_program_id"])

        # Get rollouts
        parent_rollouts = list(self.storage.get_rollouts(iteration_id, "parent_minibatch"))
        candidate_rollouts = list(self.storage.get_rollouts(iteration_id, "candidate_minibatch"))
        validation_rollouts = list(self.storage.get_rollouts(iteration_id, "validation"))

        # Get reflection
        reflection = self.storage.get_reflection(iteration_id)

        iteration_data = {
            "iteration_number": iteration["iteration_number"],
            "timestamp": iteration["timestamp"],
            "iteration_type": iteration["iteration_type"],
            "duration_ms": iteration["duration_ms"],
            "parent_program": self._export_program(parent_program) if parent_program else None,
            "parent_val_score": iteration["parent_val_score"],
            "parent_minibatch": {
                "rollouts": [self._export_rollout(r) for r in parent_rollouts],
                "aggregate_score": iteration["parent_minibatch_score"],
            },
            "reflection": self._export_reflection(reflection) if reflection else None,
            "candidate_program": self._export_program(candidate_program) if candidate_program else None,
            "candidate_minibatch": {
                "rollouts": [self._export_rollout(r) for r in candidate_rollouts],
                "aggregate_score": iteration["candidate_minibatch_score"],
            },
            "validation": {
                "rollouts": [self._export_rollout(r) for r in validation_rollouts],
                "aggregate_score": iteration["val_aggregate_score"],
            },
            "accepted": bool(iteration["accepted"]),
            "acceptance_reason": iteration["acceptance_reason"],
        }

        # Add LM calls if requested
        if include_lm_calls:
            lm_calls = list(self.storage.get_lm_calls(iteration_id=iteration_id))
            iteration_data["lm_calls"] = [self._export_lm_call(call) for call in lm_calls]

        return iteration_data

    def _export_program(self, program: dict[str, Any]) -> dict[str, Any]:
        """Export program data.

        Args:
            program: Program record from database

        Returns:
            Program dictionary
        """
        return {
            "program_id": program["program_id"],
            "signature": program["signature"],
            "instructions": json.loads(program["instructions_json"]) if program["instructions_json"] else {},
            "created_at": program["created_at"],
        }

    def _export_rollout(self, rollout: dict[str, Any]) -> dict[str, Any]:
        """Export rollout data.

        Args:
            rollout: Rollout record from database

        Returns:
            Rollout dictionary
        """
        return {
            "rollout_id": rollout["rollout_id"],
            "program_id": rollout["program_id"],
            "rollout_type": rollout["rollout_type"],
            "example_id": rollout["example_id"],
            "input": json.loads(rollout["input_json"]) if rollout["input_json"] else None,
            "output": json.loads(rollout["output_json"]) if rollout["output_json"] else None,
            "evaluation": {
                "score": rollout["score"],
                "feedback": rollout["feedback"],
            },
        }

    def _export_reflection(self, reflection: dict[str, Any]) -> dict[str, Any]:
        """Export reflection data.

        Args:
            reflection: Reflection record from database

        Returns:
            Reflection dictionary
        """
        return {
            "components_to_update": json.loads(reflection["components_json"]) if reflection["components_json"] else [],
            "reflective_dataset": json.loads(reflection["reflective_dataset_json"]) if reflection["reflective_dataset_json"] else None,
            "proposed_instructions": json.loads(reflection["proposed_instructions_json"]) if reflection["proposed_instructions_json"] else {},
            "duration_ms": reflection["duration_ms"],
        }

    def _export_lm_call(self, lm_call: dict[str, Any]) -> dict[str, Any]:
        """Export LM call data.

        Args:
            lm_call: LM call record from database

        Returns:
            LM call dictionary
        """
        return {
            "call_id": lm_call["call_id"],
            "call_type": lm_call["call_type"],
            "predictor_name": lm_call["predictor_name"],
            "signature_name": lm_call["signature_name"],
            "timestamp": lm_call["timestamp"],
            "model": lm_call["model"],
            "tokens": {
                "input": lm_call["input_tokens"],
                "output": lm_call["output_tokens"],
                "total": lm_call["total_tokens"],
            },
            "latency_ms": lm_call["latency_ms"],
        }

    def _export_pareto_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Export pareto snapshot data.

        Args:
            snapshot: Pareto snapshot record from database

        Returns:
            Pareto snapshot dictionary
        """
        snapshot_id = snapshot["snapshot_id"]

        # Get tasks
        tasks = list(self.storage.get_pareto_tasks(snapshot_id))

        return {
            "snapshot_type": snapshot["snapshot_type"],
            "iteration_id": snapshot["iteration_id"],
            "timestamp": snapshot["timestamp"],
            "best_program_id": snapshot["best_program_id"],
            "best_score": snapshot["best_score"],
            "tasks": [
                {
                    "example_id": task["example_id"],
                    "dominant_program_id": task["dominant_program_id"],
                    "dominant_score": task["dominant_score"],
                }
                for task in tasks
            ],
        }

    def export_all_runs(
        self,
        output_dir: str | Path,
        pretty: bool = True,
        include_lm_calls: bool = False,
    ) -> list[str]:
        """Export all runs to separate JSON files.

        Args:
            output_dir: Directory to write JSON files
            pretty: Whether to pretty-print JSON
            include_lm_calls: Whether to include LM call traces

        Returns:
            List of exported run IDs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_runs = []
        for run_id in self.storage.list_runs():
            try:
                output_path = output_dir / f"{run_id}.json"
                self.export_run(
                    run_id=run_id,
                    output_path=output_path,
                    pretty=pretty,
                    include_lm_calls=include_lm_calls,
                )
                exported_runs.append(run_id)
            except Exception as e:
                logger.error(f"Error exporting run {run_id}: {e}")

        logger.info(f"Exported {len(exported_runs)} runs to {output_dir}")
        return exported_runs
