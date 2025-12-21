"""Export GEPA run data for visualization tools."""

import json
from pathlib import Path
from typing import Any

from dspy_gepa_logger.storage.base import StorageBackend
from dspy_gepa_logger.models.run import GEPARunRecord
from dspy_gepa_logger.models.iteration import IterationRecord


class VisualizationExporter:
    """Export GEPA run data in formats suitable for visualization.

    Provides methods to export run data as JSON for use with
    external visualization tools and dashboards.
    """

    def __init__(self, storage: StorageBackend):
        """Initialize the exporter.

        Args:
            storage: Storage backend to load run data from
        """
        self.storage = storage

    def export_summary_json(self, run_id: str, output_path: str | Path | None = None) -> dict[str, Any]:
        """Export high-level summary for dashboard view.

        Args:
            run_id: The run to export
            output_path: Optional path to write JSON file

        Returns:
            Summary dictionary
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        summary = self._build_summary(run)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

        return summary

    def _build_summary(self, run: GEPARunRecord) -> dict[str, Any]:
        """Build summary dictionary from run record."""
        # Build score history
        score_history = []
        best_so_far = run.seed_aggregate_score

        for it in run.iterations:
            if it.accepted and it.val_aggregate_score:
                best_so_far = max(best_so_far, it.val_aggregate_score)

            score_history.append({
                "iteration": it.iteration_number,
                "parent_score": it.parent_val_score,
                "parent_minibatch_score": it.parent_minibatch_aggregate,
                "candidate_minibatch_score": it.candidate_minibatch_aggregate,
                "candidate_val_score": it.val_aggregate_score,
                "accepted": it.accepted,
                "best_so_far": best_so_far,
            })

        # Build Pareto history
        pareto_history = []
        for it in run.iterations:
            if it.pareto_update:
                pareto_history.append({
                    "iteration": it.iteration_number,
                    "score_before": it.pareto_update.pareto_aggregate_score_before,
                    "score_after": it.pareto_update.pareto_aggregate_score_after,
                    "new_best_instances": it.pareto_update.new_best_instances,
                    "ties_added": it.pareto_update.ties_added,
                })

        return {
            "run_id": run.run_id,
            "status": run.status,
            "duration_seconds": run.duration_seconds,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "metrics": {
                "total_iterations": run.total_iterations,
                "accepted_iterations": len(run.accepted_iterations),
                "acceptance_rate": run.acceptance_rate,
                "total_metric_calls": run.total_metric_calls,
                "seed_score": run.seed_aggregate_score,
                "final_best_score": run.best_aggregate_score,
                "improvement": run.improvement,
                "best_candidate_idx": run.best_candidate_idx,
            },
            "config": run.config.to_dict(),
            "score_history": score_history,
            "pareto_history": pareto_history,
            "candidates_count": len(run.candidates),
        }

    def export_iteration_detail(
        self,
        run_id: str,
        iteration_num: int,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Export detailed data for a single iteration.

        Args:
            run_id: The run to export from
            iteration_num: Iteration number to export
            output_path: Optional path to write JSON file

        Returns:
            Detailed iteration dictionary
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        iteration = None
        for it in run.iterations:
            if it.iteration_number == iteration_num:
                iteration = it
                break

        if iteration is None:
            raise ValueError(f"Iteration not found: {iteration_num}")

        detail = self._build_iteration_detail(iteration)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(detail, f, indent=2, default=str)

        return detail

    def _build_iteration_detail(self, iteration: IterationRecord) -> dict[str, Any]:
        """Build detailed dictionary for an iteration."""
        # Build minibatch evaluations
        minibatch_evals = []
        for i, ex_id in enumerate(iteration.minibatch_ids):
            eval_data = {
                "example_id": ex_id,
                "input": iteration.minibatch_inputs[i] if i < len(iteration.minibatch_inputs) else None,
                "output": iteration.minibatch_outputs[i] if i < len(iteration.minibatch_outputs) else None,
                "score": iteration.minibatch_scores[i] if i < len(iteration.minibatch_scores) else None,
                "feedback": iteration.minibatch_feedback[i] if i < len(iteration.minibatch_feedback) else None,
            }

            # Add new candidate output/score if available
            if iteration.new_candidate_minibatch_outputs and i < len(iteration.new_candidate_minibatch_outputs):
                eval_data["new_candidate_output"] = iteration.new_candidate_minibatch_outputs[i]
            if iteration.new_candidate_minibatch_scores and i < len(iteration.new_candidate_minibatch_scores):
                eval_data["new_candidate_score"] = iteration.new_candidate_minibatch_scores[i]

            minibatch_evals.append(eval_data)

        result = {
            "iteration_number": iteration.iteration_number,
            "timestamp": iteration.timestamp.isoformat(),
            "iteration_type": iteration.iteration_type,
            "duration_ms": iteration.duration_ms,
            "parent": {
                "candidate_idx": iteration.parent_candidate_idx,
                "prompt": iteration.parent_prompt,
                "val_score": iteration.parent_val_score,
                "selection_strategy": iteration.selection_strategy,
            },
            "minibatch": {
                "example_ids": iteration.minibatch_ids,
                "evaluations": minibatch_evals,
                "parent_aggregate": iteration.parent_minibatch_aggregate,
                "candidate_aggregate": iteration.candidate_minibatch_aggregate,
            },
            "decision": {
                "accepted": iteration.accepted,
                "reason": iteration.acceptance_reason,
                "new_candidate_idx": iteration.new_candidate_idx,
                "val_aggregate_score": iteration.val_aggregate_score,
            },
        }

        # Add reflection data if present
        if iteration.reflection:
            result["reflection"] = {
                "components_updated": iteration.reflection.components_to_update,
                "proposed_instructions": iteration.reflection.proposed_instructions,
                "duration_ms": iteration.reflection.duration_ms,
                "lm_calls_count": len(iteration.reflection.lm_calls),
            }

            # Include reflective dataset summary
            if iteration.reflection.reflective_datasets:
                result["reflection"]["dataset_sizes"] = {
                    comp: len(examples)
                    for comp, examples in iteration.reflection.reflective_datasets.items()
                }

        # Add Pareto update if present
        if iteration.pareto_update:
            result["pareto_update"] = {
                "new_best_instances": iteration.pareto_update.new_best_instances,
                "ties_added": iteration.pareto_update.ties_added,
                "score_before": iteration.pareto_update.pareto_aggregate_score_before,
                "score_after": iteration.pareto_update.pareto_aggregate_score_after,
            }

        return result

    def export_all_iterations(
        self,
        run_id: str,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Export all iterations for a run.

        Args:
            run_id: The run to export
            output_path: Optional path to write JSON file

        Returns:
            List of iteration detail dictionaries
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        iterations = [self._build_iteration_detail(it) for it in run.iterations]

        if output_path:
            with open(output_path, "w") as f:
                json.dump(iterations, f, indent=2, default=str)

        return iterations

    def export_candidates(
        self,
        run_id: str,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Export all candidates for a run.

        Args:
            run_id: The run to export
            output_path: Optional path to write JSON file

        Returns:
            List of candidate dictionaries
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        candidates = [c.to_dict() for c in run.candidates]

        if output_path:
            with open(output_path, "w") as f:
                json.dump(candidates, f, indent=2, default=str)

        return candidates

    def export_full_run(
        self,
        run_id: str,
        output_dir: str | Path,
    ) -> None:
        """Export complete run data to a directory.

        Creates:
            - summary.json: High-level summary
            - iterations.json: All iterations
            - candidates.json: All candidates

        Args:
            run_id: The run to export
            output_dir: Directory to write files to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.export_summary_json(run_id, output_dir / "summary.json")
        self.export_all_iterations(run_id, output_dir / "iterations.json")
        self.export_candidates(run_id, output_dir / "candidates.json")
