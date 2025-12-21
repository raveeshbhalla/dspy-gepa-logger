"""Export GEPA run data as Pandas DataFrames."""

from typing import Any

from dspy_gepa_logger.storage.base import StorageBackend

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


class DataFrameExporter:
    """Export GEPA run data as Pandas DataFrames.

    Provides convenient access to run data for analysis in
    notebooks and data science workflows.
    """

    def __init__(self, storage: StorageBackend):
        """Initialize the exporter.

        Args:
            storage: Storage backend to load run data from
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrameExporter. Install with: pip install pandas")

        self.storage = storage

    def iterations_df(self, run_id: str) -> "pd.DataFrame":
        """Get iterations as a DataFrame.

        Args:
            run_id: The run to export

        Returns:
            DataFrame with one row per iteration
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        records = []
        for it in run.iterations:
            records.append({
                "iteration": it.iteration_number,
                "parent_idx": it.parent_candidate_idx,
                "parent_val_score": it.parent_val_score,
                "parent_minibatch_score": it.parent_minibatch_aggregate,
                "candidate_minibatch_score": it.candidate_minibatch_aggregate,
                "minibatch_improvement": it.minibatch_improvement,
                "accepted": it.accepted,
                "acceptance_reason": it.acceptance_reason,
                "val_aggregate_score": it.val_aggregate_score,
                "new_candidate_idx": it.new_candidate_idx,
                "iteration_type": it.iteration_type,
                "duration_ms": it.duration_ms,
                "minibatch_size": len(it.minibatch_ids),
                "selection_strategy": it.selection_strategy,
                "components_updated": (
                    ",".join(it.reflection.components_to_update) if it.reflection else None
                ),
                "reflection_duration_ms": it.reflection.duration_ms if it.reflection else None,
                "pareto_improvement": (
                    it.pareto_update.pareto_aggregate_score_after - it.pareto_update.pareto_aggregate_score_before
                    if it.pareto_update else None
                ),
            })

        return pd.DataFrame(records)

    def evaluations_df(self, run_id: str) -> "pd.DataFrame":
        """Get all minibatch evaluations as a DataFrame.

        Args:
            run_id: The run to export

        Returns:
            DataFrame with one row per example evaluation
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        records = []
        for it in run.iterations:
            for i, ex_id in enumerate(it.minibatch_ids):
                record = {
                    "iteration": it.iteration_number,
                    "example_id": ex_id,
                    "parent_score": it.minibatch_scores[i] if i < len(it.minibatch_scores) else None,
                    "parent_feedback": it.minibatch_feedback[i] if i < len(it.minibatch_feedback) else None,
                }

                # Add candidate score if available
                if it.new_candidate_minibatch_scores and i < len(it.new_candidate_minibatch_scores):
                    record["candidate_score"] = it.new_candidate_minibatch_scores[i]
                    record["score_improvement"] = (
                        it.new_candidate_minibatch_scores[i] - record["parent_score"]
                        if record["parent_score"] is not None else None
                    )

                records.append(record)

        return pd.DataFrame(records)

    def candidates_df(self, run_id: str) -> "pd.DataFrame":
        """Get candidates as a DataFrame.

        Args:
            run_id: The run to export

        Returns:
            DataFrame with one row per candidate
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        records = []
        for c in run.candidates:
            records.append({
                "candidate_idx": c.candidate_idx,
                "creation_iteration": c.creation_iteration,
                "creation_type": c.creation_type,
                "parent_indices": ",".join(map(str, c.parent_indices)),
                "val_aggregate_score": c.val_aggregate_score,
                "metric_calls_at_discovery": c.metric_calls_at_discovery,
                "num_components": len(c.instructions),
            })

        return pd.DataFrame(records)

    def score_history_df(self, run_id: str) -> "pd.DataFrame":
        """Get score history for plotting.

        Args:
            run_id: The run to export

        Returns:
            DataFrame with running best scores per iteration
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        records = []
        best_so_far = run.seed_aggregate_score

        for it in run.iterations:
            if it.accepted and it.val_aggregate_score:
                best_so_far = max(best_so_far, it.val_aggregate_score)

            records.append({
                "iteration": it.iteration_number,
                "parent_score": it.parent_val_score,
                "candidate_score": it.val_aggregate_score,
                "accepted": it.accepted,
                "best_so_far": best_so_far,
            })

        return pd.DataFrame(records)

    def lm_calls_df(self, run_id: str) -> "pd.DataFrame":
        """Get LM calls as a DataFrame.

        Args:
            run_id: The run to export

        Returns:
            DataFrame with one row per LM call
        """
        run = self.storage.load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        records = []
        for it in run.iterations:
            # Get LM calls from reflection
            if it.reflection:
                for lm in it.reflection.lm_calls:
                    records.append({
                        "iteration": it.iteration_number,
                        "call_id": lm.call_id,
                        "model": lm.model,
                        "is_reflection": lm.is_reflection,
                        "component_name": lm.component_name,
                        "duration_ms": lm.duration_ms,
                        "timestamp": lm.timestamp,
                        "input_tokens": lm.usage.get("input_tokens") if lm.usage else None,
                        "output_tokens": lm.usage.get("output_tokens") if lm.usage else None,
                    })

            # Get LM calls from traces
            if it.minibatch_traces:
                for trace in it.minibatch_traces:
                    for lm in trace.lm_calls:
                        records.append({
                            "iteration": it.iteration_number,
                            "call_id": lm.call_id,
                            "model": lm.model,
                            "is_reflection": lm.is_reflection,
                            "component_name": lm.component_name,
                            "duration_ms": lm.duration_ms,
                            "timestamp": lm.timestamp,
                            "input_tokens": lm.usage.get("input_tokens") if lm.usage else None,
                            "output_tokens": lm.usage.get("output_tokens") if lm.usage else None,
                        })

        return pd.DataFrame(records)

    def export_all(self, run_id: str, output_dir: str) -> dict[str, "pd.DataFrame"]:
        """Export all DataFrames to CSV files.

        Args:
            run_id: The run to export
            output_dir: Directory to write CSV files to

        Returns:
            Dictionary of DataFrames
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dfs = {
            "iterations": self.iterations_df(run_id),
            "evaluations": self.evaluations_df(run_id),
            "candidates": self.candidates_df(run_id),
            "score_history": self.score_history_df(run_id),
        }

        # Only include LM calls if there are any
        try:
            lm_calls = self.lm_calls_df(run_id)
            if len(lm_calls) > 0:
                dfs["lm_calls"] = lm_calls
        except Exception:
            pass

        for name, df in dfs.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)

        return dfs
