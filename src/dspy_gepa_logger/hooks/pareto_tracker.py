"""Custom experiment tracker that captures Pareto frontier data from GEPA."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ParetoCapturingTracker:
    """Wraps GEPA's ExperimentTracker to capture Pareto frontier data.

    This tracker intercepts log_metrics() calls to extract Pareto frontier
    information and forwards it to the GEPARunTracker for storage.

    Metrics of interest from GEPA's logging (see gepa/logging/utils.py):
    - valset_pareto_front_scores: list of best scores for each validation task
    - valset_pareto_front_programs: list of sets showing which programs are best for each task
    - valset_pareto_front_agg: aggregate Pareto frontier score
    - individual_valset_score_new_program: new candidate's scores on each validation task
    """

    def __init__(self, wrapped_tracker: Any, gepa_run_tracker: Any):
        """Initialize the Pareto capturing tracker.

        Args:
            wrapped_tracker: The actual ExperimentTracker instance from GEPA
            gepa_run_tracker: GEPARunTracker instance for storing captured data
        """
        self.wrapped = wrapped_tracker
        self.gepa_tracker = gepa_run_tracker
        self.previous_pareto_state = None
        logger.info(f"ParetoCapturingTracker initialized - wrapping {type(wrapped_tracker).__name__}")

    def __enter__(self):
        """Context manager entry."""
        if hasattr(self.wrapped, '__enter__'):
            self.wrapped.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self.wrapped, '__exit__'):
            return self.wrapped.__exit__(exc_type, exc_val, exc_tb)
        return False

    def initialize(self):
        """Initialize the wrapped tracker."""
        if hasattr(self.wrapped, 'initialize'):
            self.wrapped.initialize()

    def start_run(self):
        """Start a run in the wrapped tracker."""
        if hasattr(self.wrapped, 'start_run'):
            self.wrapped.start_run()

    def end_run(self):
        """End a run in the wrapped tracker."""
        if hasattr(self.wrapped, 'end_run'):
            self.wrapped.end_run()

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Intercept log_metrics to capture Pareto data.

        Args:
            metrics: Dictionary of metrics being logged by GEPA
            step: Iteration number (1-indexed in GEPA logs)
        """
        logger.info(f"ParetoCapturingTracker.log_metrics called - step={step}, metrics keys={list(metrics.keys())}")

        # Forward to wrapped tracker first
        if hasattr(self.wrapped, 'log_metrics'):
            self.wrapped.log_metrics(metrics, step=step)

        # Extract Pareto frontier data if present
        if not self.gepa_tracker.is_tracking:
            logger.warning("Pareto tracker: gepa_tracker.is_tracking is False")
            return

        if self.gepa_tracker._current_iteration is None:
            logger.warning("Pareto tracker: _current_iteration is None")
            return

        # GEPA logs different metrics at different stages
        # Validation and Pareto data come in the same log_metrics call when a candidate is accepted
        pareto_programs = metrics.get('valset_pareto_front_programs')
        pareto_scores = metrics.get('valset_pareto_front_scores')
        pareto_agg = metrics.get('valset_pareto_front_agg')
        individual_val_scores = metrics.get('individual_valset_score_new_program')
        val_program_average = metrics.get('val_program_average')  # Newer GEPA versions
        new_program_idx = metrics.get('new_program_idx')

        logger.info(f"Pareto data found: programs={pareto_programs is not None}, scores={pareto_scores is not None}, agg={pareto_agg}, individual_val={individual_val_scores is not None}, val_avg={val_program_average}, new_idx={new_program_idx}")

        # Record acceptance if we have validation scores
        # Older GEPA: individual_valset_score_new_program (dict with per-task scores)
        # Newer GEPA: val_program_average (scalar average)
        if individual_val_scores is not None:
            # Older GEPA format
            val_scores_dict = {i: score for i, score in enumerate(individual_val_scores)}
            val_aggregate_score = sum(individual_val_scores) / len(individual_val_scores) if individual_val_scores else 0.0

            self.gepa_tracker.record_acceptance(
                accepted=True,
                reason="minibatch_improved_valset_evaluated",
                val_scores=val_scores_dict,
                val_aggregate_score=val_aggregate_score,
                new_candidate_idx=new_program_idx,
            )
            logger.info(f"Recorded acceptance for iteration {self.gepa_tracker._current_iteration.iteration_number}: val_score={val_aggregate_score:.4f}, idx={new_program_idx}")
        elif val_program_average is not None and new_program_idx is not None:
            # Newer GEPA format - we have aggregate but not per-task scores
            # We can infer acceptance from presence of pareto data and val_program_average
            self.gepa_tracker.record_acceptance(
                accepted=True,
                reason="minibatch_improved_valset_evaluated",
                val_scores={},  # Don't have per-task scores in newer GEPA
                val_aggregate_score=val_program_average,
                new_candidate_idx=new_program_idx,
            )
            logger.info(f"Recorded acceptance for iteration {self.gepa_tracker._current_iteration.iteration_number}: val_score={val_program_average:.4f}, idx={new_program_idx}")

        if pareto_programs is not None and pareto_agg is not None:
            # Handle both dict and list formats from different GEPA versions
            if isinstance(pareto_programs, dict):
                # Newer GEPA versions return dict directly
                frontier_after = pareto_programs
            else:
                # Older versions return list of sets - convert to dict
                frontier_after = {i: prog_set for i, prog_set in enumerate(pareto_programs)}

            # Get per-task scores from metrics if GEPA logged them
            # (requires log_individual_valset_scores_and_programs=True in GEPA 0.0.22+)
            pareto_scores = metrics.get('valset_pareto_front_scores', {})

            # If GEPA didn't log per-task scores, use validation scores we captured
            # from the _run_full_eval_and_add engine hook
            if not pareto_scores and self.gepa_tracker._current_iteration.val_scores:
                # Use the validation scores we already captured
                pareto_scores = dict(self.gepa_tracker._current_iteration.val_scores)
                logger.info(f"Using captured val_scores for pareto task scores: {len(pareto_scores)} tasks")

            # Fallback: use individual validation scores from metrics if available
            if not pareto_scores and individual_val_scores is not None:
                # Convert to dict if it's a list
                if isinstance(individual_val_scores, list):
                    individual_val_scores = {i: s for i, s in enumerate(individual_val_scores)}
                elif isinstance(individual_val_scores, dict):
                    individual_val_scores = dict(individual_val_scores)

                # For each task where this program is in the frontier, use its score
                new_idx = new_program_idx
                for task_id, prog_set in frontier_after.items():
                    if new_idx in prog_set and task_id in individual_val_scores:
                        pareto_scores[task_id] = individual_val_scores[task_id]

            # Use per-task scores if available, otherwise use 0.0 as placeholder
            if pareto_scores:
                if isinstance(pareto_scores, dict):
                    frontier_scores_after = pareto_scores
                else:
                    frontier_scores_after = {i: score for i, score in enumerate(pareto_scores)}
            else:
                frontier_scores_after = {i: 0.0 for i in frontier_after.keys()}

            # Get before state from previous iteration
            frontier_before = {}
            frontier_scores_before = {}
            pareto_agg_before = 0.0

            if self.previous_pareto_state:
                frontier_before = self.previous_pareto_state['frontier']
                frontier_scores_before = self.previous_pareto_state['scores']
                pareto_agg_before = self.previous_pareto_state['aggregate']

            # Detect changes (new best instances and ties)
            new_best_instances = []
            ties_added = []

            for task_id, programs_after in frontier_after.items():
                programs_before = frontier_before.get(task_id, set())

                # Check if frontier changed for this task
                if programs_after != programs_before:
                    # Use .get() since per-task scores may not be logged by GEPA
                    score_after = frontier_scores_after.get(task_id, 0.0)
                    score_before = frontier_scores_before.get(task_id, 0.0)

                    if score_after > score_before:
                        new_best_instances.append(task_id)
                    elif score_after == score_before and len(programs_after) > len(programs_before):
                        ties_added.append(task_id)

            # Create Pareto update record
            from dspy_gepa_logger.models.candidate import ParetoFrontierUpdate

            pareto_update = ParetoFrontierUpdate(
                iteration_number=self.gepa_tracker._current_iteration.iteration_number,
                frontier_before=frontier_before,
                frontier_scores_before=frontier_scores_before,
                frontier_after=frontier_after,
                frontier_scores_after=frontier_scores_after,
                new_best_instances=new_best_instances,
                ties_added=ties_added,
                pareto_aggregate_score_before=pareto_agg_before,
                pareto_aggregate_score_after=pareto_agg if pareto_agg is not None else 0.0,
            )

            # Store in current iteration
            self.gepa_tracker._current_iteration.pareto_update = pareto_update

            # Update previous state for next iteration
            self.previous_pareto_state = {
                'frontier': frontier_after,
                'scores': frontier_scores_after,
                'aggregate': pareto_agg if pareto_agg is not None else 0.0,
            }

            logger.info(
                f"Captured Pareto update for iteration {self.gepa_tracker._current_iteration.iteration_number}: "
                f"agg_score={pareto_agg:.4f}, tasks={len(frontier_after)}, "
                f"new_best={len(new_best_instances)}, ties={len(ties_added)}"
            )

    def __getattr__(self, name: str):
        """Delegate all other attributes to wrapped tracker."""
        return getattr(self.wrapped, name)
