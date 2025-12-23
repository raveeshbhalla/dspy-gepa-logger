"""Unified GEPA tracker combining all v2 hooks.

GEPATracker provides a single interface to:
1. Capture GEPA state via stop_callbacks (GEPAStateLogger)
2. Capture all LM calls with context tags (DSPyLMLogger)
3. Capture evaluation details via wrapped metric (LoggedMetric)
4. Optionally track reflection/proposal phases (LoggedInstructionProposer)

This is the v2.2 architecture that uses public hooks instead of monkey-patching.

Usage:
    from dspy_gepa_logger import GEPATracker

    tracker = GEPATracker()

    gepa = GEPA(
        metric=tracker.wrap_metric(my_metric),
        gepa_kwargs={'stop_callbacks': [tracker.get_stop_callback()]},
    )

    # Configure DSPy with LM logging
    import dspy
    dspy.configure(callbacks=tracker.get_dspy_callbacks())

    # Run optimization
    result = gepa.compile(student, trainset=train, valset=val)

    # Access captured data
    print(tracker.get_summary())
    print(tracker.get_candidate_diff(0, 5))
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .context import clear_ctx, get_ctx
from .state_logger import GEPAStateLogger, IterationDelta, IterationMetadata
from .lm_logger import DSPyLMLogger, LMCall
from .logged_metric import LoggedMetric, EvaluationRecord
from .logged_proposer import (
    LoggedInstructionProposer,
    LoggedSelector,
    ReflectionCall,
    ProposalCall,
)


logger = logging.getLogger(__name__)


@dataclass
class CandidateDiff:
    """Diff between two candidates showing prompts and evaluation changes.

    Useful for understanding what changed and why.
    """

    from_idx: int
    to_idx: int
    prompt_changes: dict[str, tuple[str, str]]  # key -> (old, new)
    evaluation_changes: list[dict[str, Any]]  # Per-example changes
    lineage: list[int]  # Full lineage from to_idx to seed


class GEPATracker:
    """Unified tracker that combines all GEPA logging hooks.

    This class integrates:
    - GEPAStateLogger: Captures iteration state incrementally via stop_callbacks
    - DSPyLMLogger: Captures all LM calls with context tags
    - LoggedMetric: Captures evaluation details (score, feedback, prediction)
    - LoggedInstructionProposer: Optionally tracks reflection/proposal phases

    The tracker automatically manages context propagation between components.

    Attributes:
        state_logger: GEPAStateLogger instance for state capture
        lm_logger: DSPyLMLogger instance for LM call capture
        metric_logger: LoggedMetric wrapper (created via wrap_metric())
        proposer_logger: Optional LoggedInstructionProposer (created via wrap_proposer())
    """

    def __init__(
        self,
        capture_lm_calls: bool = True,
        dspy_version: str | None = None,
        gepa_version: str | None = None,
    ):
        """Initialize the tracker.

        Args:
            capture_lm_calls: Whether to capture LM calls (default: True)
            dspy_version: DSPy version (for compatibility info)
            gepa_version: GEPA version (for compatibility info)
        """
        self._start_time: float | None = None
        self._capture_lm_calls = capture_lm_calls

        # Core components
        self.state_logger = GEPAStateLogger(
            dspy_version=dspy_version,
            gepa_version=gepa_version,
        )
        self.lm_logger = DSPyLMLogger() if capture_lm_calls else None
        self.metric_logger: LoggedMetric | None = None
        self.proposer_logger: LoggedInstructionProposer | None = None
        self.selector_logger: LoggedSelector | None = None

        # Wrapped objects (returned to user)
        self._wrapped_metric: Callable[..., Any] | None = None
        self._wrapped_proposer: Any | None = None
        self._wrapped_selector: Any | None = None

        # Valset example IDs for filtering comparisons
        self._valset_example_ids: set[str] | None = None

    def set_valset(self, valset: list[Any]) -> None:
        """Set the validation set for comparison filtering.

        Call this before running GEPA optimization to ensure performance
        comparisons only include validation examples (not training examples).

        Args:
            valset: List of validation examples (dspy.Example or similar)

        Example:
            tracker.set_valset(val_data)
            result = gepa.compile(student, trainset=train_data, valset=val_data)
        """
        from .logged_metric import _deterministic_example_id

        self._valset_example_ids = {
            _deterministic_example_id(example) for example in valset
        }

    def wrap_metric(
        self,
        metric: Callable[..., Any],
        capture_prediction: bool = True,
        max_prediction_preview: int = 200,
        failure_score: float = 0.0,
    ) -> LoggedMetric:
        """Wrap a metric function to capture evaluation details.

        The wrapped metric will:
        1. Set phase="eval" in context before calling the metric
        2. Capture score, feedback, and prediction for each call
        3. Restore the previous phase after the call
        4. Handle exceptions gracefully by returning failure_score

        Args:
            metric: The metric function to wrap
            capture_prediction: Whether to capture predictions (default: True)
            max_prediction_preview: Max length for prediction preview
            failure_score: Score to return when the metric throws an exception

        Returns:
            LoggedMetric wrapper that can be used in place of the original metric
        """
        self.metric_logger = LoggedMetric(
            metric_fn=metric,
            capture_prediction=capture_prediction,
            max_prediction_preview=max_prediction_preview,
            failure_score=failure_score,
        )
        self._wrapped_metric = self.metric_logger
        return self.metric_logger

    def wrap_proposer(self, proposer: Any) -> LoggedInstructionProposer:
        """Wrap an instruction proposer to track reflection/proposal phases.

        The wrapped proposer will:
        1. Set phase="reflection" during reflection analysis
        2. Set phase="proposal" during proposal generation
        3. Record all reflection and proposal calls

        Args:
            proposer: The instruction proposer to wrap

        Returns:
            LoggedInstructionProposer wrapper
        """
        self.proposer_logger = LoggedInstructionProposer(proposer)
        self._wrapped_proposer = self.proposer_logger
        return self.proposer_logger

    def wrap_selector(self, selector: Any) -> LoggedSelector:
        """Wrap a selector to track selection phases.

        Args:
            selector: The selector to wrap

        Returns:
            LoggedSelector wrapper
        """
        self.selector_logger = LoggedSelector(selector)
        self._wrapped_selector = self.selector_logger
        return self.selector_logger

    def get_stop_callback(self) -> GEPAStateLogger:
        """Get the stop_callback for use in gepa_kwargs.

        Returns the GEPAStateLogger which implements the stop_callbacks interface.
        Always returns False (never stops optimization).

        Usage:
            gepa = GEPA(
                metric=logged_metric,
                gepa_kwargs={'stop_callbacks': [tracker.get_stop_callback()]},
            )

        Returns:
            GEPAStateLogger instance
        """
        return self.state_logger

    def get_dspy_callbacks(self) -> list[DSPyLMLogger]:
        """Get DSPy callbacks for LM call capture.

        Returns list of callbacks to configure with dspy.configure().

        Usage:
            import dspy
            dspy.configure(callbacks=tracker.get_dspy_callbacks())

        Returns:
            List containing DSPyLMLogger (or empty if capture_lm_calls=False)
        """
        if self.lm_logger is not None:
            return [self.lm_logger]
        return []

    # ==================== Query Methods ====================

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of the optimization run.

        Returns:
            Dict with:
            - state_summary: From GEPAStateLogger
            - lm_summary: From DSPyLMLogger (if enabled)
            - metric_summary: From LoggedMetric (if wrapped)
            - proposer_summary: From LoggedInstructionProposer (if wrapped)
        """
        summary: dict[str, Any] = {
            "state": self.state_logger.get_summary(),
        }

        if self.lm_logger is not None:
            summary["lm_calls"] = self.lm_logger.get_summary()

        if self.metric_logger is not None:
            summary["evaluations"] = {
                "total_evaluations": len(self.metric_logger.evaluations),
                "unique_examples": len(
                    set(e.example_id for e in self.metric_logger.evaluations)
                ),
                "unique_candidates": len(
                    set(
                        e.candidate_idx
                        for e in self.metric_logger.evaluations
                        if e.candidate_idx is not None
                    )
                ),
            }

        if self.proposer_logger is not None:
            summary["proposer"] = {
                "total_reflections": len(self.proposer_logger.reflection_calls),
                "total_proposals": len(self.proposer_logger.proposal_calls),
            }

        return summary

    def get_candidate_diff(
        self,
        from_idx: int,
        to_idx: int,
    ) -> CandidateDiff:
        """Get diff between two candidates.

        Shows what changed in prompts and evaluations.

        Args:
            from_idx: Source candidate index
            to_idx: Target candidate index

        Returns:
            CandidateDiff with prompt changes, evaluation changes, and lineage
        """
        candidates = self.state_logger.get_all_candidates()

        # Get prompts
        from_prompt = candidates[from_idx] if from_idx < len(candidates) else {}
        to_prompt = candidates[to_idx] if to_idx < len(candidates) else {}

        # Compute prompt changes
        all_keys = set(from_prompt.keys()) | set(to_prompt.keys())
        prompt_changes: dict[str, tuple[str, str]] = {}
        for key in all_keys:
            old_val = from_prompt.get(key, "")
            new_val = to_prompt.get(key, "")
            if old_val != new_val:
                prompt_changes[key] = (old_val, new_val)

        # Compute evaluation changes if metric logger is available
        evaluation_changes: list[dict[str, Any]] = []
        if self.metric_logger is not None:
            from_evals = self.metric_logger.get_evaluations_for_candidate(from_idx)
            to_evals = self.metric_logger.get_evaluations_for_candidate(to_idx)

            # Group by example_id
            from_by_example = {e.example_id: e for e in from_evals}
            to_by_example = {e.example_id: e for e in to_evals}

            all_examples = set(from_by_example.keys()) | set(to_by_example.keys())
            for example_id in sorted(all_examples):
                from_eval = from_by_example.get(example_id)
                to_eval = to_by_example.get(example_id)

                change: dict[str, Any] = {"example_id": example_id}
                if from_eval and to_eval:
                    change["from_score"] = from_eval.score
                    change["to_score"] = to_eval.score
                    change["score_delta"] = to_eval.score - from_eval.score
                    if from_eval.feedback != to_eval.feedback:
                        change["feedback_changed"] = True
                        change["from_feedback"] = from_eval.feedback
                        change["to_feedback"] = to_eval.feedback
                elif from_eval:
                    change["from_score"] = from_eval.score
                    change["to_score"] = None
                    change["removed"] = True
                elif to_eval:
                    change["from_score"] = None
                    change["to_score"] = to_eval.score
                    change["added"] = True

                evaluation_changes.append(change)

        # Get lineage
        lineage = self.state_logger.get_lineage(to_idx)

        return CandidateDiff(
            from_idx=from_idx,
            to_idx=to_idx,
            prompt_changes=prompt_changes,
            evaluation_changes=evaluation_changes,
            lineage=lineage,
        )

    def get_lm_calls_for_iteration(self, iteration: int) -> list[LMCall]:
        """Get all LM calls for a specific iteration.

        Args:
            iteration: The iteration number

        Returns:
            List of LMCall records
        """
        if self.lm_logger is None:
            return []
        return self.lm_logger.get_calls_for_iteration(iteration)

    def get_lm_calls_for_phase(self, phase: str) -> list[LMCall]:
        """Get all LM calls for a specific phase.

        Args:
            phase: The phase ('reflection', 'proposal', 'eval', etc.)

        Returns:
            List of LMCall records
        """
        if self.lm_logger is None:
            return []
        return self.lm_logger.get_calls_for_phase(phase)

    def get_evaluations_for_example(self, example_id: str) -> list[EvaluationRecord]:
        """Get all evaluations for a specific example.

        Args:
            example_id: The example ID

        Returns:
            List of EvaluationRecord
        """
        if self.metric_logger is None:
            return []
        return self.metric_logger.get_evaluations_for_example(example_id)

    def get_evaluations_for_candidate(
        self, candidate_idx: int
    ) -> list[EvaluationRecord]:
        """Get all evaluations for a specific candidate.

        Args:
            candidate_idx: The candidate index

        Returns:
            List of EvaluationRecord
        """
        if self.metric_logger is None:
            return []
        return self.metric_logger.get_evaluations_for_candidate(candidate_idx)

    def compute_lift(
        self,
        baseline_candidate_idx: int,
        candidate_idx: int,
    ) -> dict[str, Any]:
        """Compute score lift from baseline to candidate.

        Compares scores across all examples evaluated by both candidates.

        Args:
            baseline_candidate_idx: Baseline candidate index
            candidate_idx: Candidate to compare

        Returns:
            Dict with lift statistics: mean_lift, total_examples, etc.
        """
        if self.metric_logger is None:
            return {"error": "No metric logger available"}

        # Get evaluations for both candidates
        baseline_evals = self.metric_logger.get_evaluations_for_candidate(
            baseline_candidate_idx
        )
        candidate_evals = self.metric_logger.get_evaluations_for_candidate(candidate_idx)

        # Build lookup by example_id
        baseline_by_example = {e.example_id: e for e in baseline_evals}
        candidate_by_example = {e.example_id: e for e in candidate_evals}

        # Find common examples
        common_examples = set(baseline_by_example.keys()) & set(
            candidate_by_example.keys()
        )

        if not common_examples:
            return {
                "mean_lift": 0.0,
                "total_examples": 0,
                "improved": 0,
                "regressed": 0,
                "unchanged": 0,
            }

        # Compute lift per example
        lifts = []
        improved = 0
        regressed = 0
        unchanged = 0

        for ex_id in common_examples:
            baseline_score = baseline_by_example[ex_id].score
            candidate_score = candidate_by_example[ex_id].score
            lift = candidate_score - baseline_score
            lifts.append(lift)

            if lift > 0:
                improved += 1
            elif lift < 0:
                regressed += 1
            else:
                unchanged += 1

        return {
            "mean_lift": sum(lifts) / len(lifts),
            "total_examples": len(common_examples),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        }

    def get_regressions(
        self,
        baseline_candidate_idx: int,
        candidate_idx: int,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Find examples where candidate regressed vs baseline.

        Args:
            baseline_candidate_idx: Baseline candidate index
            candidate_idx: Candidate to compare
            threshold: Minimum score drop to count as regression

        Returns:
            List of regression details
        """
        if self.metric_logger is None:
            return []

        # Get evaluations for both candidates
        baseline_evals = self.metric_logger.get_evaluations_for_candidate(
            baseline_candidate_idx
        )
        candidate_evals = self.metric_logger.get_evaluations_for_candidate(candidate_idx)

        # Build lookup by example_id
        baseline_by_example = {e.example_id: e for e in baseline_evals}
        candidate_by_example = {e.example_id: e for e in candidate_evals}

        # Find regressions
        regressions = []
        for ex_id in baseline_by_example:
            if ex_id not in candidate_by_example:
                continue

            baseline_score = baseline_by_example[ex_id].score
            candidate_score = candidate_by_example[ex_id].score
            delta = candidate_score - baseline_score

            if delta < -threshold:  # Score dropped more than threshold
                regressions.append(
                    {
                        "example_id": ex_id,
                        "baseline_score": baseline_score,
                        "baseline_feedback": baseline_by_example[ex_id].feedback,
                        "candidate_score": candidate_score,
                        "candidate_feedback": candidate_by_example[ex_id].feedback,
                        "delta": delta,
                    }
                )

        # Sort by delta (worst first)
        return sorted(regressions, key=lambda x: x["delta"])

    def get_pareto_evolution(self) -> list[dict[str, tuple[float, set[int]]]]:
        """Get Pareto frontier state at each iteration.

        Returns:
            List of dicts: data_id -> (score, set of program indices)
        """
        return self.state_logger.get_pareto_evolution()

    def get_lineage(self, candidate_idx: int) -> list[int]:
        """Trace a candidate back to its ancestors.

        Args:
            candidate_idx: The candidate to trace

        Returns:
            List of candidate indices from candidate to seed
        """
        return self.state_logger.get_lineage(candidate_idx)

    def get_all_candidates(self) -> list[dict[str, str]]:
        """Get all candidate prompts.

        Returns:
            List of candidate prompt dicts
        """
        return self.state_logger.get_all_candidates()

    # ==================== Properties ====================

    @property
    def seed_candidate(self) -> dict[str, str] | None:
        """Get the seed candidate prompt."""
        return self.state_logger.seed_candidate

    @property
    def seed_candidate_idx(self) -> int | None:
        """Get the seed candidate index."""
        return self.state_logger.seed_candidate_idx

    @property
    def final_candidates(self) -> list[dict[str, str]]:
        """Get final candidate prompts."""
        return self.state_logger.final_candidates

    @property
    def final_pareto(self) -> dict[str, float]:
        """Get final Pareto frontier scores."""
        return self.state_logger.final_pareto

    def get_best_candidate_idx(self) -> int:
        """Determine which candidate GEPA selected as best.

        Uses Pareto frontier data to identify the most frequently appearing
        candidate across all data points. Falls back to seed if no data.

        Note: This is an approximation - GEPA selects based on aggregate score,
        not Pareto frequency. For accurate comparison, use get_evaluation_comparison()
        which compares seed vs Pareto-optimal evaluation per example.

        Returns:
            The index of the best/selected candidate
        """
        pareto_programs = self.state_logger.final_pareto_programs
        if not pareto_programs:
            return self.seed_candidate_idx or 0

        # Count how often each candidate appears on the Pareto frontier
        candidate_counts: dict[int, int] = {}
        for prog_set in pareto_programs.values():
            if prog_set:
                # Use min() for deterministic selection when multiple candidates tie
                best = min(prog_set)
                candidate_counts[best] = candidate_counts.get(best, 0) + 1

        if not candidate_counts:
            return self.seed_candidate_idx or 0

        # Return most frequent candidate (or lowest index in case of tie)
        best_candidate = max(
            candidate_counts.keys(),
            key=lambda x: (candidate_counts[x], -x)  # Higher count, then lower index
        )
        return best_candidate

    @property
    def iterations(self) -> list[IterationDelta]:
        """Get all iteration deltas."""
        return self.state_logger.deltas

    @property
    def metadata(self) -> list[IterationMetadata]:
        """Get all iteration metadata."""
        return self.state_logger.metadata

    @property
    def lm_calls(self) -> list[LMCall]:
        """Get all LM calls."""
        if self.lm_logger is None:
            return []
        return self.lm_logger.calls

    @property
    def evaluations(self) -> list[EvaluationRecord]:
        """Get all evaluation records."""
        if self.metric_logger is None:
            return []
        return self.metric_logger.evaluations

    def get_evaluation_comparison(
        self,
        baseline_idx: int | None = None,
        optimized_idx: int | None = None,
    ) -> dict[str, Any]:
        """Get evaluation comparison between baseline and optimized candidates.

        Groups examples into improvements, regressions, and same categories.
        Each entry includes example inputs, baseline/optimized outputs, and scores.

        IMPORTANT: This compares seed (baseline) vs the SELECTED candidate's
        evaluations, not just the last evaluated candidate. Uses Pareto frontier
        data to determine which candidate GEPA actually selected as best.

        Since GEPA doesn't expose candidate_idx through public hooks, we use
        timestamp ordering to map evaluations to candidates:
        - Evaluations are sorted by timestamp
        - Evaluation[0] = candidate 0 (seed)
        - Evaluation[N] = candidate N
        - We compare eval[0] (seed) vs eval[selected_idx] (selected candidate)

        Args:
            baseline_idx: Baseline candidate index (default: seed)
            optimized_idx: Optimized candidate index (default: SELECTED best candidate)

        Returns:
            Dict with:
            - improvements: List of examples that improved
            - regressions: List of examples that regressed
            - same: List of examples with no change
            - summary: Overall stats (counts, avg scores)
        """
        if self.metric_logger is None:
            return {
                "improvements": [],
                "regressions": [],
                "same": [],
                "summary": {"error": "No evaluation data available"},
            }

        if baseline_idx is None:
            baseline_idx = self.seed_candidate_idx or 0
        if optimized_idx is None:
            # Use SELECTED best candidate, not just last by index
            optimized_idx = self.get_best_candidate_idx()

        # Try candidate_idx-based comparison first
        baseline_evals = self.metric_logger.get_evaluations_for_candidate(baseline_idx)
        optimized_evals = self.metric_logger.get_evaluations_for_candidate(optimized_idx)

        # Fallback to timestamp-based comparison if candidate_idx not available
        # (GEPA doesn't expose candidate_idx through public hooks)
        if not baseline_evals or not optimized_evals:
            all_evals = self.metric_logger.evaluations
            if not all_evals:
                return {
                    "improvements": [],
                    "regressions": [],
                    "same": [],
                    "summary": {"error": "No evaluation data available"},
                }

            # Filter to valset examples only if valset was set
            if self._valset_example_ids:
                all_evals = [
                    e for e in all_evals if e.example_id in self._valset_example_ids
                ]

            # GEPA evaluation flow:
            # - Candidates are evaluated in order: 0 (seed), 1, 2, 3, etc.
            # - Each candidate gets evaluated on the valset
            # - Evaluations sorted by timestamp correspond to candidate indices
            #
            # We need to compare seed (candidate 0) vs the SELECTED candidate.
            # The selected candidate is determined from Pareto frontier data,
            # NOT just the last candidate evaluated.

            # Group by example_id and sort by timestamp
            evals_by_example: dict[str, list] = {}
            for e in all_evals:
                if e.example_id not in evals_by_example:
                    evals_by_example[e.example_id] = []
                evals_by_example[e.example_id].append(e)

            baseline_evals = []
            optimized_evals = []

            for example_id, evals in evals_by_example.items():
                # Sort by timestamp to get chronological order (candidate 0, 1, 2, ...)
                sorted_evals = sorted(evals, key=lambda x: x.timestamp)

                # Baseline = candidate 0 (seed) = first evaluation
                if len(sorted_evals) >= 1:
                    baseline_evals.append(sorted_evals[0])

                # Optimized = BEST evaluation for this example (Pareto-optimal)
                # This represents the actual improvement achieved, regardless of
                # which candidate GEPA ultimately selected.
                #
                # Why best instead of specific candidate:
                # - GEPA selects based on aggregate score, not per-example
                # - We can't reliably map evaluations to candidates (candidate_idx=None)
                # - The Pareto frontier IS the set of best per-example scores
                # - Comparing seed vs best shows the actual optimization gain
                if len(sorted_evals) >= 1:
                    best_eval = max(sorted_evals, key=lambda x: x.score)
                    optimized_evals.append(best_eval)

        # Build lookup by example_id
        baseline_by_example = {e.example_id: e for e in baseline_evals}
        optimized_by_example = {e.example_id: e for e in optimized_evals}

        improvements = []
        regressions = []
        same = []

        # Find common examples
        all_examples = set(baseline_by_example.keys()) | set(optimized_by_example.keys())

        for example_id in sorted(all_examples):
            baseline_eval = baseline_by_example.get(example_id)
            optimized_eval = optimized_by_example.get(example_id)

            if not baseline_eval or not optimized_eval:
                continue  # Skip examples not evaluated by both

            delta = optimized_eval.score - baseline_eval.score

            entry = {
                "example_id": example_id,
                "inputs": baseline_eval.example_inputs or optimized_eval.example_inputs,
                "baseline_score": baseline_eval.score,
                "baseline_output": baseline_eval.prediction_preview,
                "baseline_feedback": baseline_eval.feedback,
                "optimized_score": optimized_eval.score,
                "optimized_output": optimized_eval.prediction_preview,
                "optimized_feedback": optimized_eval.feedback,
                "delta": delta,
            }

            if delta > 0.01:  # Small threshold to avoid float comparison issues
                improvements.append(entry)
            elif delta < -0.01:
                regressions.append(entry)
            else:
                same.append(entry)

        # Sort by delta magnitude
        improvements.sort(key=lambda x: x["delta"], reverse=True)
        regressions.sort(key=lambda x: x["delta"])

        # Compute summary stats
        total = len(improvements) + len(regressions) + len(same)
        baseline_avg = (
            sum(e["baseline_score"] for e in improvements + regressions + same) / total
            if total > 0
            else 0
        )
        optimized_avg = (
            sum(e["optimized_score"] for e in improvements + regressions + same) / total
            if total > 0
            else 0
        )

        return {
            "improvements": improvements,
            "regressions": regressions,
            "same": same,
            "summary": {
                "total_examples": total,
                "num_improvements": len(improvements),
                "num_regressions": len(regressions),
                "num_same": len(same),
                "baseline_avg_score": baseline_avg,
                "optimized_avg_score": optimized_avg,
                "avg_lift": optimized_avg - baseline_avg,
                "baseline_idx": baseline_idx,
                "optimized_idx": optimized_idx,
            },
        }

    # ==================== Visualization ====================

    def print_prompt_diff(
        self,
        from_idx: int | None = None,
        to_idx: int | None = None,
        show_full: bool = False,
        max_width: int = 80,
    ) -> None:
        """Print a formatted diff between original and optimized prompts.

        Args:
            from_idx: Source candidate index (default: seed)
            to_idx: Target candidate index (default: best/last)
            show_full: Show full prompt text (default: truncated)
            max_width: Max width for each column
        """
        if from_idx is None:
            from_idx = self.seed_candidate_idx or 0
        if to_idx is None:
            to_idx = len(self.final_candidates) - 1 if self.final_candidates else 0

        if from_idx == to_idx:
            print("No changes - same candidate")
            return

        diff = self.get_candidate_diff(from_idx, to_idx)
        lineage_str = " → ".join(str(i) for i in reversed(diff.lineage))

        print("\n" + "=" * 70)
        print("PROMPT COMPARISON")
        print("=" * 70)
        print(f"From: Candidate {from_idx} (seed)" if from_idx == self.seed_candidate_idx else f"From: Candidate {from_idx}")
        print(f"To:   Candidate {to_idx} (optimized)")
        print(f"Lineage: {lineage_str}")
        print("=" * 70)

        if not diff.prompt_changes:
            print("\nNo prompt changes detected.")
            return

        for key, (old_val, new_val) in diff.prompt_changes.items():
            print(f"\n📝 {key}")
            print("-" * 70)

            # Format old value
            print("\n❌ ORIGINAL:")
            self._print_wrapped(old_val, max_width, show_full)

            # Format new value
            print("\n✅ OPTIMIZED:")
            self._print_wrapped(new_val, max_width, show_full)

        print("\n" + "=" * 70)

    def _print_wrapped(self, text: str, max_width: int, show_full: bool) -> None:
        """Print text with word wrapping."""
        if not text:
            print("   (empty)")
            return

        # Truncate if needed
        if not show_full and len(text) > 500:
            text = text[:500] + "...\n   [truncated - use show_full=True to see all]"

        # Word wrap
        import textwrap
        wrapped = textwrap.fill(text, width=max_width, initial_indent="   ", subsequent_indent="   ")
        print(wrapped)

    def print_summary(self) -> None:
        """Print a formatted summary of the optimization run."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("GEPA OPTIMIZATION SUMMARY")
        print("=" * 60)

        # State summary
        state = summary.get("state", {})
        print(f"\n📊 State:")
        print(f"   Iterations:     {state.get('total_iterations', 0)}")
        print(f"   Candidates:     {state.get('total_candidates', 0)}")
        print(f"   Evaluations:    {state.get('total_evaluations', 0)}")
        duration = state.get('duration_seconds')
        if duration:
            print(f"   Duration:       {duration:.1f}s")

        # LM calls
        lm = summary.get("lm_calls", {})
        if lm:
            print(f"\n🤖 LM Calls:")
            print(f"   Total:          {lm.get('total_calls', 0)}")
            print(f"   Duration:       {lm.get('total_duration_ms', 0):.0f}ms")
            phases = lm.get('calls_by_phase', {})
            if phases:
                phase_str = ", ".join(f"{k}: {v}" for k, v in phases.items())
                print(f"   By Phase:       {phase_str}")

        # Evaluations
        evals = summary.get("evaluations", {})
        if evals:
            print(f"\n📋 Evaluations:")
            print(f"   Total:          {evals.get('total_evaluations', 0)}")
            print(f"   Unique Examples: {evals.get('unique_examples', 0)}")

        # Candidates
        if self.final_candidates:
            print(f"\n🎯 Candidates:")
            print(f"   Seed (idx {self.seed_candidate_idx}):")
            if self.seed_candidate:
                for key, val in self.seed_candidate.items():
                    truncated = val[:60] + "..." if len(val) > 60 else val
                    print(f"      {key}: {truncated}")

            best_idx = len(self.final_candidates) - 1
            if best_idx != self.seed_candidate_idx:
                print(f"   Best (idx {best_idx}):")
                for key, val in self.final_candidates[best_idx].items():
                    truncated = val[:60] + "..." if len(val) > 60 else val
                    print(f"      {key}: {truncated}")

        print("\n" + "=" * 60)

    def get_optimization_report(self) -> dict[str, Any]:
        """Get a structured optimization report.

        Returns:
            Dict with:
            - summary: Overall statistics
            - seed_prompt: Original prompt(s)
            - optimized_prompt: Best prompt(s)
            - prompt_changes: What changed
            - lineage: Evolution path
        """
        seed_idx = self.seed_candidate_idx or 0
        best_idx = len(self.final_candidates) - 1 if self.final_candidates else 0

        diff = self.get_candidate_diff(seed_idx, best_idx)

        return {
            "summary": self.get_summary(),
            "seed_prompt": self.seed_candidate,
            "seed_idx": seed_idx,
            "optimized_prompt": self.final_candidates[best_idx] if self.final_candidates else None,
            "optimized_idx": best_idx,
            "prompt_changes": diff.prompt_changes,
            "lineage": diff.lineage,
            "total_candidates": len(self.final_candidates),
        }

    def export_html(
        self,
        output_path: str | None = None,
        title: str = "GEPA Optimization Report",
    ) -> str:
        """Generate an HTML report of the optimization run.

        Works in both CLI and Jupyter notebooks:
        - If output_path is provided, writes to file and returns the path
        - If output_path is None, returns HTML string (for notebook display)

        For Jupyter notebooks, use:
            from IPython.display import HTML, display
            display(HTML(tracker.export_html()))

        Args:
            output_path: Optional path to write HTML file
            title: Report title

        Returns:
            HTML string if output_path is None, otherwise the output path
        """
        import html
        from datetime import datetime

        report = self.get_optimization_report()
        summary = report["summary"]
        state = summary.get("state", {})
        lm_summary = summary.get("lm_calls", {})
        eval_summary = summary.get("evaluations", {})

        # Build candidate cards
        candidates_html = ""
        for idx, candidate in enumerate(self.final_candidates):
            is_seed = idx == report["seed_idx"]
            is_best = idx == report["optimized_idx"]
            badge = ""
            if is_seed:
                badge = '<span class="badge badge-seed">SEED</span>'
            if is_best:
                badge += '<span class="badge badge-best">BEST</span>'

            prompts_html = ""
            for key, val in candidate.items():
                escaped_val = html.escape(val).replace("\n", "<br>")
                prompts_html += f"""
                <div class="prompt-section">
                    <div class="prompt-key">{html.escape(key)}</div>
                    <div class="prompt-value">{escaped_val}</div>
                </div>
                """

            candidates_html += f"""
            <div class="candidate-card" id="candidate-{idx}">
                <div class="candidate-header">
                    <span class="candidate-idx">Candidate {idx}</span>
                    {badge}
                </div>
                <div class="candidate-body">
                    {prompts_html}
                </div>
            </div>
            """

        # Build lineage visualization
        lineage = report.get("lineage", [])
        lineage_html = " → ".join(
            f'<a href="#candidate-{i}" class="lineage-link">{i}</a>'
            for i in reversed(lineage)
        )

        # Build prompt diff
        diff_html = ""
        for key, (old_val, new_val) in report.get("prompt_changes", {}).items():
            escaped_old = html.escape(old_val).replace("\n", "<br>")
            escaped_new = html.escape(new_val).replace("\n", "<br>")
            diff_html += f"""
            <div class="diff-section">
                <h4>{html.escape(key)}</h4>
                <div class="diff-container">
                    <div class="diff-panel diff-old">
                        <div class="diff-label">❌ Original (Candidate {report['seed_idx']})</div>
                        <div class="diff-content">{escaped_old}</div>
                    </div>
                    <div class="diff-panel diff-new">
                        <div class="diff-label">✅ Optimized (Candidate {report['optimized_idx']})</div>
                        <div class="diff-content">{escaped_new}</div>
                    </div>
                </div>
            </div>
            """

        # Phase breakdown
        phases = lm_summary.get("calls_by_phase", {})
        phase_rows = "".join(
            f"<tr><td>{html.escape(str(phase))}</td><td>{count}</td></tr>"
            for phase, count in phases.items()
        )

        # Get evaluation comparison
        eval_comparison = self.get_evaluation_comparison()
        eval_comp_summary = eval_comparison.get("summary", {})

        def build_eval_table(entries: list, category: str) -> str:
            if not entries:
                return f'<p class="empty-message">No {category} examples</p>'

            # Get input field names from first entry
            input_fields = list(entries[0].get("inputs", {}).keys()) if entries else []

            header_cells = "".join(f"<th>{html.escape(f)}</th>" for f in input_fields)
            rows = ""
            for i, entry in enumerate(entries):
                inputs = entry.get("inputs", {})
                input_cells = "".join(
                    f'<td class="input-cell">{html.escape(str(inputs.get(f, "")))[:100]}</td>'
                    for f in input_fields
                )

                delta_class = "delta-positive" if entry["delta"] > 0 else "delta-negative" if entry["delta"] < 0 else ""
                delta_str = f'+{entry["delta"]:.2f}' if entry["delta"] > 0 else f'{entry["delta"]:.2f}'

                rows += f"""
                <tr>
                    <td class="row-num">{i + 1}</td>
                    {input_cells}
                    <td class="score-cell">{entry["baseline_score"]:.2f}</td>
                    <td class="output-cell">{html.escape(str(entry.get("baseline_output", ""))[:150])}</td>
                    <td class="score-cell">{entry["optimized_score"]:.2f}</td>
                    <td class="output-cell">{html.escape(str(entry.get("optimized_output", ""))[:150])}</td>
                    <td class="delta-cell {delta_class}">{delta_str}</td>
                </tr>
                """

            return f"""
            <table class="eval-table">
                <thead>
                    <tr>
                        <th>#</th>
                        {header_cells}
                        <th>Base Score</th>
                        <th>Baseline Output</th>
                        <th>Opt Score</th>
                        <th>Optimized Output</th>
                        <th>Delta</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            """

        improvements_table = build_eval_table(eval_comparison.get("improvements", []), "improvement")
        regressions_table = build_eval_table(eval_comparison.get("regressions", []), "regression")
        same_table = build_eval_table(eval_comparison.get("same", []), "unchanged")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
            --accent-orange: #d29922;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }}

        h2 {{
            font-size: 1.5rem;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        h3 {{
            font-size: 1.25rem;
            margin: 1.5rem 0 1rem;
            color: var(--text-primary);
        }}

        h4 {{
            font-size: 1rem;
            margin: 1rem 0 0.5rem;
            color: var(--accent-blue);
        }}

        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-bottom: 2rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}

        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--accent-blue);
        }}

        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        .lineage-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 1.25rem;
            text-align: center;
        }}

        .lineage-link {{
            color: var(--accent-purple);
            text-decoration: none;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: background 0.2s;
        }}

        .lineage-link:hover {{
            background: var(--bg-tertiary);
        }}

        .diff-section {{
            margin: 1.5rem 0;
        }}

        .diff-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}

        .diff-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}

        .diff-label {{
            padding: 0.75rem 1rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
        }}

        .diff-old .diff-label {{
            background: rgba(248, 81, 73, 0.1);
            color: var(--accent-red);
        }}

        .diff-new .diff-label {{
            background: rgba(63, 185, 80, 0.1);
            color: var(--accent-green);
        }}

        .diff-content {{
            padding: 1rem;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
        }}

        .candidate-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
        }}

        .candidate-header {{
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .candidate-idx {{
            font-weight: 600;
        }}

        .badge {{
            font-size: 0.75rem;
            padding: 0.125rem 0.5rem;
            border-radius: 12px;
            font-weight: 600;
        }}

        .badge-seed {{
            background: var(--accent-orange);
            color: var(--bg-primary);
        }}

        .badge-best {{
            background: var(--accent-green);
            color: var(--bg-primary);
        }}

        .candidate-body {{
            padding: 1rem;
        }}

        .prompt-section {{
            margin: 0.5rem 0;
        }}

        .prompt-key {{
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 0.25rem;
        }}

        .prompt-value {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.875rem;
            background: var(--bg-tertiary);
            padding: 0.75rem;
            border-radius: 6px;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}

        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--bg-tertiary);
            font-weight: 600;
        }}

        tr:hover {{
            background: var(--bg-secondary);
        }}

        .section {{
            margin: 2rem 0;
        }}

        /* Tabs */
        .tabs {{
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }}

        .tab {{
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}

        .tab:hover {{
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }}

        .tab.active {{
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
        }}

        .tab-count {{
            margin-left: 0.5rem;
            padding: 0.125rem 0.5rem;
            border-radius: 10px;
            font-size: 0.75rem;
        }}

        .tab-count.green {{
            background: rgba(63, 185, 80, 0.2);
            color: var(--accent-green);
        }}

        .tab-count.red {{
            background: rgba(248, 81, 73, 0.2);
            color: var(--accent-red);
        }}

        .tab-count.gray {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* Evaluation table */
        .eval-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}

        .eval-table th {{
            background: var(--bg-tertiary);
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
            white-space: nowrap;
        }}

        .eval-table td {{
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            vertical-align: top;
        }}

        .eval-table tr:hover {{
            background: var(--bg-secondary);
        }}

        .row-num {{
            color: var(--text-secondary);
            font-weight: 500;
            width: 40px;
        }}

        .input-cell {{
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .output-cell {{
            max-width: 250px;
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .score-cell {{
            font-weight: 600;
            width: 80px;
            text-align: center;
        }}

        .delta-cell {{
            font-weight: 600;
            width: 70px;
            text-align: center;
        }}

        .delta-positive {{
            color: var(--accent-green);
        }}

        .delta-negative {{
            color: var(--accent-red);
        }}

        .empty-message {{
            color: var(--text-secondary);
            padding: 2rem;
            text-align: center;
        }}

        .score-summary {{
            display: flex;
            gap: 2rem;
            margin-bottom: 1rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
        }}

        .score-summary-item {{
            text-align: center;
        }}

        .score-summary-value {{
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .score-summary-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        @media (max-width: 768px) {{
            .diff-container {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 {html.escape(title)}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

        <div class="section">
            <h2>📊 Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{state.get('total_iterations', 0)}</div>
                    <div class="stat-label">Iterations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{state.get('total_candidates', 0)}</div>
                    <div class="stat-label">Candidates</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{lm_summary.get('total_calls', 0)}</div>
                    <div class="stat-label">LM Calls</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{eval_summary.get('total_evaluations', 0)}</div>
                    <div class="stat-label">Evaluations</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🧬 Lineage</h2>
            <div class="lineage-box">
                {lineage_html if lineage_html else "No lineage data"}
            </div>
        </div>

        <div class="section">
            <h2>🔄 Prompt Changes</h2>
            {diff_html if diff_html else "<p>No prompt changes detected.</p>"}
        </div>

        <div class="section">
            <h2>📈 Performance Comparison</h2>
            <div class="score-summary">
                <div class="score-summary-item">
                    <div class="score-summary-value">{eval_comp_summary.get('baseline_avg_score', 0):.2f}</div>
                    <div class="score-summary-label">Baseline Avg</div>
                </div>
                <div class="score-summary-item">
                    <div class="score-summary-value" style="color: var(--accent-green);">{eval_comp_summary.get('optimized_avg_score', 0):.2f}</div>
                    <div class="score-summary-label">Optimized Avg</div>
                </div>
                <div class="score-summary-item">
                    <div class="score-summary-value" style="color: {'var(--accent-green)' if eval_comp_summary.get('avg_lift', 0) >= 0 else 'var(--accent-red)'};">{'+' if eval_comp_summary.get('avg_lift', 0) >= 0 else ''}{eval_comp_summary.get('avg_lift', 0):.2f}</div>
                    <div class="score-summary-label">Avg Lift</div>
                </div>
                <div class="score-summary-item">
                    <div class="score-summary-value">{eval_comp_summary.get('total_examples', 0)}</div>
                    <div class="score-summary-label">Examples</div>
                </div>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="showTab('improvements')">
                    Improvements <span class="tab-count green">{eval_comp_summary.get('num_improvements', 0)}</span>
                </button>
                <button class="tab" onclick="showTab('regressions')">
                    Regressions <span class="tab-count red">{eval_comp_summary.get('num_regressions', 0)}</span>
                </button>
                <button class="tab" onclick="showTab('same')">
                    Same <span class="tab-count gray">{eval_comp_summary.get('num_same', 0)}</span>
                </button>
            </div>

            <div id="improvements" class="tab-content active">
                {improvements_table}
            </div>
            <div id="regressions" class="tab-content">
                {regressions_table}
            </div>
            <div id="same" class="tab-content">
                {same_table}
            </div>
        </div>

        {"<div class='section'><h2>📞 LM Calls by Phase</h2><table><thead><tr><th>Phase</th><th>Count</th></tr></thead><tbody>" + phase_rows + "</tbody></table></div>" if phase_rows else ""}

        <div class="section">
            <h2>📝 All Candidates</h2>
            {candidates_html}
        </div>
    </div>

    <script>
        function showTab(tabId) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            // Deactivate all tabs
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            // Show selected tab content
            document.getElementById(tabId).classList.add('active');
            // Activate clicked tab
            event.target.closest('.tab').classList.add('active');
        }}
    </script>
</body>
</html>"""

        if output_path:
            with open(output_path, "w") as f:
                f.write(html_content)
            return output_path

        return html_content

    # ==================== Lifecycle ====================

    def clear(self) -> None:
        """Clear all captured data."""
        self._start_time = None
        self.state_logger.clear()
        if self.lm_logger is not None:
            self.lm_logger.clear()
        if self.metric_logger is not None:
            self.metric_logger.clear()
        if self.proposer_logger is not None:
            self.proposer_logger.clear()
        if self.selector_logger is not None:
            self.selector_logger.clear()
        clear_ctx()

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        total_iters = summary.get("state", {}).get("total_iterations", 0)
        total_candidates = summary.get("state", {}).get("total_candidates", 0)
        total_lm = summary.get("lm_calls", {}).get("total_calls", 0)
        total_evals = summary.get("evaluations", {}).get("total_evaluations", 0)

        return (
            f"GEPATracker(iterations={total_iters}, candidates={total_candidates}, "
            f"lm_calls={total_lm}, evaluations={total_evals})"
        )
