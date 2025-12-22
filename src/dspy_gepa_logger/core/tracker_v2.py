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

    def wrap_metric(
        self,
        metric: Callable[..., Any],
        capture_prediction: bool = True,
        max_prediction_preview: int = 200,
    ) -> LoggedMetric:
        """Wrap a metric function to capture evaluation details.

        The wrapped metric will:
        1. Set phase="eval" in context before calling the metric
        2. Capture score, feedback, and prediction for each call
        3. Restore the previous phase after the call

        Args:
            metric: The metric function to wrap
            capture_prediction: Whether to capture predictions (default: True)
            max_prediction_preview: Max length for prediction preview

        Returns:
            LoggedMetric wrapper that can be used in place of the original metric
        """
        self.metric_logger = LoggedMetric(
            metric_fn=metric,
            capture_prediction=capture_prediction,
            max_prediction_preview=max_prediction_preview,
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
