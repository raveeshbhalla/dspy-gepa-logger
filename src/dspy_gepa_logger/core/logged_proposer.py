"""Proposer wrapper for reflection/proposal phase tagging.

LoggedInstructionProposer wraps an instruction proposer to set context tags
for reflection and proposal phases. This enables proper phase attribution
for ALL LM calls, not just eval.

Without this wrapper, LM calls made during reflection and proposal would
not be tagged with the correct phase.
"""

from dataclasses import dataclass, field
from typing import Any

from .context import set_ctx, get_ctx


@dataclass
class ReflectionCall:
    """Record of a reflection call."""

    iteration: int
    candidate_idx: int
    current_instructions: dict[str, str]
    timestamp: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class ProposalCall:
    """Record of a proposal call."""

    iteration: int
    candidate_idx: int
    num_proposals: int
    proposals: list[dict[str, str]]
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class LoggedInstructionProposer:
    """Wraps an instruction proposer to set context for reflection/proposal LM calls.

    This enables proper phase attribution for ALL LM calls, not just eval.

    Usage:
        from dspy.teleprompt.gepa import DefaultInstructionProposer

        # Wrap the default proposer
        logged_proposer = LoggedInstructionProposer(DefaultInstructionProposer())

        gepa = GEPA(
            metric=logged_metric,
            instruction_proposer=logged_proposer,
            ...
        )

    How it works:
        1. Before reflection: sets phase="reflection" in context
        2. During reflection: LM calls will be tagged with phase="reflection"
        3. Before proposal: sets phase="proposal" in context
        4. During proposal: LM calls will be tagged with phase="proposal"
        5. After proposal: clears phase from context
    """

    def __init__(self, base_proposer: Any):
        """Initialize the logged proposer wrapper.

        Args:
            base_proposer: The instruction proposer to wrap
        """
        self.base_proposer = base_proposer
        self.reflection_calls: list[ReflectionCall] = []
        self.proposal_calls: list[ProposalCall] = []

    def propose(
        self,
        current_instructions: dict[str, str],
        candidate_idx: int,
        reflection_data: Any,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Propose new instructions with context tagging.

        Sets phase="reflection" during reflection, phase="proposal" during generation.

        Args:
            current_instructions: Current instruction set
            candidate_idx: Index of the candidate being mutated
            reflection_data: Data from reflection step
            **kwargs: Additional arguments for the base proposer

        Returns:
            List of proposed new instruction sets
        """
        ctx = get_ctx()
        iteration = ctx.get("iteration", 0)

        # Phase 1: Reflection - analyzing current performance
        set_ctx(phase="reflection", candidate_idx=candidate_idx)

        # Record reflection input
        self.reflection_calls.append(
            ReflectionCall(
                iteration=iteration,
                candidate_idx=candidate_idx,
                current_instructions=dict(current_instructions),
            )
        )

        # Phase 2: Proposal - generating new candidates
        set_ctx(phase="proposal", candidate_idx=candidate_idx)

        # Call base proposer
        proposals = self.base_proposer.propose(
            current_instructions=current_instructions,
            candidate_idx=candidate_idx,
            reflection_data=reflection_data,
            **kwargs,
        )

        # Ensure proposals is a list
        if not isinstance(proposals, list):
            proposals = [proposals] if proposals else []

        # Record proposals
        self.proposal_calls.append(
            ProposalCall(
                iteration=iteration,
                candidate_idx=candidate_idx,
                num_proposals=len(proposals),
                proposals=[dict(p) for p in proposals],
            )
        )

        # Reset phase
        set_ctx(phase=None)

        return proposals

    def __call__(
        self,
        current_instructions: dict[str, str],
        candidate_idx: int,
        reflection_data: Any,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Allow using the proposer as a callable.

        Some GEPA versions may call the proposer directly instead of .propose().
        """
        return self.propose(
            current_instructions=current_instructions,
            candidate_idx=candidate_idx,
            reflection_data=reflection_data,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base proposer.

        This allows the wrapper to be a drop-in replacement for the base proposer.
        """
        return getattr(self.base_proposer, name)

    def get_reflection_calls_for_iteration(
        self, iteration: int
    ) -> list[ReflectionCall]:
        """Get reflection calls for a specific iteration."""
        return [r for r in self.reflection_calls if r.iteration == iteration]

    def get_proposal_calls_for_iteration(self, iteration: int) -> list[ProposalCall]:
        """Get proposal calls for a specific iteration."""
        return [p for p in self.proposal_calls if p.iteration == iteration]

    def get_proposals_for_candidate(
        self, candidate_idx: int
    ) -> list[ProposalCall]:
        """Get all proposals generated for a specific candidate."""
        return [p for p in self.proposal_calls if p.candidate_idx == candidate_idx]

    def clear(self) -> None:
        """Clear all recorded calls."""
        self.reflection_calls = []
        self.proposal_calls = []


class LoggedSelector:
    """Optional: Wraps a selector to set candidate_idx context during selection.

    Use this if you need to track which candidate is being evaluated
    when GEPA selects candidates for the next iteration.
    """

    def __init__(self, base_selector: Any):
        """Initialize the logged selector wrapper.

        Args:
            base_selector: The selector to wrap
        """
        self.base_selector = base_selector
        self.selection_calls: list[dict[str, Any]] = []

    def select(
        self, candidates: list[Any], scores: list[float], **kwargs: Any
    ) -> list[int]:
        """Select candidates with context tracking.

        Args:
            candidates: List of candidates to select from
            scores: Scores for each candidate
            **kwargs: Additional arguments for the base selector

        Returns:
            List of selected candidate indices
        """
        # Set phase for any LM calls during selection
        set_ctx(phase="selection")

        selected = self.base_selector.select(candidates, scores, **kwargs)

        # Record selection
        self.selection_calls.append(
            {
                "num_candidates": len(candidates),
                "scores": list(scores),
                "selected": list(selected),
                "iteration": get_ctx().get("iteration"),
            }
        )

        set_ctx(phase=None)
        return selected

    def __call__(
        self, candidates: list[Any], scores: list[float], **kwargs: Any
    ) -> list[int]:
        """Allow using the selector as a callable."""
        return self.select(candidates, scores, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base selector."""
        return getattr(self.base_selector, name)

    def clear(self) -> None:
        """Clear all recorded calls."""
        self.selection_calls = []
