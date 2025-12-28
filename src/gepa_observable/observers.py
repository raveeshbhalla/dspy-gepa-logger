# Copyright (c) 2025 - GEPA Observable Fork
# Observer protocol and event dataclasses for GEPA optimization observability

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SeedValidationEvent:
    """Fired after the seed candidate is evaluated on the validation set."""

    seed_candidate: dict[str, str]
    valset_scores: dict[Any, float]
    valset_outputs: dict[Any, Any]
    total_evals: int
    valset_feedbacks: dict[Any, str | None] | None = None  # Per-example feedback from metric


@dataclass
class IterationStartEvent:
    """Fired at the start of each optimization iteration."""

    iteration: int
    selected_candidate_idx: int
    selected_candidate: dict[str, str]
    parent_score: float


@dataclass
class MiniBatchEvalEvent:
    """Fired after evaluating a candidate on a mini-batch.

    This event fires twice per iteration:
    1. When evaluating the parent candidate (is_new_candidate=False)
    2. When evaluating the proposed new candidate (is_new_candidate=True)
    """

    iteration: int
    candidate_idx: int
    candidate: dict[str, str]
    batch_ids: list[Any]
    scores: list[float]
    outputs: list[Any]
    trajectories: list[Any] | None
    is_new_candidate: bool  # False=parent eval, True=new candidate eval
    feedbacks: list[str | None] | None = None  # Per-example feedback from metric


@dataclass
class ReflectionEvent:
    """Fired after reflection proposes new text for components.

    Contains both the input (reflective_dataset) and output (proposed_texts)
    of the reflection process.
    """

    iteration: int
    parent_candidate_idx: int
    components_to_update: list[str]
    reflective_dataset: dict[str, list[dict[str, Any]]]  # input to reflection
    proposed_texts: dict[str, str]  # output from reflection


@dataclass
class AcceptanceDecisionEvent:
    """Fired after the acceptance decision is made for a proposed candidate."""

    iteration: int
    parent_score_sum: float
    new_score_sum: float
    accepted: bool
    proceed_to_valset: bool


@dataclass
class ValsetEvalEvent:
    """Fired after evaluating a candidate on the full validation set."""

    iteration: int
    candidate_idx: int
    candidate: dict[str, str]
    val_ids: list[Any]
    scores: dict[Any, float]
    outputs: dict[Any, Any]
    is_new_best: bool
    valset_score: float
    feedbacks: dict[Any, str | None] | None = None  # Per-example feedback from metric


@dataclass
class MergeEvent:
    """Fired when a merge operation is attempted."""

    iteration: int
    parent_candidate_ids: list[int]
    merged_candidate: dict[str, str]
    subsample_scores_before: list[float] | None
    subsample_scores_after: list[float] | None
    accepted: bool


@dataclass
class OptimizationCompleteEvent:
    """Fired when optimization completes."""

    total_iterations: int
    total_evals: int
    best_candidate_idx: int
    best_score: float
    best_candidate: dict[str, str]  # The actual best candidate prompt text


@runtime_checkable
class GEPAObserver(Protocol):
    """Protocol for observing GEPA optimization events.

    Implement any subset of these methods to receive callbacks.
    All methods are optional - unimplemented methods will be skipped.
    """

    def on_seed_validation(self, event: SeedValidationEvent) -> None:
        """Called after seed candidate validation."""
        ...

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        """Called at the start of each iteration."""
        ...

    def on_minibatch_eval(self, event: MiniBatchEvalEvent) -> None:
        """Called after mini-batch evaluation."""
        ...

    def on_reflection(self, event: ReflectionEvent) -> None:
        """Called after reflection produces new texts."""
        ...

    def on_acceptance_decision(self, event: AcceptanceDecisionEvent) -> None:
        """Called after acceptance decision is made."""
        ...

    def on_valset_eval(self, event: ValsetEvalEvent) -> None:
        """Called after full validation set evaluation."""
        ...

    def on_merge(self, event: MergeEvent) -> None:
        """Called when a merge operation is attempted."""
        ...

    def on_optimization_complete(self, event: OptimizationCompleteEvent) -> None:
        """Called when optimization completes."""
        ...


@dataclass
class ObserverManager:
    """Manages a list of observers and dispatches events to them."""

    observers: list[GEPAObserver] = field(default_factory=list)
    mlflow_tracing: bool = False
    _mlflow_root_span: Any = None

    def _notify(self, method_name: str, event: Any) -> None:
        """Notify all observers that have the specified method."""
        for observer in self.observers:
            method = getattr(observer, method_name, None)
            if method is not None and callable(method):
                try:
                    method(event)
                except Exception as e:
                    # Log but don't fail optimization due to observer errors
                    import logging

                    logging.warning(f"Observer {observer} raised exception in {method_name}: {e}")

    def notify_seed_validation(self, event: SeedValidationEvent) -> None:
        self._notify("on_seed_validation", event)

    def notify_iteration_start(self, event: IterationStartEvent) -> None:
        self._notify("on_iteration_start", event)

    def notify_minibatch_eval(self, event: MiniBatchEvalEvent) -> None:
        self._notify("on_minibatch_eval", event)

    def notify_reflection(self, event: ReflectionEvent) -> None:
        self._notify("on_reflection", event)

    def notify_acceptance_decision(self, event: AcceptanceDecisionEvent) -> None:
        self._notify("on_acceptance_decision", event)

    def notify_valset_eval(self, event: ValsetEvalEvent) -> None:
        self._notify("on_valset_eval", event)

    def notify_merge(self, event: MergeEvent) -> None:
        self._notify("on_merge", event)

    def notify_optimization_complete(self, event: OptimizationCompleteEvent) -> None:
        self._notify("on_optimization_complete", event)
