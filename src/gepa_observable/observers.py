# Copyright (c) 2025 - GEPA Observable Fork
# Observer protocol and event dataclasses for GEPA optimization observability

from __future__ import annotations

import io
import json
import logging
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

# Import from local core and server modules
from gepa_observable.core.serialization import serialize_output, serialize_example_inputs
from gepa_observable.server.client import ServerClient
from gepa_observable.core.lm_logger import DSPyLMLogger, LMCall


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


# =============================================================================
# Built-in Observers
# =============================================================================


class LogCapture(io.TextIOBase):
    """Captures stdout and sends to server while preserving original output."""

    def __init__(self, original_stream: Any, server_client: ServerClient, stream_type: str = "stdout"):
        super().__init__()
        self._original = original_stream
        self._client = server_client
        self._stream_type = stream_type
        self._buffer: list[str] = []
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        if not s:
            return 0

        # Always write to original stream
        result = self._original.write(s)

        # Buffer and push to server
        with self._lock:
            self._buffer.append(s)
            # Flush on newlines - always clear buffer to prevent memory leaks
            if "\n" in s:
                content = "".join(self._buffer)
                self._buffer = []
                # Only push to server if connected
                if self._client and self._client.is_connected:
                    try:
                        self._client.push_logs([{
                            "logType": self._stream_type,
                            "content": content,
                            "timestamp": time.time(),
                        }])
                    except Exception:
                        pass  # Don't fail if server push fails

        return len(s) if result is None else result

    def flush(self) -> None:
        self._original.flush()
        with self._lock:
            if self._buffer:
                content = "".join(self._buffer)
                self._buffer = []
                # Only push to server if connected
                if self._client and self._client.is_connected:
                    try:
                        self._client.push_logs([{
                            "logType": self._stream_type,
                            "content": content,
                            "timestamp": time.time(),
                        }])
                    except Exception:
                        pass

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def encoding(self) -> str:
        return self._original.encoding

    def isatty(self) -> bool:
        return self._original.isatty()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


class LoggingObserver:
    """Built-in observer that logs optimization progress to console.

    This observer provides a clean console output during optimization,
    showing progress through iterations, scores, and key events.

    Example:
        >>> from gepa_observable.observers import LoggingObserver
        >>> observer = LoggingObserver(verbose=True)
        >>> result = optimize(..., observers=[observer])
        >>> print(observer.get_summary())
    """

    def __init__(self, verbose: bool = True, show_prompts: bool = False):
        """Initialize the logging observer.

        Args:
            verbose: If True, print detailed per-event logs. If False, only show summary.
            show_prompts: If True, show full prompt text in reflection events.
        """
        self.verbose = verbose
        self.show_prompts = show_prompts

        # State tracking for summary
        self.seed_scores: list[float] = []
        self.iterations: list[int] = []
        self.reflections: list[ReflectionEvent] = []
        self.accepted_candidates: list[int] = []

    def on_seed_validation(self, event: SeedValidationEvent) -> None:
        avg_score = (sum(event.valset_scores.values()) / len(event.valset_scores)) if event.valset_scores else 0.0
        if self.verbose:
            print(f"\n[Seed] Validated seed candidate: avg score = {avg_score:.2%}")
            print(f"       Total evals: {event.total_evals}")
        self.seed_scores = list(event.valset_scores.values())

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        if self.verbose:
            print(f"\n[Iter {event.iteration}] Starting iteration")
            print(f"         Selected candidate {event.selected_candidate_idx} (score: {event.parent_score:.2%})")
        self.iterations.append(event.iteration)

    def on_minibatch_eval(self, event: MiniBatchEvalEvent) -> None:
        if self.verbose:
            avg_score = (sum(event.scores) / len(event.scores)) if event.scores else 0.0
            candidate_type = "NEW" if event.is_new_candidate else "parent"
            print(f"         [{candidate_type}] Minibatch eval: avg = {avg_score:.2%} (n={len(event.scores)})")
            if event.trajectories:
                print(f"                   Trajectories captured: {len(event.trajectories)}")

    def on_reflection(self, event: ReflectionEvent) -> None:
        if self.verbose:
            print(f"         [Reflection] Components updated: {event.components_to_update}")
            if self.show_prompts:
                for name, text in event.proposed_texts.items():
                    preview = text[:80] + "..." if len(text) > 80 else text
                    print(f"                       {name}: {preview}")
        self.reflections.append(event)

    def on_acceptance_decision(self, event: AcceptanceDecisionEvent) -> None:
        if self.verbose:
            status = "ACCEPTED" if event.accepted else "REJECTED"
            print(f"         [Decision] {status}: parent={event.parent_score_sum:.2f} vs new={event.new_score_sum:.2f}")
            if event.proceed_to_valset:
                print("                    Proceeding to full valset evaluation")

    def on_valset_eval(self, event: ValsetEvalEvent) -> None:
        if self.verbose:
            is_best = " (NEW BEST!)" if event.is_new_best else ""
            print(f"         [Valset] Candidate {event.candidate_idx}: score = {event.valset_score:.2%}{is_best}")
        if event.is_new_best:
            self.accepted_candidates.append(event.candidate_idx)

    def on_merge(self, event: MergeEvent) -> None:
        if self.verbose:
            status = "ACCEPTED" if event.accepted else "REJECTED"
            print(f"         [Merge] {status}: merged candidates {event.parent_candidate_ids}")

    def on_optimization_complete(self, event: OptimizationCompleteEvent) -> None:
        print(f"\n{'='*60}")
        print("[Complete] Optimization finished!")
        print(f"           Total iterations: {event.total_iterations}")
        print(f"           Total evaluations: {event.total_evals}")
        print(f"           Best candidate: {event.best_candidate_idx} (score: {event.best_score:.2%})")
        print(f"{'='*60}")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the optimization run."""
        return {
            "total_iterations": len(self.iterations),
            "total_reflections": len(self.reflections),
            "accepted_candidates": self.accepted_candidates,
            "seed_avg_score": sum(self.seed_scores) / len(self.seed_scores) if self.seed_scores else 0,
        }


class ServerObserver:
    """Built-in observer that sends events to the GEPA web dashboard.

    This observer handles all the complexity of server communication:
    - Automatic run lifecycle management (start_run/complete_run)
    - LM call capture via DSPyLMLogger (auto-registered)
    - Example serialization for inputs/outputs
    - Stdout capture (optional)

    Example:
        >>> from gepa_observable.observers import ServerObserver
        >>> server = ServerObserver.create(
        ...     server_url="http://localhost:3000",
        ...     trainset=train_data,
        ...     valset=val_data,
        ... )
        >>> result = optimize(..., observers=[server])

    For full control, use the constructor directly:
        >>> server = ServerObserver(
        ...     server_url="http://localhost:3000",
        ...     project_name="My Project",
        ...     capture_lm_calls=True,
        ... )
        >>> server.set_examples(train_data, val_data)
        >>> # Register LM logger with your DSPy LM (guard for None)
        >>> logger = server.get_lm_logger()
        >>> lm = dspy.LM("...", callbacks=[logger] if logger else [])
    """

    def __init__(
        self,
        server_url: str,
        project_name: str = "GEPA Run",
        run_name: str | None = None,
        capture_lm_calls: bool = True,
        capture_stdout: bool = False,
    ):
        """Initialize the server observer.

        Args:
            server_url: URL of the GEPA dashboard server.
            project_name: Name of the project in the dashboard.
            run_name: Name for this run. Auto-generated if None.
            capture_lm_calls: If True, create an LM logger for capturing LM calls.
            capture_stdout: If True, capture stdout and send to server.
        """
        self.server_url = server_url
        self.project_name = project_name
        self.run_name = run_name
        self.capture_lm_calls = capture_lm_calls
        self.capture_stdout = capture_stdout

        # Initialize server client
        self.client = ServerClient(server_url, project_name=project_name)

        # Initialize LM logger if requested
        self._lm_logger: DSPyLMLogger | None = None
        if capture_lm_calls:
            self._lm_logger = DSPyLMLogger(on_call_complete=self._on_lm_call)

        # Stdout capture state
        self._original_stdout: Any = None
        self._log_capture: LogCapture | None = None

        # Run state
        self.run_id: str | None = None
        self.seed_candidate: dict[str, str] | None = None
        self.seed_score: float | None = None
        self.total_evals: int = 0
        self.num_candidates: int = 1  # Start with seed candidate
        self.candidates_pushed: set[int] = set()
        self.pareto_candidates: set[int] = set()

        # Per-iteration tracking
        self.current_iteration: int = -1
        self.iteration_evals: int = 0
        self.iteration_parent_idx: int | None = None
        self.iteration_child_idxs: list[int] = []
        self.iteration_reflection_input: str | None = None
        self.iteration_reflection_output: str | None = None
        self.iteration_proposed_changes: list[dict[str, str]] = []
        self.iteration_parent_score: float = 0.0
        self.iteration_new_score: float = 0.0
        self.iteration_accepted: bool = False

        # LM call tracking
        self.lm_call_count: int = 0

        # Example lookup maps
        self._trainset_map: dict[Any, Any] = {}
        self._valset_map: dict[Any, Any] = {}

    @classmethod
    def create(
        cls,
        server_url: str,
        trainset: list[Any],
        valset: list[Any],
        project_name: str = "GEPA Run",
        run_name: str | None = None,
        capture_lm_calls: bool = True,
        capture_stdout: bool = False,
    ) -> ServerObserver:
        """Factory method to create a fully-configured ServerObserver.

        This is the recommended way to create a ServerObserver when you have
        access to the training and validation sets.

        Args:
            server_url: URL of the GEPA dashboard server.
            trainset: Training dataset for example lookup.
            valset: Validation dataset for example lookup.
            project_name: Name of the project in the dashboard.
            run_name: Name for this run. Auto-generated if None.
            capture_lm_calls: If True, create an LM logger for capturing LM calls.
            capture_stdout: If True, capture stdout and send to server.

        Returns:
            A configured ServerObserver instance.
        """
        observer = cls(
            server_url=server_url,
            project_name=project_name,
            run_name=run_name,
            capture_lm_calls=capture_lm_calls,
            capture_stdout=capture_stdout,
        )
        observer.set_examples(trainset, valset)
        return observer

    def set_examples(self, trainset: list[Any], valset: list[Any]) -> None:
        """Store example references for input serialization.

        Call this before optimization starts to enable input capture.
        Examples are indexed by their position (0, 1, 2, ...) which matches
        how GEPA assigns batch_ids.
        """
        self._trainset_map = {i: ex for i, ex in enumerate(trainset)}
        self._valset_map = {i: ex for i, ex in enumerate(valset)}

    def get_lm_logger(self) -> DSPyLMLogger | None:
        """Get the LM logger for manual registration with DSPy.

        Returns:
            The DSPyLMLogger instance, or None if capture_lm_calls=False.

        Example:
            >>> server = ServerObserver(...)
            >>> logger = server.get_lm_logger()
            >>> lm = dspy.LM("openai/gpt-4o", callbacks=[logger] if logger else [])
        """
        return self._lm_logger

    def start_stdout_capture(self) -> None:
        """Start capturing stdout to send to the server."""
        if self.capture_stdout and self._original_stdout is None:
            self._original_stdout = sys.stdout
            self._log_capture = LogCapture(self._original_stdout, self.client)
            sys.stdout = self._log_capture  # type: ignore

    def stop_stdout_capture(self) -> None:
        """Stop capturing stdout and restore original stream."""
        if self._original_stdout is not None:
            if self._log_capture:
                self._log_capture.flush()
            sys.stdout = self._original_stdout
            self._original_stdout = None
            self._log_capture = None

    def _get_example_inputs(self, example_id: Any) -> dict[str, Any] | None:
        """Look up and serialize example inputs by ID."""
        example = self._valset_map.get(example_id) or self._trainset_map.get(example_id)
        if example is not None:
            return serialize_example_inputs(example)
        return None

    def _on_lm_call(self, call: LMCall) -> None:
        """Handle an LM call captured by DSPyLMLogger."""
        self.lm_call_count += 1

        if not self.client.is_connected:
            return

        # Determine iteration (use current iteration or None if before optimization starts)
        iteration = self.current_iteration + 1 if self.current_iteration >= 0 else None

        self.client.push_lm_calls([{
            "callId": call.call_id,
            "model": call.model,
            "startTime": call.start_time,
            "endTime": call.end_time,
            "durationMs": call.duration_ms,
            "iteration": iteration,
            "phase": call.phase,
            "candidateIdx": call.candidate_idx,
            "inputs": call.inputs,
            "outputs": call.outputs,
        }])

    def _start_run(self, seed_candidate: dict[str, str]) -> str | None:
        """Start a new run on the server."""
        self.seed_candidate = seed_candidate
        valset_ids = [str(i) for i in range(len(self._valset_map))] if self._valset_map else None

        run_name = self.run_name or f"GEPA Run {time.strftime('%Y-%m-%d %H:%M:%S')}"

        self.run_id = self.client.start_run(
            config={"observer_type": "gepa-observable"},
            seed_prompt=seed_candidate,
            name=run_name,
            valset_example_ids=valset_ids,
        )

        if self.run_id:
            # Push seed candidate (index 0)
            self.client.push_candidates([
                (0, seed_candidate, None, 0)
            ])
            self.candidates_pushed.add(0)
            logging.info(f"[ServerObserver] Started run: {self.run_id}")

        # Start stdout capture after connection established
        self.start_stdout_capture()

        return self.run_id

    def _push_current_iteration(self) -> None:
        """Push accumulated iteration data to the server."""
        if self.current_iteration < 0 or not self.client.is_connected:
            return

        # Use 1-indexed iteration numbers for the UI
        ui_iteration = self.current_iteration + 1

        # Build pareto_programs dict - server expects dict[str, int] not dict[str, list]
        # Use first pareto candidate index for each slot, or None if not available
        pareto_programs = None
        if self.pareto_candidates:
            # pareto_candidates is a set, use next(iter(...)) to get first element
            first_pareto = next(iter(self.pareto_candidates))
            pareto_programs = {str(i): first_pareto for i in range(5)}

        self.client.push_iteration(
            iteration_number=ui_iteration,
            timestamp=time.time(),
            total_evals=self.total_evals,
            num_candidates=self.num_candidates,
            pareto_size=len(self.pareto_candidates),
            pareto_programs=pareto_programs,
            reflection_input=self.iteration_reflection_input,
            reflection_output=self.iteration_reflection_output,
            proposed_changes=self.iteration_proposed_changes if self.iteration_proposed_changes else None,
            parent_candidate_idx=self.iteration_parent_idx,
            child_candidate_idxs=self.iteration_child_idxs if self.iteration_child_idxs else None,
        )

    def _reset_iteration_state(self) -> None:
        """Reset per-iteration tracking state."""
        self.iteration_evals = 0
        self.iteration_parent_idx = None
        self.iteration_child_idxs = []
        self.iteration_reflection_input = None
        self.iteration_reflection_output = None
        self.iteration_proposed_changes = []
        self.iteration_parent_score = 0.0
        self.iteration_new_score = 0.0
        self.iteration_accepted = False

    def on_seed_validation(self, event: SeedValidationEvent) -> None:
        # Start the run on first event
        if self.run_id is None:
            self._start_run(event.seed_candidate)

        if not self.client.is_connected:
            return

        self.seed_score = (sum(event.valset_scores.values()) / len(event.valset_scores)) if event.valset_scores else None
        self.total_evals = event.total_evals
        self.pareto_candidates.add(0)  # Seed starts on pareto front

        # Push seed evaluations
        evaluations = []
        for example_id, score in event.valset_scores.items():
            output = event.valset_outputs.get(example_id)
            feedback = event.valset_feedbacks.get(example_id) if event.valset_feedbacks else None
            evaluations.append({
                "evalId": str(uuid.uuid4()),
                "exampleId": str(example_id),
                "candidateIdx": 0,
                "iteration": None,  # NULL for seed/baseline evaluations
                "phase": "seed_validation",
                "score": score,
                "feedback": feedback,
                "exampleInputs": self._get_example_inputs(example_id),
                "predictionPreview": str(output) if output else None,
                "predictionRef": serialize_output(output),
                "timestamp": time.time(),
            })
        self.client.push_evaluations(evaluations)

        # Push iteration 0 (seed validation)
        self.client.push_iteration(
            iteration_number=0,
            timestamp=time.time(),
            total_evals=self.total_evals,
            num_candidates=1,
            pareto_size=1,
            pareto_programs={"0": [0]},
        )

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        if not self.client.is_connected:
            return

        # Push previous iteration's final state
        if self.current_iteration >= 0:
            self._push_current_iteration()

        # Start new iteration
        self._reset_iteration_state()
        self.current_iteration = event.iteration
        self.iteration_parent_idx = event.selected_candidate_idx
        self.iteration_parent_score = event.parent_score

    def on_minibatch_eval(self, event: MiniBatchEvalEvent) -> None:
        if not self.client.is_connected:
            return

        # Track evaluations
        num_evals = len(event.scores)
        self.iteration_evals += num_evals
        self.total_evals += num_evals

        # Track scores
        batch_sum = sum(event.scores)
        if event.is_new_candidate:
            self.iteration_new_score = batch_sum
        else:
            self.iteration_parent_score = batch_sum

        # Use 1-indexed iteration for UI
        ui_iteration = event.iteration + 1

        # Push evaluations
        evaluations = []
        for i, (batch_id, score) in enumerate(zip(event.batch_ids, event.scores)):
            output = event.outputs[i] if i < len(event.outputs) else None
            feedback = event.feedbacks[i] if event.feedbacks and i < len(event.feedbacks) else None
            evaluations.append({
                "evalId": str(uuid.uuid4()),
                "exampleId": str(batch_id),
                "candidateIdx": event.candidate_idx,
                "iteration": ui_iteration,
                "phase": "minibatch_new" if event.is_new_candidate else "minibatch_parent",
                "score": score,
                "feedback": feedback,
                "exampleInputs": self._get_example_inputs(batch_id),
                "predictionPreview": str(output) if output else None,
                "predictionRef": serialize_output(output),
                "timestamp": time.time(),
            })
        self.client.push_evaluations(evaluations)

    def on_reflection(self, event: ReflectionEvent) -> None:
        if not self.client.is_connected:
            return

        self.iteration_reflection_input = json.dumps(event.reflective_dataset, default=str)
        self.iteration_reflection_output = json.dumps(event.proposed_texts, default=str)
        self.iteration_proposed_changes = [
            {"component": name, "newText": text}
            for name, text in event.proposed_texts.items()
        ]

    def on_acceptance_decision(self, event: AcceptanceDecisionEvent) -> None:
        if not self.client.is_connected:
            return

        self.iteration_accepted = event.accepted
        ui_iteration = event.iteration + 1

        self.client.push_logs([{
            "logType": "info",
            "content": f"{'ACCEPTED' if event.accepted else 'REJECTED'}: parent={event.parent_score_sum:.2f} vs new={event.new_score_sum:.2f}",
            "timestamp": time.time(),
            "iteration": ui_iteration,
            "phase": "acceptance_decision",
        }])

    def on_valset_eval(self, event: ValsetEvalEvent) -> None:
        if not self.client.is_connected:
            return

        num_evals = len(event.scores)
        self.total_evals += num_evals

        if event.is_new_best:
            self.pareto_candidates.add(event.candidate_idx)

        # Push new candidate if not already pushed
        if event.candidate_idx not in self.candidates_pushed:
            parent_idx = self.iteration_parent_idx if self.iteration_parent_idx is not None else 0
            self.client.push_candidates([
                (event.candidate_idx, event.candidate, parent_idx, event.iteration + 1)
            ])
            self.candidates_pushed.add(event.candidate_idx)
            self.num_candidates += 1
            self.iteration_child_idxs.append(event.candidate_idx)

        ui_iteration = event.iteration + 1

        # Push valset evaluations
        evaluations = []
        for val_id, score in event.scores.items():
            output = event.outputs.get(val_id)
            feedback = event.feedbacks.get(val_id) if event.feedbacks else None
            evaluations.append({
                "evalId": str(uuid.uuid4()),
                "exampleId": str(val_id),
                "candidateIdx": event.candidate_idx,
                "iteration": ui_iteration,
                "phase": "valset",
                "score": score,
                "feedback": feedback,
                "exampleInputs": self._get_example_inputs(val_id),
                "predictionPreview": str(output) if output else None,
                "predictionRef": serialize_output(output),
                "timestamp": time.time(),
            })
        self.client.push_evaluations(evaluations)

    def on_merge(self, event: MergeEvent) -> None:
        # Optional: Log merge events
        pass

    def on_optimization_complete(self, event: OptimizationCompleteEvent) -> None:
        # Push final iteration state
        if self.current_iteration >= 0:
            self._push_current_iteration()

        # Stop stdout capture before completing
        self.stop_stdout_capture()

        if not self.client.is_connected:
            return

        # Complete the run
        self.client.complete_run(
            status="COMPLETED",
            best_prompt=event.best_candidate,
            best_candidate_idx=event.best_candidate_idx,
            best_score=event.best_score,
            seed_score=self.seed_score,
        )
        logging.info(f"[ServerObserver] Run completed: {self.run_id}")
