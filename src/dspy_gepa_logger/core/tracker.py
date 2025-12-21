"""Main GEPA run tracker class."""

import uuid
import time
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Iterator

from dspy_gepa_logger.core.config import TrackerConfig
from dspy_gepa_logger.models.run import GEPARunRecord, GEPARunConfig
from dspy_gepa_logger.models.iteration import IterationRecord
from dspy_gepa_logger.models.candidate import CandidateRecord, ParetoFrontierUpdate
from dspy_gepa_logger.models.reflection import ReflectionRecord
from dspy_gepa_logger.models.trace import LMCallRecord, TraceRecord
from dspy_gepa_logger.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Context variable to track if we're in a reflection context
_reflection_context: ContextVar[str | None] = ContextVar("reflection_context", default=None)


class GEPARunTracker:
    """Main tracker class that coordinates logging across all hooks.

    This class manages the lifecycle of tracking a GEPA optimization run,
    including:
    - Starting/stopping tracking sessions
    - Recording iterations, evaluations, and reflections
    - Managing LM call capture via callbacks
    - Persisting data to storage backends

    Usage:
        tracker = GEPARunTracker(storage=JSONLStorageBackend("./logs"))

        with tracker.track():
            optimized = gepa.compile(student=program, trainset=train, valset=val)

        # Access results
        run = tracker.get_run()
    """

    def __init__(
        self,
        storage: StorageBackend,
        config: TrackerConfig | None = None,
    ):
        """Initialize the tracker.

        Args:
            storage: Storage backend for persisting run data
            config: Tracker configuration options
        """
        self.storage = storage
        self.config = config or TrackerConfig()

        # Current run state
        self._current_run: GEPARunRecord | None = None
        self._current_iteration: IterationRecord | None = None
        self._iteration_counter = 0

        # LM call tracking
        self._active_lm_calls: dict[str, dict[str, Any]] = {}
        self._current_lm_calls: list[LMCallRecord] = []

        # Callback management
        self._callback: Any = None  # Will be GEPALoggingCallback
        self._original_callbacks: list[Any] = []

        # Event callbacks for custom processing
        self._on_iteration_complete: list[Callable[[IterationRecord], None]] = []
        self._on_candidate_accepted: list[Callable[[CandidateRecord, IterationRecord], None]] = []

    @property
    def is_tracking(self) -> bool:
        """Check if currently tracking a run."""
        return self._current_run is not None

    @property
    def current_run_id(self) -> str | None:
        """Get the current run ID if tracking."""
        return self._current_run.run_id if self._current_run else None

    def is_in_reflection_context(self) -> bool:
        """Check if currently in a reflection context."""
        return _reflection_context.get() is not None

    def get_reflection_component(self) -> str | None:
        """Get the current reflection component name."""
        return _reflection_context.get()

    @contextmanager
    def reflection_context(self, component_name: str):
        """Context manager to mark LM calls as reflection calls.

        Args:
            component_name: The component being reflected on
        """
        token = _reflection_context.set(component_name)
        try:
            yield
        finally:
            _reflection_context.reset(token)

    def start_run(
        self,
        run_id: str | None = None,
        seed_candidate: dict[str, str] | None = None,
        seed_val_scores: dict[int, float] | None = None,
        gepa_config: dict[str, Any] | None = None,
    ) -> str:
        """Start tracking a new GEPA run.

        Args:
            run_id: Optional run ID (generated if not provided)
            seed_candidate: Initial prompt configuration
            seed_val_scores: Initial validation scores per example
            gepa_config: GEPA configuration parameters

        Returns:
            The run ID
        """
        run_id = run_id or str(uuid.uuid4())

        # Calculate seed aggregate score
        seed_aggregate = 0.0
        if seed_val_scores:
            seed_aggregate = sum(seed_val_scores.values()) / len(seed_val_scores)

        # Create run record
        self._current_run = GEPARunRecord(
            run_id=run_id,
            started_at=datetime.utcnow(),
            config=GEPARunConfig(**(gepa_config or {})),
            seed_candidate=seed_candidate or {},
            seed_val_scores=seed_val_scores or {},
            seed_aggregate_score=seed_aggregate,
            status="running",
        )

        # Reset counters
        self._iteration_counter = 0
        self._current_iteration = None
        self._current_lm_calls = []

        # Persist initial state
        self.storage.save_run_start(self._current_run)

        # Install DSPy callback if configured
        if self.config.capture_lm_calls:
            self._install_callback()

        logger.info(f"Started GEPA run tracking: {run_id}")
        return run_id

    def register_examples(
        self,
        trainset: list | None = None,
        valset: list | None = None,
    ) -> None:
        """Register train and validation examples for the current run.

        This should be called after start_run() to populate the examples table.

        Args:
            trainset: List of training examples
            valset: List of validation examples
        """
        if self._current_run is None:
            logger.warning("register_examples called but no run is active")
            return

        run_id = self._current_run.run_id

        # Check if storage supports example creation
        if not hasattr(self.storage, 'storage') or not hasattr(self.storage.storage, 'create_example'):
            logger.warning("Storage backend doesn't support example registration")
            return

        # Register training examples
        if trainset:
            for i, example in enumerate(trainset):
                # Extract inputs and outputs from DSPy Example
                inputs = {}
                outputs = {}

                if hasattr(example, 'inputs') and callable(example.inputs):
                    inputs = example.inputs()
                elif hasattr(example, 'toDict'):
                    data = example.toDict()
                    # Split into inputs and outputs based on _input_keys if available
                    if hasattr(example, '_input_keys'):
                        input_keys = example._input_keys
                        inputs = {k: v for k, v in data.items() if k in input_keys}
                        outputs = {k: v for k, v in data.items() if k not in input_keys and not k.startswith('_')}
                    else:
                        inputs = data

                self.storage.storage.create_example(
                    run_id=run_id,
                    dataset_type='train',
                    example_index=i,
                    inputs=inputs,
                    outputs=outputs,
                )

        # Register validation examples
        if valset:
            for i, example in enumerate(valset):
                # Extract inputs and outputs from DSPy Example
                inputs = {}
                outputs = {}

                if hasattr(example, 'inputs') and callable(example.inputs):
                    inputs = example.inputs()
                elif hasattr(example, 'toDict'):
                    data = example.toDict()
                    # Split into inputs and outputs based on _input_keys if available
                    if hasattr(example, '_input_keys'):
                        input_keys = example._input_keys
                        inputs = {k: v for k, v in data.items() if k in input_keys}
                        outputs = {k: v for k, v in data.items() if k not in input_keys and not k.startswith('_')}
                    else:
                        inputs = data

                self.storage.storage.create_example(
                    run_id=run_id,
                    dataset_type='val',
                    example_index=i,
                    inputs=inputs,
                    outputs=outputs,
                )

        logger.info(f"Registered {len(trainset) if trainset else 0} train examples and {len(valset) if valset else 0} val examples")

    def end_run(
        self,
        status: str = "completed",
        error: str | None = None,
        final_pareto_frontier: dict[int, set[int]] | None = None,
        total_metric_calls: int = 0,
    ) -> None:
        """End the current run.

        Args:
            status: Final status ("completed", "failed", "stopped")
            error: Error message if failed
            final_pareto_frontier: Final Pareto frontier state
            total_metric_calls: Total metric calls made
        """
        if self._current_run is None:
            logger.warning("end_run called but no run is active")
            return

        # End any active iteration before ending the run
        if self._current_iteration is not None:
            logger.info(f"Ending final iteration {self._current_iteration.iteration_number} before run completion")
            self.end_iteration()

        self._current_run.completed_at = datetime.utcnow()
        self._current_run.status = status
        self._current_run.error_message = error
        self._current_run.total_metric_calls = total_metric_calls

        if final_pareto_frontier:
            self._current_run.final_pareto_frontier = final_pareto_frontier

        # Persist final state
        self.storage.save_run_end(self._current_run)

        # Remove callback
        self._remove_callback()

        logger.info(
            f"Ended GEPA run {self._current_run.run_id}: "
            f"status={status}, iterations={self._current_run.total_iterations}, "
            f"best_score={self._current_run.best_aggregate_score:.4f}"
        )

        # Clear current run state
        self._current_run = None
        self._current_iteration = None

    @contextmanager
    def track(self, run_id: str | None = None):
        """Context manager for tracking a GEPA run.

        Usage:
            with tracker.track() as run_id:
                optimized = gepa.compile(...)
        """
        run_id = self.start_run(run_id=run_id)
        try:
            yield run_id
            self.end_run(status="completed")
        except Exception as e:
            self.end_run(status="failed", error=str(e))
            raise

    # ==================== Iteration Recording ====================

    def start_iteration(
        self,
        parent_candidate_idx: int,
        parent_prompt: dict[str, str],
        parent_val_score: float,
        selection_strategy: str = "pareto",
        iteration_type: str = "reflective_mutation",
    ) -> int:
        """Start recording a new iteration.

        Args:
            parent_candidate_idx: Index of the parent candidate
            parent_prompt: Parent's prompt configuration
            parent_val_score: Parent's validation score
            selection_strategy: How the parent was selected
            iteration_type: Type of iteration

        Returns:
            The iteration number
        """
        if self._current_run is None:
            raise RuntimeError("No active run - call start_run first")

        self._iteration_counter += 1

        self._current_iteration = IterationRecord(
            run_id=self._current_run.run_id,
            iteration_number=self._iteration_counter,
            timestamp=datetime.utcnow(),
            parent_candidate_idx=parent_candidate_idx,
            parent_prompt=parent_prompt,
            parent_val_score=parent_val_score,
            selection_strategy=selection_strategy,
            iteration_type=iteration_type,
        )

        self._current_lm_calls = []

        return self._iteration_counter

    def auto_start_iteration(
        self,
        parent_candidate: dict[str, str],
        parent_val_score: float = 0.0,
    ) -> None:
        """Auto-start iteration if not already started.

        This is called by InstrumentedGEPAAdapter when it detects a new iteration.

        Args:
            parent_candidate: Parent prompt configuration
            parent_val_score: Parent validation score
        """
        if self._current_iteration is None:
            # Determine parent candidate index from our records
            parent_idx = 0
            if self._current_run and self._current_run.candidates:
                # Try to find matching candidate
                for cand in self._current_run.candidates:
                    if cand.instructions == parent_candidate:
                        parent_idx = cand.candidate_idx
                        break

            self.start_iteration(
                parent_candidate_idx=parent_idx,
                parent_prompt=parent_candidate,
                parent_val_score=parent_val_score,
            )

    def record_parent_evaluation(
        self,
        minibatch_ids: list[int],
        minibatch_inputs: list[dict[str, Any]],
        minibatch_outputs: list[dict[str, Any]],
        minibatch_scores: list[float],
        minibatch_feedback: list[str],
        traces: list[TraceRecord] | None = None,
    ) -> None:
        """Record the parent evaluation on the minibatch.

        Args:
            minibatch_ids: IDs of examples in the minibatch
            minibatch_inputs: Input data for each example
            minibatch_outputs: Output predictions for each example
            minibatch_scores: Scores for each example
            minibatch_feedback: Feedback for each example
            traces: Execution traces if captured
        """
        if self._current_iteration is None:
            logger.warning("record_parent_evaluation called but no iteration is active - auto-starting")
            self.auto_start_iteration(parent_candidate={}, parent_val_score=0.0)

        self._current_iteration.minibatch_ids = minibatch_ids
        self._current_iteration.minibatch_inputs = minibatch_inputs
        self._current_iteration.minibatch_outputs = minibatch_outputs
        self._current_iteration.minibatch_scores = minibatch_scores
        self._current_iteration.minibatch_feedback = minibatch_feedback

        if self.config.capture_traces and traces:
            self._current_iteration.minibatch_traces = traces

    def record_reflection(
        self,
        components_to_update: list[str],
        reflective_datasets: dict[str, list[dict[str, Any]]],
        proposed_instructions: dict[str, str],
        duration_ms: float = 0.0,
    ) -> None:
        """Record the reflection step.

        Args:
            components_to_update: Components being updated
            reflective_datasets: Feedback data sent to reflection LM
            proposed_instructions: New instructions proposed
            duration_ms: Time taken for reflection
        """
        if self._current_iteration is None:
            logger.warning("record_reflection called but no iteration is active")
            return

        # Collect reflection LM calls
        reflection_lm_calls = [lm for lm in self._current_lm_calls if lm.is_reflection]

        self._current_iteration.reflection = ReflectionRecord(
            iteration_number=self._current_iteration.iteration_number,
            components_to_update=components_to_update,
            reflective_datasets=reflective_datasets,
            lm_calls=reflection_lm_calls,
            proposed_instructions=proposed_instructions,
            duration_ms=duration_ms,
        )

        self._current_iteration.new_candidate_prompt = proposed_instructions

    def record_candidate_evaluation(
        self,
        minibatch_outputs: list[dict[str, Any]],
        minibatch_scores: list[float],
        minibatch_feedback: list[str] | None = None,
    ) -> None:
        """Record the new candidate's evaluation on the minibatch.

        Args:
            minibatch_outputs: Output predictions for each example
            minibatch_scores: Scores for each example
            minibatch_feedback: Optional feedback for each example
        """
        if self._current_iteration is None:
            logger.warning("record_candidate_evaluation called but no iteration is active")
            return

        self._current_iteration.new_candidate_minibatch_outputs = minibatch_outputs
        self._current_iteration.new_candidate_minibatch_scores = minibatch_scores
        if minibatch_feedback:
            self._current_iteration.new_candidate_minibatch_feedback = minibatch_feedback

    def record_validation_rollouts(
        self,
        validation_examples: list[Any],
        outputs: list[Any],
        scores: list[float],
        program_instructions: dict[str, str] | None = None,
    ) -> None:
        """Record validation rollouts with full output data.

        This is called when a candidate is evaluated on the full validation set
        after being accepted on the minibatch. It captures the actual outputs
        for each validation example, enabling detailed comparison reports.

        Args:
            validation_examples: The validation set examples (DSPy Examples)
            outputs: Model outputs for each validation example
            scores: Scores for each validation example
            program_instructions: The program/prompt that was evaluated
        """
        if self._current_iteration is None:
            logger.warning("record_validation_rollouts called but no iteration is active")
            return

        # Build dictionaries keyed by example index
        val_scores = {}
        val_outputs = {}
        val_inputs = {}

        for i, (example, output, score) in enumerate(zip(validation_examples, outputs, scores)):
            val_scores[i] = score

            # Extract output data
            if hasattr(output, 'toDict'):
                val_outputs[i] = output.toDict()
            elif hasattr(output, '__dict__'):
                val_outputs[i] = {k: v for k, v in output.__dict__.items() if not k.startswith('_')}
            elif isinstance(output, dict):
                val_outputs[i] = output
            else:
                val_outputs[i] = {"value": str(output)}

            # Extract input data from example
            if hasattr(example, 'inputs') and callable(example.inputs):
                val_inputs[i] = example.inputs()
            elif hasattr(example, 'toDict'):
                val_inputs[i] = example.toDict()
            elif isinstance(example, dict):
                val_inputs[i] = example
            else:
                val_inputs[i] = {}

        # Store in current iteration
        self._current_iteration.val_scores = val_scores
        self._current_iteration.val_outputs = val_outputs
        self._current_iteration.val_inputs = val_inputs
        self._current_iteration.val_aggregate_score = sum(scores) / len(scores) if scores else 0.0

        # Also store the program instructions if provided
        if program_instructions and not self._current_iteration.new_candidate_prompt:
            self._current_iteration.new_candidate_prompt = program_instructions

        logger.info(
            f"Recorded {len(scores)} validation rollouts for iteration "
            f"{self._current_iteration.iteration_number}, avg_score={self._current_iteration.val_aggregate_score:.4f}"
        )

    def record_acceptance(
        self,
        accepted: bool,
        reason: str | None = None,
        val_scores: dict[int, float] | None = None,
        val_aggregate_score: float | None = None,
        new_candidate_idx: int | None = None,
    ) -> None:
        """Record the acceptance decision.

        Args:
            accepted: Whether the candidate was accepted
            reason: Reason for acceptance/rejection
            val_scores: Validation scores if accepted
            val_aggregate_score: Aggregate validation score
            new_candidate_idx: Index assigned to new candidate
        """
        if self._current_iteration is None:
            logger.warning("record_acceptance called but no iteration is active")
            return

        self._current_iteration.accepted = accepted
        self._current_iteration.acceptance_reason = reason

        # Only set val_scores if not already populated by record_validation_rollouts()
        # (the engine hook provides more complete data including outputs)
        if val_scores and not self._current_iteration.val_scores:
            self._current_iteration.val_scores = val_scores

        # Only set aggregate if not already set
        if val_aggregate_score is not None and self._current_iteration.val_aggregate_score is None:
            self._current_iteration.val_aggregate_score = val_aggregate_score

        self._current_iteration.new_candidate_idx = new_candidate_idx

    def record_pareto_update(self, pareto_update: ParetoFrontierUpdate) -> None:
        """Record Pareto frontier update.

        Args:
            pareto_update: The frontier update record
        """
        if self._current_iteration is None:
            logger.warning("record_pareto_update called but no iteration is active")
            return

        self._current_iteration.pareto_update = pareto_update

    def end_iteration(self, duration_ms: float | None = None) -> None:
        """End the current iteration and persist it.

        Args:
            duration_ms: Total iteration duration
        """
        if self._current_iteration is None:
            logger.warning("end_iteration called but no iteration is active")
            return

        if self._current_run is None:
            logger.warning("end_iteration called but no run is active")
            return

        self._current_iteration.duration_ms = duration_ms

        # Attach LM calls to iteration if capture is enabled
        if self.config.capture_lm_calls and self._current_lm_calls:
            self._current_iteration.lm_calls = self._current_lm_calls.copy()
            logger.info(f"Attached {len(self._current_lm_calls)} LM calls to iteration {self._current_iteration.iteration_number}")

        # Persist iteration (wrap in try/except to handle duplicate save attempts)
        try:
            self.storage.save_iteration(self._current_run.run_id, self._current_iteration)
        except Exception as e:
            # Check if this is a duplicate save error
            if "UNIQUE constraint failed" in str(e) and "iteration_number" in str(e):
                logger.warning(f"Iteration {self._current_iteration.iteration_number} already saved, skipping duplicate save")
            else:
                raise

        # Add to run record
        self._current_run.add_iteration(self._current_iteration)

        # Fire callbacks
        for callback in self._on_iteration_complete:
            try:
                callback(self._current_iteration)
            except Exception as e:
                logger.warning(f"Error in iteration callback: {e}")

        # Reset
        self._current_iteration = None
        self._current_lm_calls = []

    # ==================== Candidate Recording ====================

    def record_candidate(
        self,
        candidate_idx: int,
        instructions: dict[str, str],
        parent_indices: list[int],
        creation_iteration: int,
        creation_type: str,
        val_subscores: dict[int, float],
        val_aggregate_score: float,
        metric_calls_at_discovery: int = 0,
    ) -> None:
        """Record a new candidate.

        Args:
            candidate_idx: Index of the candidate
            instructions: Prompt configuration
            parent_indices: Indices of parent candidates
            creation_iteration: Iteration when created
            creation_type: How the candidate was created
            val_subscores: Per-example validation scores
            val_aggregate_score: Aggregate validation score
            metric_calls_at_discovery: Budget spent to discover this candidate
        """
        if self._current_run is None:
            logger.warning("record_candidate called but no run is active")
            return

        candidate = CandidateRecord(
            candidate_idx=candidate_idx,
            instructions=instructions,
            parent_indices=parent_indices,
            creation_iteration=creation_iteration,
            creation_type=creation_type,
            val_subscores=val_subscores,
            val_aggregate_score=val_aggregate_score,
            metric_calls_at_discovery=metric_calls_at_discovery,
        )

        # Persist candidate
        self.storage.save_candidate(self._current_run.run_id, candidate)

        # Add to run record
        self._current_run.add_candidate(candidate)

        # Fire callbacks
        if self._current_iteration:
            for callback in self._on_candidate_accepted:
                try:
                    callback(candidate, self._current_iteration)
                except Exception as e:
                    logger.warning(f"Error in candidate callback: {e}")

    # ==================== LM Call Recording ====================

    def start_lm_call(
        self,
        call_id: str,
        model: str,
        inputs: dict[str, Any],
    ) -> None:
        """Record the start of an LM call.

        Args:
            call_id: Unique ID for the call
            model: Model name
            inputs: Call inputs
        """
        self._active_lm_calls[call_id] = {
            "start_time": time.time(),
            "model": model,
            "inputs": inputs,
            "is_reflection": self.is_in_reflection_context(),
            "component_name": self.get_reflection_component(),
        }

    def end_lm_call(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Record the end of an LM call with comprehensive MLflow-style tracking.

        Args:
            call_id: Unique ID for the call
            outputs: Call outputs
            exception: Exception if call failed
        """
        call_data = self._active_lm_calls.pop(call_id, None)
        if call_data is None:
            return

        start_time = datetime.fromtimestamp(call_data["start_time"])
        end_time = datetime.utcnow()
        duration_ms = (time.time() - call_data["start_time"]) * 1000

        # Extract comprehensive response data
        response_text = ""
        usage = None
        finish_reason = None
        raw_response = None

        if outputs:
            raw_response = outputs  # Store full response
            if isinstance(outputs, list) and len(outputs) > 0:
                response_text = str(outputs[0])
            elif isinstance(outputs, dict):
                response_text = outputs.get("text", outputs.get("content", outputs.get("message", str(outputs))))
                usage = outputs.get("usage", {})
                finish_reason = outputs.get("finish_reason")

        # Normalize token usage fields
        input_tokens = None
        output_tokens = None
        total_tokens = None

        if usage:
            # Handle different API response formats
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")

            # Calculate total if not provided
            if total_tokens is None and input_tokens and output_tokens:
                total_tokens = input_tokens + output_tokens

        # Extract request parameters
        inputs = call_data["inputs"]
        raw_request = inputs  # Store full request

        lm_record = LMCallRecord(
            call_id=call_id,
            model=call_data["model"],
            # Request details
            messages=inputs.get("messages", []),
            temperature=inputs.get("temperature"),
            max_tokens=inputs.get("max_tokens"),
            top_p=inputs.get("top_p"),
            frequency_penalty=inputs.get("frequency_penalty"),
            presence_penalty=inputs.get("presence_penalty"),
            stop_sequences=inputs.get("stop"),
            raw_request=raw_request,
            raw_response=raw_response,
            # Response
            response_text=response_text,
            finish_reason=finish_reason,
            # Token usage
            usage=usage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            # Timing
            duration_ms=duration_ms,
            timestamp=start_time,
            start_time=start_time,
            end_time=end_time,
            # Error handling
            success=exception is None,
            error_type=type(exception).__name__ if exception else None,
            error_message=str(exception) if exception else None,
            # Context
            is_reflection=call_data["is_reflection"],
            component_name=call_data["component_name"],
            iteration_number=self._current_iteration.iteration_number if self._current_iteration else None,
        )

        # Calculate cost if tokens are available
        if input_tokens and output_tokens:
            try:
                lm_record.calculate_cost()
            except Exception as e:
                logger.debug(f"Could not calculate cost for LM call: {e}")

        self._current_lm_calls.append(lm_record)

    # ==================== Callback Management ====================

    def _install_callback(self) -> None:
        """Install DSPy callback for LM call capture."""
        try:
            import dspy
            from dspy_gepa_logger.hooks.callback_handler import GEPALoggingCallback

            self._callback = GEPALoggingCallback(self)
            self._original_callbacks = dspy.settings.get("callbacks", [])
            dspy.configure(callbacks=self._original_callbacks + [self._callback])
        except ImportError:
            logger.warning("Could not import dspy - LM call capture disabled")

    def _remove_callback(self) -> None:
        """Remove DSPy callback."""
        if self._callback is None:
            return

        try:
            import dspy

            callbacks = dspy.settings.get("callbacks", [])
            if self._callback in callbacks:
                callbacks.remove(self._callback)
                dspy.configure(callbacks=callbacks)
        except ImportError:
            pass

        self._callback = None

    # ==================== Event Callbacks ====================

    def on_iteration_complete(self, callback: Callable[[IterationRecord], None]) -> None:
        """Register a callback for iteration completion.

        Args:
            callback: Function to call with the iteration record
        """
        self._on_iteration_complete.append(callback)

    def on_candidate_accepted(
        self,
        callback: Callable[[CandidateRecord, IterationRecord], None],
    ) -> None:
        """Register a callback for candidate acceptance.

        Args:
            callback: Function to call with candidate and iteration records
        """
        self._on_candidate_accepted.append(callback)

    # ==================== Query API ====================

    def get_run(self, run_id: str | None = None) -> GEPARunRecord | None:
        """Get a run record.

        Args:
            run_id: Run ID to load (defaults to current run)

        Returns:
            The run record or None if not found
        """
        if run_id is None:
            return self._current_run
        return self.storage.load_run(run_id)

    def get_iterations(self, run_id: str | None = None) -> Iterator[IterationRecord]:
        """Get iterations for a run.

        Args:
            run_id: Run ID (defaults to current run)

        Yields:
            Iteration records
        """
        run_id = run_id or self.current_run_id
        if run_id is None:
            return
        yield from self.storage.load_iterations(run_id)

    def get_candidates(self, run_id: str | None = None) -> Iterator[CandidateRecord]:
        """Get candidates for a run.

        Args:
            run_id: Run ID (defaults to current run)

        Yields:
            Candidate records
        """
        run_id = run_id or self.current_run_id
        if run_id is None:
            return
        yield from self.storage.load_candidates(run_id)

    def list_runs(self) -> list[str]:
        """List all available run IDs.

        Returns:
            List of run IDs
        """
        return self.storage.list_runs()
