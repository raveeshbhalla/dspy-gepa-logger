"""Instrumented GEPA adapter that automatically captures all iteration data."""

import time
import logging
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
DspyAdapter = None


def get_dspy_adapter():
    """Lazy import DspyAdapter to avoid import errors if dspy not installed."""
    global DspyAdapter
    if DspyAdapter is None:
        try:
            from dspy.teleprompt.gepa.gepa_utils import DspyAdapter as _DspyAdapter
            DspyAdapter = _DspyAdapter
        except ImportError:
            raise ImportError(
                "dspy.teleprompt.gepa.gepa_utils.DspyAdapter not found. "
                "This feature requires DSPy >= 2.5.0"
            )
    return DspyAdapter


class InstrumentedGEPAAdapter:
    """Wraps DspyAdapter to automatically capture GEPA iteration data.

    This adapter intercepts all calls to evaluate(), make_reflective_dataset(),
    and propose_new_texts() to automatically record iteration data in the tracker.

    Usage:
        tracker = track_gepa_run(log_dir="./logs")

        # In your GEPA compile call:
        gepa = GEPA(...)
        with tracker.track():
            # Create instrumented adapter
            adapter = tracker.create_instrumented_adapter(
                program=student,
                metric=my_metric,
                ...  # other adapter args
            )
            # Use GEPA with instrumented adapter
            optimized = gepa.compile(student=student, trainset=train, valset=val)
    """

    def __init__(self, wrapped_adapter, tracker, log_file: Path | None = None):
        """Initialize the instrumented adapter.

        Args:
            wrapped_adapter: The DspyAdapter instance to wrap
            tracker: GEPARunTracker instance for recording data
            log_file: Optional path to write detailed logs
        """
        self.wrapped = wrapped_adapter
        self.tracker = tracker
        self.log_file = log_file

        # Store original methods before they get replaced
        # Use getattr with None default in case methods don't exist yet
        self._original_evaluate = getattr(wrapped_adapter, 'evaluate', None)
        self._original_make_reflective_dataset = getattr(wrapped_adapter, 'make_reflective_dataset', None)
        self._original_propose_new_texts = getattr(wrapped_adapter, 'propose_new_texts', None)

        # State tracking
        self.evaluation_count = 0
        self.reflection_count = 0
        self.current_parent_candidate = None
        self.current_parent_eval_result = None
        self.expecting_candidate_eval = False  # True after propose_new_texts, False after recording it

        # Delegate all other attributes to wrapped adapter
        for attr in dir(wrapped_adapter):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(wrapped_adapter, attr))

    def _log(self, message: str):
        """Write to log file if enabled."""
        if self.log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")

    def evaluate(self, batch, candidate, capture_traces=False):
        """Wrap evaluate() to capture evaluation data.

        This is called multiple times per iteration:
        0. First call is baseline eval on full valset (iteration 0)
        1. Then with capture_traces=True on reflection minibatch (parent eval)
        2. Then with capture_traces=False on same minibatch (candidate eval)
        3. Optionally on full validation set if accepted
        """
        start_time = time.time()
        self.evaluation_count += 1

        # Determine if this is parent or candidate evaluation
        is_parent_eval = capture_traces  # Parent evals capture traces

        # Detect baseline evaluation (iteration 0)
        # This is the very first evaluation on the full validation set
        is_baseline = (self.evaluation_count == 1 and
                      self.tracker.is_tracking and
                      self.tracker._current_iteration is None and
                      not capture_traces)

        # End previous iteration if this is a parent eval and we have an active iteration
        if is_parent_eval and self.tracker.is_tracking and self.tracker._current_iteration is not None:
            # Save the previous iteration before starting a new one
            prev_iter_num = self.tracker._current_iteration.iteration_number
            self.tracker.end_iteration(duration_ms=(time.time() - start_time) * 1000)
            self._log(f"Ended previous iteration {prev_iter_num}")

        # Create iteration 0 for baseline if this is the first eval
        if is_baseline:
            self._log("Detected baseline evaluation - creating iteration 0")
            # Start iteration (will be iteration 1 by default due to counter)
            self.tracker.start_iteration(
                parent_candidate_idx=0,
                parent_prompt=candidate if isinstance(candidate, dict) else {},
                parent_val_score=0.0,  # Will be updated after eval
                selection_strategy="baseline",
                iteration_type="baseline",
            )
            # Manually set to iteration 0 and adjust counter
            if self.tracker._current_iteration:
                self.tracker._current_iteration.iteration_number = 0
                self.tracker._iteration_counter = 0  # Reset so next will be 1

        # Auto-start iteration if this is a parent eval and tracker is active
        if is_parent_eval and self.tracker.is_tracking and self.tracker._current_iteration is None:
            self.tracker.auto_start_iteration(
                parent_candidate=candidate if isinstance(candidate, dict) else {},
                parent_val_score=0.0,  # Will be updated later
            )

        self._log(f"{'='*60}")
        self._log(f"Evaluation #{self.evaluation_count} ({'parent' if is_parent_eval else 'candidate'})")
        self._log(f"Batch size: {len(batch)}")
        self._log(f"Capture traces: {capture_traces}")
        self._log(f"Candidate instructions: {candidate}")

        # Call original evaluate method
        try:
            result = self._original_evaluate(batch, candidate, capture_traces)
        except Exception as e:
            self._log(f"Error in evaluate: {e}")
            raise

        duration_ms = (time.time() - start_time) * 1000

        # Extract data from result safely
        scores = getattr(result, 'scores', []) or []
        outputs = getattr(result, 'outputs', []) or []
        trajectories = getattr(result, 'trajectories', None)
        feedbacks = getattr(result, 'feedbacks', None)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        self._log(f"Duration: {duration_ms:.2f}ms")
        self._log(f"Scores: {scores}")
        self._log(f"Feedbacks: {feedbacks[:3] if feedbacks else None}...")  # Log first 3
        self._log(f"Average: {avg_score:.4f}")

        # Record data in tracker (only if iteration is active)
        current_iter = self.tracker._current_iteration
        if is_parent_eval and current_iter:
            # This is the parent evaluation on reflection minibatch
            self.current_parent_candidate = candidate
            self.current_parent_eval_result = result

            # Extract example IDs and inputs safely
            example_ids = []
            inputs = []
            for i, ex in enumerate(batch):
                example_ids.append(getattr(ex, 'example_id', i))
                try:
                    inputs.append(ex.inputs() if hasattr(ex, 'inputs') else {})
                except Exception:
                    inputs.append({})

            # Extract feedback - try result.feedbacks first, then trajectories
            feedback = []
            if feedbacks:
                # Use feedbacks from result if available
                feedback = [str(f) if f is not None else "" for f in feedbacks]
            elif trajectories:
                # Fall back to extracting from trajectories
                for traj in trajectories:
                    traj_feedback = ""
                    try:
                        if isinstance(traj, dict) and 'feedback' in traj:
                            traj_feedback = str(traj['feedback'])
                        elif hasattr(traj, 'feedback'):
                            traj_feedback = str(traj.feedback)
                    except Exception:
                        pass
                    feedback.append(traj_feedback)
            else:
                feedback = ["" for _ in batch]

            # Record parent evaluation
            try:
                self.tracker.record_parent_evaluation(
                    minibatch_ids=example_ids,
                    minibatch_inputs=inputs,
                    minibatch_outputs=outputs,
                    minibatch_scores=scores,
                    minibatch_feedback=feedback,
                    traces=self._safe_convert_trajectories(trajectories),
                )
                self._log(f"Recorded parent evaluation in iteration {current_iter.iteration_number}")
            except Exception as e:
                self._log(f"Error recording parent evaluation: {e}")

        elif not is_parent_eval and current_iter:
            # This is candidate evaluation
            # Only record if we're expecting it (right after propose_new_texts)
            if self.expecting_candidate_eval:
                # Extract feedback - try result.feedbacks first, then trajectories
                feedback = []
                if feedbacks:
                    # Use feedbacks from result if available
                    feedback = [str(f) if f is not None else "" for f in feedbacks]
                elif trajectories:
                    # Fall back to extracting from trajectories
                    for traj in trajectories:
                        traj_feedback = ""
                        try:
                            if isinstance(traj, dict) and 'feedback' in traj:
                                traj_feedback = str(traj['feedback'])
                            elif hasattr(traj, 'feedback'):
                                traj_feedback = str(traj.feedback)
                        except Exception:
                            pass
                        feedback.append(traj_feedback)
                else:
                    feedback = ["" for _ in batch]

                try:
                    self.tracker.record_candidate_evaluation(
                        minibatch_outputs=outputs,
                        minibatch_scores=scores,
                        minibatch_feedback=feedback if any(feedback) else None,
                    )
                    self._log(f"Recorded NEW CANDIDATE evaluation in iteration {current_iter.iteration_number}")
                    self.expecting_candidate_eval = False  # Reset flag
                except Exception as e:
                    self._log(f"Error recording candidate evaluation: {e}")
            else:
                # This is a validation eval or other candidate eval - log but don't record in iteration
                self._log(f"Skipping candidate eval (not the new candidate from iteration {current_iter.iteration_number})")

        # If this was the baseline evaluation, record validation scores and outputs, then close iteration 0
        if is_baseline and current_iter and current_iter.iteration_number == 0:
            # Record validation rollouts with outputs for baseline
            # This ensures the report can show baseline answers
            try:
                self.tracker.record_validation_rollouts(
                    validation_examples=batch,
                    outputs=outputs,
                    scores=scores,
                    program_instructions=candidate if isinstance(candidate, dict) else {},
                )
                self._log(f"Recorded baseline validation rollouts with outputs: {len(outputs)} examples")
            except Exception as e:
                self._log(f"Error recording baseline validation rollouts: {e}")

            # Record validation scores for baseline
            val_scores = {i: score for i, score in enumerate(scores)}
            self.tracker.record_acceptance(
                accepted=False,  # Baseline is not "accepted", it's the starting point
                reason="baseline",
                val_scores=val_scores,
                val_aggregate_score=avg_score,
                new_candidate_idx=0,
            )
            self._log(f"Recorded baseline validation scores: avg={avg_score:.4f}")

            # End iteration 0 immediately
            self.tracker.end_iteration(duration_ms=duration_ms)
            self._log("Ended iteration 0 (baseline)")

        self._log(f"{'='*60}\n")

        return result

    def _safe_convert_trajectories(self, trajectories):
        """Safely convert trajectories, returning None on any error."""
        if not trajectories:
            return None
        try:
            return self._convert_trajectories_to_traces(trajectories)
        except Exception as e:
            self._log(f"Error converting trajectories: {e}")
            return None

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Wrap make_reflective_dataset() to capture reflection data."""
        start_time = time.time()
        self.reflection_count += 1

        self._log(f"{'='*60}")
        self._log(f"Reflection #{self.reflection_count}")
        self._log(f"Components to update: {components_to_update}")

        # Call original method
        result = self._original_make_reflective_dataset(candidate, eval_batch, components_to_update)

        duration_ms = (time.time() - start_time) * 1000

        # Log reflective dataset details
        for component, examples in result.items():
            self._log(f"Component '{component}': {len(examples)} reflective examples")
            for i, ex in enumerate(examples):  # Log all examples
                self._log(f"  Example {i+1}:")
                self._log(f"    Inputs: {ex.get('Inputs', {})}")
                self._log(f"    Outputs: {ex.get('Generated_Outputs', {})}")
                self._log(f"    Feedback: {ex.get('Feedback', '')[:100]}...")

        self._log(f"Reflection dataset creation took {duration_ms:.2f}ms")
        self._log(f"{'='*60}\n")

        # Store for use in propose_new_texts (if it exists)
        self.current_reflective_dataset = result
        self.current_components_to_update = components_to_update

        # If propose_new_texts doesn't exist (newer GEPA versions), record reflection here
        if self._original_propose_new_texts is None:
            current_iter = self.tracker._current_iteration
            if current_iter:
                # Record reflection without proposed instructions (GEPA handles that internally now)
                self.tracker.record_reflection(
                    components_to_update=components_to_update,
                    reflective_datasets=result,
                    proposed_instructions={},  # Empty - GEPA doesn't expose this in current version
                    duration_ms=duration_ms,
                )
                self._log(f"Recorded reflection in iteration {current_iter.iteration_number} (no propose_new_texts)")

        return result

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        """Wrap propose_new_texts() to capture reflection LM calls and proposals."""
        start_time = time.time()

        self._log(f"{'='*60}")
        self._log(f"Proposing new instructions for: {components_to_update}")

        # Call the original method directly (not per-component, which breaks the API)
        # The reflection context tracking via DSPy callbacks still works
        try:
            if self._original_propose_new_texts is None:
                self._log("Error: _original_propose_new_texts is None - method not found on adapter")
                raise RuntimeError(
                    "propose_new_texts method not found on DspyAdapter. "
                    "This may indicate an incompatible DSPy version."
                )

            proposed_texts = self._original_propose_new_texts(
                candidate,
                reflective_dataset,
                components_to_update
            )
        except Exception as e:
            self._log(f"Error in propose_new_texts: {e}")
            raise

        duration_ms = (time.time() - start_time) * 1000

        for component in components_to_update:
            old_instr = candidate.get(component, '') if isinstance(candidate, dict) else ''
            new_instr = proposed_texts.get(component, '') if isinstance(proposed_texts, dict) else ''
            self._log(f"Component '{component}':")
            self._log(f"  Old instruction: {str(old_instr)[:100]}...")
            self._log(f"  New instruction: {str(new_instr)[:100]}...")

        # Record reflection in tracker
        current_iter = self.tracker._current_iteration
        if current_iter:
            iter_num = current_iter.iteration_number
            self.tracker.record_reflection(
                components_to_update=components_to_update,
                reflective_datasets=reflective_dataset,
                proposed_instructions=proposed_texts,
                duration_ms=duration_ms,
            )
            self._log(f"Recorded reflection in iteration {iter_num}")

            # DON'T end iteration here - we still need to capture the new candidate evaluation
            # The iteration will end when the next parent eval starts (see evaluate method)

            # Set flag to expect candidate evaluation next
            self.expecting_candidate_eval = True

        self._log(f"Total proposal duration: {duration_ms:.2f}ms")
        self._log(f"{'='*60}\n")

        return proposed_texts

    def _convert_trajectories_to_traces(self, trajectories):
        """Convert GEPA trajectories to our TraceRecord format."""
        from dspy_gepa_logger.models.trace import TraceRecord, PredictorCallRecord

        if not trajectories:
            return None

        traces = []
        for traj in trajectories:
            if not isinstance(traj, dict):
                continue

            # Extract trace data
            trace_data = traj.get('trace', [])
            example = traj.get('example')
            prediction = traj.get('prediction')
            score = traj.get('score', 0.0)

            # Convert to predictor calls
            predictor_calls = []
            for i, item in enumerate(trace_data):
                # Handle different trace formats
                if isinstance(item, tuple) and len(item) >= 3:
                    predictor, inputs, outputs = item[0], item[1], item[2]
                else:
                    continue

                # Get predictor name safely
                pred_name = getattr(predictor, '__class__', type(predictor)).__name__
                sig_name = ""
                if hasattr(predictor, 'signature'):
                    sig = predictor.signature
                    sig_name = getattr(sig, '__name__', str(type(sig).__name__))

                # Convert outputs to dict if it's a Prediction object
                outputs_dict = {}
                if hasattr(outputs, 'toDict'):
                    outputs_dict = outputs.toDict()
                elif hasattr(outputs, '__dict__'):
                    outputs_dict = {k: v for k, v in outputs.__dict__.items() if not k.startswith('_')}
                elif isinstance(outputs, dict):
                    outputs_dict = outputs

                predictor_calls.append(PredictorCallRecord(
                    predictor_name=pred_name,
                    signature_name=sig_name,
                    inputs=inputs if isinstance(inputs, dict) else {},
                    outputs=outputs_dict,
                ))

            # Get example ID safely
            example_id = 0
            if hasattr(example, 'example_id'):
                example_id = example.example_id
            elif hasattr(example, '_example_id'):
                example_id = example._example_id

            traces.append(TraceRecord(
                example_id=example_id,
                predictor_calls=predictor_calls,
                lm_calls=[],  # LM calls tracked separately via callbacks
            ))

        return traces


def create_instrumented_gepa(gepa_instance, tracker, log_file: Path | None = None):
    """Create an instrumented GEPA instance that uses the tracking adapter.

    This is a helper function that wraps a GEPA instance to use InstrumentedGEPAAdapter
    and injects Pareto frontier tracking.

    Args:
        gepa_instance: The GEPA optimizer instance
        tracker: GEPARunTracker instance
        log_file: Optional path to write detailed logs

    Returns:
        The GEPA instance (modified in place)

    Example:
        tracker = track_gepa_run(log_dir="./logs")
        gepa = GEPA(metric=my_metric, ...)

        with tracker.track():
            create_instrumented_gepa(gepa, tracker, log_file=Path("./logs/gepa_run.log"))
            optimized = gepa.compile(student=program, trainset=train, valset=val)
    """
    # We need to wrap the DspyAdapter methods DURING __init__, BEFORE optimization starts
    # This is done by patching DspyAdapter.__init__ to wrap the methods immediately
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    original_adapter_init = DspyAdapter.__init__

    def wrapping_adapter_init(adapter_self, *args, **kwargs):
        # Call original init first
        original_adapter_init(adapter_self, *args, **kwargs)

        # Log that we're wrapping
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Wrapping DspyAdapter methods for tracking\n")

        # Create instrumented wrapper with the ORIGINAL methods
        # We must capture them NOW before we replace them
        original_evaluate = getattr(adapter_self, 'evaluate', None)
        original_make_reflective_dataset = getattr(adapter_self, 'make_reflective_dataset', None)
        original_propose_new_texts = getattr(adapter_self, 'propose_new_texts', None)

        # Debug log what methods we found
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found methods: evaluate={original_evaluate is not None}, make_reflective_dataset={original_make_reflective_dataset is not None}, propose_new_texts={original_propose_new_texts is not None}\n")

        # Create wrapper that uses these originals
        instrumented = InstrumentedGEPAAdapter(adapter_self, tracker, log_file)
        # Override the stored originals with the REAL originals
        if original_evaluate is not None:
            instrumented._original_evaluate = original_evaluate
        if original_make_reflective_dataset is not None:
            instrumented._original_make_reflective_dataset = original_make_reflective_dataset
        if original_propose_new_texts is not None:
            instrumented._original_propose_new_texts = original_propose_new_texts

        # Replace adapter's methods with instrumented versions (only if originals exist)
        if instrumented._original_evaluate is not None:
            adapter_self.evaluate = instrumented.evaluate
        if instrumented._original_make_reflective_dataset is not None:
            adapter_self.make_reflective_dataset = instrumented.make_reflective_dataset
        if instrumented._original_propose_new_texts is not None:
            adapter_self.propose_new_texts = instrumented.propose_new_texts

    # Patch DspyAdapter.__init__ globally
    # This will be active for the duration of the GEPA compile
    DspyAdapter.__init__ = wrapping_adapter_init

    # Patch GEPAEngine._run_full_eval_and_add for capturing validation rollouts
    # This is called when a candidate is accepted and evaluated on full valset
    try:
        from gepa.core.engine import GEPAEngine

        original_run_full_eval_and_add = GEPAEngine._run_full_eval_and_add

        def wrapped_run_full_eval_and_add(engine_self, new_program, state, parent_program_idx):
            # Get valset outputs before calling original (to capture them)
            valset_outputs, valset_subscores = engine_self._val_evaluator()(new_program)

            # Now manually do what the original method does, but we capture the data
            num_metric_calls_by_discovery = state.total_num_evals
            valset_score = sum(valset_subscores) / len(valset_subscores)

            state.num_full_ds_evals += 1
            state.total_num_evals += len(valset_subscores)

            new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
                parent_program_idx=parent_program_idx,
                new_program=new_program,
                valset_score=valset_score,
                valset_outputs=valset_outputs,
                valset_subscores=valset_subscores,
                run_dir=engine_self.run_dir,
                num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
            )
            state.full_program_trace[-1]["new_program_idx"] = new_program_idx

            if new_program_idx == linear_pareto_front_program_idx:
                engine_self.logger.log(f"Iteration {state.i + 1}: New program is on the linear pareto front")

            from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
            log_detailed_metrics_after_discovering_new_program(
                logger=engine_self.logger,
                gepa_state=state,
                valset_score=valset_score,
                new_program_idx=new_program_idx,
                valset_subscores=valset_subscores,
                experiment_tracker=engine_self.experiment_tracker,
                linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            )

            # Capture validation rollouts
            if tracker.is_tracking and tracker._current_iteration is not None:
                try:
                    # Get validation examples directly from engine's valset
                    validation_examples = engine_self.valset if engine_self.valset else []

                    tracker.record_validation_rollouts(
                        validation_examples=validation_examples,
                        outputs=valset_outputs,
                        scores=valset_subscores,
                        program_instructions=new_program,
                    )
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Captured {len(valset_subscores)} validation rollouts via engine hook\n")
                except Exception as e:
                    logger.warning(f"Error capturing validation rollouts: {e}")
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error capturing validation rollouts: {e}\n")

            return new_program_idx, linear_pareto_front_program_idx

        GEPAEngine._run_full_eval_and_add = wrapped_run_full_eval_and_add
        gepa_instance._original_run_full_eval_and_add = original_run_full_eval_and_add
        gepa_instance._gepa_engine_hook_active = True

        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Patched GEPAEngine._run_full_eval_and_add for validation rollout capture\n")

    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not patch GEPAEngine for validation rollout capture: {e}")

    # Also patch GEPA's create_experiment_tracker to inject our Pareto tracker
    try:
        from gepa.logging.experiment_tracker import create_experiment_tracker
        from dspy_gepa_logger.hooks.pareto_tracker import ParetoCapturingTracker

        original_create_experiment_tracker = create_experiment_tracker

        def wrapped_create_experiment_tracker(*args, **kwargs):
            """Create experiment tracker and wrap it with Pareto capturing."""
            logger.info(f"wrapped_create_experiment_tracker called with args={args}, kwargs={kwargs}")
            real_tracker = original_create_experiment_tracker(*args, **kwargs)
            logger.info(f"Created real tracker: {type(real_tracker).__name__}")
            pareto_tracker = ParetoCapturingTracker(real_tracker, tracker)
            logger.info(f"Wrapped with ParetoCapturingTracker")
            return pareto_tracker

        # Monkey-patch the function in the gepa.api module (where it's imported)
        import gepa.api
        gepa.api.create_experiment_tracker = wrapped_create_experiment_tracker
        logger.info(f"Monkey-patched gepa.api.create_experiment_tracker")

        # Store original for cleanup
        gepa_instance._original_create_experiment_tracker = original_create_experiment_tracker
        gepa_instance._gepa_pareto_tracking_active = True

        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Injected Pareto frontier tracking\n")

    except ImportError as e:
        logger.warning(f"Could not inject Pareto tracker: {e}")

    # Set up LM call tracking via callback
    if tracker.config.capture_lm_calls:
        try:
            import dspy
            from dspy_gepa_logger.hooks.callback_handler import GEPALoggingCallback

            # Create callback
            callback = GEPALoggingCallback(tracker)

            # Get current DSPy settings
            settings = dspy.settings
            original_callbacks = getattr(settings, 'callbacks', None) or []

            # Add our callback
            new_callbacks = list(original_callbacks) + [callback]
            dspy.configure(callbacks=new_callbacks)

            # Store for cleanup
            gepa_instance._gepa_callback = callback
            gepa_instance._original_callbacks = original_callbacks

            logger.info(f"Registered GEPALoggingCallback for LM call tracking")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Registered LM call tracking callback\n")
        except Exception as e:
            logger.warning(f"Could not set up LM call tracking: {e}")

    # Store original for cleanup
    gepa_instance._original_adapter_init = original_adapter_init
    gepa_instance._gepa_logger_active = True

    return gepa_instance


def cleanup_instrumented_gepa(gepa_instance):
    """Restore original DspyAdapter.__init__, GEPAEngine.__init__, and experiment tracker after GEPA run.

    Call this after gepa.compile() completes to restore the original behavior.
    """
    if hasattr(gepa_instance, '_original_adapter_init') and hasattr(gepa_instance, '_gepa_logger_active'):
        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
        DspyAdapter.__init__ = gepa_instance._original_adapter_init
        del gepa_instance._original_adapter_init
        del gepa_instance._gepa_logger_active

    # Restore original GEPAEngine._run_full_eval_and_add
    if hasattr(gepa_instance, '_original_run_full_eval_and_add') and hasattr(gepa_instance, '_gepa_engine_hook_active'):
        try:
            from gepa.core.engine import GEPAEngine
            GEPAEngine._run_full_eval_and_add = gepa_instance._original_run_full_eval_and_add
            del gepa_instance._original_run_full_eval_and_add
            del gepa_instance._gepa_engine_hook_active
            logger.info("Restored original GEPAEngine._run_full_eval_and_add")
        except Exception as e:
            logger.warning(f"Could not restore GEPAEngine._run_full_eval_and_add: {e}")

    # Restore original experiment tracker creator
    if hasattr(gepa_instance, '_original_create_experiment_tracker') and hasattr(gepa_instance, '_gepa_pareto_tracking_active'):
        try:
            import gepa.api
            gepa.api.create_experiment_tracker = gepa_instance._original_create_experiment_tracker
            del gepa_instance._original_create_experiment_tracker
            del gepa_instance._gepa_pareto_tracking_active
        except Exception as e:
            logger.warning(f"Could not restore experiment tracker: {e}")

    # Restore original callbacks
    if hasattr(gepa_instance, '_original_callbacks'):
        try:
            import dspy
            dspy.configure(callbacks=gepa_instance._original_callbacks)
            del gepa_instance._original_callbacks
            if hasattr(gepa_instance, '_gepa_callback'):
                del gepa_instance._gepa_callback
            logger.info("Restored original DSPy callbacks")
        except Exception as e:
            logger.warning(f"Could not restore callbacks: {e}")
