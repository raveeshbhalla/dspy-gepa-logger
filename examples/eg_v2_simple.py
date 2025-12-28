"""Example using gepa-observable with first-class observer callbacks.

This example shows how to use the gepa-observable fork which provides:

1. GEPAObserver protocol - Receive callbacks at every stage of optimization
2. Event dataclasses - Typed events for seed validation, iterations, reflections, etc.
3. MLflow tracing (optional) - Hierarchical spans with GEPA context attributes

Optionally connects to the web dashboard for real-time monitoring.
Set USE_SERVER=True or pass --server flag to enable.
"""

import argparse
import dspy
from dspy.teleprompt import GEPA
from dotenv import load_dotenv
import io
import json
import os
import random
import sys
import threading
import time
import uuid
from typing import Any
import pandas as pd

# Import from gepa-observable
from gepa_observable import (
    optimize,
    GEPAObserver,
    SeedValidationEvent,
    IterationStartEvent,
    MiniBatchEvalEvent,
    ReflectionEvent,
    AcceptanceDecisionEvent,
    ValsetEvalEvent,
    OptimizationCompleteEvent,
    MergeEvent,
)

# Import server client for web dashboard integration
from dspy_gepa_logger.server.client import ServerClient
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger


def serialize_output(output):
    """Serialize an output object to JSON-safe format.

    For DSPy predictions, extracts the actual fields (reasoning, answer, etc.)
    instead of internal attributes like _completions.
    """
    if output is None:
        return None
    try:
        # For DSPy predictions, extract public fields (skip _prefixed ones)
        if hasattr(output, "__dict__"):
            fields = {k: v for k, v in output.__dict__.items() if not k.startswith("_")}
            if fields:
                return json.dumps(fields, default=str)
            # Fallback to full dict if no public fields
            return json.dumps(output.__dict__, default=str)
        elif hasattr(output, "toDict"):
            return json.dumps(output.toDict(), default=str)
        else:
            return json.dumps(output, default=str)
    except (TypeError, ValueError):
        return str(output)


def serialize_example_inputs(example):
    """Serialize example inputs to a dict for display.

    Args:
        example: The DSPy example object

    Returns:
        Dict of input field names to values, or None if serialization fails
    """
    if example is None:
        return None
    try:
        if hasattr(example, "inputs") and callable(example.inputs):
            # DSPy Example with .inputs() method - get only input fields
            input_keys = example.inputs()
            return {k: getattr(example, k, None) for k in input_keys}
        elif hasattr(example, "toDict") and callable(example.toDict):
            return example.toDict()
        elif hasattr(example, "__dict__"):
            return {k: v for k, v in example.__dict__.items() if not k.startswith("_")}
        else:
            return {"value": str(example)}
    except Exception:
        return None


class LogCapture(io.TextIOBase):
    """Captures stdout and sends to server while preserving original output."""

    def __init__(self, original_stream, server_client, stream_type: str = "stdout"):
        super().__init__()
        self._original = original_stream
        self._client = server_client
        self._stream_type = stream_type
        self._buffer = []
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        if not s:
            return 0

        # Always write to original stream
        result = self._original.write(s)

        # Buffer and push to server
        with self._lock:
            self._buffer.append(s)
            # Flush on newlines
            if '\n' in s and self._client and self._client.is_connected:
                content = "".join(self._buffer)
                self._buffer = []
                try:
                    self._client.push_logs([{
                        "logType": self._stream_type,
                        "content": content,
                        "timestamp": time.time(),
                    }])
                except Exception:
                    pass  # Don't fail if server push fails

        return len(s) if result is None else result

    def flush(self):
        self._original.flush()
        with self._lock:
            if self._buffer and self._client and self._client.is_connected:
                content = "".join(self._buffer)
                self._buffer = []
                try:
                    self._client.push_logs([{
                        "logType": self._stream_type,
                        "content": content,
                        "timestamp": time.time(),
                    }])
                except Exception:
                    pass

    def fileno(self):
        return self._original.fileno()

    @property
    def encoding(self):
        return self._original.encoding

    def isatty(self):
        return self._original.isatty()

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False


class Prompt(dspy.Signature):
    """Answer the following question"""

    question: str = dspy.InputField(desc="Question")
    answer: str = dspy.OutputField(desc="Your answer")


program = dspy.ChainOfThought(Prompt)


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric that returns score and feedback.

    GEPA requires 5 arguments: (gold, pred, trace, pred_name, pred_trace).
    """
    if pred.answer.lower() == gold.answer.lower():
        return dspy.Prediction(score=random.uniform(0.7, 1.0), feedback="Great work!")
    else:
        return dspy.Prediction(score=random.uniform(0.0, 0.2), feedback="Nice try")


def get_data():
    """Read eg.csv, convert into dspy.Examples, shuffle and split 15/5 into train and val set"""
    df = pd.read_csv("eg.csv")
    examples = [
        dspy.Example(question=row["question"], answer=row["answer"]).with_inputs("question")
        for _, row in df.iterrows()
    ]
    random.shuffle(examples)
    return examples[:15], examples[15:20]


class LoggingObserver:
    """Example observer that logs all GEPA optimization events."""

    def __init__(self):
        self.seed_scores = []
        self.iterations = []
        self.reflections = []
        self.accepted_candidates = []

    def on_seed_validation(self, event: SeedValidationEvent):
        avg_score = sum(event.valset_scores.values()) / len(event.valset_scores)
        print(f"\n[Seed] Validated seed candidate: avg score = {avg_score:.2%}")
        print(f"       Total evals: {event.total_evals}")
        self.seed_scores = list(event.valset_scores.values())

    def on_iteration_start(self, event: IterationStartEvent):
        print(f"\n[Iter {event.iteration}] Starting iteration")
        print(f"         Selected candidate {event.selected_candidate_idx} (score: {event.parent_score:.2%})")
        self.iterations.append(event.iteration)

    def on_minibatch_eval(self, event: MiniBatchEvalEvent):
        avg_score = sum(event.scores) / len(event.scores)
        candidate_type = "NEW" if event.is_new_candidate else "parent"
        print(f"         [{candidate_type}] Minibatch eval: avg = {avg_score:.2%} (n={len(event.scores)})")
        if event.trajectories:
            print(f"                   Trajectories captured: {len(event.trajectories)}")

    def on_reflection(self, event: ReflectionEvent):
        print(f"         [Reflection] Components updated: {event.components_to_update}")
        for name, text in event.proposed_texts.items():
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"                       {name}: {preview}")
        self.reflections.append(event)

    def on_acceptance_decision(self, event: AcceptanceDecisionEvent):
        status = "ACCEPTED" if event.accepted else "REJECTED"
        print(f"         [Decision] {status}: parent={event.parent_score_sum:.2f} vs new={event.new_score_sum:.2f}")
        if event.proceed_to_valset:
            print(f"                    Proceeding to full valset evaluation")

    def on_valset_eval(self, event: ValsetEvalEvent):
        is_best = " (NEW BEST!)" if event.is_new_best else ""
        print(f"         [Valset] Candidate {event.candidate_idx}: score = {event.valset_score:.2%}{is_best}")
        if event.is_new_best:
            self.accepted_candidates.append(event.candidate_idx)

    def on_optimization_complete(self, event: OptimizationCompleteEvent):
        print(f"\n{'='*60}")
        print(f"[Complete] Optimization finished!")
        print(f"           Total iterations: {event.total_iterations}")
        print(f"           Total evaluations: {event.total_evals}")
        print(f"           Best candidate: {event.best_candidate_idx} (score: {event.best_score:.2%})")
        print(f"{'='*60}")

    def get_summary(self):
        return {
            "total_iterations": len(self.iterations),
            "total_reflections": len(self.reflections),
            "accepted_candidates": self.accepted_candidates,
            "seed_avg_score": sum(self.seed_scores) / len(self.seed_scores) if self.seed_scores else 0,
        }


class ServerObserver:
    """Observer that sends GEPA events to the web dashboard server.

    Tracks state across the optimization run and sends structured data
    to the web dashboard for visualization.
    """

    def __init__(self, server_url: str, project_name: str = "GEPA Observable"):
        self.client = ServerClient(server_url, project_name=project_name)
        self.run_id = None

        # State tracking
        self.seed_candidate = None
        self.seed_score = None
        self.total_evals = 0
        self.num_candidates = 1  # Start with seed candidate
        self.candidates_pushed = set()
        self.pareto_candidates = set()  # Track candidates on pareto front

        # Per-iteration tracking
        self.current_iteration = -1
        self.iteration_evals = 0
        self.iteration_parent_idx = None
        self.iteration_child_idxs = []
        self.iteration_reflection_input = None
        self.iteration_reflection_output = None
        self.iteration_proposed_changes = []
        self.iteration_parent_score = 0.0
        self.iteration_new_score = 0.0
        self.iteration_accepted = False

        # LM call tracking
        self.lm_call_count = 0

        # Example lookup maps (set by set_examples)
        self._trainset_map: dict[Any, Any] = {}
        self._valset_map: dict[Any, Any] = {}

    def set_examples(self, trainset: list, valset: list):
        """Store example references for input serialization.

        Call this before optimization starts to enable input capture.
        Examples are indexed by their position (0, 1, 2, ...) which matches
        how GEPA assigns batch_ids.
        """
        self._trainset_map = {i: ex for i, ex in enumerate(trainset)}
        self._valset_map = {i: ex for i, ex in enumerate(valset)}

    def _get_example_inputs(self, example_id: Any) -> dict | None:
        """Look up and serialize example inputs by ID."""
        # Try valset first (most common), then trainset
        example = self._valset_map.get(example_id) or self._trainset_map.get(example_id)
        if example is not None:
            return serialize_example_inputs(example)
        return None

    def start_run(self, seed_candidate: dict[str, str], valset_ids: list[str] | None = None):
        """Start a new run on the server."""
        self.seed_candidate = seed_candidate
        self.run_id = self.client.start_run(
            config={"observer_type": "gepa-observable"},
            seed_prompt=seed_candidate,
            name=f"GEPA Run {time.strftime('%Y-%m-%d %H:%M:%S')}",
            valset_example_ids=valset_ids,
        )
        if self.run_id:
            # Push seed candidate (index 0)
            self.client.push_candidates([
                (0, seed_candidate, None, 0)
            ])
            self.candidates_pushed.add(0)
            print(f"[Server] Started run: {self.run_id}")
        return self.run_id

    def _push_current_iteration(self):
        """Push accumulated iteration data to the server."""
        if self.current_iteration < 0 or not self.client.is_connected:
            return

        # Use 1-indexed iteration numbers for the UI (GEPA uses 0-indexed internally)
        ui_iteration = self.current_iteration + 1

        # Build pareto_programs dict - maps example IDs to candidate indices on pareto front
        pareto_programs = {str(i): list(self.pareto_candidates) for i in range(5)} if self.pareto_candidates else None

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

    def _reset_iteration_state(self):
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

    def on_seed_validation(self, event: SeedValidationEvent):
        if not self.client.is_connected:
            return

        self.seed_score = (sum(event.valset_scores.values()) / len(event.valset_scores)) if event.valset_scores else None
        self.total_evals = event.total_evals
        self.pareto_candidates.add(0)  # Seed starts on pareto front

        # Push seed evaluations
        evaluations = []
        for example_id, score in event.valset_scores.items():
            output = event.valset_outputs.get(example_id)
            # Get feedback from the event if available
            feedback = event.valset_feedbacks.get(example_id) if event.valset_feedbacks else None
            evaluations.append({
                "evalId": str(uuid.uuid4()),
                "exampleId": str(example_id),
                "candidateIdx": 0,
                "iteration": None,  # NULL for seed/baseline evaluations (frontend expects this)
                "phase": "seed_validation",
                "score": score,
                "feedback": feedback,
                "exampleInputs": self._get_example_inputs(example_id),
                "predictionPreview": str(output) if output else None,  # Full output, no truncation
                "predictionRef": serialize_output(output),  # Full JSON serialization
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
            pareto_programs={"0": [0]},  # Seed is on pareto front
        )

    def on_iteration_start(self, event: IterationStartEvent):
        if not self.client.is_connected:
            return

        # If we have a previous iteration, push its final state
        if self.current_iteration >= 0:
            self._push_current_iteration()

        # Start new iteration
        self._reset_iteration_state()
        self.current_iteration = event.iteration
        self.iteration_parent_idx = event.selected_candidate_idx
        self.iteration_parent_score = event.parent_score

    def on_minibatch_eval(self, event: MiniBatchEvalEvent):
        if not self.client.is_connected:
            return

        # Track evaluations
        num_evals = len(event.scores)
        self.iteration_evals += num_evals
        self.total_evals += num_evals

        # Track scores for parent vs new comparison
        batch_sum = sum(event.scores)
        if event.is_new_candidate:
            self.iteration_new_score = batch_sum
        else:
            self.iteration_parent_score = batch_sum

        # Use 1-indexed iteration for UI
        ui_iteration = event.iteration + 1

        # Push evaluations for this minibatch
        evaluations = []
        for i, (batch_id, score) in enumerate(zip(event.batch_ids, event.scores)):
            output = event.outputs[i] if i < len(event.outputs) else None
            # Get feedback directly from the event
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
                "predictionPreview": str(output) if output else None,  # Full output, no truncation
                "predictionRef": serialize_output(output),  # Full JSON serialization
                "timestamp": time.time(),
            })
        self.client.push_evaluations(evaluations)

    def on_reflection(self, event: ReflectionEvent):
        if not self.client.is_connected:
            return

        # Store reflection data for iteration push (full content, no truncation)
        self.iteration_reflection_input = json.dumps(event.reflective_dataset, default=str)
        self.iteration_reflection_output = json.dumps(event.proposed_texts, default=str)
        self.iteration_proposed_changes = [
            {"component": name, "newText": text}  # Full text, no truncation
            for name, text in event.proposed_texts.items()
        ]
        # Note: Feedback is now captured directly from MiniBatchEvalEvent.feedbacks
        # and ValsetEvalEvent.feedbacks, so we don't need to extract from reflective_dataset

    def on_acceptance_decision(self, event: AcceptanceDecisionEvent):
        if not self.client.is_connected:
            return

        self.iteration_accepted = event.accepted

        # Use 1-indexed iteration for UI
        ui_iteration = event.iteration + 1

        # Log acceptance/rejection with parent vs new comparison
        self.client.push_logs([{
            "logType": "info",
            "content": f"{'ACCEPTED' if event.accepted else 'REJECTED'}: parent={event.parent_score_sum:.2f} vs new={event.new_score_sum:.2f}",
            "timestamp": time.time(),
            "iteration": ui_iteration,
            "phase": "acceptance_decision",
        }])

    def on_valset_eval(self, event: ValsetEvalEvent):
        if not self.client.is_connected:
            return

        # Track evaluations
        num_evals = len(event.scores)
        self.total_evals += num_evals

        # Track pareto candidates - if is_new_best, this candidate is on the pareto front
        if event.is_new_best:
            self.pareto_candidates.add(event.candidate_idx)

        # Push new candidate if not already pushed
        if event.candidate_idx not in self.candidates_pushed:
            # Find parent idx from current iteration tracking
            parent_idx = self.iteration_parent_idx if self.iteration_parent_idx is not None else 0
            self.client.push_candidates([
                (event.candidate_idx, event.candidate, parent_idx, event.iteration + 1)
            ])
            self.candidates_pushed.add(event.candidate_idx)
            self.num_candidates += 1
            self.iteration_child_idxs.append(event.candidate_idx)

        # Use 1-indexed iteration for UI
        ui_iteration = event.iteration + 1

        # Push valset evaluations
        evaluations = []
        for val_id, score in event.scores.items():
            output = event.outputs.get(val_id)
            # Get feedback from the event if available
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
                "predictionPreview": str(output) if output else None,  # Full output, no truncation
                "predictionRef": serialize_output(output),  # Full JSON serialization
                "timestamp": time.time(),
            })
        self.client.push_evaluations(evaluations)

    def on_merge(self, event: MergeEvent):
        # Optional: Log merge events
        pass

    def on_lm_call(self, call):
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
            "phase": call.phase,  # Use phase from the LM logger context
            "candidateIdx": call.candidate_idx,
            "inputs": call.inputs,
            "outputs": call.outputs,
        }])

    def on_optimization_complete(self, event: OptimizationCompleteEvent):
        if not self.client.is_connected:
            return

        # Push final iteration state
        if self.current_iteration >= 0:
            self._push_current_iteration()

        # Complete the run
        self.client.complete_run(
            status="COMPLETED",
            best_prompt=event.best_candidate,
            best_candidate_idx=event.best_candidate_idx,
            best_score=event.best_score,
            seed_score=self.seed_score,
        )
        print(f"[Server] Run completed: {self.run_id}")


def main(use_server: bool = False, server_url: str = "http://localhost:3000", use_mlflow: bool = False):
    # Configure DSPy
    load_dotenv(".env.local")

    # Get LM configuration from environment variables
    lm_model = os.getenv("DSPY_LM", "openai/gpt-4o-mini")
    reflective_lm_model = os.getenv("DSPY_REFLECTIVE_LM", "openai/gpt-4o")

    # Create observers first (needed for LM logger callback)
    observers = []
    logging_observer = LoggingObserver()
    observers.append(logging_observer)

    # Add server observer if enabled
    server_observer = None
    lm_logger = None
    original_stdout = None
    log_capture = None

    if use_server:
        server_observer = ServerObserver(server_url, project_name="GEPA Observable")
        observers.append(server_observer)

        # Setup LM call capture
        lm_logger = DSPyLMLogger(on_call_complete=server_observer.on_lm_call)

    # Configure LM with callbacks for LM call capture
    # Callbacks must be passed directly to the LM constructor, not via dspy.configure()
    if lm_logger is not None:
        lm = dspy.LM(lm_model, temperature=1.0, callbacks=[lm_logger])
        _reflective_lm_base = dspy.LM(reflective_lm_model, temperature=1.0, callbacks=[lm_logger])
    else:
        lm = dspy.LM(lm_model, temperature=1.0)
        _reflective_lm_base = dspy.LM(reflective_lm_model, temperature=1.0)

    # Wrap reflective_lm to return a string (GEPA expects lm(prompt) -> str, not list)
    # dspy.LM returns a list of strings, so we wrap it to extract the first element
    def reflective_lm(prompt: str) -> str:
        result = _reflective_lm_base(prompt)
        if isinstance(result, list):
            return result[0] if result else ""
        return result

    dspy.configure(lm=lm)

    print("=" * 60)
    print("GEPA Observable - First-Class Observer Callbacks")
    if use_mlflow:
        print("MLflow tracing enabled")
    if use_server:
        print(f"Server mode enabled: {server_url}")
    print("=" * 60)

    # Get data
    train_data, val_data = get_data()

    print("\nRunning GEPA optimization with observers...")
    print("-" * 40)

    # Create DSPy adapter for GEPA (using our forked version with feedback support)
    from gepa_observable.adapters.dspy_adapter.dspy_adapter import DspyAdapter

    # Create feedback map for predictors
    # DspyAdapter calls feedback_fn with: predictor_output, predictor_inputs,
    # module_inputs, module_outputs, captured_trace
    def feedback_fn_creator(pred_name, predictor):
        def feedback_fn(
            predictor_output=None,
            predictor_inputs=None,
            module_inputs=None,
            module_outputs=None,
            captured_trace=None,
            module_score=None,  # The score already computed by the metric
            **kwargs
        ):
            # module_inputs is the gold example, module_outputs is the prediction
            # module_score is passed from make_reflective_dataset and must be returned as-is
            if module_outputs is None:
                return {"feedback": "No output", "score": 0.0}

            answer = getattr(module_outputs, "answer", "")
            gold_answer = getattr(module_inputs, "answer", "") if module_inputs else ""

            # Use the already-computed score from the metric, but generate feedback
            # GEPA requires the feedback function to return the same score as the module metric
            if module_score is not None:
                score = module_score
            else:
                # Fallback if module_score not provided (shouldn't happen in GEPA flow)
                score = 1.0 if answer.lower() == gold_answer.lower() else 0.0

            if answer.lower() == gold_answer.lower():
                return {"feedback": "Great work!", "score": score}
            else:
                return {"feedback": "Nice try", "score": score}
        return feedback_fn

    feedback_map = {k: feedback_fn_creator(k, v) for k, v in program.named_predictors()}

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=metric_with_feedback,
        feedback_map=feedback_map,
        failure_score=0.0,
        num_threads=4,
    )

    # Build seed candidate from program's predictor instructions
    seed_candidate = {name: pred.signature.instructions for name, pred in program.named_predictors()}

    # Start server run BEFORE optimization (so we can track seed validation)
    if server_observer:
        valset_ids = [str(i) for i in range(len(val_data))]
        server_observer.set_examples(train_data, val_data)  # Enable input capture
        server_observer.start_run(seed_candidate, valset_ids)

        # Setup stdout capture AFTER server connection is established
        original_stdout = sys.stdout
        log_capture = LogCapture(original_stdout, server_observer.client)
        sys.stdout = log_capture

    # Run optimization using gepa-observable
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,
        adapter=adapter,
        reflection_lm=reflective_lm,
        reflection_minibatch_size=5,
        max_metric_calls=100,  # Budget
        skip_perfect_score=False,
        run_dir="logs",
        # Observer support (gepa-observable addition)
        observers=observers,
        mlflow_tracing=use_mlflow,
        # MLflow integration (optional)
        use_mlflow=use_mlflow,
    )

    # Restore stdout before printing final results
    if original_stdout:
        sys.stdout = original_stdout
        if log_capture:
            log_capture.flush()

    # ==================== RESULTS ====================

    print("\n" + "=" * 60)
    print("OBSERVER SUMMARY")
    print("=" * 60)

    summary = logging_observer.get_summary()
    print(f"Total iterations observed: {summary['total_iterations']}")
    print(f"Total reflections: {summary['total_reflections']}")
    print(f"Seed avg score: {summary['seed_avg_score']:.2%}")
    print(f"Accepted candidates: {summary['accepted_candidates']}")

    # Get best candidate from result
    print(f"\nBest candidate index: {result.best_idx}")
    print(f"Best candidate score: {result.val_aggregate_scores[result.best_idx]:.2%}")

    if server_observer:
        print(f"Total LM calls captured: {server_observer.lm_call_count}")

    print("\n" + "=" * 60)
    print("Done!")
    if use_server:
        print(f"View run at: {server_url}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEPA optimization with observer callbacks")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Enable web dashboard server integration",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:3000",
        help="Web dashboard server URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracing for hierarchical LM call spans",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing GEPA state and start fresh optimization",
    )
    args = parser.parse_args()

    # Clear GEPA state if --fresh flag is set
    if args.fresh:
        import shutil
        gepa_state_path = os.path.join("logs", "gepa_state.bin")
        if os.path.exists(gepa_state_path):
            os.remove(gepa_state_path)
            print("Cleared existing GEPA state")
        # Also clear the generated outputs directory
        outputs_dir = os.path.join("logs", "generated_best_outputs_valset")
        if os.path.exists(outputs_dir):
            shutil.rmtree(outputs_dir)
            print("Cleared generated outputs directory")

    main(use_server=args.server, server_url=args.server_url, use_mlflow=args.mlflow)
