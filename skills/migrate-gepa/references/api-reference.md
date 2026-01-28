# API Reference

Complete reference for gepa-observable parameters, observer protocol, and event types.

## GEPA Class Parameters

### Budget Options (exactly one required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `auto` | `"light" \| "medium" \| "heavy"` | Preset budget - "light" (6 candidates), "medium" (12), or "heavy" (18) |
| `max_full_evals` | `int` | Maximum number of full validation evaluations |
| `max_metric_calls` | `int` | Maximum total metric calls |

### DSPy GEPA Parameters (100% compatible)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metric` | Required | 5-argument GEPA feedback metric function |
| `reflection_lm` | `None` | Language model for reflection (recommend strong model like gpt-5.2) |
| `reflection_minibatch_size` | `3` | Examples per reflection step |
| `candidate_selection_strategy` | `"pareto"` | `"pareto"` or `"current_best"` |
| `skip_perfect_score` | `True` | Skip reflection if perfect score achieved |
| `add_format_failure_as_feedback` | `False` | Add format failures as feedback |
| `component_selector` | `"round_robin"` | `"round_robin"` or `"all"` |
| `use_merge` | `True` | Enable merge-based optimization |
| `max_merge_invocations` | `5` | Maximum merge attempts |
| `num_threads` | `None` | Threads for parallel evaluation |
| `failure_score` | `0.0` | Score for failed evaluations |
| `perfect_score` | `1.0` | Perfect score value |
| `seed` | `0` | Random seed for reproducibility |
| `log_dir` | `None` | Directory for saving state |
| `track_stats` | `False` | Track optimization statistics |
| `track_best_outputs` | `False` | Track best outputs per validation task |
| `warn_on_score_mismatch` | `True` | Warn on score/feedback mismatch |
| `enable_tool_optimization` | `False` | Jointly optimize ReAct tool descriptions |
| `gepa_kwargs` | `None` | Additional kwargs passed to gepa.optimize |

### External Logging Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_wandb` | `False` | Enable Weights & Biases logging |
| `wandb_api_key` | `None` | W&B API key |
| `wandb_init_kwargs` | `None` | W&B init kwargs |
| `use_mlflow` | `False` | Enable MLflow logging |
| `mlflow_tracking_uri` | `None` | MLflow tracking URI |
| `mlflow_experiment_name` | `None` | MLflow experiment name |

### Observable Parameters (NEW in gepa-observable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server_url` | `None` | Web dashboard URL - enables ServerObserver |
| `project_name` | `"GEPA Run"` | Project name for dashboard |
| `run_name` | `None` | Run name (auto-generated if None) |
| `verbose` | `True` | Auto-creates LoggingObserver for console output |
| `capture_lm_calls` | `True` | Capture LM calls to dashboard |
| `capture_stdout` | `True` | Capture stdout to dashboard |
| `observers` | `None` | Custom GEPAObserver instances |
| `mlflow_tracing` | `False` | Enable MLflow tracing spans |

## GEPAObserver Protocol

Implement any subset of these methods to receive callbacks. All methods are optional.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class GEPAObserver(Protocol):
    def on_seed_validation(self, event: SeedValidationEvent) -> None: ...
    def on_iteration_start(self, event: IterationStartEvent) -> None: ...
    def on_minibatch_eval(self, event: MiniBatchEvalEvent) -> None: ...
    def on_reflection(self, event: ReflectionEvent) -> None: ...
    def on_acceptance_decision(self, event: AcceptanceDecisionEvent) -> None: ...
    def on_valset_eval(self, event: ValsetEvalEvent) -> None: ...
    def on_merge(self, event: MergeEvent) -> None: ...
    def on_optimization_complete(self, event: OptimizationCompleteEvent) -> None: ...
```

## Event Dataclasses

### SeedValidationEvent

Fired after the seed candidate is evaluated on the validation set.

```python
@dataclass
class SeedValidationEvent:
    seed_candidate: dict[str, str]           # Initial prompt instructions
    valset_scores: dict[Any, float]          # Score per validation example
    valset_outputs: dict[Any, Any]           # Output per validation example
    total_evals: int                          # Total evaluations so far
    valset_feedbacks: dict[Any, str | None] | None  # Per-example feedback
```

### IterationStartEvent

Fired at the start of each optimization iteration.

```python
@dataclass
class IterationStartEvent:
    iteration: int                    # Current iteration number (0-indexed)
    selected_candidate_idx: int       # Index of selected parent candidate
    selected_candidate: dict[str, str]  # Parent candidate instructions
    parent_score: float               # Parent candidate's score
```

### MiniBatchEvalEvent

Fired after evaluating a candidate on a mini-batch. Fires twice per iteration:
1. When evaluating the parent candidate (`is_new_candidate=False`)
2. When evaluating the proposed new candidate (`is_new_candidate=True`)

```python
@dataclass
class MiniBatchEvalEvent:
    iteration: int                    # Current iteration
    candidate_idx: int                # Candidate being evaluated
    candidate: dict[str, str]         # Candidate instructions
    batch_ids: list[Any]              # IDs of examples in batch
    scores: list[float]               # Scores for each example
    outputs: list[Any]                # Outputs for each example
    trajectories: list[Any] | None    # Execution traces (if available)
    is_new_candidate: bool            # False=parent, True=new candidate
    feedbacks: list[str | None] | None  # Per-example feedback
```

### ReflectionEvent

Fired after reflection proposes new text for components.

```python
@dataclass
class ReflectionEvent:
    iteration: int                    # Current iteration
    parent_candidate_idx: int         # Parent candidate index
    components_to_update: list[str]   # Component names being updated
    reflective_dataset: dict[str, list[dict[str, Any]]]  # Input to reflection
    proposed_texts: dict[str, str]    # Proposed new instructions
```

### AcceptanceDecisionEvent

Fired after the acceptance decision is made for a proposed candidate.

```python
@dataclass
class AcceptanceDecisionEvent:
    iteration: int              # Current iteration
    parent_score_sum: float     # Sum of parent scores on minibatch
    new_score_sum: float        # Sum of new candidate scores
    accepted: bool              # Whether new candidate was accepted
    proceed_to_valset: bool     # Whether proceeding to full validation
```

### ValsetEvalEvent

Fired after evaluating a candidate on the full validation set.

```python
@dataclass
class ValsetEvalEvent:
    iteration: int                    # Current iteration
    candidate_idx: int                # Candidate being evaluated
    candidate: dict[str, str]         # Candidate instructions
    val_ids: list[Any]                # Validation example IDs
    scores: dict[Any, float]          # Score per validation example
    outputs: dict[Any, Any]           # Output per validation example
    is_new_best: bool                 # Whether this is new best candidate
    valset_score: float               # Average validation score
    feedbacks: dict[Any, str | None] | None  # Per-example feedback
```

### MergeEvent

Fired when a merge operation is attempted.

```python
@dataclass
class MergeEvent:
    iteration: int                        # Current iteration
    parent_candidate_ids: list[int]       # Candidates being merged
    merged_candidate: dict[str, str]      # Resulting merged candidate
    subsample_scores_before: list[float] | None  # Scores before merge
    subsample_scores_after: list[float] | None   # Scores after merge
    accepted: bool                        # Whether merge was accepted
```

### OptimizationCompleteEvent

Fired when optimization completes.

```python
@dataclass
class OptimizationCompleteEvent:
    total_iterations: int             # Total iterations completed
    total_evals: int                  # Total metric evaluations
    best_candidate_idx: int           # Index of best candidate
    best_score: float                 # Best candidate's score
    best_candidate: dict[str, str]    # Best candidate instructions
```

## Built-in Observers

### LoggingObserver

Console output with configurable verbosity.

```python
from gepa_observable import LoggingObserver

observer = LoggingObserver(
    verbose=True,        # Print detailed per-event logs
    show_prompts=False,  # Show full prompt text in reflections
)

# After optimization, get summary
summary = observer.get_summary()
# Returns: {
#     "total_iterations": int,
#     "total_reflections": int,
#     "accepted_candidates": list[int],
#     "seed_avg_score": float,
# }
```

### ServerObserver

Dashboard integration with LM call capture.

```python
from gepa_observable import ServerObserver

# Factory method (recommended)
observer = ServerObserver.create(
    server_url="http://localhost:3000",
    trainset=train_data,
    valset=val_data,
    project_name="My Project",
    run_name="Experiment 1",
    capture_lm_calls=True,
    capture_stdout=True,
)

# Or manual setup
observer = ServerObserver(
    server_url="http://localhost:3000",
    project_name="My Project",
    capture_lm_calls=True,
)
observer.set_examples(train_data, val_data)

# Get LM logger for manual DSPy integration
lm_logger = observer.get_lm_logger()
if lm_logger:
    lm = dspy.LM("openai/gpt-5.2", callbacks=[lm_logger])
```

## Observer Composition

Multiple observers work together - all receive the same events:

```python
from gepa_observable import GEPA, LoggingObserver

class MetricsObserver:
    def __init__(self):
        self.scores = []

    def on_valset_eval(self, event):
        self.scores.append(event.valset_score)

class AlertObserver:
    def on_valset_eval(self, event):
        if event.is_new_best:
            send_slack_notification(f"New best: {event.valset_score:.2%}")

metrics = MetricsObserver()
alerts = AlertObserver()

optimizer = GEPA(
    metric=my_metric,
    auto="medium",
    server_url="http://localhost:3000",  # ServerObserver
    verbose=True,                         # LoggingObserver
    observers=[metrics, alerts],          # Custom observers
)

# All 4 observers receive all events:
# 1. ServerObserver (from server_url)
# 2. LoggingObserver (from verbose=True)
# 3. MetricsObserver (from observers list)
# 4. AlertObserver (from observers list)
```

## Metric Function Signature

GEPA metrics must accept 5 arguments:

```python
def my_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Args:
        gold: The ground truth Example
        pred: The model Prediction
        trace: Full execution trace (may be None)
        pred_name: Name of the predictor (for predictor-level feedback)
        pred_trace: Predictor-specific trace

    Returns:
        Either:
        - A float score (0.0 to 1.0)
        - A dspy.Prediction with 'score' field
        - A dspy.Prediction with 'score' and 'feedback' fields
    """
    is_correct = pred.answer.lower() == gold.answer.lower()

    # Return with feedback for better reflection
    return dspy.Prediction(
        score=1.0 if is_correct else 0.0,
        feedback="Correct!" if is_correct else f"Expected '{gold.answer}', got '{pred.answer}'"
    )
```
