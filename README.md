# gepa-observable

Observable GEPA optimizer for DSPy with real-time web dashboard integration. Drop-in replacement for `dspy.GEPA` with built-in observability.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/raveeshbhalla/dspy-gepa-logger)

## Features

- **DSPy-Compatible API**: `GEPA` Teleprompter class with `compile()` method, matching `dspy.GEPA`
- **Full DSPy GEPA Parity**: Supports `auto`, `max_full_evals`, `max_metric_calls`, `gepa_kwargs`, etc.
- **Observer Pattern Architecture**: First-class observer callbacks for complete optimization visibility
- **Web Dashboard Integration**: Real-time monitoring with just `server_url` parameter
- **LM Call Capture**: Automatic capture of all LM calls with context tags
- **MLflow Integration**: Optional hierarchical tracing with GEPA context attributes

## Installation

```bash
pip install dspy-gepa-logger
```

Or from source:

```bash
git clone https://github.com/raveeshbhalla/dspy-gepa-logger.git
cd dspy-gepa-logger
pip install -e .
```

## Claude Code Plugin (Migration Helper)

If you're using Claude Code, install our migration skill to get AI-assisted help migrating your existing DSPy GEPA code.

### Quick Install

```bash
/plugin marketplace add raveeshbhalla/dspy-gepa-logger
```

Then select "Browse and install plugins" and install `gepa-observable-migration`.

### What It Does

Once installed, Claude Code will automatically help when you ask about:
- Migrating from DSPy GEPA to gepa-observable
- Adding observability to GEPA optimization
- Setting up dashboard monitoring for GEPA
- Creating custom observers

Example prompts:
- "Help me migrate my notebook from dspy.GEPA to gepa-observable"
- "Add web dashboard logging to my GEPA optimization script"
- "Set up custom observers for my GEPA run"

## Quick Start

```python
import dspy
from gepa_observable import GEPA

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define your program
class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

program = dspy.ChainOfThought(MySignature)

# Define your metric (GEPA requires 5 arguments)
def my_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    is_correct = pred.answer.lower() == gold.answer.lower()
    return dspy.Prediction(
        score=1.0 if is_correct else 0.0,
        feedback="Correct!" if is_correct else f"Expected {gold.answer}"
    )

# Create optimizer with dashboard integration
optimizer = GEPA(
    metric=my_metric,
    auto="medium",  # or max_full_evals=10, or max_metric_calls=500
    server_url="http://localhost:3000",  # Enables web dashboard
    project_name="My Project",
)

# Compile (optimize) - just like dspy.GEPA!
optimized = optimizer.compile(
    student=program,
    trainset=train_data,
    valset=val_data,
)

# Use your optimized program
result = optimized(question="What is 2+2?")
print(result.answer)
```

## API Reference

### Budget Options (exactly one required)

| Parameter | Description |
|-----------|-------------|
| `auto` | `"light"` (6 candidates), `"medium"` (12), or `"heavy"` (18) - matches DSPy's GEPA |
| `max_full_evals` | Maximum number of full validation evaluations |
| `max_metric_calls` | Maximum total metric calls |

### DSPy GEPA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metric` | Required | 5-argument GEPA feedback metric function |
| `reflection_lm` | `None` | LM for reflection (recommend strong model like gpt-4o) |
| `reflection_minibatch_size` | `3` | Examples per reflection step |
| `candidate_selection_strategy` | `"pareto"` | `"pareto"` or `"current_best"` |
| `skip_perfect_score` | `True` | Skip reflection if perfect score achieved |
| `use_merge` | `True` | Enable merge-based optimization |
| `num_threads` | `None` | Threads for parallel evaluation |
| `log_dir` | `None` | Directory for saving state |
| `track_stats` | `False` | Track optimization statistics |
| `enable_tool_optimization` | `False` | Jointly optimize ReAct tool descriptions |
| `gepa_kwargs` | `None` | Additional kwargs passed to gepa.optimize |

### Observable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server_url` | `None` | Web dashboard URL - enables ServerObserver |
| `project_name` | `"GEPA Run"` | Project name for dashboard |
| `run_name` | `None` | Run name (auto-generated if None) |
| `verbose` | `True` | Auto-creates LoggingObserver for console output |
| `capture_lm_calls` | `True` | Capture LM calls to dashboard |
| `capture_stdout` | `True` | Capture stdout to dashboard |
| `observers` | `None` | Custom GEPAObserver instances |

## Metric Functions

GEPA metrics must accept 5 arguments and return score + optional feedback:

```python
def my_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    # Return dspy.Prediction with score and feedback
    return dspy.Prediction(
        score=0.8,
        feedback="Almost correct - missing detail X"
    )

    # Or just return a float score
    return 0.8
```

## Custom Observers

Implement your own observer with any subset of callbacks:

```python
from gepa_observable import GEPA, GEPAObserver

class MyObserver:
    def on_seed_validation(self, event):
        print(f"Seed score: {sum(event.valset_scores.values())/len(event.valset_scores):.2%}")

    def on_iteration_start(self, event):
        print(f"Starting iteration {event.iteration}")

    def on_valset_eval(self, event):
        if event.is_new_best:
            print(f"New best: {event.valset_score:.2%}")

    def on_optimization_complete(self, event):
        print(f"Done! Best score: {event.best_score:.2%}")

optimizer = GEPA(
    metric=my_metric,
    max_metric_calls=100,
    observers=[MyObserver()],
    verbose=False,  # Disable auto-LoggingObserver
)
```

### Observer Events

| Event | Description |
|-------|-------------|
| `SeedValidationEvent` | Initial validation of seed candidate |
| `IterationStartEvent` | Start of each optimization iteration |
| `MiniBatchEvalEvent` | Minibatch evaluation (parent or new candidate) |
| `ReflectionEvent` | Reflection/proposal phase with proposed changes |
| `AcceptanceDecisionEvent` | Accept/reject decision for new candidate |
| `ValsetEvalEvent` | Full validation set evaluation |
| `MergeEvent` | Candidate merge attempts |
| `OptimizationCompleteEvent` | Optimization finished |

## Direct API (Advanced)

For more control, use `optimize()` directly with a custom adapter:

```python
from gepa_observable import optimize
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

adapter = DspyAdapter(
    student_module=program,
    metric_fn=my_metric,
    feedback_map={name: my_metric for name, _ in program.named_predictors()},
)
seed_candidate = {name: pred.signature.instructions
                  for name, pred in program.named_predictors()}

result = optimize(
    seed_candidate=seed_candidate,
    trainset=train_data,
    valset=val_data,
    adapter=adapter,
    reflection_lm="openai/gpt-4o",
    max_metric_calls=100,
    server_url="http://localhost:3000",
)
```

## Web Dashboard

For real-time monitoring with persistent history and project organization.

### Requirements

- Node.js 20.19+, 22.12+, or 24.0+

### Setup

```bash
cd web
npm install
echo 'DATABASE_URL="file:./dev.db"' > .env
npx prisma generate
npx prisma migrate deploy
npm run dev
```

Open http://localhost:3000 to view:

- **Projects**: Organize runs by project
- **Run History**: Browse all past optimization runs
- **Real-time Updates**: Watch ongoing runs with live stats
- **Evaluation Comparison**: Interactive tables showing improvements/regressions
- **Prompt Comparison**: Side-by-side view of original vs optimized prompts

## Example

See `examples/eg_v2_simple.py` for a complete working example:

```bash
cd examples
python eg_v2_simple.py --server
```

## Architecture

```
src/gepa_observable/
├── __init__.py            # Main exports: GEPA, optimize, observers
├── gepa.py                # GEPA Teleprompter class (DSPy-compatible)
├── api.py                 # optimize() function
├── observers.py           # GEPAObserver protocol, events, built-in observers
├── core/
│   ├── context.py         # Thread-safe context for phase tagging
│   ├── lm_logger.py       # DSPyLMLogger - LM call capture
│   ├── serialization.py   # JSON serialization utilities
│   └── engine.py          # GEPAEngine with observer notifications
├── server/
│   └── client.py          # ServerClient for dashboard integration
└── adapters/
    └── dspy_adapter/      # DSPy program optimization adapter
```

## Requirements

- Python >= 3.10
- dspy >= 2.5.0
- gepa >= 0.0.22
- requests (for web dashboard)
