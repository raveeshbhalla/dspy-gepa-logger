# dspy-gepa-logger

Lightweight logging and visualization for DSPy GEPA optimization runs. Track evaluations, compare prompts, and generate HTML reports - all using public hooks (no monkey-patching).

## Features

- **Public Hooks Architecture**: Uses DSPy callbacks and GEPA's `stop_callbacks` - no monkey-patching
- **In-Memory Capture**: Fast, lightweight data capture with no storage backend required
- **Evaluation Tracking**: Captures scores, feedback, and predictions for each evaluation
- **LM Call Logging**: Records all language model calls with context tags (eval, reflection, proposal)
- **Interactive HTML Reports**: Dark-themed reports with prompt comparison, performance analysis, and clickable rows for detailed evaluation view
- **Jupyter Support**: Display reports inline in notebooks

## Installation

```bash
pip install git+https://github.com/raveeshbhalla/dspy-gepa-logger.git
```

Or from source:

```bash
git clone https://github.com/raveeshbhalla/dspy-gepa-logger.git
cd dspy-gepa-logger
pip install -e .
```

## Quick Start

```python
import dspy
from dspy_gepa_logger import create_logged_gepa, configure_dspy_logging

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define your metric (can return score + feedback)
def my_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    is_correct = pred.answer.lower() == gold.answer.lower()
    score = 1.0 if is_correct else 0.0
    feedback = "Correct!" if is_correct else f"Expected {gold.answer}"
    return dspy.Prediction(score=score, feedback=feedback)

# Create GEPA with logging
gepa, tracker, logged_metric = create_logged_gepa(
    metric=my_metric,
    num_threads=4,
    max_full_evals=3,
)

# Enable LM call logging
configure_dspy_logging(tracker)

# Set validation set for performance comparison
tracker.set_valset(val_data)

# Run optimization
optimized = gepa.compile(
    student=my_program,
    trainset=train_data,
    valset=val_data,
)

# View results
tracker.print_summary()
tracker.print_prompt_diff()

# Export HTML report
tracker.export_html("report.html")
```

## API Reference

### `create_logged_gepa()`

Factory function that creates a GEPA optimizer with logging enabled.

```python
gepa, tracker, logged_metric = create_logged_gepa(
    metric=my_metric,              # Your metric function
    reflection_lm=None,            # Optional: LM for reflection (defaults to configured LM)
    num_threads=4,                 # Parallel evaluation threads
    max_full_evals=3,              # Budget for full evaluations
    track_stats=True,              # Enable statistics tracking
    log_dir=None,                  # Optional: directory for GEPA logs
    # ... other GEPA kwargs
)
```

Returns:
- `gepa`: Configured GEPA optimizer with logging hooks
- `tracker`: GEPATracker instance for accessing captured data
- `logged_metric`: Wrapped metric function (already passed to GEPA)

### `GEPATracker`

Main class for accessing optimization data.

#### Visualization Methods

```python
# Print formatted summary
tracker.print_summary()

# Print prompt comparison (original vs optimized)
tracker.print_prompt_diff()
tracker.print_prompt_diff(show_full=True)  # Show full text, not truncated

# Export HTML report
tracker.export_html("report.html")

# For Jupyter notebooks
from IPython.display import HTML, display
display(HTML(tracker.export_html()))
```

#### Data Access

```python
# Get structured report
report = tracker.get_optimization_report()
print(f"Candidates: {report['total_candidates']}")
print(f"Lineage: {report['lineage']}")

# Get evaluation comparison (improvements/regressions)
comparison = tracker.get_evaluation_comparison()
print(f"Improvements: {comparison['summary']['num_improvements']}")
print(f"Regressions: {comparison['summary']['num_regressions']}")

# Access raw data
tracker.evaluations          # All evaluation records
tracker.lm_calls             # All LM call records
tracker.final_candidates     # All candidate prompts
tracker.get_lineage(idx)     # Trace candidate ancestry
```

### Metric Functions

Your metric can return scores in several formats:

```python
# Simple score (float)
def metric(gold, pred, trace=None):
    return 1.0 if correct else 0.0

# Score with feedback (tuple)
def metric(gold, pred, trace=None):
    return (score, "Feedback text")

# dspy.Prediction with score and feedback
def metric(gold, pred, trace=None):
    return dspy.Prediction(score=0.8, feedback="Almost correct")
```

GEPA metrics use 5 arguments:

```python
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=score, feedback=feedback)
```

## HTML Report

The HTML report includes:

- **Summary Statistics**: Iterations, candidates, LM calls, evaluations
- **Lineage Visualization**: Click to navigate between candidates
- **Prompt Comparison**: Side-by-side original vs optimized prompts
- **Performance Comparison**: Tabbed view of improvements, regressions, and unchanged examples
  - Shows input, baseline/optimized outputs, and score delta
  - **Click any row** to open a modal with full evaluation details
  - Filterable by validation set

![HTML Report Screenshot](docs/report-screenshot.png)

## Example

See `examples/eg_v2_simple.py` for a complete working example:

```bash
cd examples
python eg_v2_simple.py
open optimization_report.html
```

## Architecture

```
dspy_gepa_logger/
├── core/
│   ├── tracker_v2.py      # GEPATracker - unified logging interface
│   ├── logged_metric.py   # LoggedMetric - evaluation capture
│   ├── state_logger.py    # GEPAStateLogger - iteration state via stop_callbacks
│   ├── lm_logger.py       # DSPyLMLogger - LM call capture via callbacks
│   ├── logged_proposer.py # Optional reflection/proposal phase tagging
│   └── context.py         # Thread-safe context for phase tagging
└── __init__.py            # Public API exports
```

### How It Works

1. **LoggedMetric** wraps your metric to capture evaluations with scores and feedback
2. **GEPAStateLogger** uses GEPA's `stop_callbacks` to capture iteration state incrementally
3. **DSPyLMLogger** uses DSPy's callback system to capture all LM calls with context tags
4. **GEPATracker** combines all hooks and provides query/visualization methods

No monkey-patching required - all hooks use public APIs.

## Requirements

- Python >= 3.10
- dspy >= 2.5.0
