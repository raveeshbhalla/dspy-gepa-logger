# Migration Examples

Complete before/after code examples for migrating from DSPy GEPA to gepa-observable.

## Minimal Migration (Import Change Only)

The simplest migration - just change the import. Everything else works identically.

### Before (DSPy GEPA)

```python
import dspy
from dspy.teleprompt import GEPA

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define program
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

program = dspy.ChainOfThought(QA)

# Define metric
def accuracy_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return 1.0 if pred.answer.lower() == gold.answer.lower() else 0.0

# Optimize
optimizer = GEPA(
    metric=accuracy_metric,
    auto="medium",
)
optimized = optimizer.compile(student=program, trainset=train, valset=val)
```

### After (gepa-observable)

```python
import dspy
from gepa_observable import GEPA  # <-- Only this line changes!

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define program
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

program = dspy.ChainOfThought(QA)

# Define metric
def accuracy_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return 1.0 if pred.answer.lower() == gold.answer.lower() else 0.0

# Optimize
optimizer = GEPA(
    metric=accuracy_metric,
    auto="medium",
)
optimized = optimizer.compile(student=program, trainset=train, valset=val)
```

## Dashboard Integration

Add real-time web dashboard monitoring with a single parameter.

```python
import dspy
from gepa_observable import GEPA

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

program = dspy.ChainOfThought(QA)

optimizer = GEPA(
    metric=accuracy_metric,
    auto="medium",
    # Add dashboard:
    server_url="http://localhost:3000",
    project_name="QA Optimization",
)
optimized = optimizer.compile(student=program, trainset=train, valset=val)
```

## Full Observability

Enable all observability features for complete visibility.

```python
import dspy
from gepa_observable import GEPA

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

program = dspy.ChainOfThought(QA)

optimizer = GEPA(
    metric=accuracy_metric,
    auto="medium",
    # Full observability:
    server_url="http://localhost:3000",
    project_name="QA Optimization",
    run_name="Experiment v1.2",
    capture_lm_calls=True,      # Log all LM calls
    capture_stdout=True,        # Capture console output
    verbose=True,               # Console logging
)
optimized = optimizer.compile(student=program, trainset=train, valset=val)
```

## Custom Observer Implementation

Create custom observers for your specific needs.

```python
import dspy
from gepa_observable import GEPA

class ProgressObserver:
    """Track optimization progress and log to custom system."""

    def __init__(self):
        self.iterations = []
        self.best_scores = []

    def on_seed_validation(self, event):
        avg = sum(event.valset_scores.values()) / len(event.valset_scores)
        print(f"Starting optimization with seed score: {avg:.2%}")
        self.best_scores.append(avg)

    def on_iteration_start(self, event):
        self.iterations.append(event.iteration)
        print(f"\n--- Iteration {event.iteration + 1} ---")
        print(f"Parent candidate: {event.selected_candidate_idx}")
        print(f"Parent score: {event.parent_score:.2%}")

    def on_reflection(self, event):
        print(f"Reflecting on: {event.components_to_update}")
        for comp, text in event.proposed_texts.items():
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"  {comp}: {preview}")

    def on_acceptance_decision(self, event):
        status = "ACCEPTED" if event.accepted else "REJECTED"
        print(f"Decision: {status}")
        print(f"  Parent: {event.parent_score_sum:.2f}")
        print(f"  New: {event.new_score_sum:.2f}")

    def on_valset_eval(self, event):
        if event.is_new_best:
            print(f"NEW BEST! Score: {event.valset_score:.2%}")
            self.best_scores.append(event.valset_score)

    def on_optimization_complete(self, event):
        print(f"\n{'='*50}")
        print(f"Optimization Complete!")
        print(f"  Iterations: {event.total_iterations}")
        print(f"  Evaluations: {event.total_evals}")
        print(f"  Best score: {event.best_score:.2%}")
        print(f"  Improvement: {self.best_scores[-1] - self.best_scores[0]:.2%}")
        print(f"{'='*50}")


# Use custom observer alongside dashboard
progress = ProgressObserver()

optimizer = GEPA(
    metric=accuracy_metric,
    auto="medium",
    server_url="http://localhost:3000",
    observers=[progress],
    verbose=False,  # Disable built-in logging since we have custom
)
optimized = optimizer.compile(student=program, trainset=train, valset=val)

# Access tracked data after optimization
print(f"Score history: {progress.best_scores}")
```

## Notebook Migration Walkthrough

Step-by-step for Jupyter notebooks.

### Cell 1: Imports (CHANGE THIS)

```python
# Before:
# from dspy.teleprompt import GEPA

# After:
from gepa_observable import GEPA
import dspy
```

### Cell 2: Configuration (NO CHANGE)

```python
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

### Cell 3: Program Definition (NO CHANGE)

```python
class QA(dspy.Signature):
    """Answer questions accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

program = dspy.ChainOfThought(QA)
```

### Cell 4: Data Loading (NO CHANGE)

```python
train_data = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
    # ... more examples
]
val_data = train_data[:10]  # Use subset for validation
```

### Cell 5: Metric Definition (NO CHANGE)

```python
def accuracy_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA feedback metric - 5 arguments required."""
    is_correct = pred.answer.strip().lower() == gold.answer.strip().lower()
    return dspy.Prediction(
        score=1.0 if is_correct else 0.0,
        feedback="Correct!" if is_correct else f"Expected '{gold.answer}'"
    )
```

### Cell 6: Optimization (ADD OBSERVABILITY)

```python
# Add observability parameters
optimizer = GEPA(
    metric=accuracy_metric,
    auto="medium",
    server_url="http://localhost:3000",  # NEW: Dashboard
    project_name="Notebook Experiment",   # NEW: Project name
)

# This call is unchanged
optimized = optimizer.compile(
    student=program,
    trainset=train_data,
    valset=val_data,
)
```

### Cell 7: Use Optimized Program (NO CHANGE)

```python
result = optimized(question="What is the speed of light?")
print(result.answer)
```

## Script Migration (Full File)

### Before: `optimize_qa.py`

```python
#!/usr/bin/env python3
"""QA optimization script using DSPy GEPA."""

import dspy
from dspy.teleprompt import GEPA

def main():
    # Setup
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Program
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    program = dspy.ChainOfThought(QA)

    # Data
    train = load_training_data()
    val = load_validation_data()

    # Metric
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if pred.answer == gold.answer else 0.0

    # Optimize
    optimizer = GEPA(metric=metric, auto="medium")
    optimized = optimizer.compile(student=program, trainset=train, valset=val)

    # Save
    optimized.save("optimized_qa.json")

if __name__ == "__main__":
    main()
```

### After: `optimize_qa.py`

```python
#!/usr/bin/env python3
"""QA optimization script using gepa-observable."""

import dspy
from gepa_observable import GEPA  # <-- Changed import

def main():
    # Setup
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Program
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    program = dspy.ChainOfThought(QA)

    # Data
    train = load_training_data()
    val = load_validation_data()

    # Metric
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if pred.answer == gold.answer else 0.0

    # Optimize with observability
    optimizer = GEPA(
        metric=metric,
        auto="medium",
        server_url="http://localhost:3000",  # <-- Added
        project_name="QA Optimization",       # <-- Added
    )
    optimized = optimizer.compile(student=program, trainset=train, valset=val)

    # Save
    optimized.save("optimized_qa.json")

if __name__ == "__main__":
    main()
```

## Metric with Feedback

Providing feedback improves reflection quality.

### Basic Metric (Float Return)

```python
def basic_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Returns just a score - works but no feedback for reflection."""
    return 1.0 if pred.answer == gold.answer else 0.0
```

### Feedback Metric (Recommended)

```python
def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Returns score + feedback - improves reflection quality."""
    is_correct = pred.answer.strip().lower() == gold.answer.strip().lower()

    if is_correct:
        return dspy.Prediction(
            score=1.0,
            feedback="Correct answer."
        )
    else:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Incorrect. Expected '{gold.answer}', but got '{pred.answer}'."
        )
```

### Partial Credit Metric

```python
def partial_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Partial credit with detailed feedback."""
    gold_lower = gold.answer.strip().lower()
    pred_lower = pred.answer.strip().lower()

    if pred_lower == gold_lower:
        return dspy.Prediction(score=1.0, feedback="Exact match!")
    elif gold_lower in pred_lower:
        return dspy.Prediction(
            score=0.7,
            feedback=f"Contains correct answer but has extra content: '{pred.answer}'"
        )
    elif any(word in pred_lower for word in gold_lower.split()):
        return dspy.Prediction(
            score=0.3,
            feedback=f"Partially correct. Expected '{gold.answer}', got '{pred.answer}'"
        )
    else:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Incorrect. Expected '{gold.answer}', got '{pred.answer}'"
        )
```

## Running the Dashboard

Before running optimization with `server_url`, start the dashboard:

```bash
cd web
npm install
echo 'DATABASE_URL="file:./dev.db"' > .env
npx prisma generate
npx prisma migrate deploy
npm run dev
```

Then open http://localhost:3000 in your browser.
