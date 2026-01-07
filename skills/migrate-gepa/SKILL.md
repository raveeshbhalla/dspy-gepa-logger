---
name: observable-gepa-migration
description: Migrate DSPy GEPA usage in DSPy from the original to gepa-observable. This makes it possible for teams to clearly review each iteration and the lineage to understand how their prompt is evolving. The repository offers a web dashboard for monitoring, but requires a custom GEPA fork that provides custom observers and LM call logging. Use when developers want to add observability to GEPA optimization.
---

# Migrate DSPy GEPA to gepa-observable

Migrate existing DSPy GEPA code to gepa-observable for integrated observability:
real-time dashboard, custom observer callbacks, and LM call capture working together.

## Quick Migration

### Step 1: Install gepa-observable

```bash
pip install dspy-gepa-logger
```

### Step 2: Change Import

```python
# Before
from dspy.teleprompt import GEPA

# After
from gepa_observable import GEPA
```

### Step 3: Add Observability (Full System)

```python
from gepa_observable import GEPA

optimizer = GEPA(
    metric=my_metric,
    auto="medium",
    # Observability system:
    server_url="http://your-server:3000",  # Dashboard integration
    project_name="My Project",
    capture_lm_calls=True,                  # LM call logging
    capture_stdout=True,                    # Console capture
    verbose=True,                           # LoggingObserver
    observers=[MyCustomObserver()],         # Custom callbacks
)
```

### Step 4: Everything Else Stays the Same

- Metric functions (5-arg signature already required by DSPy GEPA)
- Program definitions (ChainOfThought, ReAct, etc.)
- Training/validation data format
- Budget params (auto, max_full_evals, max_metric_calls)
- compile() call: `optimizer.compile(student=program, trainset=train, valset=val)`

## The Observability System

gepa-observable provides an **integrated observability system** where all components work together:

| Component | Purpose | Enabled By |
|-----------|---------|------------|
| **ServerObserver** | Sends events to web dashboard | `server_url` param |
| **LoggingObserver** | Console output with summaries | `verbose=True` |
| **LM Call Logger** | Captures all LM invocations | `capture_lm_calls=True` |
| **Custom Observers** | Your callbacks for any event | `observers=[...]` |

All observers receive the **same 8 lifecycle events**:

1. `SeedValidationEvent` - Initial validation scores
2. `IterationStartEvent` - Each optimization iteration
3. `MiniBatchEvalEvent` - Minibatch evaluations
4. `ReflectionEvent` - Proposed prompt changes
5. `AcceptanceDecisionEvent` - Accept/reject decisions
6. `ValsetEvalEvent` - Full validation evaluations
7. `MergeEvent` - Candidate merge operations
8. `OptimizationCompleteEvent` - Final results

## Custom Observer Example

```python
class MyObserver:
    def on_seed_validation(self, event):
        avg = sum(event.valset_scores.values()) / len(event.valset_scores)
        print(f"Seed score: {avg:.2%}")

    def on_iteration_start(self, event):
        print(f"Iteration {event.iteration}, parent score: {event.parent_score:.2%}")

    def on_reflection(self, event):
        for comp, text in event.proposed_instructions.items():
            print(f"Proposing for {comp}: {text[:100]}...")

    def on_valset_eval(self, event):
        if event.is_new_best:
            print(f"NEW BEST: {event.valset_score:.2%}")

    def on_optimization_complete(self, event):
        print(f"Done! Best: {event.best_score:.2%} in {event.num_iterations} iters")

# Use with other observers - they all work together
optimizer = GEPA(
    metric=my_metric,
    auto="medium",
    server_url="http://localhost:3000",
    observers=[MyObserver()],
    verbose=True,  # Also keep LoggingObserver
)
```

## Migration Checklist

### For Notebooks

1. [ ] Install: `pip install dspy-gepa-logger`
2. [ ] Change import: `from gepa_observable import GEPA`
3. [ ] Add `server_url` for dashboard
4. [ ] Add custom observers if needed
5. [ ] Run cells in order - API is 100% compatible

### For Scripts

1. [ ] Install: `pip install dspy-gepa-logger`
2. [ ] Find all `from dspy.teleprompt import GEPA`
3. [ ] Replace import
4. [ ] Add observability parameters to GEPA constructor
5. [ ] No other changes needed

## References

- `references/api-reference.md` - Complete parameter docs, observer protocol, all event types
- `references/examples.md` - Full before/after code examples
