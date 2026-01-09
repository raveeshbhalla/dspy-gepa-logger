---
name: observable-gepa-optimizer
description: Build and optimize DSPy programs with observability. Migrate existing GEPA code or create optimized programs from scratch using labelled datasets. Integrated dashboard monitoring, custom observers, and LM call logging.
---

# Observable GEPA Optimizer

Build production-ready DSPy programs with built-in observability using gepa-observable.

## Choose Your Use Case

### Use Case A: Migration

**You have**: Existing DSPy GEPA code
**You want**: Add observability (dashboard, observers, LM logging)

This is for developers who already have working DSPy GEPA optimization code and want to add the gepa-observable features without changing their optimization logic.

**See**: `use-cases/migration.md`

---

### Use Case B: Create from Scratch

**You have**: Labelled dataset (CSV with ground truth, optionally with file references like PDFs)
**You want**: Complete DSPy program with GEPA optimization, ready to run

This is for developers who have labelled data but no existing DSPy code. The agent will interactively guide you through:
1. Understanding your data structure
2. Defining the task and DSPy Signature
3. Choosing an evaluation strategy
4. Generating complete, runnable code

**See**: `use-cases/create-from-scratch.md`

---

## Quick Reference

### Installation

```bash
pip install dspy-gepa-logger --pre  # Include pre-release versions
```

> **Note**: The package is currently in alpha. Check [PyPI](https://pypi.org/project/dspy-gepa-logger/) for the latest version.

### Minimal Example

```python
from gepa_observable import GEPA

optimizer = GEPA(
    metric=my_metric,
    auto="medium",
    server_url="http://localhost:3000",  # Dashboard integration
    project_name="My Project",
)
optimized = optimizer.compile(student=program, trainset=train, valset=val)
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Dashboard** | Real-time web UI for monitoring optimization |
| **Observers** | Custom callbacks for all 8 optimization lifecycle events |
| **LM Logging** | Capture all LLM calls with context (iteration, phase, candidate) |
| **100% Compatible** | Drop-in replacement for `dspy.GEPA` |

---

## References

- `references/api-reference.md` - Complete API documentation
- `references/migration-examples.md` - Before/after migration examples
- `references/task-types.md` - Task type signatures and patterns
- `references/metric-templates.md` - Metric function templates
- `references/multimodal-patterns.md` - PDF/image handling with multimodal LLMs
- `references/output-templates/` - Code generation templates
