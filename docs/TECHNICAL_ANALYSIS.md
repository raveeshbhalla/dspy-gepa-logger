# GEPA Logger Technical Analysis

**Date:** 2025-12-22
**Purpose:** Comprehensive analysis of data capture strategies for GEPA optimization logging

---

## Executive Summary

This document analyzes three potential data sources for capturing GEPA optimization runs:

1. **GEPA's Native Logging** - What GEPA logs out of the box
2. **MLflow Tracing** - How MLflow can capture LLM calls and provide visualization
3. **DSPy's Built-in Tracing** - What DSPy exposes via callbacks and trajectories

The recommended approach combines all three into a robust, layered architecture that maximizes data capture while minimizing fragility.

---

## Table of Contents

1. [Current Implementation Issues](#1-current-implementation-issues)
2. [GEPA's Native Logging](#2-gepas-native-logging)
3. [MLflow Tracing Capabilities](#3-mlflow-tracing-capabilities)
4. [DSPy's Built-in Tracing](#4-dspys-built-in-tracing)
5. [Combined Strategy](#5-combined-strategy)
6. [Data Coverage Matrix](#6-data-coverage-matrix)
7. [Recommended Architecture](#7-recommended-architecture)
8. [Implementation Plan](#8-implementation-plan)
9. [Appendix: Code Examples](#9-appendix-code-examples)

---

## 1. Current Implementation Issues

### The Error

```python
AttributeError: 'GEPAEngine' object has no attribute '_val_evaluator'
```

**Location:** `gepa_adapter.py:549`

```python
def wrapped_run_full_eval_and_add(engine_self, new_program, state, parent_program_idx):
    # This line fails - _val_evaluator doesn't exist
    valset_outputs, valset_subscores = engine_self._val_evaluator()(new_program)
```

### Root Cause

The hook attempts to access `GEPAEngine._val_evaluator`, a private attribute that:
- Doesn't exist in the current GEPA version
- May have been renamed or restructured
- Was never part of GEPA's public API

### The Fundamental Problem

The current implementation is **too tightly coupled to GEPA internals**:

| Hook Target | Stability | Risk |
|-------------|-----------|------|
| DSPy Callbacks (`on_lm_start/end`) | ✅ Public API | Low |
| DspyAdapter methods | ⚠️ Semi-private | Medium |
| GEPAEngine._run_full_eval_and_add | ❌ Private | **High** |
| GEPAEngine._val_evaluator | ❌ Private | **Broken** |
| create_experiment_tracker | ⚠️ Semi-public | Medium |

---

## 2. GEPA's Native Logging

### 2.1 Console Output

GEPA logs detailed information to the console via `logger.log()`:

#### Run Configuration
```
INFO dspy.teleprompt.gepa.gepa: Running GEPA for approx 420 metric calls of the program.
INFO dspy.teleprompt.gepa.gepa: Using 10 examples for tracking Pareto scores.
```

#### Baseline Evaluation (Iteration 0)
```
INFO dspy.evaluate.evaluate: Average Metric: 0.25 / 10 (2.5%)
INFO dspy.teleprompt.gepa.gepa: Iteration 0: Base program full valset score: 0.025 over 10 / 10 examples
```

#### Parent Selection
```
INFO dspy.teleprompt.gepa.gepa: Iteration 1: Selected program 0 score: 0.025
```

#### Proposed Instructions (FULL TEXT!)
```
INFO dspy.teleprompt.gepa.gepa: Iteration 1: Proposed new text for self: You are an expert...
[COMPLETE INSTRUCTION TEXT IS LOGGED]
```

#### Minibatch Evaluation
```
INFO dspy.evaluate.evaluate: Average Metric: 2.0 / 5 (40.0%)
INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 5 (60.0%)
```

#### Acceptance Decision
```
INFO dspy.teleprompt.gepa.gepa: Iteration 1: New subsample score 3.0 is better than old score 2.0. Continue to full eval and add to candidate pool.
```

#### Pareto Frontier
```
INFO dspy.teleprompt.gepa.gepa: Iteration 1: New program is on the linear pareto front
```

### 2.2 Files Written to `log_dir`

| File Pattern | Content | Format |
|--------------|---------|--------|
| `generated_best_outputs_valset/task_{id}/iter_{N}_prog_{M}.json` | Single validation score | JSON (just a number) |
| `gepa_state.bin` | Full GEPA state | Pickled Python object |

### 2.3 ExperimentTracker Metrics

GEPA logs these metrics via `ExperimentTracker.log_metrics()`:

| Metric Key | Description | Type |
|------------|-------------|------|
| `val_program_average` | Aggregate validation score | float |
| `new_program_idx` | Candidate program index | int |
| `valset_pareto_front_programs` | Per-task dominating programs | dict |
| `valset_pareto_front_scores` | Per-task best scores | dict |
| `valset_pareto_front_agg` | Aggregate Pareto score | float |
| `individual_valset_score_new_program` | Per-task scores for new candidate | list |

**Note:** Metrics are only logged when a candidate is **accepted**.

### 2.4 What GEPA Does NOT Log

| Missing Data | Impact |
|--------------|--------|
| Structured program instructions | Must parse console logs |
| Minibatch examples (inputs, outputs) | Can't see what drove improvements |
| Reflection feedback/reasoning | Can't understand why changes were proposed |
| LM calls (tokens, latency, cost) | No cost/performance visibility |
| Rejection reasons | Don't know why candidates failed |
| Validation outputs | Can't compare actual predictions |

### 2.5 Log Parser Potential

Since GEPA logs **full proposed instructions** to console, we can parse them:

```python
import re

PATTERNS = {
    'selected_program': r'Iteration (\d+): Selected program (\d+) score: ([\d.]+)',
    'proposed_text': r'Iteration (\d+): Proposed new text for (\w+): (.+)',
    'score_comparison': r'New subsample score ([\d.]+) is better than old score ([\d.]+)',
    'baseline': r'Iteration 0: Base program full valset score: ([\d.]+)',
    'pareto_front': r'Iteration (\d+): New program is on the linear pareto front',
}
```

This provides a **fallback data source** that doesn't require any code instrumentation.

---

## 3. MLflow Tracing Capabilities

### 3.1 Overview

MLflow Tracing (introduced in MLflow 2.14, September 2024) provides OpenTelemetry-compatible observability for LLM applications.

### 3.2 Trace Data Model

```
Trace
├── trace_id (UUID)
├── request_id (UUID)
├── timestamp_ms (int64)
├── execution_time_ms (int64)
├── status (SUCCESS | ERROR | IN_PROGRESS)
├── tags (dict[str, str])
└── spans: List[Span]

Span
├── span_id (UUID)
├── parent_span_id (UUID | None)  # For nesting
├── name (str)
├── start_time_ns / end_time_ns (int64)
├── span_type (CHAIN | AGENT | TOOL | LLM | RETRIEVER | etc.)
├── inputs (dict)
├── outputs (dict)
├── attributes (dict)  # Custom metadata
└── events: List[SpanEvent]
```

### 3.3 Automatic LLM Call Capture

MLflow can automatically capture LLM calls:

```python
import mlflow

mlflow.openai.autolog()  # For OpenAI
mlflow.langchain.autolog()  # For LangChain
```

**Captured automatically:**

| Field | Description | Location |
|-------|-------------|----------|
| Model | Model identifier | `span.attributes["llm.model"]` |
| Prompts | Input messages | `span.inputs["messages"]` |
| Completions | Response text | `span.outputs["text"]` |
| Input Tokens | Prompt tokens | `span.attributes["llm.token_count.prompt"]` |
| Output Tokens | Completion tokens | `span.attributes["llm.token_count.completion"]` |
| Latency | Call duration | `span.end_time_ns - span.start_time_ns` |
| Temperature | Sampling temp | `span.attributes["llm.temperature"]` |
| Finish Reason | Why stopped | `span.attributes["llm.finish_reason"]` |

### 3.4 Manual Tracing API

For custom instrumentation:

```python
import mlflow
from mlflow.entities import SpanType

with mlflow.start_trace(name="gepa_iteration") as trace:
    trace.set_tags({"iteration": 5, "optimizer": "GEPA"})

    with mlflow.start_span(
        name="parent_evaluation",
        span_type=SpanType.CHAIN,
        inputs={"examples": minibatch_ids},
    ) as span:
        outputs = evaluate_parent(minibatch)
        span.set_outputs({"scores": scores})
        span.set_attributes({
            "minibatch_size": len(minibatch),
            "avg_score": sum(scores) / len(scores),
        })
```

### 3.5 Custom Attributes for GEPA

MLflow allows arbitrary attributes:

```python
span.set_attributes({
    # Standard MLflow/OpenTelemetry
    "llm.model": "gpt-4o-mini",
    "llm.token_count.prompt": 1200,

    # GEPA-specific (custom)
    "gepa.iteration_number": 5,
    "gepa.is_reflection": True,
    "gepa.component_name": "predict",
    "gepa.accepted": True,
    "gepa.acceptance_reason": "score_improved",
})
```

### 3.6 Query API

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="tags.optimizer = 'GEPA'",
    order_by=["metrics.final_score DESC"],
)

# Search traces
traces = client.search_traces(
    experiment_ids=[experiment_id],
    filter_string="attributes.`gepa.is_reflection` = 'true'",
)
```

### 3.7 DSPy + MLflow Integration

**Current state:** No official DSPy-MLflow integration exists.

**Possible approaches:**

1. **DSPy Callback → MLflow Bridge:**
```python
class DSPyMLflowCallback(BaseCallback):
    def on_lm_start(self, call_id, instance, inputs):
        self._spans[call_id] = mlflow.start_span(
            name=f"dspy_lm_{call_id}",
            span_type=SpanType.LLM,
            inputs=inputs,
        )

    def on_lm_end(self, call_id, outputs, exception=None):
        span = self._spans.pop(call_id)
        span.set_outputs(outputs)
        span.__exit__(None, None, None)
```

2. **Patch dspy.LM directly:**
```python
original_call = dspy.LM.__call__

def traced_call(self, *args, **kwargs):
    with mlflow.start_span(name="dspy_lm", span_type=SpanType.LLM):
        return original_call(self, *args, **kwargs)

dspy.LM.__call__ = traced_call
```

### 3.8 MLflow Benefits for GEPA

| Feature | Value for GEPA |
|---------|---------------|
| Web UI | Visual exploration of runs |
| Trace visualization | See LM call hierarchies |
| Experiment comparison | Compare optimization runs |
| Cost analysis | Track token usage over time |
| Model registry | Version optimized prompts |
| Cloud deployment | Scale tracking infrastructure |

---

## 4. DSPy's Built-in Tracing

### 4.1 Callback System

DSPy provides a callback system for LM call tracking:

```python
from dspy.utils.callback import BaseCallback

class BaseCallback:
    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        """Called when an LM call starts."""
        pass

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None,
                  exception: Exception | None = None):
        """Called when an LM call completes."""
        pass
```

**Registration:**
```python
import dspy

dspy.configure(callbacks=[MyCallback()])
```

**Note:** There are NO `on_module_start/on_module_end` callbacks - only LM-level hooks.

### 4.2 Data Available in Callbacks

**`on_lm_start` receives:**

| Field | Description |
|-------|-------------|
| `call_id` | Unique identifier for the call |
| `instance` | LM instance (has `.model` attribute) |
| `inputs` | Full request dict: messages, temperature, max_tokens, etc. |

**`on_lm_end` receives:**

| Field | Description |
|-------|-------------|
| `call_id` | Same ID from start |
| `outputs` | Response dict: text, usage (tokens), finish_reason |
| `exception` | Exception if call failed |

### 4.3 Execution Trajectories

DSPy can capture execution traces during evaluation:

```python
result = adapter.evaluate(batch, candidate, capture_traces=True)

# result.trajectories contains:
for trajectory in result.trajectories:
    trace_data = trajectory['trace']      # List of (predictor, inputs, outputs)
    example = trajectory['example']        # Input example
    prediction = trajectory['prediction']  # Final output
    score = trajectory['score']           # Metric score
    feedback = trajectory.get('feedback')  # Optional feedback
```

### 4.4 Token Data Availability

DSPy exposes token counts through the callback outputs:

```python
def on_lm_end(self, call_id, outputs, exception):
    usage = outputs.get("usage", {})

    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
```

### 4.5 What DSPy Does NOT Provide

| Missing Feature | Workaround |
|-----------------|------------|
| `dspy.settings.trace` | Use callbacks instead |
| `on_module_start/end` | Only LM-level callbacks |
| Built-in MLflow integration | Create custom bridge |
| Global trace history | Must capture via callbacks |

---

## 5. Combined Strategy

### 5.1 Layered Data Capture Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA CAPTURE LAYERS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 1: DSPy Callbacks (MOST STABLE - Public API)                   │   │
│  │ ────────────────────────────────────────────────────────────────────  │   │
│  │ • on_lm_start/on_lm_end for ALL LM calls                             │   │
│  │ • Token counts, latency, cost                                        │   │
│  │ • Full request/response capture                                      │   │
│  │ • Context marking (is_reflection, component_name)                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 2: DspyAdapter Hooks (SEMI-STABLE)                             │   │
│  │ ────────────────────────────────────────────────────────────────────  │   │
│  │ • evaluate() → minibatch evals, parent/candidate distinction         │   │
│  │ • make_reflective_dataset() → reflection input data                  │   │
│  │ • Execution trajectories from capture_traces=True                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 3: ExperimentTracker Wrapper (STABLE)                          │   │
│  │ ────────────────────────────────────────────────────────────────────  │   │
│  │ • log_metrics() → Pareto frontier, validation scores                 │   │
│  │ • Acceptance decisions (inferred from presence of metrics)           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 4: GEPA Log Parser (FALLBACK - Zero Code Changes)              │   │
│  │ ────────────────────────────────────────────────────────────────────  │   │
│  │ • Parse console output for proposed instructions                     │   │
│  │ • Parse iteration progress messages                                  │   │
│  │ • Backup data source if instrumentation fails                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 5: Post-Run Re-evaluation (OPTIONAL - Complete Data)           │   │
│  │ ────────────────────────────────────────────────────────────────────  │   │
│  │ • Re-run accepted candidates on valset after optimization            │   │
│  │ • Capture validation OUTPUTS (not just scores)                       │   │
│  │ • Generate comparison artifacts                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Dual Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────┐          ┌───────────────────────────┐       │
│  │ SQLite (Primary)          │          │ MLflow (Optional)         │       │
│  │ ─────────────────────────  │          │ ─────────────────────────  │       │
│  │ • Normalized schema       │          │ • Trace visualization    │       │
│  │ • SQL queries             │◄────────►│ • Web UI                 │       │
│  │ • Pareto frontier queries │          │ • Experiment comparison  │       │
│  │ • Fast lookups            │          │ • Cost analysis          │       │
│  │ • HTML report generation  │          │ • Model registry         │       │
│  └───────────────────────────┘          └───────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Coverage Matrix

### 6.1 Data Source Comparison

| Data Point | DSPy Callback | DspyAdapter | ExperimentTracker | Log Parser | Post-Run Eval |
|------------|:-------------:|:-----------:|:-----------------:|:----------:|:-------------:|
| LM calls (request/response) | ✅ Primary | | | | |
| Token counts | ✅ Primary | | | | |
| Latency/cost | ✅ Primary | | | | |
| Is reflection call | ✅ Context var | | | | |
| Parent minibatch inputs | | ✅ Primary | | | |
| Parent minibatch outputs | | ✅ Primary | | | |
| Parent minibatch scores | | ✅ Primary | | | |
| Candidate minibatch outputs | | ✅ Primary | | | |
| Candidate minibatch scores | | ✅ Primary | | | |
| Execution trajectories | | ✅ capture_traces | | | |
| Reflection feedback | | ✅ Primary | | | |
| Proposed instructions | | ✅ Primary | | ✅ Backup | |
| Pareto frontier | | | ✅ Primary | | |
| Validation scores | | | ✅ Primary | | |
| Acceptance decisions | | | ✅ Inferred | ✅ Backup | |
| Validation outputs | | | ❌ | | ✅ Primary |
| Full iteration history | | | | ✅ Backup | |

### 6.2 Strategy Comparison

| Strategy | Stability | Data Completeness | Complexity | Best For |
|----------|:---------:|:-----------------:|:----------:|----------|
| DSPy Callbacks only | ✅ Very stable | 40% | Low | Cost/token tracking |
| Current approach (fixed) | ⚠️ Semi-stable | 85% | Medium | Most use cases |
| + Log Parser backup | ✅ Very stable | 90% | Medium | Production resilience |
| + MLflow integration | ✅ Stable | 90% + visualization | Medium-High | Teams wanting UI |
| + Post-run eval | ✅ Very stable | 100% | High | Full reproducibility |

---

## 7. Recommended Architecture

### 7.1 Component Diagram

```
User Code
    │
    ├─ GEPA.compile(student, trainset, valset)
    │
    └─ GEPARunTracker ─────────────────────────────────────────────┐
        │                                                           │
        ├─ DSPy Callback ──────────────────────────────────────────┤
        │   └─ GEPALoggingCallback                                 │
        │       ├─ on_lm_start() → start_lm_call()                 │
        │       └─ on_lm_end() → end_lm_call()                     │
        │                                                           │
        ├─ DspyAdapter Hooks ──────────────────────────────────────┤
        │   └─ InstrumentedGEPAAdapter                             │
        │       ├─ evaluate() → record_parent/candidate_evaluation │
        │       └─ make_reflective_dataset() → record_reflection   │
        │                                                           │
        ├─ ExperimentTracker Wrapper ──────────────────────────────┤
        │   └─ ParetoCapturingTracker                              │
        │       └─ log_metrics() → record_pareto_update            │
        │                                                           │
        └─ Storage Backends ───────────────────────────────────────┤
            ├─ SQLiteStorageAdapter (primary)                      │
            └─ MLflowBackend (optional)                            │
```

### 7.2 What to Keep

✅ **Keep these components (working well):**
- `GEPARunTracker` core logic
- `InstrumentedGEPAAdapter` (except GEPAEngine hook)
- `ParetoCapturingTracker`
- `GEPALoggingCallback`
- Data models (`GEPARunRecord`, `IterationRecord`, etc.)
- SQLite storage schema
- Context variable for reflection tracking

### 7.3 What to Remove

❌ **Remove (broken/fragile):**
- `GEPAEngine._run_full_eval_and_add` hook (lines 540-616 in gepa_adapter.py)
- Any access to `engine_self._val_evaluator`

### 7.4 What to Add

➕ **Add for robustness:**
- Log parser as backup data source
- Post-run evaluation for validation outputs
- MLflow backend (optional)

---

## 8. Implementation Plan

### Phase 1: Fix Immediate Error

**Goal:** Get the logger working again

**Changes:**
1. Remove lines 540-616 in `gepa_adapter.py` (the `_run_full_eval_and_add` hook)
2. Remove references to `_val_evaluator`
3. Test that basic logging works

**Trade-off:** We lose validation outputs, but keep:
- All minibatch data
- All LM calls
- Pareto frontier updates
- Validation scores

### Phase 2: Add Log Parser Backup

**Goal:** Fallback data source for resilience

**New file:** `src/dspy_gepa_logger/parsers/log_parser.py`

**Features:**
- Parse GEPA console output
- Extract proposed instructions
- Extract iteration progress
- Validate against instrumentation data

### Phase 3: Add Post-Run Evaluation

**Goal:** Capture validation outputs

**New file:** `src/dspy_gepa_logger/evaluation/post_run.py`

**Features:**
- Store accepted candidates during optimization
- Re-run on valset after optimization completes
- Capture full outputs for comparison reports

### Phase 4: Add MLflow Integration (Optional)

**Goal:** Visualization and cost analysis

**New file:** `src/dspy_gepa_logger/storage/mlflow_backend.py`

**Features:**
- Log LM calls as MLflow spans
- Log iterations as traces
- Store artifacts (prompts, Pareto data)
- Enable MLflow UI

### Phase 5: Fix SQLite Read Path

**Goal:** Query stored data

**Changes to:** `src/dspy_gepa_logger/storage/sqlite_adapter.py`

**Features:**
- Implement `load_iterations()`
- Implement `load_candidates()`
- Reconstruct full objects from normalized data

---

## 9. Appendix: Code Examples

### 9.1 Log Parser

```python
# src/dspy_gepa_logger/parsers/log_parser.py

import re
from dataclasses import dataclass, field

@dataclass
class ParsedIteration:
    iteration_number: int
    parent_idx: int = 0
    parent_score: float = 0.0
    proposed_instructions: dict[str, str] = field(default_factory=dict)
    minibatch_old_score: float = 0.0
    minibatch_new_score: float = 0.0
    accepted: bool = False

class GEPALogParser:
    """Parse GEPA console output to extract iteration data."""

    PATTERNS = {
        'selected_program': re.compile(
            r'Iteration (\d+): Selected program (\d+) score: ([\d.]+)'
        ),
        'proposed_text': re.compile(
            r'Iteration (\d+): Proposed new text for (\w+): (.+)'
        ),
        'score_comparison': re.compile(
            r'New subsample score ([\d.]+) is better than old score ([\d.]+)'
        ),
        'baseline': re.compile(
            r'Iteration 0: Base program full valset score: ([\d.]+)'
        ),
        'acceptance': re.compile(
            r'Continue to full eval and add to candidate pool'
        ),
    }

    def parse_log(self, log_content: str) -> list[ParsedIteration]:
        iterations = []
        current_iter = None

        for line in log_content.split('\n'):
            # Parent selection
            if match := self.PATTERNS['selected_program'].search(line):
                iter_num, parent_idx, parent_score = match.groups()
                current_iter = ParsedIteration(
                    iteration_number=int(iter_num),
                    parent_idx=int(parent_idx),
                    parent_score=float(parent_score),
                )
                iterations.append(current_iter)

            # Proposed instructions
            elif match := self.PATTERNS['proposed_text'].search(line):
                iter_num, component, instruction = match.groups()
                if current_iter and current_iter.iteration_number == int(iter_num):
                    current_iter.proposed_instructions[component] = instruction

            # Score comparison
            elif match := self.PATTERNS['score_comparison'].search(line):
                new_score, old_score = match.groups()
                if current_iter:
                    current_iter.minibatch_new_score = float(new_score)
                    current_iter.minibatch_old_score = float(old_score)

            # Acceptance
            elif self.PATTERNS['acceptance'].search(line):
                if current_iter:
                    current_iter.accepted = True

        return iterations
```

### 9.2 Post-Run Evaluator

```python
# src/dspy_gepa_logger/evaluation/post_run.py

from typing import Any, Callable
from dataclasses import dataclass

@dataclass
class ValidationOutput:
    candidate_idx: int
    example_id: int
    input_data: dict
    output: Any
    score: float

class PostRunEvaluator:
    """Re-evaluate accepted candidates to capture validation outputs."""

    def __init__(self, valset: list, metric: Callable):
        self.valset = valset
        self.metric = metric
        self.validation_outputs: list[ValidationOutput] = []

    def evaluate_candidate(
        self,
        program,
        candidate_instructions: dict,
        candidate_idx: int
    ) -> tuple[list, list[float]]:
        """Run program on full valset and capture outputs."""

        # Apply instructions to program
        for name, instruction in candidate_instructions.items():
            if hasattr(program, name):
                predictor = getattr(program, name)
                if hasattr(predictor, 'signature'):
                    predictor.signature.instructions = instruction

        outputs = []
        scores = []

        for i, example in enumerate(self.valset):
            # Run program
            try:
                output = program(**example.inputs())
            except Exception as e:
                output = {"error": str(e)}

            outputs.append(output)

            # Score
            try:
                score = self.metric(example, output)
                if hasattr(score, 'score'):
                    score = score.score
                score = float(score)
            except Exception:
                score = 0.0

            scores.append(score)

            # Store
            self.validation_outputs.append(ValidationOutput(
                candidate_idx=candidate_idx,
                example_id=getattr(example, 'example_id', i),
                input_data=example.inputs(),
                output=output,
                score=score,
            ))

        return outputs, scores

    def get_outputs_for_candidate(self, candidate_idx: int) -> list[ValidationOutput]:
        """Get all validation outputs for a specific candidate."""
        return [o for o in self.validation_outputs if o.candidate_idx == candidate_idx]
```

### 9.3 MLflow Backend

```python
# src/dspy_gepa_logger/storage/mlflow_backend.py

import mlflow
from mlflow.entities import SpanType
from typing import Any

class MLflowBackend:
    """MLflow tracing backend for GEPA runs."""

    def __init__(self, experiment_name: str = "gepa_experiments", tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self._current_trace = None
        self._current_run = None

    def start_run(self, run_id: str, config: dict):
        """Start MLflow run."""
        self._current_run = mlflow.start_run(run_name=f"gepa_{run_id}")
        mlflow.set_tags({
            "gepa_run_id": run_id,
            "optimizer": "GEPA",
        })
        mlflow.log_params({k: str(v) for k, v in config.items()})

    def start_iteration(self, iteration_num: int, parent_idx: int):
        """Start trace for iteration."""
        self._current_trace = mlflow.start_trace(f"iteration_{iteration_num}")
        self._current_trace.set_tags({
            "iteration_number": str(iteration_num),
            "parent_candidate_idx": str(parent_idx),
        })

    def log_lm_call(self, lm_record):
        """Log LM call as MLflow span."""
        with mlflow.start_span(
            name=f"lm_{lm_record.call_id[:8]}",
            span_type=SpanType.LLM,
            inputs={"messages": lm_record.messages[:2] if lm_record.messages else []},
        ) as span:
            span.set_outputs({"response": lm_record.response_text[:500] if lm_record.response_text else ""})
            span.set_attributes({
                "llm.model": lm_record.model or "unknown",
                "llm.token_count.prompt": lm_record.input_tokens or 0,
                "llm.token_count.completion": lm_record.output_tokens or 0,
                "llm.temperature": lm_record.temperature or 0.0,
                "gepa.is_reflection": lm_record.is_reflection,
                "gepa.component_name": lm_record.component_name or "",
            })

    def end_iteration(self, iteration, accepted: bool, val_score: float = None):
        """End iteration trace and log metrics."""
        step = iteration.iteration_number

        mlflow.log_metrics({
            "parent_val_score": iteration.parent_val_score or 0,
            "accepted": int(accepted),
            "val_aggregate_score": val_score or 0,
        }, step=step)

        # Log candidate prompt as artifact
        if iteration.new_candidate_prompt:
            mlflow.log_dict(
                iteration.new_candidate_prompt,
                f"iterations/iter_{step}/candidate_prompt.json"
            )

        if self._current_trace:
            self._current_trace.__exit__(None, None, None)
            self._current_trace = None

    def log_pareto_update(self, iteration_num: int, pareto_data: dict):
        """Log Pareto frontier snapshot."""
        mlflow.log_dict(
            pareto_data,
            f"pareto/iteration_{iteration_num}.json"
        )

    def end_run(self, final_score: float = None, improvement: float = None):
        """End MLflow run."""
        if final_score is not None:
            mlflow.log_metric("final_score", final_score)
        if improvement is not None:
            mlflow.log_metric("improvement", improvement)

        mlflow.end_run()
        self._current_run = None
```

### 9.4 Hybrid Tracker Usage

```python
# Example usage with all components

from dspy_gepa_logger import GEPARunTracker
from dspy_gepa_logger.storage import SQLiteStorage, SQLiteStorageAdapter
from dspy_gepa_logger.storage.mlflow_backend import MLflowBackend
from dspy_gepa_logger.hooks import create_instrumented_gepa, cleanup_instrumented_gepa
from dspy_gepa_logger.evaluation.post_run import PostRunEvaluator

# Setup storage
sqlite = SQLiteStorage("./gepa_runs.db")
adapter = SQLiteStorageAdapter(sqlite)

# Create tracker with optional MLflow
tracker = GEPARunTracker(
    storage=adapter,
    mlflow_backend=MLflowBackend(experiment_name="my_experiment"),  # Optional
)

# Create GEPA optimizer
gepa = GEPA(metric=my_metric, auto="light")

# Instrument
create_instrumented_gepa(gepa, tracker, log_file="./gepa.log")

# Run optimization
tracker.start_run(config={"auto": "light", "model": "gpt-4o-mini"})
tracker.register_examples(trainset, valset)

try:
    optimized = gepa.compile(
        student=MyProgram(),
        trainset=trainset,
        valset=valset,
    )
finally:
    cleanup_instrumented_gepa(gepa)
    tracker.end_run()

# Optional: Post-run evaluation for validation outputs
evaluator = PostRunEvaluator(valset, my_metric)
for candidate in tracker.get_accepted_candidates():
    evaluator.evaluate_candidate(
        program=MyProgram(),
        candidate_instructions=candidate.instructions,
        candidate_idx=candidate.candidate_idx,
    )

# Generate reports
from dspy_gepa_logger.export import generate_html_report
generate_html_report(tracker.current_run, output_path="./report.html")
```

---

## Conclusion

The recommended approach combines multiple data sources:

1. **DSPy Callbacks** for LM call tracking (most stable)
2. **DspyAdapter hooks** for iteration data (remove fragile engine hooks)
3. **ExperimentTracker wrapper** for Pareto data (stable)
4. **Log parser** as backup (zero coupling)
5. **Post-run evaluation** for complete validation outputs (optional)

With **dual storage** (SQLite + optional MLflow) for both queryability and visualization.

This architecture provides:
- **Resilience** - Multiple data sources, graceful degradation
- **Completeness** - All GEPA data captured
- **Stability** - No dependency on GEPA private internals
- **Flexibility** - Optional MLflow for visualization
- **Queryability** - SQL access to all data
