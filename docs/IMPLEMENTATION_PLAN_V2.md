# GEPA Logger Implementation Plan v2.2

**Date:** 2025-12-22
**Status:** Recommended Approach (Final revision with GPT feedback)
**Key Insight:** Use GEPA's public hooks + custom proposer for full phase attribution

---

## Executive Summary

We use **five hook points** for complete data capture:

1. **`stop_callbacks`** via `gepa_kwargs` - Iteration-level state with lineage and Pareto data
2. **`track_stats=True`** - Comprehensive post-run detailed results (canonical "final truth")
3. **DSPy Callbacks** - All LM calls with prompts/responses (context-tagged)
4. **`LoggedMetric` wrapper** - Captures score + feedback + prediction per evaluation
5. **`LoggedInstructionProposer`** (NEW) - Sets context for reflection/proposal phase attribution

This eliminates the fragile `_val_evaluator` hook that was causing errors.

### Key Design Decisions (v2.1 + v2.2 fixes)

| Issue | Problem | Solution |
|-------|---------|----------|
| Memory blowup | Deep-copying full state each iteration | **Incremental snapshots** - only store diffs |
| Timestamp correlation | Fragile with `num_threads > 1` | **Context tags** via single shared contextvar |
| Missing feedback | GEPA state doesn't retain feedback strings | **LoggedMetric** - capture `(score, feedback)` directly |
| API wrapper bug | `gepa_kwargs` double-nesting | **Clean merge** - explicit parameter handling |
| **Contextvar mismatch** | v2.1 had two different contextvars | **Single `context.py`** module |
| **Phase not set at eval time** | Context only set at iteration end | **LoggedMetric sets phase="eval"** before metric call |
| **Reflection/proposal not tagged** | No hook before proposer runs | **LoggedInstructionProposer** wrapper sets context |
| **Example ID unstable** | `hash()` is salted per process | **Deterministic SHA256** hash |
| **Pareto set nondeterministic** | `list(set)[0]` is random | **Store full set** or use `min()` |
| **Prediction storage bloat** | Raw objects can be huge | **Store refs + preview** |

---

## What We Want to Capture (Requirements Recap)

| Goal | Data Needed |
|------|-------------|
| Visual comparison: initial vs optimized prompts | Seed program, final optimized program |
| Performance comparison on validation set | Validation scores per candidate |
| Pareto frontier visualization | `pareto_front_valset`, `program_at_pareto_front_valset` |
| Program lineage traversal | `parent_program_for_candidate` |
| "Where did this prompt piece come from?" | Lineage + candidate content |
| "Why did performance change?" | Score deltas + what changed between parent/child |

---

## The Three Public Hooks

### Hook 1: `stop_callbacks` via `gepa_kwargs`

**How it works:** GEPA accepts a list of callback functions that are called each iteration. Each callback receives the full `state` object and returns `True` to stop or `False` to continue.

**What we get:**

```python
class GEPAState:
    # CANDIDATES
    program_candidates: list[dict[str, str]]          # All candidate programs (the actual prompt text!)

    # LINEAGE (answers "where did this come from?")
    parent_program_for_candidate: list[list[int|None]] # Parent indices for each candidate

    # SCORES
    prog_candidate_val_subscores: list[dict[DataId, float]]  # Validation scores per candidate per data point

    # PARETO FRONTIER
    pareto_front_valset: dict[DataId, float]          # Best scores achieved per validation sample
    program_at_pareto_front_valset: dict[DataId, set[int]]  # Which programs are on the Pareto front

    # HISTORY
    full_program_trace: list[dict[str, Any]]          # Complete optimization history log

    # METADATA
    i: int                                            # Current iteration
    total_num_evals: int                              # Cumulative evaluations
    num_full_ds_evals: int                            # Full dataset evaluation count

    # OUTPUTS (optional)
    best_outputs_valset: dict[DataId, list[tuple[int, RolloutOutput]]] | None
```

**Usage:**

```python
from dspy import GEPA

class StateLogger:
    def __init__(self):
        self.snapshots = []

    def __call__(self, state) -> bool:
        self.snapshots.append({
            'iteration': state.i,
            'total_evals': state.total_num_evals,
            'num_candidates': len(state.program_candidates),
            'candidates': [dict(c) for c in state.program_candidates],  # Deep copy
            'lineage': [list(p) for p in state.parent_program_for_candidate],
            'pareto_frontier': dict(state.pareto_front_valset),
            'pareto_programs': {k: list(v) for k, v in state.program_at_pareto_front_valset.items()},
            'validation_scores': [dict(s) for s in state.prog_candidate_val_subscores],
            'full_trace': list(state.full_program_trace),
        })
        return False  # Don't stop optimization

# Instantiate and pass to GEPA
state_logger = StateLogger()
gepa = GEPA(
    metric=my_metric,
    gepa_kwargs={'stop_callbacks': [state_logger]}
)
```

**Why this is powerful:**
- `program_candidates` contains the actual prompt text for each candidate
- `parent_program_for_candidate` tells us exactly which candidate spawned which
- `pareto_front_valset` gives us the Pareto frontier directly
- `full_program_trace` may contain the complete history of what happened

---

### Hook 2: `track_stats=True` + `detailed_results`

**How it works:** When you enable `track_stats=True`, GEPA collects comprehensive statistics and makes them available in `result.detailed_results` after optimization completes.

**What we get:**

```python
result = gepa.compile(
    student,
    trainset=trainset,
    valset=valset,
    track_stats=True  # Enable detailed tracking
)

# After completion:
detailed = result.detailed_results
# Contains comprehensive post-run data including:
# - All candidates that were tried
# - Final Pareto frontier
# - Performance statistics
# - Lineage information
```

**Why this is useful:**
- Provides a clean, final summary after optimization
- Useful for post-run analysis and reporting
- Complements the per-iteration data from `stop_callbacks`

---

### Hook 3: DSPy Callbacks

**How it works:** DSPy has a built-in callback system that fires on every LM call.

**What we get:**

```python
import dspy

class LMCallLogger:
    def __init__(self):
        self.calls = []

    def on_lm_start(self, call_id, instance, inputs):
        self.calls.append({
            'call_id': call_id,
            'event': 'start',
            'timestamp': time.time(),
            'model': instance.model if hasattr(instance, 'model') else str(instance),
            'inputs': inputs,
        })

    def on_lm_end(self, call_id, instance, outputs):
        self.calls.append({
            'call_id': call_id,
            'event': 'end',
            'timestamp': time.time(),
            'outputs': outputs,
        })

lm_logger = LMCallLogger()
dspy.configure(
    callbacks=[lm_logger]  # or lm=dspy.LM(..., callbacks=[lm_logger])
)
```

**What this captures:**
- Every LM call (prompts, responses, model, timing)
- Includes calls from: metric evaluation, reflection, proposal generation
- Can correlate with iteration data via timing

---

## Hook 4: `LoggedMetric` Wrapper (Required)

**Why it's required (not optional):**

Your key UX question is **"why did it change?"** and your metric returns `(score, feedback)`. Without a metric wrapper:
- You **lose feedback strings entirely** - GEPA state only stores scores, not feedback
- You can't do **paired example-level deltas** - need actual outputs per example
- You can't answer **"what did the model say for this example before vs after?"**

### What `LoggedMetric` captures that nothing else does:

| Data | `stop_callbacks` | DSPy Callbacks | `LoggedMetric` |
|------|:----------------:|:--------------:|:--------------:|
| Metric feedback string | ❌ | ❌ | ✅ |
| Per-example prediction | ❌ (only best) | ❌ | ✅ |
| Direct example→score correlation | ❌ | ❌ | ✅ |
| Candidate idx for each eval | ❌ | ❌ | ✅ (via context) |

### Implementation (v2.2 - fixed context, deterministic IDs, prediction refs):

```python
"""core/logged_metric.py - Metric wrapper for evaluation capture."""

import json
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .context import get_ctx, set_ctx  # Single shared contextvar!


def _deterministic_example_id(example: Any) -> str:
    """
    Generate a stable, deterministic example ID.

    IMPORTANT: Python's hash() is salted per process - NOT stable across runs.
    Use SHA256 of canonical JSON instead.
    """
    # Try to use explicit ID first
    if hasattr(example, 'id') and example.id is not None:
        return str(example.id)

    # Build canonical representation
    if hasattr(example, 'toDict'):
        data = example.toDict()
    elif hasattr(example, '__dict__'):
        data = {k: v for k, v in example.__dict__.items() if not k.startswith('_')}
    else:
        data = str(example)

    # Deterministic hash
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _serialize_prediction(prediction: Any, max_preview_len: int = 200) -> tuple[str | None, str]:
    """
    Serialize prediction to ref + preview.

    Returns:
        (prediction_ref, prediction_preview)
        - prediction_ref: JSON blob (or None if not serializable)
        - prediction_preview: Short string for display
    """
    # Build preview
    preview_str = str(prediction)[:max_preview_len]
    if len(str(prediction)) > max_preview_len:
        preview_str += "..."

    # Try to serialize
    try:
        if hasattr(prediction, 'toDict'):
            ref = json.dumps(prediction.toDict(), default=str)
        elif hasattr(prediction, '__dict__'):
            ref = json.dumps(prediction.__dict__, default=str)
        else:
            ref = json.dumps(str(prediction))
        return ref, preview_str
    except Exception:
        return None, preview_str


@dataclass
class EvaluationRecord:
    """Single metric evaluation record."""
    eval_id: str
    example_id: str  # Deterministic hash, stable across runs
    candidate_idx: int | None  # Which program candidate (from context)
    iteration: int | None  # Which GEPA iteration (from context)
    phase: str  # 'eval', 'validation', 'minibatch' (set by LoggedMetric)
    score: float
    feedback: str | None  # The "why" - critical for your UX

    # Prediction stored as ref + preview (not raw object!)
    prediction_ref: str | None = None  # JSON blob for full reconstruction
    prediction_preview: str = ""  # Short string for display


class LoggedMetric:
    """
    Wraps a metric function to capture evaluations.

    This is a PUBLIC API hook - no monkey-patching required.
    Just wrap your metric before passing to GEPA.

    CRITICAL: This class SETS context (phase="eval") before calling the metric,
    so DSPy LM callbacks will be correctly tagged.
    """

    def __init__(
        self,
        metric_fn: Callable,
        *,
        capture_prediction: bool = True,
        max_prediction_preview: int = 200,
    ):
        self.metric_fn = metric_fn
        self.capture_prediction = capture_prediction
        self.max_prediction_preview = max_prediction_preview
        self.evaluations: list[EvaluationRecord] = []

    def __call__(self, example, prediction, trace=None, **kwargs):
        # Get current context (iteration, candidate_idx set by state logger or proposer)
        ctx = get_ctx()

        # SET phase="eval" BEFORE calling metric (so any LM calls inside are tagged)
        set_ctx(phase="eval")

        # Generate deterministic example ID
        example_id = _deterministic_example_id(example)

        # Call actual metric
        result = self.metric_fn(example, prediction, trace=trace, **kwargs)

        # Extract score and feedback
        if isinstance(result, tuple) and len(result) == 2:
            score, feedback = result
        else:
            score, feedback = result, None

        # Serialize prediction (ref + preview, not raw object)
        if self.capture_prediction:
            pred_ref, pred_preview = _serialize_prediction(
                prediction, self.max_prediction_preview
            )
        else:
            pred_ref, pred_preview = None, ""

        # Record evaluation
        record = EvaluationRecord(
            eval_id=str(uuid.uuid4()),
            example_id=example_id,
            candidate_idx=ctx.get('candidate_idx'),
            iteration=ctx.get('iteration'),
            phase="eval",  # We just set this
            score=float(score) if score is not None else 0.0,
            feedback=str(feedback) if feedback else None,
            prediction_ref=pred_ref,
            prediction_preview=pred_preview,
        )
        self.evaluations.append(record)

        return result

    def get_evaluations_for_example(self, example_id: str) -> list[EvaluationRecord]:
        """Get all evaluations for a specific example (for delta analysis)."""
        return [e for e in self.evaluations if e.example_id == example_id]

    def get_evaluations_for_candidate(self, candidate_idx: int) -> list[EvaluationRecord]:
        """Get all evaluations for a specific candidate."""
        return [e for e in self.evaluations if e.candidate_idx == candidate_idx]

    def compute_lift(self, example_id: str, before_candidate: int, after_candidate: int) -> dict:
        """Compute score lift for an example between two candidates."""
        before = [e for e in self.evaluations
                  if e.example_id == example_id and e.candidate_idx == before_candidate]
        after = [e for e in self.evaluations
                 if e.example_id == example_id and e.candidate_idx == after_candidate]

        if not before or not after:
            return {'lift': None, 'before': None, 'after': None}

        return {
            'lift': after[0].score - before[0].score,
            'before': before[0],
            'after': after[0],
        }

    def get_regressions(self, seed_candidate: int = 0) -> list[dict]:
        """Find examples where optimized candidates scored worse than seed."""
        regressions = []
        example_ids = set(e.example_id for e in self.evaluations)

        for ex_id in example_ids:
            seed_evals = [e for e in self.evaluations
                         if e.example_id == ex_id and e.candidate_idx == seed_candidate]
            if not seed_evals:
                continue

            seed_score = seed_evals[0].score

            # Find any candidate that scored worse
            for e in self.evaluations:
                if e.example_id == ex_id and e.candidate_idx != seed_candidate:
                    if e.score < seed_score:
                        regressions.append({
                            'example_id': ex_id,
                            'seed_score': seed_score,
                            'seed_feedback': seed_evals[0].feedback,
                            'regressed_candidate': e.candidate_idx,
                            'regressed_score': e.score,
                            'regressed_feedback': e.feedback,
                            'delta': e.score - seed_score,
                        })

        return sorted(regressions, key=lambda x: x['delta'])
```

---

## Hook 5: `LoggedInstructionProposer` (NEW in v2.2)

**Why it's needed:**

With only `stop_callbacks` + `LoggedMetric`, we can tag **eval** LM calls but NOT **reflection/proposal** LM calls. The proposer runs inside GEPA and we have no hook before it executes.

**Solution:** Wrap the default instruction proposer to set context before reflection/proposal phases.

```python
"""core/logged_proposer.py - Proposer wrapper for reflection/proposal phase tagging."""

from typing import Any, Callable
from .context import set_ctx, get_ctx


class LoggedInstructionProposer:
    """
    Wraps an instruction proposer to set context for reflection/proposal LM calls.

    This enables proper phase attribution for ALL LM calls, not just eval.

    Usage:
        from dspy.teleprompt.gepa import DefaultInstructionProposer

        # Wrap the default proposer
        logged_proposer = LoggedInstructionProposer(DefaultInstructionProposer())

        gepa = GEPA(
            metric=logged_metric,
            instruction_proposer=logged_proposer,
            ...
        )
    """

    def __init__(self, base_proposer: Any):
        self.base_proposer = base_proposer
        self.reflection_calls: list[dict] = []
        self.proposal_calls: list[dict] = []

    def propose(
        self,
        current_instructions: dict[str, str],
        candidate_idx: int,
        reflection_data: Any,
        **kwargs
    ) -> list[dict[str, str]]:
        """
        Propose new instructions with context tagging.

        Sets phase="reflection" during reflection, phase="proposal" during generation.
        """
        ctx = get_ctx()
        iteration = ctx.get('iteration', 0)

        # Phase 1: Reflection - analyzing current performance
        set_ctx(phase="reflection", candidate_idx=candidate_idx)

        # Record reflection input
        self.reflection_calls.append({
            'iteration': iteration,
            'candidate_idx': candidate_idx,
            'current_instructions': dict(current_instructions),
        })

        # Phase 2: Proposal - generating new candidates
        set_ctx(phase="proposal", candidate_idx=candidate_idx)

        # Call base proposer
        proposals = self.base_proposer.propose(
            current_instructions=current_instructions,
            candidate_idx=candidate_idx,
            reflection_data=reflection_data,
            **kwargs
        )

        # Record proposals
        self.proposal_calls.append({
            'iteration': iteration,
            'candidate_idx': candidate_idx,
            'num_proposals': len(proposals),
            'proposals': [dict(p) for p in proposals],
        })

        # Reset phase
        set_ctx(phase=None)

        return proposals

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base proposer."""
        return getattr(self.base_proposer, name)


class LoggedSelector:
    """
    Optional: Wraps a selector to set candidate_idx context during selection.

    Use this if you need to track which candidate is being evaluated
    when GEPA selects candidates for the next iteration.
    """

    def __init__(self, base_selector: Any):
        self.base_selector = base_selector

    def select(self, candidates: list, scores: list, **kwargs) -> list[int]:
        """Select candidates with context tracking."""
        # Set phase for any LM calls during selection
        set_ctx(phase="selection")

        selected = self.base_selector.select(candidates, scores, **kwargs)

        set_ctx(phase=None)
        return selected

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_selector, name)
```

### When to use LoggedInstructionProposer:

| Use Case | Need Proposer? | Why |
|----------|:--------------:|-----|
| Just want scores/feedback per example | ❌ | `LoggedMetric` alone is enough |
| Want to tag eval LM calls | ❌ | `LoggedMetric` sets `phase="eval"` |
| Want to tag reflection/proposal LM calls | ✅ | Only way to know which LM call is which |
| Building prompt provenance ("where did this come from?") | ✅ | Need to capture reflection reasoning |

---

## Unified Context Module (v2.2 fix)

**Critical fix:** v2.1 had two different contextvars (`_eval_context` and `_gepa_context`).
v2.2 unifies into a single module.

```python
"""core/context.py - Single shared context for all logging hooks."""

import contextvars
from typing import Any

# THE context - used by ALL hooks
_gepa_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "gepa_context",
    default={}
)


def set_ctx(**kwargs) -> None:
    """
    Update context with new values. Preserves existing values not overwritten.

    Args:
        iteration: Current GEPA iteration
        phase: Current phase ('eval', 'reflection', 'proposal', 'selection', None)
        candidate_idx: Index of candidate being processed
        Any other keys you want to track

    Example:
        set_ctx(iteration=5, phase="eval", candidate_idx=3)
    """
    current = dict(_gepa_context.get() or {})
    # Only update non-None values (allows clearing with explicit None)
    for k, v in kwargs.items():
        if v is not None:
            current[k] = v
        elif k in current:
            del current[k]
    _gepa_context.set(current)


def get_ctx() -> dict:
    """
    Get current context dict.

    Returns:
        dict with keys: iteration, phase, candidate_idx, etc.
    """
    return dict(_gepa_context.get() or {})


def clear_ctx() -> None:
    """Clear all context (call at start of new optimization run)."""
    _gepa_context.set({})


def with_ctx(**kwargs):
    """
    Context manager for temporary context changes.

    Example:
        with with_ctx(phase="reflection"):
            # LM calls here will be tagged with phase="reflection"
            ...
        # Context restored after block
    """
    class ContextManager:
        def __init__(self):
            self.old_ctx = None

        def __enter__(self):
            self.old_ctx = get_ctx()
            set_ctx(**kwargs)
            return self

        def __exit__(self, *args):
            _gepa_context.set(self.old_ctx)

    return ContextManager()
```

### How context flows through the system:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GEPA Iteration Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. GEPAStateLogger.__call__(state)                            │
│     └─ set_ctx(iteration=state.i)                              │
│                                                                 │
│  2. LoggedInstructionProposer.propose(...)                     │
│     └─ set_ctx(phase="reflection", candidate_idx=...)          │
│     └─ [LM calls for reflection - tagged!]                     │
│     └─ set_ctx(phase="proposal", candidate_idx=...)            │
│     └─ [LM calls for proposal - tagged!]                       │
│                                                                 │
│  3. GEPA evaluates candidates via LoggedMetric                 │
│     └─ set_ctx(phase="eval")                                   │
│     └─ [LM calls during metric - tagged!]                      │
│     └─ Records: example_id, score, feedback, candidate_idx     │
│                                                                 │
│  4. DSPyLMLogger.on_lm_start/end(...)                          │
│     └─ Reads get_ctx() to tag each LM call                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture (Revised v2.2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GEPA.compile()                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │  LoggedMetric   │  │  Reflector  │  │  Proposer   │            │
│   │  (your metric)  │  │   (LLM)     │  │   (LLM)     │            │
│   │                 │  │             │  │             │            │
│   │ Captures:       │  └──────┬──────┘  └──────┬──────┘            │
│   │ • score         │         │                │                    │
│   │ • feedback      │         └────────────────┘                    │
│   │ • prediction    │                  │                            │
│   │ • example_id    │                  ▼                            │
│   │ • candidate_idx │         DSPy Callbacks (context-tagged)       │
│   └────────┬────────┘         • iteration, phase, candidate_idx     │
│            │                  • prompts, responses, model           │
│            │                           │                            │
├────────────┼───────────────────────────┼────────────────────────────┤
│            │                           │                            │
│            │   Each Iteration:         │                            │
│            │   ┌───────────────────────┼──────────────────────┐    │
│            │   │   stop_callbacks (incremental snapshots):     │    │
│            │   │                                                │    │
│            │   │   • NEW candidates only (not full copy)       │    │
│            │   │   • NEW lineage links                         │    │
│            │   │   • Pareto DIFFS (additions/removals)         │    │
│            │   │   • Sets context for correlation              │    │
│            │   │                                                │    │
│            │   └────────────────────────────────────────────────┘    │
│            │                                                        │
├────────────┼────────────────────────────────────────────────────────┤
│   After Completion (track_stats=True):                              │
│   • result.detailed_results   (canonical "final truth")             │
│   • result.program            (best program)                        │
└────────────┼────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        GEPATracker                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   state_logger.deltas[]       ←── Incremental state (memory-safe)   │
│   lm_logger.calls[]           ←── Context-tagged LM calls           │
│   metric_logger.evaluations[] ←── Score + feedback + predictions    │
│   result.detailed_results     ←── Canonical final summary           │
│                                                                     │
│                         ▼                                           │
│               Correlation by Context (not timestamps!)              │
│   • LM calls tagged with iteration/phase/candidate_idx              │
│   • Evaluations tagged with example_id/candidate_idx                │
│   • Lineage graph reconstructed from deltas                         │
│   • Pareto evolution reconstructed from diffs                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Coverage Matrix (Revised)

| Data Needed | stop_callbacks | track_stats | DSPy Callbacks | LoggedMetric |
|-------------|:--------------:|:-----------:|:--------------:|:------------:|
| Candidate prompts | ✅ (incremental) | ✅ | | |
| Lineage (parent→child) | ✅ (incremental) | ✅ | | |
| Validation scores | ✅ | ✅ | | ✅ |
| Pareto frontier | ✅ (diffs) | ✅ | | |
| Which programs on Pareto | ✅ (diffs) | ✅ | | |
| Iteration metadata | ✅ | ✅ | | |
| LM prompts/responses | | | ✅ | |
| LM call attribution | | | ✅ (context-tagged) | |
| **Metric feedback (WHY)** | ❌ | ❌ | ❌ | ✅ |
| **Per-example predictions** | ❌ | ❌ | ❌ | ✅ |
| **Example-level deltas** | ❌ | ❌ | ❌ | ✅ |

**Conclusion:** All four hooks together cover 100% of requirements:
- `stop_callbacks` → Candidates, lineage, Pareto (incremental, memory-safe)
- `track_stats=True` → Canonical final summary
- DSPy callbacks → LM calls (context-tagged, thread-safe)
- `LoggedMetric` → Feedback, predictions, example-level analysis (the "why")

---

## Implementation Plan

### Phase 1: Core Logger (No Monkey-Patching)

```
src/dspy_gepa_logger/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── state_logger.py      # stop_callbacks implementation
│   ├── lm_logger.py         # DSPy callback implementation
│   └── tracker.py           # Combines both + stores data
├── storage/
│   ├── __init__.py
│   └── sqlite_store.py      # Persist to SQLite (simple start)
└── api.py                   # Public API for users
```

### Phase 2: Implementation

#### Step 1: State Logger (`state_logger.py`) - Incremental Snapshots (v2.2)

**Key fixes in v2.2:**
- Uses unified `context.py` module (not separate contextvar)
- Stores full Pareto program sets (not arbitrary first element)
- Uses deterministic selection (`min()`) when single index needed
- Seed detection doesn't assume iteration 0

```python
"""core/state_logger.py - Captures GEPA state incrementally via stop_callbacks."""

import time
import logging
from dataclasses import dataclass, field
from typing import Any

import dspy

from .context import set_ctx, get_ctx, clear_ctx  # Unified context!


@dataclass
class IterationDelta:
    """Incremental changes from one iteration to the next."""
    iteration: int
    timestamp: float
    total_evals: int

    # Only NEW candidates added this iteration
    new_candidates: list[tuple[int, dict[str, str]]]  # (idx, content)

    # Only NEW parent links
    new_lineage: list[tuple[int, list[int | None]]]  # (child_idx, parent_idxs)

    # Pareto frontier changes - store FULL sets, not single index
    pareto_additions: dict[str, tuple[float, set[int]]]  # data_id -> (score, program_idxs)
    pareto_removals: set[str]  # data_ids removed from frontier


@dataclass
class IterationMetadata:
    """Lightweight metadata per iteration (always stored)."""
    iteration: int
    timestamp: float
    total_evals: int
    num_candidates: int
    pareto_size: int


class GEPAStateLogger:
    """
    Callback for stop_callbacks that captures state INCREMENTALLY.

    Avoids O(iterations × state_size) memory by only storing diffs.

    NOTE: This is NOT a "stopper" - it always returns False.
    We just use stop_callbacks as the iteration hook.
    """

    def __init__(self, dspy_version: str | None = None, gepa_version: str | None = None):
        self.deltas: list[IterationDelta] = []
        self.metadata: list[IterationMetadata] = []
        self._start_time: float | None = None

        # Track previous state for diffing
        self._prev_num_candidates: int = 0
        self._prev_pareto: dict[str, float] = {}

        # Version info for compatibility
        self.versions = {
            'dspy': dspy_version or getattr(dspy, '__version__', 'unknown'),
            'gepa': gepa_version or 'unknown',
        }

        # Seed candidate - detected by lineage, NOT by iteration==0
        self.seed_candidate: dict[str, str] | None = None
        self.seed_candidate_idx: int | None = None

        # Final state (populated at end)
        self.final_candidates: list[dict[str, str]] = []
        self.final_lineage: list[list[int | None]] = []
        self.final_pareto: dict[str, float] = {}
        self.final_pareto_programs: dict[str, set[int]] = {}  # Store full sets!

        self._logger = logging.getLogger(__name__)

    def __call__(self, state) -> bool:
        """Called by GEPA each iteration. Returns False to continue (never stops)."""
        if self._start_time is None:
            self._start_time = time.time()
            clear_ctx()  # Start fresh

        try:
            self._record_iteration(state)
        except Exception as e:
            # Don't crash GEPA if logging fails
            self._logger.warning(f"State logging failed at iteration {getattr(state, 'i', '?')}: {e}")

        # Update context for downstream hooks (LoggedMetric, LM callbacks)
        set_ctx(iteration=self._safe_get(state, 'i', 0))

        return False  # Never stop - we're just logging

    def _record_iteration(self, state):
        """Record incremental changes from this iteration."""
        iteration = self._safe_get(state, 'i', 0)
        timestamp = time.time() - self._start_time
        total_evals = self._safe_get(state, 'total_num_evals', 0)

        candidates = self._safe_get(state, 'program_candidates', [])
        lineage = self._safe_get(state, 'parent_program_for_candidate', [])
        pareto = dict(self._safe_get(state, 'pareto_front_valset', {}))
        pareto_programs = self._safe_get(state, 'program_at_pareto_front_valset', {})

        # Detect seed candidate by lineage (parent is None), NOT by iteration==0
        if self.seed_candidate is None and candidates and lineage:
            for idx, parents in enumerate(lineage):
                if not parents or parents[0] is None:
                    self.seed_candidate = dict(candidates[idx])
                    self.seed_candidate_idx = idx
                    break
            # Fallback: first candidate if no parentless candidate found
            if self.seed_candidate is None and candidates:
                self.seed_candidate = dict(candidates[0])
                self.seed_candidate_idx = 0

        # Compute NEW candidates (indices >= prev count)
        new_candidates = []
        for idx in range(self._prev_num_candidates, len(candidates)):
            new_candidates.append((idx, dict(candidates[idx])))

        # Compute NEW lineage entries
        new_lineage = []
        for idx in range(self._prev_num_candidates, len(lineage)):
            new_lineage.append((idx, list(lineage[idx])))

        # Compute Pareto frontier changes - store FULL sets
        pareto_additions: dict[str, tuple[float, set[int]]] = {}
        pareto_removals: set[str] = set()

        for data_id, score in pareto.items():
            if data_id not in self._prev_pareto or self._prev_pareto[data_id] != score:
                # New or improved - store the FULL set of program indices
                prog_set = pareto_programs.get(data_id, set())
                if isinstance(prog_set, set):
                    pareto_additions[data_id] = (score, set(prog_set))
                else:
                    # Handle case where it's not a set
                    pareto_additions[data_id] = (score, {prog_set} if prog_set is not None else set())

        for data_id in self._prev_pareto:
            if data_id not in pareto:
                pareto_removals.add(data_id)

        # Store delta
        delta = IterationDelta(
            iteration=iteration,
            timestamp=timestamp,
            total_evals=total_evals,
            new_candidates=new_candidates,
            new_lineage=new_lineage,
            pareto_additions=pareto_additions,
            pareto_removals=pareto_removals,
        )
        self.deltas.append(delta)

        # Store lightweight metadata
        meta = IterationMetadata(
            iteration=iteration,
            timestamp=timestamp,
            total_evals=total_evals,
            num_candidates=len(candidates),
            pareto_size=len(pareto),
        )
        self.metadata.append(meta)

        # Update tracking for next iteration
        self._prev_num_candidates = len(candidates)
        self._prev_pareto = pareto

        # Keep final state reference (shallow, for end-of-run)
        self.final_candidates = [dict(c) for c in candidates]
        self.final_lineage = [list(p) for p in lineage]
        self.final_pareto = pareto
        self.final_pareto_programs = {
            k: set(v) if isinstance(v, set) else {v}
            for k, v in pareto_programs.items()
        }

    def _safe_get(self, state, attr: str, default: Any) -> Any:
        """Safely get attribute with fallback (version compatibility)."""
        try:
            return getattr(state, attr, default)
        except Exception:
            return default

    def get_all_candidates(self) -> list[dict[str, str]]:
        """Reconstruct all candidates from deltas."""
        candidates = []
        for delta in self.deltas:
            for idx, content in delta.new_candidates:
                while len(candidates) <= idx:
                    candidates.append({})
                candidates[idx] = content
        return candidates

    def get_lineage(self, candidate_idx: int) -> list[int]:
        """Trace a candidate back to its ancestors."""
        if not self.final_lineage or candidate_idx >= len(self.final_lineage):
            return [candidate_idx]

        lineage = [candidate_idx]
        current = candidate_idx

        while current is not None and current < len(self.final_lineage):
            parents = self.final_lineage[current]
            if parents and parents[0] is not None:
                lineage.append(parents[0])
                current = parents[0]
            else:
                break

        return lineage

    def get_pareto_evolution(self) -> list[dict[str, tuple[float, set[int]]]]:
        """
        Get Pareto frontier state at each iteration.

        Returns list of dicts: data_id -> (score, set of program indices)
        """
        pareto_states = []
        current_pareto: dict[str, tuple[float, set[int]]] = {}

        for delta in self.deltas:
            # Apply removals
            for data_id in delta.pareto_removals:
                current_pareto.pop(data_id, None)
            # Apply additions
            for data_id, (score, prog_set) in delta.pareto_additions.items():
                current_pareto[data_id] = (score, set(prog_set))

            pareto_states.append(dict(current_pareto))

        return pareto_states

    def get_pareto_best_candidate(self, data_id: str) -> int | None:
        """
        Get deterministic "best" candidate for a Pareto data point.

        Uses min() for deterministic selection when multiple candidates tie.
        """
        prog_set = self.final_pareto_programs.get(data_id)
        if prog_set:
            return min(prog_set)  # Deterministic: lowest index
        return None
```

#### Step 2: LM Logger (`lm_logger.py`) - Context-Tagged (v2.2)

**Key fix: Uses unified `context.py` module (not separate import).**
Robust with `num_threads > 1` and eliminates timing drift issues.

```python
"""core/lm_logger.py - Captures all LM calls via DSPy callbacks with context tagging."""

import time
from dataclasses import dataclass, field
from typing import Any

from .context import get_ctx  # Unified context module!


@dataclass
class LMCall:
    call_id: str
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    model: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    # Context tags (from contextvar, not timestamps!)
    iteration: int | None = None
    phase: str | None = None  # 'reflection', 'proposal', 'eval', 'validation'
    candidate_idx: int | None = None


class DSPyLMLogger:
    """
    DSPy callback that captures all LM calls WITH context tags.

    Instead of correlating by timestamp (fragile with threads),
    we read the current context set by the state logger / metric wrapper.
    """

    def __init__(self):
        self.calls: list[LMCall] = []
        self._pending: dict[str, LMCall] = {}
        self._start_time: float | None = None

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        if self._start_time is None:
            self._start_time = time.time()

        # Read current context (set by state logger, proposer, or metric wrapper)
        ctx = get_ctx()

        call = LMCall(
            call_id=call_id,
            start_time=time.time() - self._start_time,
            model=getattr(instance, 'model', str(type(instance).__name__)),
            inputs=inputs,
            # Tag with context!
            iteration=ctx.get('iteration'),
            phase=ctx.get('phase'),
            candidate_idx=ctx.get('candidate_idx'),
        )
        self._pending[call_id] = call

    def on_lm_end(self, call_id: str, instance: Any, outputs: dict[str, Any]):
        if call_id in self._pending:
            call = self._pending.pop(call_id)
            call.end_time = time.time() - (self._start_time or time.time())
            call.duration_ms = (call.end_time - call.start_time) * 1000
            call.outputs = outputs
            self.calls.append(call)

    def get_calls_for_iteration(self, iteration: int) -> list[LMCall]:
        """Get all LM calls for a specific iteration (by tag, not time!)."""
        return [c for c in self.calls if c.iteration == iteration]

    def get_calls_for_phase(self, phase: str) -> list[LMCall]:
        """Get all LM calls for a specific phase (reflection, proposal, eval)."""
        return [c for c in self.calls if c.phase == phase]

    def get_calls_for_candidate(self, candidate_idx: int) -> list[LMCall]:
        """Get all LM calls related to a specific candidate."""
        return [c for c in self.calls if c.candidate_idx == candidate_idx]

    def get_reflection_calls(self) -> list[LMCall]:
        """Get all reflection LM calls."""
        return self.get_calls_for_phase('reflection')

    def get_proposal_calls(self) -> list[LMCall]:
        """Get all proposal generation LM calls."""
        return self.get_calls_for_phase('proposal')

    def get_eval_calls(self) -> list[LMCall]:
        """Get all evaluation LM calls."""
        return [c for c in self.calls if c.phase in ('eval', 'validation', 'minibatch')]
```

#### Step 3: Combined Tracker (`tracker.py`)

```python
"""Main tracker that combines state, LM, and metric logging."""

from typing import Callable, Any

from .state_logger import GEPAStateLogger
from .lm_logger import DSPyLMLogger
from .logged_metric import LoggedMetric


class GEPATracker:
    """
    Main tracker that coordinates all logging hooks.

    Combines:
    - State logger (stop_callbacks) - iteration state, lineage, Pareto
    - LM logger (DSPy callbacks) - all LM calls
    - Metric logger (LoggedMetric) - evaluations with feedback
    """

    def __init__(self):
        self.state_logger = GEPAStateLogger()
        self.lm_logger = DSPyLMLogger()
        self.metric_logger: LoggedMetric | None = None
        self.run_metadata: dict = {}

    def wrap_metric(
        self,
        metric_fn: Callable,
        *,
        capture_prediction: bool = True,
        capture_trace: bool = False,
    ) -> LoggedMetric:
        """Wrap a metric function for logging. Returns the wrapped metric."""
        self.metric_logger = LoggedMetric(
            metric_fn,
            capture_prediction=capture_prediction,
            capture_trace=capture_trace,
        )
        return self.metric_logger

    def get_stop_callback(self):
        """Returns the callback to pass to gepa_kwargs['stop_callbacks']."""
        return self.state_logger

    def get_dspy_callbacks(self):
        """Returns callbacks to pass to dspy.configure(callbacks=[...])."""
        return [self.lm_logger]

    def get_lm_calls_for_iteration(self, iteration: int) -> list:
        """Get LM calls for iteration (by context tag, not timestamp)."""
        return self.lm_logger.get_calls_for_iteration(iteration)

    def get_evaluations_for_candidate(self, candidate_idx: int) -> list:
        """Get all metric evaluations for a candidate."""
        if self.metric_logger:
            return self.metric_logger.get_evaluations_for_candidate(candidate_idx)
        return []

    def get_summary(self) -> dict:
        """Generate a summary of the optimization run."""
        if not self.state_logger.metadata:
            return {}

        first_meta = self.state_logger.metadata[0]
        last_meta = self.state_logger.metadata[-1]

        return {
            'total_iterations': last_meta.iteration + 1,
            'total_evaluations': last_meta.total_evals,
            'total_lm_calls': len(self.lm_logger.calls),
            'total_metric_calls': len(self.metric_logger.evaluations) if self.metric_logger else 0,
            'seed_candidate': self.state_logger.seed_candidate,
            'final_candidates': self.state_logger.final_candidates,
            'final_pareto_frontier': self.state_logger.final_pareto,
            'duration_seconds': last_meta.timestamp,
            'versions': self.state_logger.versions,
        }

    def get_candidate_diff(self, candidate_a: int, candidate_b: int) -> dict:
        """Compare two candidates (prompts + scores)."""
        candidates = self.state_logger.final_candidates

        if candidate_a >= len(candidates) or candidate_b >= len(candidates):
            return {'error': 'Invalid candidate index'}

        return {
            'candidate_a': {
                'idx': candidate_a,
                'content': candidates[candidate_a],
                'evaluations': self.get_evaluations_for_candidate(candidate_a),
            },
            'candidate_b': {
                'idx': candidate_b,
                'content': candidates[candidate_b],
                'evaluations': self.get_evaluations_for_candidate(candidate_b),
            },
            'lineage_a': self.state_logger.get_lineage(candidate_a),
            'lineage_b': self.state_logger.get_lineage(candidate_b),
        }
```

#### Step 4: Public API (`api.py`) - Fixed kwargs Handling

**Key fix: Clean separation of `gepa_kwargs` to avoid double-nesting.**

```python
"""Public API for dspy-gepa-logger."""

import dspy
from dspy import GEPA
from typing import Callable, Any

from .core.tracker import GEPATracker


def create_logged_gepa(
    metric: Callable,
    *,
    # GEPA constructor args (explicit, not **kwargs)
    num_threads: int = 8,
    max_iterations: int = 10,
    max_proposals_per_step: int = 4,
    track_stats: bool = True,
    # Logger options
    capture_prediction: bool = True,
    capture_trace: bool = False,
    # User's additional gepa_kwargs (clean merge)
    gepa_kwargs: dict[str, Any] | None = None,
) -> tuple[GEPA, GEPATracker, Callable]:
    """
    Create a GEPA optimizer with full logging enabled.

    Args:
        metric: Your metric function (will be wrapped automatically)
        num_threads: GEPA num_threads parameter
        max_iterations: GEPA max_iterations parameter
        max_proposals_per_step: GEPA max_proposals_per_step parameter
        track_stats: Enable GEPA's detailed_results (recommended: True)
        capture_prediction: Store model predictions in LoggedMetric
        capture_trace: Store full traces (large, usually False)
        gepa_kwargs: Additional kwargs to pass to GEPA's gepa_kwargs

    Returns:
        tuple: (gepa_instance, tracker, logged_metric)

    Example:
        gepa, tracker, logged_metric = create_logged_gepa(my_metric)

        # Configure DSPy with LM callbacks
        dspy.configure(
            lm=my_lm,
            callbacks=tracker.get_dspy_callbacks()
        )

        # Run optimization
        result = gepa.compile(student, trainset=trainset, valset=valset)

        # Access logged data
        summary = tracker.get_summary()
        print(f"Ran {summary['total_iterations']} iterations")

        # Compare candidates
        diff = tracker.get_candidate_diff(0, 5)
        print(f"Seed prompt: {diff['candidate_a']['content']}")
        print(f"Optimized prompt: {diff['candidate_b']['content']}")

        # Analyze regressions
        for eval in logged_metric.evaluations:
            if eval.score < 0.5:
                print(f"Low score on {eval.example_id}: {eval.feedback}")
    """
    tracker = GEPATracker()

    # Wrap the metric for logging
    logged_metric = tracker.wrap_metric(
        metric,
        capture_prediction=capture_prediction,
        capture_trace=capture_trace,
    )

    # Build gepa_kwargs with clean merge (no double-nesting!)
    user_gepa_kwargs = gepa_kwargs or {}

    # Extract and merge stop_callbacks
    user_stop_callbacks = user_gepa_kwargs.pop('stop_callbacks', [])
    merged_stop_callbacks = [tracker.get_stop_callback()] + list(user_stop_callbacks)

    merged_gepa_kwargs = {
        **user_gepa_kwargs,
        'stop_callbacks': merged_stop_callbacks,
    }

    # Create GEPA with wrapped metric
    gepa = GEPA(
        metric=logged_metric,  # Wrapped metric!
        num_threads=num_threads,
        max_iterations=max_iterations,
        max_proposals_per_step=max_proposals_per_step,
        track_stats=track_stats,
        gepa_kwargs=merged_gepa_kwargs,
    )

    return gepa, tracker, logged_metric


def configure_dspy_logging(tracker: GEPATracker, lm: Any, **dspy_kwargs):
    """
    Helper to configure DSPy with logging callbacks.

    Args:
        tracker: GEPATracker instance from create_logged_gepa
        lm: DSPy LM instance
        **dspy_kwargs: Additional args for dspy.configure

    Example:
        gepa, tracker, _ = create_logged_gepa(my_metric)
        configure_dspy_logging(tracker, dspy.LM("openai/gpt-4o-mini"))
    """
    existing_callbacks = dspy_kwargs.pop('callbacks', [])
    all_callbacks = list(existing_callbacks) + tracker.get_dspy_callbacks()

    dspy.configure(
        lm=lm,
        callbacks=all_callbacks,
        **dspy_kwargs,
    )
```

---

## Usage Example (Updated)

```python
import dspy
from dspy_gepa_logger import create_logged_gepa, configure_dspy_logging

# Your metric that returns (score, feedback)
def my_metric(example, prediction, trace=None):
    is_correct = prediction.answer == example.answer
    score = 1.0 if is_correct else 0.0
    feedback = f"Expected '{example.answer}', got '{prediction.answer}'"
    return score, feedback  # LoggedMetric captures BOTH!

# Create logged GEPA (metric is wrapped automatically)
gepa, tracker, logged_metric = create_logged_gepa(
    my_metric,
    max_iterations=10,
    capture_prediction=True,  # Store model outputs
)

# Configure DSPy with LM logging (context-tagged, thread-safe)
configure_dspy_logging(tracker, dspy.LM("openai/gpt-4o-mini"))

# Define your program
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

# Run optimization
student = MyProgram()
result = gepa.compile(student, trainset=trainset, valset=valset)

# ============================================
# Access all logged data
# ============================================

summary = tracker.get_summary()
print(f"Total iterations: {summary['total_iterations']}")
print(f"Total LM calls: {summary['total_lm_calls']}")
print(f"Total metric calls: {summary['total_metric_calls']}")

# Compare initial vs final prompts
print(f"\nSeed prompt:\n{summary['seed_candidate']}")
print(f"\nFinal prompts:\n{summary['final_candidates']}")

# ============================================
# Lineage analysis ("where did this come from?")
# ============================================

# Get lineage for best candidate
lineage = tracker.state_logger.get_lineage(candidate_idx=5)
print(f"\nLineage for candidate 5: {lineage}")
# Output: [5, 3, 1, 0] - candidate 5 came from 3, which came from 1, which came from seed (0)

# Compare two candidates in the lineage
diff = tracker.get_candidate_diff(0, 5)
print(f"\nPrompt diff:")
print(f"  Before: {diff['candidate_a']['content']}")
print(f"  After: {diff['candidate_b']['content']}")

# ============================================
# Example-level analysis ("why did performance change?")
# ============================================

# Find regressions
for eval_record in logged_metric.evaluations:
    if eval_record.candidate_idx == 5 and eval_record.score < 0.5:
        print(f"\nRegression on example {eval_record.example_id}:")
        print(f"  Feedback: {eval_record.feedback}")
        print(f"  Prediction: {eval_record.prediction}")

# Compute lift between seed and optimized for a specific example
lift = logged_metric.compute_lift(
    example_id="example_42",
    before_candidate=0,
    after_candidate=5
)
print(f"\nLift on example_42: {lift['lift']}")
print(f"  Before: {lift['before'].feedback}")
print(f"  After: {lift['after'].feedback}")

# ============================================
# LM call analysis (by context, not timestamps!)
# ============================================

# Get all LM calls for iteration 5
iter5_calls = tracker.get_lm_calls_for_iteration(5)
print(f"\nLM calls in iteration 5: {len(iter5_calls)}")

# Get reflection vs proposal calls
reflection_calls = tracker.lm_logger.get_reflection_calls()
proposal_calls = tracker.lm_logger.get_proposal_calls()
print(f"Reflection calls: {len(reflection_calls)}")
print(f"Proposal calls: {len(proposal_calls)}")

# ============================================
# Pareto frontier evolution
# ============================================

pareto_history = tracker.state_logger.get_pareto_evolution()
print(f"\nPareto frontier at each iteration:")
for i, pareto in enumerate(pareto_history):
    print(f"  Iteration {i}: {len(pareto)} examples covered, avg score {sum(pareto.values())/len(pareto):.2f}")
```

---

## What We've Eliminated

| Old Approach | Problem | New Approach |
|--------------|---------|--------------|
| Monkey-patch `_run_full_eval_and_add` | Accesses private `_val_evaluator` that doesn't exist | Use `stop_callbacks` (public API) |
| Hook into `DspyAdapter` methods | Adapter created internally, hard to inject | State already contains adapter outputs |
| Parse GEPA console logs | Fragile, incomplete | `full_program_trace` contains structured data |
| Wrap internal evaluator | Private API | `prog_candidate_val_subscores` in state |

---

## Next Steps

1. **Implement the core logger** - Use the code above as starting point
2. **Test context correlation** - Verify that `contextvar` tags propagate correctly with `num_threads > 1`
3. **Inspect `state` fields empirically** - Confirm exact field names/types in the live GEPA version
4. **Add storage layer** - SQLite for persistence (or event log for replay)
5. **Build visualization** - Lineage graph, Pareto evolution, regression tables

---

## Open Questions (Mostly Resolved)

| Question | Status | Answer |
|----------|--------|--------|
| What's in `full_program_trace`? | To verify | Need runtime inspection, but not critical since we capture everything else |
| Does `best_outputs_valset` have predictions? | Bypassed | `LoggedMetric` captures all predictions directly |
| Can we identify reflection vs proposal LM calls? | ✅ Solved | Context tags (`phase`) set before each operation |
| Will context tags work with threading? | To verify | `contextvars` are designed for this, but test with `num_threads > 1` |

---

## Summary of Changes from v2 → v2.1

| Aspect | v2 (Original) | v2.1 (Revised) |
|--------|---------------|----------------|
| LoggedMetric | "Probably not needed" | **Required** - only source for feedback |
| State snapshots | Deep copy every iteration | **Incremental diffs** - memory-safe |
| LM correlation | Timestamp ranges | **Context tags** - thread-safe |
| API wrapper | Buggy kwargs handling | **Clean merge** - no double-nesting |
| Field access | Direct attribute access | **Safe getattr** with version info |

---

## File Structure (v2.2)

```
src/dspy_gepa_logger/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── context.py           # Single shared contextvar (THE source of truth)
│   ├── state_logger.py      # GEPAStateLogger (incremental snapshots)
│   ├── lm_logger.py         # DSPyLMLogger (context-tagged)
│   ├── logged_metric.py     # LoggedMetric (score + feedback + prediction refs)
│   ├── logged_proposer.py   # LoggedInstructionProposer (phase tagging)
│   └── tracker.py           # GEPATracker (combines all hooks)
├── storage/
│   ├── __init__.py
│   └── sqlite_store.py      # Persist to SQLite
└── api.py                   # create_logged_gepa, configure_dspy_logging
```

---

## Summary: v2.1 → v2.2 Changes

| Issue | v2.1 (Bug) | v2.2 (Fixed) |
|-------|------------|--------------|
| Contextvar mismatch | `_eval_context` vs `_gepa_context` | Single `context.py` with `get_ctx()`/`set_ctx()` |
| Phase not set at eval | Context set only at iteration end | `LoggedMetric` sets `phase="eval"` before metric call |
| Reflection/proposal not tagged | No hook | `LoggedInstructionProposer` sets phase |
| Example ID unstable | `hash()` (salted per process) | SHA256 deterministic hash |
| Pareto set nondeterministic | `list(set)[0]` | Store full sets, use `min()` |
| Seed assumption | `iteration == 0` | Detect by lineage (parent is None) |
| Prediction bloat | Raw objects | Refs + preview |

All fixes maintain the "public hooks only" architecture - no monkey-patching.
