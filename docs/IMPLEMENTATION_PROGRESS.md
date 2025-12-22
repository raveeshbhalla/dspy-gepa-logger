# GEPA Logger v2.2 Implementation Progress

**Branch:** `feat/implement-v2.2-logger`
**Started:** 2025-12-22
**Reference:** [IMPLEMENTATION_PLAN_V2.md](./IMPLEMENTATION_PLAN_V2.md)

---

## Overview

This document tracks the phase-wise implementation of the v2.2 logger architecture that replaces monkey-patching with public hooks.

### Key Changes from v1 (Current Implementation)

| Aspect | v1 (Current) | v2.2 (New) |
|--------|--------------|------------|
| State capture | Monkey-patch `_run_full_eval_and_add` | Use `stop_callbacks` via `gepa_kwargs` |
| LM call tagging | Timestamp correlation | Context tags via `contextvars` |
| Feedback capture | Parse trajectories | `LoggedMetric` wrapper |
| Phase attribution | Not available | `LoggedInstructionProposer` wrapper |
| Memory usage | Deep copy full state | Incremental snapshots (diffs only) |

---

## Phase 1: Core Context System

**File:** `src/dspy_gepa_logger/core/context.py`

**Goal:** Create a unified contextvar system for tagging LM calls with iteration/phase/candidate info.

### Tasks
- [ ] Create `context.py` with `set_ctx()`, `get_ctx()`, `clear_ctx()`
- [ ] Add `with_ctx()` context manager for temporary context changes
- [ ] Write unit tests for context operations
- [ ] Verify thread-safety with concurrent context access

### Passing Criteria
- All unit tests pass
- Context operations are thread-safe
- `set_ctx(key=None)` clears that key
- Context manager restores previous state on exit

### Test File
`tests/test_context.py`

---

## Phase 2: LoggedMetric Wrapper

**File:** `src/dspy_gepa_logger/core/logged_metric.py`

**Goal:** Wrap user metrics to capture score, feedback, and prediction for every evaluation.

### Dependencies
- Phase 1: Context system (for reading iteration/phase/candidate_idx)

### Tasks
- [ ] Create `EvaluationRecord` dataclass
- [ ] Implement `_deterministic_example_id()` using SHA256
- [ ] Implement `_serialize_prediction()` with ref + preview
- [ ] Create `LoggedMetric` class that wraps metrics
- [ ] Add `set_ctx(phase="eval")` before calling wrapped metric
- [ ] Add helper methods: `get_evaluations_for_example()`, `get_evaluations_for_candidate()`, `compute_lift()`, `get_regressions()`
- [ ] Write unit tests with mock metrics

### Passing Criteria
- Wrapped metric returns same result as original
- Score, feedback, and prediction captured correctly
- Deterministic example IDs are stable across runs
- Context phase is set to "eval" during metric execution
- Helper methods return correct filtered data

### Test File
`tests/test_logged_metric.py`

---

## Phase 3: LM Logger

**File:** `src/dspy_gepa_logger/core/lm_logger.py`

**Goal:** Create DSPy callback that captures all LM calls with context tags.

### Dependencies
- Phase 1: Context system (for reading tags)

### Tasks
- [ ] Create `LMCall` dataclass with context fields
- [ ] Implement `DSPyLMLogger` with `on_lm_start()` and `on_lm_end()` hooks
- [ ] Tag LM calls with iteration/phase/candidate_idx from context
- [ ] Add query methods: `get_calls_for_iteration()`, `get_calls_for_phase()`, etc.
- [ ] Write unit tests with mock LM calls

### Passing Criteria
- LM calls are captured with correct start/end times
- Context tags (iteration, phase, candidate_idx) are attached
- Query methods filter correctly
- Handles concurrent calls via pending call tracking

### Test File
`tests/test_lm_logger.py`

---

## Phase 4: State Logger (stop_callbacks Hook)

**File:** `src/dspy_gepa_logger/core/state_logger.py`

**Goal:** Capture GEPA state incrementally via `stop_callbacks`.

### Dependencies
- Phase 1: Context system (for setting iteration context)

### Tasks
- [ ] Create `IterationDelta` and `IterationMetadata` dataclasses
- [ ] Implement `GEPAStateLogger` as a callable for `stop_callbacks`
- [ ] Capture incremental changes only (new candidates, new lineage, Pareto diffs)
- [ ] Detect seed candidate by lineage (parent is None), not iteration==0
- [ ] Store full Pareto program sets, use `min()` for deterministic selection
- [ ] Add helper methods: `get_all_candidates()`, `get_lineage()`, `get_pareto_evolution()`
- [ ] Write unit tests with mock GEPA state objects

### Passing Criteria
- State logger always returns `False` (never stops optimization)
- Only incremental changes stored (memory-efficient)
- Seed candidate detected correctly
- Lineage tracing works correctly
- Pareto evolution can be reconstructed from diffs

### Test File
`tests/test_state_logger.py`

---

## Phase 5: LoggedInstructionProposer

**File:** `src/dspy_gepa_logger/core/logged_proposer.py`

**Goal:** Wrap instruction proposer to set context for reflection/proposal phases.

### Dependencies
- Phase 1: Context system

### Tasks
- [ ] Create `LoggedInstructionProposer` that wraps a base proposer
- [ ] Set `phase="reflection"` before reflection, `phase="proposal"` before proposal
- [ ] Record reflection and proposal calls
- [ ] Delegate unknown attributes to base proposer
- [ ] Write unit tests with mock proposer

### Passing Criteria
- Context phase is set correctly during reflection and proposal
- Base proposer is called with correct arguments
- Unknown attributes are delegated to base proposer
- Reflection/proposal calls are recorded

### Test File
`tests/test_logged_proposer.py`

---

## Phase 6: GEPATracker

**File:** `src/dspy_gepa_logger/core/tracker.py` (REPLACE existing)

**Goal:** Combine all hooks into a unified tracker.

### Dependencies
- Phase 1-5: All individual components

### Tasks
- [ ] Create new `GEPATracker` class
- [ ] Integrate `GEPAStateLogger`, `DSPyLMLogger`, `LoggedMetric`
- [ ] Add `wrap_metric()` helper
- [ ] Add `get_stop_callback()` for `gepa_kwargs`
- [ ] Add `get_dspy_callbacks()` for DSPy configuration
- [ ] Add `get_summary()`, `get_candidate_diff()` methods
- [ ] Write integration tests

### Passing Criteria
- All components work together
- Context flows correctly between components
- Summary includes all expected fields
- Candidate diff shows prompts and evaluations

### Test File
`tests/test_tracker_v2.py`

---

## Phase 7: Public API

**File:** `src/dspy_gepa_logger/api.py`

**Goal:** Create user-friendly API that sets up everything correctly.

### Dependencies
- Phase 6: GEPATracker

### Tasks
- [ ] Create `create_logged_gepa()` function
- [ ] Create `configure_dspy_logging()` helper
- [ ] Handle `gepa_kwargs` merging without double-nesting
- [ ] Update `__init__.py` exports
- [ ] Write integration tests with mock GEPA

### Passing Criteria
- `create_logged_gepa()` returns (gepa, tracker, logged_metric) tuple
- `gepa_kwargs` merged correctly (no double-nesting)
- DSPy callbacks configured correctly
- Existing user `stop_callbacks` preserved

### Test File
`tests/test_api_v2.py`

---

## Progress Log

### 2025-12-22

- [x] Created implementation progress document
- [x] Phase 1: Core context system (23 tests pass)
- [x] Phase 2: LoggedMetric wrapper (32 tests pass)
- [x] Phase 3: LM Logger (28 tests pass)
- [x] Phase 4: State Logger (24 tests pass)
- [x] Phase 5: LoggedInstructionProposer (24 tests pass)
- [x] Phase 6: GEPATracker (51 tests pass)
- [x] Phase 7: Public API (19 tests pass)
- [x] Integration testing with real GEPA run

**Total tests: 201 passing**

---

## Integration Testing Results

### Example: `examples/eg_v2_simple.py`

Successfully captures:
- **LM Calls**: 100 total with phase tags (`'eval': 24, 'untagged': 76`)
- **Evaluations**: 90 total with scores and feedback
- **State**: Candidates, lineage, Pareto evolution
- **Candidate Diff**: Prompt changes between candidates

### Runtime Errors Encountered & Fixed

#### Error 1: GEPA 5-argument metric requirement
**Error**: `TypeError: GEPA metric must accept five arguments`
**Cause**: GEPA metrics require `(gold, pred, trace, pred_name, pred_trace)` but LoggedMetric only accepted 3 args.
**Fix**: Updated `LoggedMetric.__call__` to accept 5 args and dynamically call the underlying metric with the correct signature using `inspect.signature()`.
**File**: `src/dspy_gepa_logger/core/logged_metric.py`

#### Error 2: DSPy 3.0+ callback interface changes
**Error**: `DSPyLMLogger.on_lm_end() missing 1 required positional argument: 'instance'`
**Cause**: DSPy 3.0+ changed `on_lm_end` signature from `(call_id, instance, outputs, exception)` to `(call_id, outputs, exception)` - `instance` is only passed to `on_lm_start`.
**Fix**: Updated `on_lm_end` signature to match DSPy 3.0+ interface. Also added stub methods for new callbacks: `on_module_start`, `on_module_end`, `on_evaluate_start`, `on_evaluate_end`, `on_adapter_format_start`, `on_adapter_format_end`, `on_adapter_parse_start`, `on_adapter_parse_end`.
**File**: `src/dspy_gepa_logger/core/lm_logger.py`

#### Error 3: State logger dict conversion error
**Error**: `cannot convert dictionary update sequence element #0 to a sequence`
**Cause**: GEPA's `pareto_front_valset` returns a dict-like object that doesn't iterate correctly for `dict()` conversion.
**Fix**: Added `_safe_dict_convert()` method that uses `.items()` for dict-like objects that don't iterate as (key, value) pairs.
**File**: `src/dspy_gepa_logger/core/state_logger.py`

#### Error 4: dspy.Prediction return type not handled
**Error**: Evaluations captured score=dspy.Prediction object instead of float
**Cause**: Metrics can return `dspy.Prediction(score=..., feedback=...)` but LoggedMetric expected `(score, feedback)` tuples.
**Fix**: Added handling for objects with `.score` attribute in addition to tuple format.
**File**: `src/dspy_gepa_logger/core/logged_metric.py`

### Known Limitations

1. **candidate_idx is None**: GEPA doesn't expose which candidate is being evaluated through public hooks. Would require monkey-patching or additional inference to capture this.

2. **Reflection/proposal phase tagging**: Requires using `wrap_proposer=True` in `create_logged_gepa()` to tag reflection and proposal LM calls. Without this, those calls show as 'untagged'.

---

## Open Questions

| Question | Status | Notes |
|----------|--------|-------|
| Keep existing v1 code alongside v2? | Yes | Both exports available for backwards compatibility |
| Storage layer changes needed? | TBD | v2 data structures differ from v1 |
| Visualization updates? | Future | After v2 core is working |
| candidate_idx tracking | Limitation | Not available through public hooks |
