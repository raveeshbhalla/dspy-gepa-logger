# GEPA/DSPy Run Logger (MLflow+) — Technical Plan + Spec

This doc defines a **GEPA-native observability system** (prompt provenance, example-level deltas, Pareto frontier, lineage) with **optional MLflow interop**. It’s designed so you can (a) reconstruct runs perfectly from an **append-only event log**, and (b) power a UX that answers “where did this prompt piece come from?” and “why did performance change?”.

---

## 0) Goals

### Primary goals

* Capture **high-fidelity traces** during GEPA optimization (compile-time) and store them in a queryable form.
* Provide **prompt archaeology**:

  * *Where did this instruction clause come from?*
  * *Which iteration introduced it?*
  * *What evidence (examples + feedback + traces) motivated it?*
* Provide **example-level performance comparisons**:

  * parent vs candidate on same minibatch
  * seed vs final on val set
  * distributions + transition matrices + top improvements/regressions
* Provide **Pareto frontier exploration** and **lineage traversal**.

### Secondary goals

* Work alongside MLflow autologging:

  * MLflow remains the generic experiment store / standard UI
  * This system remains the specialized “GEPA prompt evolution” UI and query backend

---

## 1) Terminology

* **Run**: a single GEPA optimization run.
* **ProgramVersion**: a specific version of a DSPy Program (node in lineage).
* **ComponentVersion**: a specific version of a particular prompt-bearing component (predictor instruction, tool description, etc.).
* **Rollout**: one execution of a program on a single example input.
* **Evaluation**: metric output for a rollout (score + optional feedback); can be program-level or predictor-level.
* **ReflectionEvent**: a mutation step that proposes new component instructions (and is the **edge** in lineage).
* **ParetoSnapshot**: a snapshot of candidate programs and their objective vectors, marking non-dominated programs.

---

## 2) Architecture

### 2.1 High-level components

1. **Instrumentation (in-process)**

   * `GEPAObserver` interface
   * Wrappers:

     * `LoggedMetric`
     * `LoggedInstructionProposer`
     * `LoggedComponentSelector`
   * Optional DSPy callback integration for span-level traces.

2. **Persistence**

   * **Event Log** (authoritative): JSONL (local) or Kafka topic (prod)
   * **Query Store** (derived): Postgres (or DuckDB for local dev)
   * **Blob Store**: S3/GCS/minio/local FS for heavy payloads (traces, full inputs/outputs)

3. **Ingestion + Derivations**

   * `EventIngestor` writes normalized entities to Postgres
   * `DerivationJobs` compute:

     * prompt diff hunks
     * introduced-in-iteration mapping
     * per-example deltas & transition matrices
     * Pareto non-dominance membership
     * embeddings (optional) for semantic search

4. **API + UI**

   * REST API for runs, versions, comparisons, lineage, pareto, traces
   * Web UI (Next.js recommended):

     * Run Overview, Prompt Diff, Example Compare, Pareto Explorer, Lineage Graph, Trace Viewer

### 2.2 Data flow

* GEPA/DSPy emits structured events → appended to event log
* Ingestor streams events → Postgres + blob store refs
* Derivation jobs produce analytics tables → UI queries through REST

---

## 3) Instrumentation Spec

### 3.1 Core observer interface

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Literal
import time
import uuid

EventType = Literal[
  "run_started",
  "run_finished",
  "iteration_started",
  "iteration_finished",
  "program_version_created",
  "component_selection",
  "rollout_started",
  "rollout_finished",
  "evaluation_recorded",
  "reflection_started",
  "reflection_finished",
  "pareto_snapshot",
  "artifact_recorded",
  "error"
]

@dataclass
class ObsEvent:
    event_id: str
    run_id: str
    ts_ms: int
    type: EventType
    payload: Dict[str, Any]

class GEPAObserver(Protocol):
    def emit(self, event: ObsEvent) -> None: ...
    def flush(self) -> None: ...
```

**Design notes**

* Always include: `event_id`, `run_id`, `ts_ms`, `type`, `payload`
* `payload` must be JSON-serializable OR contain blob references (see Blob Store spec).

### 3.2 Identity / correlation model (must-have IDs)

* `run_id`: uuid
* `iteration_id`: int
* `program_version_id`: uuid
* `component_version_id`: uuid
* `example_id`: stable hash ID
* `rollout_id`: uuid
* `evaluation_id`: uuid
* `reflection_id`: uuid
* `pareto_snapshot_id`: uuid

### 3.3 Wrapper: LoggedMetric

Wrap the metric used by GEPA. It must:

* create a `rollout_id`
* store input/output blobs + trace blobs
* record evaluations (program-level and/or predictor-level)

```python
class LoggedMetric:
    def __init__(self, metric_fn, observer: GEPAObserver, run_id: str, trace_level: str = "FULL"):
        self.metric_fn = metric_fn
        self.observer = observer
        self.run_id = run_id
        self.trace_level = trace_level

    def __call__(self, example, prediction, trace=None, **kwargs):
        # kwargs may include pred_name, pred_trace (GEPA)
        rollout_id = str(uuid.uuid4())
        ts = int(time.time() * 1000)

        # 1) rollout_finished event includes references to stored blobs
        self.observer.emit(ObsEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            ts_ms=ts,
            type="rollout_finished",
            payload={
                "rollout_id": rollout_id,
                "example_id": example.get("id"),  # enforce stable ids in dataset loader
                "program_version_id": kwargs.get("program_version_id"),
                "split": kwargs.get("split"),
                "inputs_ref": kwargs.get("inputs_ref"),
                "outputs_ref": kwargs.get("outputs_ref"),
                "trace_ref": kwargs.get("trace_ref"),
                "pred_name": kwargs.get("pred_name"),
                "pred_trace_ref": kwargs.get("pred_trace_ref"),
                "resource_usage": kwargs.get("resource_usage"),
            }
        ))

        # 2) call underlying metric
        score, feedback = self.metric_fn(example, prediction, trace=trace, **kwargs)

        # 3) evaluation event
        self.observer.emit(ObsEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            ts_ms=int(time.time() * 1000),
            type="evaluation_recorded",
            payload={
                "evaluation_id": str(uuid.uuid4()),
                "rollout_id": rollout_id,
                "scope": "predictor" if kwargs.get("pred_name") else "program",
                "pred_name": kwargs.get("pred_name"),
                "score": float(score),
                "feedback": feedback,
                "metric_version": kwargs.get("metric_version", "v1"),
            }
        ))

        return score, feedback
```

**Important**

* Don’t assume `example` already has an ID. Create one in dataset ingestion (see §5.3).
* Store `trace` objects in blob store; put only `trace_ref` in payload.

### 3.4 Wrapper: LoggedInstructionProposer (Reflection)

Wrap GEPA’s instruction proposer (reflection LM call + parsing updates).

Emit:

* `reflection_started` with:

  * parent program version id
  * selected components
  * evidence references (example ids, rollout ids, evaluation ids)
* `reflection_finished` with:

  * candidate program version id
  * raw prompt/response refs
  * proposed component updates
  * diff hunks (optional inline; recommended to compute derivations asynchronously)

```python
class LoggedInstructionProposer:
    def __init__(self, proposer, observer: GEPAObserver, run_id: str):
        self.proposer = proposer
        self.observer = observer
        self.run_id = run_id

    def propose(self, *, parent_program_version_id: str, components: list[str], context: dict, **kwargs):
        reflection_id = str(uuid.uuid4())
        self.observer.emit(ObsEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            ts_ms=int(time.time() * 1000),
            type="reflection_started",
            payload={
                "reflection_id": reflection_id,
                "parent_program_version_id": parent_program_version_id,
                "components": components,
                "context_example_ids": context.get("example_ids", []),
                "context_rollout_ids": context.get("rollout_ids", []),
                "context_evaluation_ids": context.get("evaluation_ids", []),
                "proposer_config": kwargs.get("proposer_config", {}),
            }
        ))

        result = self.proposer.propose(parent_program_version_id=parent_program_version_id,
                                       components=components, context=context, **kwargs)

        # result should include raw reflection prompt/response and parsed updates
        self.observer.emit(ObsEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            ts_ms=int(time.time() * 1000),
            type="reflection_finished",
            payload={
                "reflection_id": reflection_id,
                "candidate_program_version_id": result["candidate_program_version_id"],
                "raw_prompt_ref": result.get("raw_prompt_ref"),
                "raw_response_ref": result.get("raw_response_ref"),
                "proposed_component_updates": result.get("updates", {}),
                "parse_warnings": result.get("warnings", []),
            }
        ))
        return result
```

### 3.5 Wrapper: LoggedComponentSelector

Capture why specific components were selected in an iteration.

```python
class LoggedComponentSelector:
    def __init__(self, selector, observer: GEPAObserver, run_id: str):
        self.selector = selector
        self.observer = observer
        self.run_id = run_id

    def select(self, parent_program_version_id: str, iteration_id: int, **kwargs) -> list[str]:
        selected, reason = self.selector.select(parent_program_version_id, iteration_id, **kwargs)
        self.observer.emit(ObsEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            ts_ms=int(time.time() * 1000),
            type="component_selection",
            payload={
                "iteration_id": iteration_id,
                "parent_program_version_id": parent_program_version_id,
                "selected_components": selected,
                "reason": reason,
                "selector_config": kwargs.get("selector_config", {}),
            }
        ))
        return selected
```

### 3.6 Iteration lifecycle events

At minimum emit:

* `iteration_started`:

  * iteration id, parent program version id, minibatch example ids
* `iteration_finished`:

  * parent minibatch summary, candidate minibatch summary
  * accepted/rejected
  * validation results (if accepted)
  * Pareto snapshot id (if updated)

---

## 4) Event Schema (authoritative log)

### 4.1 Event envelope (JSON)

```json
{
  "event_id": "uuid",
  "run_id": "uuid",
  "ts_ms": 1730000000000,
  "type": "reflection_finished",
  "payload": { }
}
```

### 4.2 Canonical event payloads

#### run_started

```json
{
  "run_id": "uuid",
  "name": "optional",
  "config": {
    "gepa": { "pop_size": 10, "budget": 200, "...": "..." },
    "dspy": { "lm": "gpt-...", "temperature": 0.2 },
    "objectives": [
      {"name": "val_score", "type": "maximize"},
      {"name": "avg_tokens", "type": "minimize"}
    ]
  },
  "seed_program_version_id": "uuid",
  "dataset_versions": {
    "train": {"dataset_id": "uuid", "hash": "sha256..."},
    "val": {"dataset_id": "uuid", "hash": "sha256..."}
  }
}
```

#### program_version_created

```json
{
  "program_version_id": "uuid",
  "parent_program_version_ids": ["uuid"],
  "created_in_iteration": 7,
  "serialization_ref": "blob://.../program.pkl",
  "components": [
    {
      "component_name": "qa_predictor",
      "component_type": "predictor_instruction",
      "component_version_id": "uuid",
      "text_ref": "blob://.../component.txt",
      "text_hash": "sha256..."
    }
  ],
  "summary_metrics": { "train_minibatch_score": 0.62 }
}
```

#### rollout_finished

```json
{
  "rollout_id": "uuid",
  "program_version_id": "uuid",
  "example_id": "ex_...",
  "split": "train_minibatch",
  "inputs_ref": "blob://.../inputs.json",
  "outputs_ref": "blob://.../outputs.json",
  "trace_ref": "blob://.../trace.json",
  "pred_name": "qa_predictor",
  "pred_trace_ref": "blob://.../pred_trace.json",
  "resource_usage": { "tokens_in": 1200, "tokens_out": 240, "latency_ms": 900 }
}
```

#### evaluation_recorded

```json
{
  "evaluation_id": "uuid",
  "rollout_id": "uuid",
  "scope": "predictor",
  "pred_name": "qa_predictor",
  "score": 1.0,
  "feedback": "Missed entity X; add instruction to verify Y.",
  "metric_version": "v1"
}
```

#### reflection_finished

```json
{
  "reflection_id": "uuid",
  "parent_program_version_id": "uuid",
  "candidate_program_version_id": "uuid",
  "raw_prompt_ref": "blob://.../reflection_prompt.txt",
  "raw_response_ref": "blob://.../reflection_response.txt",
  "proposed_component_updates": {
    "qa_predictor": "New instruction text..."
  },
  "parse_warnings": []
}
```

#### pareto_snapshot

```json
{
  "pareto_snapshot_id": "uuid",
  "iteration_id": 7,
  "objective_defs": [
    {"name":"val_score","type":"maximize"},
    {"name":"avg_tokens","type":"minimize"}
  ],
  "members": [
    {
      "program_version_id": "uuid",
      "objective_vector": {"val_score": 0.71, "avg_tokens": 850},
      "is_nondominated": true,
      "tags": ["frontier"]
    }
  ]
}
```

---

## 5) Persistence Spec

### 5.1 Blob store layout

Blob store contains large payloads. Use content-addressed paths where possible.

Recommended path convention:

* `runs/{run_id}/programs/{program_version_id}/program.pkl`
* `runs/{run_id}/components/{component_version_id}.txt`
* `runs/{run_id}/rollouts/{rollout_id}/inputs.json`
* `runs/{run_id}/rollouts/{rollout_id}/outputs.json`
* `runs/{run_id}/rollouts/{rollout_id}/trace.json`
* `runs/{run_id}/reflections/{reflection_id}/prompt.txt`
* `runs/{run_id}/reflections/{reflection_id}/response.txt`

### 5.2 Postgres schema (normalized)

#### runs

```sql
CREATE TABLE runs (
  run_id UUID PRIMARY KEY,
  name TEXT,
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ,
  status TEXT NOT NULL, -- RUNNING|PAUSED|ABANDONED|FINISHED
  config_json JSONB NOT NULL,
  seed_program_version_id UUID,
  final_program_version_id UUID
);
```

#### datasets, examples

```sql
CREATE TABLE datasets (
  dataset_id UUID PRIMARY KEY,
  name TEXT,
  split TEXT, -- train|val|test
  hash TEXT NOT NULL,
  metadata_json JSONB
);

CREATE TABLE examples (
  example_id TEXT PRIMARY KEY, -- stable hash id
  dataset_id UUID REFERENCES datasets(dataset_id),
  inputs_ref TEXT NOT NULL,
  expected_ref TEXT,
  metadata_json JSONB
);
```

**Stable example IDs**

* `example_id = "ex_" + sha256(canonical_json(inputs) + canonical_json(expected))[:24]`

#### program_versions

```sql
CREATE TABLE program_versions (
  program_version_id UUID PRIMARY KEY,
  run_id UUID REFERENCES runs(run_id),
  created_in_iteration INT,
  serialization_ref TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  summary_metrics JSONB
);

CREATE TABLE program_parents (
  program_version_id UUID REFERENCES program_versions(program_version_id),
  parent_program_version_id UUID REFERENCES program_versions(program_version_id),
  edge_type TEXT NOT NULL, -- reflection_mutation|merge
  reflection_id UUID,
  PRIMARY KEY (program_version_id, parent_program_version_id)
);
```

#### component_versions

```sql
CREATE TABLE component_versions (
  component_version_id UUID PRIMARY KEY,
  run_id UUID REFERENCES runs(run_id),
  program_version_id UUID REFERENCES program_versions(program_version_id),
  component_name TEXT NOT NULL,
  component_type TEXT NOT NULL, -- predictor_instruction|tool_description|...
  text_ref TEXT NOT NULL,
  text_hash TEXT NOT NULL
);

CREATE INDEX idx_component_versions_by_program ON component_versions(program_version_id);
CREATE INDEX idx_component_versions_by_name ON component_versions(component_name);
```

#### iterations

```sql
CREATE TABLE iterations (
  run_id UUID REFERENCES runs(run_id),
  iteration_id INT NOT NULL,
  parent_program_version_id UUID REFERENCES program_versions(program_version_id),
  candidate_program_version_id UUID REFERENCES program_versions(program_version_id),
  minibatch_example_ids JSONB NOT NULL, -- array of example_id strings
  parent_minibatch_score DOUBLE PRECISION,
  candidate_minibatch_score DOUBLE PRECISION,
  accepted BOOLEAN,
  validation_score DOUBLE PRECISION,
  pareto_snapshot_id UUID,
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ,
  PRIMARY KEY (run_id, iteration_id)
);
```

#### rollouts + evaluations

```sql
CREATE TABLE rollouts (
  rollout_id UUID PRIMARY KEY,
  run_id UUID REFERENCES runs(run_id),
  program_version_id UUID REFERENCES program_versions(program_version_id),
  example_id TEXT REFERENCES examples(example_id),
  split TEXT NOT NULL, -- train_minibatch|val|test|...
  inputs_ref TEXT NOT NULL,
  outputs_ref TEXT NOT NULL,
  trace_ref TEXT,
  pred_name TEXT,
  pred_trace_ref TEXT,
  resource_usage JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE evaluations (
  evaluation_id UUID PRIMARY KEY,
  run_id UUID REFERENCES runs(run_id),
  rollout_id UUID REFERENCES rollouts(rollout_id),
  scope TEXT NOT NULL, -- program|predictor
  pred_name TEXT,
  score DOUBLE PRECISION NOT NULL,
  feedback TEXT,
  metric_version TEXT NOT NULL
);

CREATE INDEX idx_rollouts_by_program_example ON rollouts(program_version_id, example_id);
CREATE INDEX idx_evals_by_rollout ON evaluations(rollout_id);
```

#### reflections

```sql
CREATE TABLE reflections (
  reflection_id UUID PRIMARY KEY,
  run_id UUID REFERENCES runs(run_id),
  iteration_id INT,
  parent_program_version_id UUID REFERENCES program_versions(program_version_id),
  candidate_program_version_id UUID REFERENCES program_versions(program_version_id),
  raw_prompt_ref TEXT,
  raw_response_ref TEXT,
  proposer_config JSONB,
  parse_warnings JSONB
);

CREATE TABLE reflection_context (
  reflection_id UUID REFERENCES reflections(reflection_id),
  context_example_ids JSONB,
  context_rollout_ids JSONB,
  context_evaluation_ids JSONB
);
```

#### pareto snapshots

```sql
CREATE TABLE pareto_snapshots (
  pareto_snapshot_id UUID PRIMARY KEY,
  run_id UUID REFERENCES runs(run_id),
  iteration_id INT,
  objective_defs JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pareto_members (
  pareto_snapshot_id UUID REFERENCES pareto_snapshots(pareto_snapshot_id),
  program_version_id UUID REFERENCES program_versions(program_version_id),
  objective_vector JSONB NOT NULL,
  is_nondominated BOOLEAN NOT NULL,
  tags JSONB,
  PRIMARY KEY (pareto_snapshot_id, program_version_id)
);
```

### 5.3 Derived tables (analytics)

#### prompt_diff_hunks

```sql
CREATE TABLE prompt_diff_hunks (
  hunk_id UUID PRIMARY KEY,
  run_id UUID REFERENCES runs(run_id),
  component_name TEXT NOT NULL,
  from_component_version_id UUID REFERENCES component_versions(component_version_id),
  to_component_version_id UUID REFERENCES component_versions(component_version_id),
  reflection_id UUID REFERENCES reflections(reflection_id),
  iteration_id INT,
  added_text TEXT,
  removed_text TEXT,
  added_spans JSONB,
  removed_spans JSONB
);

CREATE INDEX idx_hunks_component_iter ON prompt_diff_hunks(component_name, iteration_id);
```

#### example_deltas

```sql
CREATE TABLE example_deltas (
  run_id UUID REFERENCES runs(run_id),
  iteration_id INT,
  example_id TEXT REFERENCES examples(example_id),
  parent_program_version_id UUID REFERENCES program_versions(program_version_id),
  candidate_program_version_id UUID REFERENCES program_versions(program_version_id),
  parent_score DOUBLE PRECISION,
  candidate_score DOUBLE PRECISION,
  delta DOUBLE PRECISION,
  PRIMARY KEY (run_id, iteration_id, example_id)
);
```

#### score_transitions (bucketed)

```sql
CREATE TABLE score_transitions (
  run_id UUID REFERENCES runs(run_id),
  iteration_id INT,
  bucket_scheme TEXT, -- e.g. "quantiles_5" or "bins_0_1_step_0_2"
  from_bucket INT,
  to_bucket INT,
  count INT,
  PRIMARY KEY (run_id, iteration_id, bucket_scheme, from_bucket, to_bucket)
);
```

---

## 6) Derivation Jobs Spec

### 6.1 Prompt diffs & introduced-at-iteration mapping

Input:

* `component_versions` across lineage edges
* `reflections` linking parent ↔ candidate

Process:

* For each reflection edge and each updated component:

  * load parent text and candidate text from blob store
  * compute diff hunks (line and char granularity)
  * store hunks in `prompt_diff_hunks` with `iteration_id` and `reflection_id`

“Introduced at iteration” query:

* for any selected substring `S` in final component text:

  * exact search backwards through ancestor component versions
  * if missing, fallback to semantic match over hunks (optional vector index)

### 6.2 Example deltas & transitions

Input:

* rollouts + evaluations for parent and candidate on same minibatch

Process:

* join on `(example_id, pred_name?)` depending on your metric semantics
* store `example_deltas`
* bucket parent and candidate scores, store `score_transitions`
* compute “top improvements/regressions” (can be derived on-the-fly, or cached)

### 6.3 Pareto membership

Input:

* `pareto_members` objective vectors

Process:

* compute non-dominance
* set `is_nondominated`
* optionally compute dominance explanations (why A dominates B)

---

## 7) REST API Spec

Base: `/api/v1`

### 7.1 Runs

* `GET /runs` → list runs (filters: status, date range)
* `POST /runs` → create run record (optional; often created by event ingestion)
* `GET /runs/{run_id}` → run summary
* `GET /runs/{run_id}/config` → config JSON
* `GET /runs/{run_id}/status` → status + progress
* `GET /runs/{run_id}/events?cursor=...` → raw event stream (debug)
* `GET /runs/{run_id}/artifacts` → artifact refs (MLflow links optional)

### 7.2 Program versions & components

* `GET /runs/{run_id}/programs` → list program versions (filters: frontier-only, accepted-only)
* `GET /runs/{run_id}/programs/{program_version_id}` → program details + summary metrics
* `GET /runs/{run_id}/programs/{program_version_id}/components` → component versions
* `GET /runs/{run_id}/components/{component_version_id}/text` → resolved text (server fetches blob)

### 7.3 Prompt diffs & provenance

* `GET /runs/{run_id}/diff?from={program_version_id}&to={program_version_id}`

  * returns per-component diffs with hunks
* `POST /runs/{run_id}/provenance/locate`

  * body: `{ "program_version_id": "...", "component_name": "...", "selected_text": "..." }`
  * response: earliest match:

    * iteration, reflection_id, parent->candidate, evidence refs
* `GET /runs/{run_id}/reflections/{reflection_id}` → reflection details + context links

### 7.4 Iterations

* `GET /runs/{run_id}/iterations`
* `GET /runs/{run_id}/iterations/{iteration_id}`

  * includes parent/candidate scores, acceptance, validation, pareto snapshot id
* `GET /runs/{run_id}/iterations/{iteration_id}/deltas`

  * includes histogram summary + top improvements/regressions + transitions

### 7.5 Rollouts & traces

* `GET /runs/{run_id}/rollouts?program_version_id=...&example_id=...`
* `GET /runs/{run_id}/rollouts/{rollout_id}`
* `GET /runs/{run_id}/rollouts/{rollout_id}/trace` → resolved trace JSON (or signed URL)
* `GET /runs/{run_id}/rollouts/{rollout_id}/io` → inputs/outputs blobs

### 7.6 Pareto

* `GET /runs/{run_id}/pareto/snapshots`
* `GET /runs/{run_id}/pareto/snapshots/{pareto_snapshot_id}`
* `GET /runs/{run_id}/pareto/frontier?iteration_id=...&x=obj1&y=obj2`

### 7.7 Lineage graph

* `GET /runs/{run_id}/lineage`

  * returns nodes (program versions) and edges (reflection/merge)
  * filters: accepted-only, frontier-only
* `GET /runs/{run_id}/lineage/path?from=seed&to={program_version_id}`

  * returns ordered steps with reflection summaries

---

## 8) UI Spec (Minimum Lovable + Full)

### 8.1 Run Overview

* Run metadata (config, status, runtime)
* Performance over iterations
* “Current best” program(s) per objective
* Quick links: Seed vs Final diff, Pareto Explorer, Lineage Graph

### 8.2 Prompt Diff Viewer (component-aware)

* Left: parent (seed or selected ancestor)
* Right: candidate/final
* Tabs per component
* Each diff hunk shows:

  * introduced iteration
  * link to reflection event
  * link to “evidence examples”

### 8.3 Example Compare

* Table of examples:

  * parent score, candidate score, delta
  * output diff (side-by-side)
  * feedback text
  * deep link to trace
* Filters:

  * regressions only
  * improvements only
  * by task/category tag (if your dataset has it)
  * by score bucket transitions

### 8.4 Lift Decomposition

* Delta histogram
* Transition matrix visualization (bucket → bucket)
* “What improved?” clusters (optional feedback clustering)

### 8.5 Pareto Explorer

* Choose x/y objectives + point size/color
* Frontier highlight
* Click point → opens Program Detail

### 8.6 Program Detail

* Summary objective vector + validation score
* Diff from parent
* Best improvements/regressions examples
* Lineage path back to seed

### 8.7 Lineage Graph

* DAG view:

  * nodes = ProgramVersions
  * edges = reflections / merges
* Clicking edge shows reflection prompt/response and evidence.

### 8.8 Trace Viewer

* Span tree (module/predictor/tool)
* Inputs/outputs per span (redacted or truncated)
* Links back to rollout + evaluation

---

## 9) MLflow Interop

### 9.1 Recommended: Dual-write

* Enable MLflow DSPy autologging for standard artifacts/traces.
* Your system stores:

  * GEPA-native event stream
  * normalized DB
  * derived analytics

### 9.2 Linking runs

In `runs.config_json`, store:

```json
{
  "mlflow": {
    "tracking_uri": "...",
    "experiment_id": "...",
    "mlflow_run_id": "..."
  }
}
```

UI can deep-link to MLflow run pages where useful.

---

## 10) Privacy / Redaction / Governance

### 10.1 Field-level policies

* Inputs/outputs may contain sensitive data; support:

  * redaction rules per key path (`$.user.email`, `$.ssn`, etc.)
  * truncation limits for logs
  * hashing for certain fields

### 10.2 Access control

* Per-run ACLs (team/project)
* Signed URLs for blob access
* Audit logging for viewing traces

---

## 11) Performance & Scaling

### 11.1 Trace level controls

Config:

* `trace_level = NONE | MINIMAL | FULL`
* `store_trace_for = accepted_only | all | sample(p)`
  Minimum lovable default: `FULL for accepted candidates, MINIMAL for rejected`.

### 11.2 Storage strategy

* Postgres stores metadata + references.
* Blob store stores heavy payloads, compressed.

### 11.3 Event ingestion

* Local: write JSONL + periodic batch ingest
* Prod: Kafka → consumer → Postgres/Blob

---

## 12) “End-to-end” sequence (happy path)

1. `run_started`
2. `program_version_created` (seed)
3. For each iteration:

   * `iteration_started` (minibatch IDs)
   * evaluate parent:

     * `rollout_finished` x N
     * `evaluation_recorded` x N
   * `component_selection`
   * `reflection_started`
   * `reflection_finished` (proposed updates)
   * `program_version_created` (candidate)
   * evaluate candidate:

     * `rollout_finished` x N
     * `evaluation_recorded` x N
   * if accepted:

     * validation rollouts + evals
     * `pareto_snapshot`
   * `iteration_finished`
4. `run_finished` (final program version id)

---

## 13) Implementation Checklist

### Phase 1 (MVP)

* [ ] Define event envelope + writer (JSONL)
* [ ] `GEPAObserver` + wrappers (metric/proposer/selector)
* [ ] Blob store writer + ref system
* [ ] Postgres schema (core tables)
* [ ] Ingestor from JSONL → Postgres
* [ ] Basic UI: Run overview, Seed vs Final component diff, Example compare table

### Phase 2 (Archaeology + lift)

* [ ] Diff hunk derivation
* [ ] Provenance “locate selected text”
* [ ] Example deltas + transitions
* [ ] UI: lift decomposition & “introduced at iteration” linking

### Phase 3 (Pareto + lineage graph)

* [ ] Objective vector definitions
* [ ] Pareto non-dominance computation
* [ ] Lineage DAG endpoints + UI graph

---

## 14) Appendix: Recommended config schema (run config)

```json
{
  "trace_level": "FULL",
  "store_trace_for": "accepted_only",
  "objectives": [
    {"name":"val_score","type":"maximize"},
    {"name":"avg_tokens","type":"minimize"}
  ],
  "dataset": {
    "train_name": "train_v3",
    "val_name": "val_v3",
    "example_id_strategy": "sha256(canonical(inputs)+canonical(expected))"
  },
  "gepa": {
    "budget": 200,
    "pop_size": 10,
    "use_merge": true
  },
  "interop": {
    "mlflow_enabled": true,
    "mlflow_tracking_uri": "..."
  }
}
```