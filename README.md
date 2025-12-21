# dspy-gepa-logger

Comprehensive logging and visualization infrastructure for DSPy GEPA optimization runs. Track every iteration, compare baseline vs optimized prompts, and generate beautiful HTML comparison reports.

## Features

- **SQLite Storage**: Normalized database schema for efficient querying and analysis
- **Automatic Instrumentation**: Hooks into GEPA internals to capture all data automatically
- **Pareto Frontier Tracking**: Per-task scores and frontier evolution across iterations
- **HTML Comparison Reports**: Side-by-side comparison of baseline vs optimized outputs
- **LM Call Capture**: Records all language model calls with tokens and latency
- **Multiple Export Formats**: JSON, DataFrames, and interactive HTML

## Installation

### From GitHub

```bash
pip install git+https://github.com/raveeshbhalla/dspy-gepa-logger.git
```

### From Source

```bash
git clone https://github.com/raveeshbhalla/dspy-gepa-logger.git
cd dspy-gepa-logger
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import dspy
from dspy.teleprompt import GEPA
from dspy_gepa_logger import GEPARunTracker
from dspy_gepa_logger.storage import SQLiteStorageAdapter, SQLiteStorage
from dspy_gepa_logger.hooks import create_instrumented_gepa, cleanup_instrumented_gepa

# Set up DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Create SQLite storage and tracker
storage = SQLiteStorage("./gepa_runs.db")
adapter = SQLiteStorageAdapter(storage)
tracker = GEPARunTracker(storage=adapter)

# Define your program
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

# Define your metric
def my_metric(example, pred, trace=None):
    return float(example.answer.lower() in pred.answer.lower())

# Create GEPA optimizer
gepa = GEPA(
    metric=my_metric,
    auto="light",
)

# Instrument GEPA for logging
create_instrumented_gepa(gepa, tracker, log_file="./gepa.log")

# Run optimization with tracking
tracker.start_run(config={"optimizer": "GEPA", "auto": "light"})
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

# Generate HTML comparison report
run_id = tracker.current_run.run_id
```

## Generate HTML Reports

After running GEPA, generate a comparison report:

```bash
# Using the CLI
gepa-report --db ./gepa_runs.db --run-id <RUN_ID> --output report.html

# Or programmatically
python -m dspy_gepa_logger.export.html_report \
    --db ./gepa_runs.db \
    --run-id <RUN_ID> \
    --output report.html
```

The report shows:
- **Summary statistics**: Improvements, regressions, scores
- **Prompt comparison**: Baseline vs optimized instructions side-by-side
- **Example comparison**: Click any example to see both outputs with scores
- **Color coding**: Green for improvements, red for regressions

## Data Captured

For each GEPA iteration, the logger captures:

1. **Parent selection**: Which candidate was selected and why
2. **Minibatch evaluation**: Inputs, outputs, scores, and feedback
3. **Reflection**: The reflective LM call with proposed changes
4. **Candidate evaluation**: New candidate's performance
5. **Acceptance decision**: Accept/reject with reasoning
6. **Validation scores**: Full validation set performance
7. **Pareto frontier**: Per-task best programs and scores

## SQLite Schema

The database uses a normalized schema:

```
runs              - Run metadata and final scores
iterations        - Per-iteration data with parent/candidate programs
programs          - Unique program instructions (deduplicated by hash)
examples          - Train and validation examples
rollouts          - Per-example outputs and scores
reflections       - Reflection LM calls and proposed changes
lm_calls          - All LM calls with tokens and latency
pareto_snapshots  - Pareto frontier state per iteration
pareto_tasks      - Per-task best programs and scores
```

### Example Queries

```sql
-- Get run summary
SELECT * FROM run_summary;

-- Get iteration history
SELECT iteration_number, accepted, val_aggregate_score
FROM iterations WHERE run_id = '<RUN_ID>'
ORDER BY iteration_number;

-- Get Pareto frontier evolution
SELECT * FROM pareto_snapshots WHERE run_id = '<RUN_ID>';

-- Compare baseline vs optimized outputs
SELECT e.inputs_json,
       baseline.output_json as baseline_output,
       optimized.output_json as optimized_output,
       baseline.score as baseline_score,
       optimized.score as optimized_score
FROM examples e
JOIN rollouts baseline ON e.example_id = baseline.example_id
JOIN rollouts optimized ON e.example_id = optimized.example_id
WHERE baseline.program_id = <BASELINE_ID>
  AND optimized.program_id = <OPTIMIZED_ID>;
```

## Export to DataFrames

```python
from dspy_gepa_logger.export import DataFrameExporter

exporter = DataFrameExporter(tracker.storage)

# Get iterations as DataFrame
iterations_df = exporter.iterations_df(run_id)
print(iterations_df[['iteration_number', 'accepted', 'val_aggregate_score']])

# Plot optimization progress
import matplotlib.pyplot as plt
plt.plot(iterations_df['iteration_number'], iterations_df['val_aggregate_score'])
plt.xlabel('Iteration')
plt.ylabel('Validation Score')
plt.show()
```

## Export to JSON

```python
from dspy_gepa_logger.export import JSONExporter

exporter = JSONExporter(storage)
exporter.export_run(run_id, './exports/run.json', pretty=True)
```

## Configuration

```python
from dspy_gepa_logger import TrackerConfig, GEPARunTracker

config = TrackerConfig(
    capture_traces=True,        # Capture execution traces
    capture_lm_calls=True,      # Capture LM call details
    capture_full_inputs=True,   # Store full input data
    capture_full_outputs=True,  # Store full output data
)

tracker = GEPARunTracker(storage=adapter, config=config)
```

## Architecture

```
dspy_gepa_logger/
├── core/
│   ├── tracker.py          # Main GEPARunTracker class
│   └── config.py           # TrackerConfig
├── storage/
│   ├── sqlite_backend.py   # SQLite storage implementation
│   ├── sqlite_adapter.py   # Adapter for tracker integration
│   ├── jsonl_backend.py    # JSONL file storage
│   └── memory_backend.py   # In-memory storage (testing)
├── hooks/
│   ├── gepa_adapter.py     # GEPA instrumentation hooks
│   ├── pareto_tracker.py   # Pareto frontier capture
│   └── callback_handler.py # DSPy callback integration
├── models/
│   ├── run.py              # GEPARunRecord
│   ├── iteration.py        # IterationRecord
│   └── ...                 # Other data models
└── export/
    ├── html_report.py      # HTML comparison reports
    ├── json_exporter.py    # JSON export
    └── dataframe.py        # DataFrame export
```

## How It Works

The logger uses monkey-patching to instrument GEPA without modifying DSPy core:

1. **`create_instrumented_gepa()`** patches:
   - `DspyAdapter.__init__` - Wraps the adapter with instrumented version
   - `GEPAEngine._run_full_eval_and_add` - Captures validation rollouts
   - `create_experiment_tracker` - Injects Pareto frontier tracking

2. **InstrumentedDspyAdapter** wraps:
   - `evaluate()` - Captures minibatch evaluations
   - `make_reflective_dataset()` - Captures reflection data
   - `propose_new_texts()` - Captures proposed changes

3. **GEPALoggingCallback** captures:
   - All LM calls via DSPy's callback system
   - Token counts, latencies, and model info

## Requirements

- Python >= 3.10
- dspy >= 2.5.0
- gepa >= 0.0.17
- pandas >= 2.0.0

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.
