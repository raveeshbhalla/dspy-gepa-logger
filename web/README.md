# GEPA Logger Web Dashboard

A web dashboard for tracking DSPy GEPA optimization runs with real-time updates, project organization, and persistent history.

## Features

- **Projects**: Organize optimization runs by project
- **Real-time Updates**: Live stats and progress via Server-Sent Events (SSE)
- **Run History**: Browse past optimization runs with full details
- **Evaluation Comparison**: Interactive tables showing improvements, regressions, and unchanged examples
- **Phase-based Filtering**: Distinguishes minibatch vs valset evaluations for accurate performance metrics
- **Prompt Comparison**: Side-by-side view of original vs optimized prompts
- **Lineage Tracking**: Visualize candidate ancestry
- **Full Prediction Display**: Modal navigation with complete prediction output (no truncation)

## Tech Stack

- **Frontend**: Next.js 16 with React
- **UI Components**: shadcn/ui
- **Database**: SQLite with Prisma ORM
- **Real-time**: Server-Sent Events (SSE)

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

Copy the example environment file (or create `.env` if it doesn't exist):

```bash
# The .env file should contain:
DATABASE_URL="file:./dev.db"
```

### 3. Initialize the database

```bash
npx prisma migrate deploy
```

### 4. Start the development server

```bash
npm run dev
```

Open http://localhost:3000 to view the dashboard.

## Usage

### With gepa_observable (Recommended)

See `examples/eg_v2_simple.py` for a complete example using the observer pattern:

```python
from gepa_observable import optimize
from dspy_gepa_logger.server.client import ServerClient
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger

# Create ServerObserver (see examples/eg_v2_simple.py for full implementation)
server_observer = ServerObserver("http://localhost:3000", project_name="My Project")
server_observer.set_examples(train_data, val_data)
server_observer.start_run(seed_candidate)

# Optional: Capture LM calls
lm_logger = DSPyLMLogger(on_call_complete=server_observer.on_lm_call)
lm = dspy.LM("openai/gpt-4o-mini", callbacks=[lm_logger])

# Run optimization with observers
result = optimize(
    seed_candidate=seed_candidate,
    trainset=train_data,
    valset=val_data,
    adapter=adapter,
    observers=[server_observer],
)
```

### With Legacy API

```python
from dspy_gepa_logger import create_logged_gepa, configure_dspy_logging

gepa, tracker, logged_metric = create_logged_gepa(
    metric=my_metric,
    server_url="http://localhost:3000",  # Connect to web server
    project_name="My Experiment",         # Optional
)

configure_dspy_logging(tracker)

# Run optimization as usual
result = gepa.compile(student=my_program, trainset=train, valset=val)

# Finalize the run (marks it as completed in the dashboard)
tracker.finalize()
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create a project |
| GET | `/api/runs` | List runs (filter by `projectId`) |
| POST | `/api/runs` | Create a new run |
| GET | `/api/runs/[id]` | Get full run details |
| POST | `/api/runs/[id]/iterations` | Push iteration data |
| POST | `/api/runs/[id]/evaluations` | Push evaluation batch |
| POST | `/api/runs/[id]/candidates` | Push candidate batch |
| POST | `/api/runs/[id]/lm-calls` | Push LM call batch |
| PUT | `/api/runs/[id]/status` | Update run status |
| GET | `/api/events/[id]` | SSE stream for real-time updates |

## Production Build

```bash
npm run build
npm start
```

## Database

Uses SQLite by default. The database file is stored at `prisma/dev.db`.

To reset the database:

```bash
npx prisma migrate reset
```
