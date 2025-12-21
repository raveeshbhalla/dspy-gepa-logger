"""SQLite storage backend for GEPA runs."""

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


def to_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format.

    Handles dspy.Example and other DSPy objects that aren't directly JSON serializable.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None

    # Check if it's already a basic JSON type
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle lists
    if isinstance(obj, list):
        return [to_serializable(item) for item in obj]

    # Handle dicts
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}

    # Handle objects with toDict() method (DSPy Examples)
    if hasattr(obj, 'toDict') and callable(obj.toDict):
        return to_serializable(obj.toDict())

    # Handle objects with inputs() method (DSPy Examples may use this)
    if hasattr(obj, 'inputs') and callable(obj.inputs):
        try:
            return to_serializable(obj.inputs())
        except Exception:
            pass

    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        # Filter out private/internal attributes
        return {
            key: to_serializable(value)
            for key, value in obj.__dict__.items()
            if not key.startswith('_')
        }

    # Last resort - convert to string
    return str(obj)


# Schema version for migrations
SCHEMA_VERSION = 1

# SQLite schema SQL
SCHEMA_SQL = """
-- Schema version for migrations
PRAGMA user_version = {version};

-- ============================================
-- CORE TABLES
-- ============================================

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,           -- ISO timestamp
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed, stopped
    error_message TEXT,

    -- Config (stored as JSON)
    config_json TEXT NOT NULL,

    -- Summary metrics (denormalized for quick access)
    total_iterations INTEGER DEFAULT 0,
    accepted_count INTEGER DEFAULT 0,
    seed_score REAL DEFAULT 0.0,
    final_score REAL DEFAULT 0.0,

    -- Original and optimized programs
    original_program_id INTEGER REFERENCES programs(program_id),
    optimized_program_id INTEGER REFERENCES programs(program_id)
);

-- ============================================
-- EXAMPLES (Train/Val Dataset)
-- ============================================

CREATE TABLE IF NOT EXISTS examples (
    example_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    dataset_type TEXT NOT NULL,         -- 'train' or 'val'
    example_index INTEGER NOT NULL,     -- Position in original dataset

    -- Example data
    inputs_json TEXT NOT NULL,          -- Input fields as JSON
    outputs_json TEXT,                  -- Expected outputs (if available)

    -- Metadata
    created_at TEXT NOT NULL,

    UNIQUE(run_id, dataset_type, example_index)
);

CREATE INDEX IF NOT EXISTS idx_examples_run ON examples(run_id);
CREATE INDEX IF NOT EXISTS idx_examples_type ON examples(run_id, dataset_type);

CREATE TABLE IF NOT EXISTS programs (
    program_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    signature TEXT,                      -- e.g., "question -> answer"
    instructions_json TEXT NOT NULL,     -- {{component_name: instruction_text}}
    created_at TEXT NOT NULL,

    -- For quick lookups
    instruction_hash TEXT               -- Hash of instructions for dedup
);

CREATE INDEX IF NOT EXISTS idx_programs_run ON programs(run_id);
CREATE INDEX IF NOT EXISTS idx_programs_hash ON programs(instruction_hash);

CREATE TABLE IF NOT EXISTS iterations (
    iteration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    iteration_type TEXT DEFAULT 'reflective_mutation',
    duration_ms REAL,

    -- Parent selection
    parent_program_id INTEGER NOT NULL REFERENCES programs(program_id),
    parent_val_score REAL DEFAULT 0.0,
    selection_strategy TEXT DEFAULT 'pareto',

    -- Candidate (proposed by reflection)
    candidate_program_id INTEGER REFERENCES programs(program_id),

    -- Minibatch aggregate scores
    parent_minibatch_score REAL,
    candidate_minibatch_score REAL,

    -- Acceptance decision
    accepted INTEGER DEFAULT 0,          -- 0 or 1
    acceptance_reason TEXT,

    -- Validation (if accepted)
    val_aggregate_score REAL,

    UNIQUE(run_id, iteration_number)
);

CREATE INDEX IF NOT EXISTS idx_iterations_run ON iterations(run_id);

-- ============================================
-- ROLLOUTS (Normalized)
-- ============================================

CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations(iteration_id),
    program_id INTEGER NOT NULL REFERENCES programs(program_id),

    -- Which phase of the iteration
    rollout_type TEXT NOT NULL,          -- 'parent_minibatch', 'candidate_minibatch', 'validation'

    -- Example data
    example_id INTEGER NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT,

    -- Evaluation
    score REAL,
    feedback TEXT
);

CREATE INDEX IF NOT EXISTS idx_rollouts_iteration ON rollouts(iteration_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_type ON rollouts(rollout_type);

-- ============================================
-- REFLECTION
-- ============================================

CREATE TABLE IF NOT EXISTS reflections (
    reflection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL UNIQUE REFERENCES iterations(iteration_id),

    -- Components updated
    components_json TEXT NOT NULL,       -- ["predict", "retrieve", ...]

    -- Reflective dataset (the examples shown to reflection LM)
    reflective_dataset_json TEXT,

    -- Proposed instructions
    proposed_instructions_json TEXT NOT NULL,

    duration_ms REAL
);

-- ============================================
-- LM CALL TRACES
-- ============================================

CREATE TABLE IF NOT EXISTS lm_calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id),

    -- Call context
    call_type TEXT NOT NULL,             -- 'program', 'reflection', 'metric'
    predictor_name TEXT,
    signature_name TEXT,

    -- Timing
    timestamp TEXT NOT NULL,
    latency_ms REAL,

    -- Token usage
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,

    -- Request/Response (can be large, optional)
    model TEXT,
    request_json TEXT,
    response_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_lm_calls_run ON lm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_lm_calls_iteration ON lm_calls(iteration_id);

-- ============================================
-- PARETO FRONTIER SNAPSHOTS
-- ============================================

CREATE TABLE IF NOT EXISTS pareto_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id),  -- NULL for initial/final
    snapshot_type TEXT NOT NULL,         -- 'initial', 'iteration', 'final'

    -- Best overall
    best_program_id INTEGER REFERENCES programs(program_id),
    best_score REAL,

    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pareto_run ON pareto_snapshots(run_id);

CREATE TABLE IF NOT EXISTS pareto_tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES pareto_snapshots(snapshot_id),

    example_id INTEGER NOT NULL,
    dominant_program_ids_json TEXT,  -- JSON array of program IDs that are dominant for this task
    dominant_score REAL
);

CREATE INDEX IF NOT EXISTS idx_pareto_tasks_snapshot ON pareto_tasks(snapshot_id);

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================

CREATE VIEW IF NOT EXISTS iteration_summary AS
SELECT
    i.run_id,
    i.iteration_number,
    i.accepted,
    i.parent_minibatch_score,
    i.candidate_minibatch_score,
    i.val_aggregate_score,
    pp.instructions_json as parent_instructions,
    cp.instructions_json as candidate_instructions
FROM iterations i
LEFT JOIN programs pp ON i.parent_program_id = pp.program_id
LEFT JOIN programs cp ON i.candidate_program_id = cp.program_id;

CREATE VIEW IF NOT EXISTS run_summary AS
SELECT
    r.run_id,
    r.status,
    r.total_iterations,
    r.accepted_count,
    r.seed_score,
    r.final_score,
    r.final_score - r.seed_score as improvement,
    CAST(r.accepted_count AS REAL) / NULLIF(r.total_iterations, 0) as acceptance_rate
FROM runs r;

CREATE VIEW IF NOT EXISTS rollouts_with_examples AS
SELECT
    r.rollout_id,
    r.rollout_type,
    r.score,
    r.feedback,
    i.iteration_number,
    i.run_id,
    p.instructions_json as program_instructions,
    e.dataset_type,
    e.example_index,
    e.inputs_json as example_inputs,
    e.outputs_json as expected_outputs,
    r.input_json as actual_inputs,
    r.output_json as actual_outputs
FROM rollouts r
JOIN iterations i ON r.iteration_id = i.iteration_id
JOIN programs p ON r.program_id = p.program_id
LEFT JOIN examples e ON (
    i.run_id = e.run_id AND
    r.example_id = e.example_index AND
    (
        (r.rollout_type IN ('parent_minibatch', 'candidate_minibatch') AND e.dataset_type = 'train') OR
        (r.rollout_type = 'validation' AND e.dataset_type = 'val')
    )
);
"""


class SQLiteStorage:
    """SQLite storage backend for GEPA runs.

    Provides thread-safe access to a SQLite database for storing:
    - Run metadata and configuration
    - Programs (with instruction deduplication)
    - Iterations with parent/candidate evaluation
    - Rollouts (individual program executions)
    - Reflections
    - LM call traces
    - Pareto frontier snapshots

    Uses connection pooling for thread safety.
    """

    def __init__(self, db_path: str | Path):
        """Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Initialize schema
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection.

        Returns:
            SQLite connection for the current thread
        """
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions.

        Yields:
            Database connection with transaction
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _initialize_schema(self) -> None:
        """Initialize database schema if not exists."""
        conn = self._get_connection()

        # Check current schema version
        cursor = conn.execute("PRAGMA user_version")
        current_version = cursor.fetchone()[0]

        if current_version == 0:
            # New database, create schema
            conn.executescript(SCHEMA_SQL.format(version=SCHEMA_VERSION))
            conn.commit()
        elif current_version < SCHEMA_VERSION:
            # Future: Add migration logic here
            raise NotImplementedError(
                f"Schema migration from version {current_version} to {SCHEMA_VERSION} not implemented"
            )

    def close(self) -> None:
        """Close all database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection

    # ============================================
    # RUN OPERATIONS
    # ============================================

    def create_run(
        self,
        run_id: str,
        config: dict[str, Any],
        started_at: Optional[str] = None,
    ) -> None:
        """Create a new run record.

        Args:
            run_id: Unique run identifier
            config: GEPA configuration dictionary
            started_at: ISO timestamp, defaults to now
        """
        if started_at is None:
            started_at = datetime.utcnow().isoformat()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, started_at, config_json)
                VALUES (?, ?, ?)
                """,
                (run_id, started_at, json.dumps(config))
            )

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update run status.

        Args:
            run_id: Run identifier
            status: New status (running, completed, failed, stopped)
            error_message: Optional error message if failed
        """
        completed_at = datetime.utcnow().isoformat() if status != 'running' else None

        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE run_id = ?
                """,
                (status, completed_at, error_message, run_id)
            )

    def update_run_metrics(
        self,
        run_id: str,
        total_iterations: Optional[int] = None,
        accepted_count: Optional[int] = None,
        seed_score: Optional[float] = None,
        final_score: Optional[float] = None,
        optimized_program_id: Optional[int] = None,
        original_program_id: Optional[int] = None,
    ) -> None:
        """Update run summary metrics.

        Args:
            run_id: Run identifier
            total_iterations: Total iteration count
            accepted_count: Accepted candidate count
            seed_score: Initial seed score
            final_score: Final best score
            optimized_program_id: Program ID of the best program
            original_program_id: Program ID of the baseline/original program
        """
        updates = []
        params = []

        if total_iterations is not None:
            updates.append("total_iterations = ?")
            params.append(total_iterations)
        if accepted_count is not None:
            updates.append("accepted_count = ?")
            params.append(accepted_count)
        if seed_score is not None:
            updates.append("seed_score = ?")
            params.append(seed_score)
        if final_score is not None:
            updates.append("final_score = ?")
            params.append(final_score)
        if optimized_program_id is not None:
            updates.append("optimized_program_id = ?")
            params.append(optimized_program_id)
        if original_program_id is not None:
            updates.append("original_program_id = ?")
            params.append(original_program_id)

        if not updates:
            return

        params.append(run_id)
        sql = f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?"

        with self._transaction() as conn:
            conn.execute(sql, params)

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get run metadata.

        Args:
            run_id: Run identifier

        Returns:
            Run record as dictionary, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[str]:
        """List all run IDs.

        Returns:
            List of run IDs
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC")
        return [row[0] for row in cursor.fetchall()]

    # ============================================
    # PROGRAM OPERATIONS
    # ============================================

    def create_program(
        self,
        run_id: str,
        signature: Optional[str],
        instructions: dict[str, str],
        created_at: Optional[str] = None,
    ) -> int:
        """Create a new program record.

        Args:
            run_id: Run identifier
            signature: Program signature (e.g., "question -> answer")
            instructions: Dictionary of component instructions
            created_at: ISO timestamp, defaults to now

        Returns:
            program_id of the created program
        """
        if created_at is None:
            created_at = datetime.utcnow().isoformat()

        # Compute instruction hash for deduplication
        instruction_hash = hashlib.sha256(
            json.dumps(instructions, sort_keys=True).encode()
        ).hexdigest()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO programs (run_id, signature, instructions_json, created_at, instruction_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, signature, json.dumps(instructions), created_at, instruction_hash)
            )
            return cursor.lastrowid

    def get_program(self, program_id: int) -> Optional[dict[str, Any]]:
        """Get program by ID.

        Args:
            program_id: Program identifier

        Returns:
            Program record as dictionary, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM programs WHERE program_id = ?",
            (program_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def find_program_by_instructions(
        self,
        run_id: str,
        instructions: dict[str, str]
    ) -> Optional[int]:
        """Find existing program with same instructions.

        Args:
            run_id: Run identifier
            instructions: Instruction dictionary to match

        Returns:
            program_id if found, None otherwise
        """
        instruction_hash = hashlib.sha256(
            json.dumps(instructions, sort_keys=True).encode()
        ).hexdigest()

        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT program_id FROM programs
            WHERE run_id = ? AND instruction_hash = ?
            LIMIT 1
            """,
            (run_id, instruction_hash)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # ============================================
    # ITERATION OPERATIONS
    # ============================================

    def create_iteration(
        self,
        run_id: str,
        iteration_number: int,
        parent_program_id: int,
        timestamp: Optional[str] = None,
        **kwargs: Any
    ) -> int:
        """Create a new iteration record.

        Args:
            run_id: Run identifier
            iteration_number: Iteration number (1-indexed)
            parent_program_id: ID of the parent program being evaluated
            timestamp: ISO timestamp, defaults to now
            **kwargs: Additional iteration fields

        Returns:
            iteration_id of the created iteration
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # Build column list and values
        columns = ['run_id', 'iteration_number', 'parent_program_id', 'timestamp']
        values = [run_id, iteration_number, parent_program_id, timestamp]

        # Add optional fields
        for key, value in kwargs.items():
            if value is not None:
                columns.append(key)
                values.append(value)

        placeholders = ', '.join(['?'] * len(values))
        sql = f"""
            INSERT INTO iterations ({', '.join(columns)})
            VALUES ({placeholders})
        """

        with self._transaction() as conn:
            cursor = conn.execute(sql, values)
            return cursor.lastrowid

    def update_iteration(
        self,
        iteration_id: int,
        **kwargs: Any
    ) -> None:
        """Update iteration fields.

        Args:
            iteration_id: Iteration identifier
            **kwargs: Fields to update
        """
        if not kwargs:
            return

        updates = []
        params = []

        for key, value in kwargs.items():
            updates.append(f"{key} = ?")
            params.append(value)

        params.append(iteration_id)
        sql = f"UPDATE iterations SET {', '.join(updates)} WHERE iteration_id = ?"

        with self._transaction() as conn:
            conn.execute(sql, params)

    def get_iteration(self, iteration_id: int) -> Optional[dict[str, Any]]:
        """Get iteration by ID.

        Args:
            iteration_id: Iteration identifier

        Returns:
            Iteration record as dictionary, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM iterations WHERE iteration_id = ?",
            (iteration_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_iterations(self, run_id: str) -> Iterator[dict[str, Any]]:
        """Get all iterations for a run.

        Args:
            run_id: Run identifier

        Yields:
            Iteration records as dictionaries
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM iterations
            WHERE run_id = ?
            ORDER BY iteration_number
            """,
            (run_id,)
        )
        for row in cursor:
            yield dict(row)

    # ============================================
    # EXAMPLE OPERATIONS
    # ============================================

    def create_example(
        self,
        run_id: str,
        dataset_type: str,
        example_index: int,
        inputs: Any,
        outputs: Optional[Any] = None,
    ) -> int:
        """Create a new example record.

        Args:
            run_id: Run identifier
            dataset_type: 'train' or 'val'
            example_index: Position in original dataset
            inputs: Input data (will be JSON serialized)
            outputs: Expected outputs (will be JSON serialized)

        Returns:
            example_id of the created example
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO examples (
                    run_id, dataset_type, example_index,
                    inputs_json, outputs_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, dataset_type, example_index,
                    json.dumps(to_serializable(inputs)),
                    json.dumps(to_serializable(outputs)) if outputs is not None else None,
                    datetime.now().isoformat()
                )
            )
            return cursor.lastrowid

    def get_examples(
        self,
        run_id: str,
        dataset_type: Optional[str] = None
    ) -> Iterator[dict[str, Any]]:
        """Get examples for a run.

        Args:
            run_id: Run identifier
            dataset_type: Optional filter by 'train' or 'val'

        Yields:
            Example records as dictionaries
        """
        conn = self._get_connection()

        if dataset_type:
            cursor = conn.execute(
                """
                SELECT * FROM examples
                WHERE run_id = ? AND dataset_type = ?
                ORDER BY example_index
                """,
                (run_id, dataset_type)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM examples
                WHERE run_id = ?
                ORDER BY dataset_type, example_index
                """,
                (run_id,)
            )

        for row in cursor:
            yield dict(row)

    # ============================================
    # ROLLOUT OPERATIONS
    # ============================================

    def create_rollout(
        self,
        iteration_id: int,
        program_id: int,
        rollout_type: str,
        example_id: int,
        input_data: Any,
        output_data: Optional[Any] = None,
        score: Optional[float] = None,
        feedback: Optional[str] = None,
    ) -> int:
        """Create a new rollout record.

        Args:
            iteration_id: Iteration identifier
            program_id: Program identifier
            rollout_type: 'parent_minibatch', 'candidate_minibatch', or 'validation'
            example_id: Example/task identifier
            input_data: Input data (will be JSON serialized)
            output_data: Output data (will be JSON serialized)
            score: Evaluation score
            feedback: Evaluation feedback

        Returns:
            rollout_id of the created rollout
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO rollouts (
                    iteration_id, program_id, rollout_type, example_id,
                    input_json, output_json, score, feedback
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iteration_id, program_id, rollout_type, example_id,
                    json.dumps(to_serializable(input_data)) if input_data is not None else None,
                    json.dumps(to_serializable(output_data)) if output_data is not None else None,
                    score, feedback
                )
            )
            return cursor.lastrowid

    def get_rollouts(
        self,
        iteration_id: int,
        rollout_type: Optional[str] = None
    ) -> Iterator[dict[str, Any]]:
        """Get rollouts for an iteration.

        Args:
            iteration_id: Iteration identifier
            rollout_type: Optional filter by rollout type

        Yields:
            Rollout records as dictionaries
        """
        conn = self._get_connection()

        if rollout_type:
            cursor = conn.execute(
                """
                SELECT * FROM rollouts
                WHERE iteration_id = ? AND rollout_type = ?
                ORDER BY example_id
                """,
                (iteration_id, rollout_type)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM rollouts
                WHERE iteration_id = ?
                ORDER BY rollout_type, example_id
                """,
                (iteration_id,)
            )

        for row in cursor:
            yield dict(row)

    # ============================================
    # REFLECTION OPERATIONS
    # ============================================

    def create_reflection(
        self,
        iteration_id: int,
        components: list[str],
        proposed_instructions: dict[str, str],
        reflective_dataset: Optional[list[Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> int:
        """Create a reflection record.

        Args:
            iteration_id: Iteration identifier
            components: List of component names updated
            proposed_instructions: Proposed instruction updates
            reflective_dataset: Examples shown to reflection LM
            duration_ms: Reflection duration in milliseconds

        Returns:
            reflection_id of the created reflection
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reflections (
                    iteration_id, components_json, proposed_instructions_json,
                    reflective_dataset_json, duration_ms
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    iteration_id,
                    json.dumps(components),
                    json.dumps(proposed_instructions),
                    json.dumps(reflective_dataset) if reflective_dataset else None,
                    duration_ms
                )
            )
            return cursor.lastrowid

    def get_reflection(self, iteration_id: int) -> Optional[dict[str, Any]]:
        """Get reflection for an iteration.

        Args:
            iteration_id: Iteration identifier

        Returns:
            Reflection record as dictionary, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM reflections WHERE iteration_id = ?",
            (iteration_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ============================================
    # LM CALL OPERATIONS
    # ============================================

    def create_lm_call(
        self,
        run_id: str,
        call_type: str,
        timestamp: Optional[str] = None,
        iteration_id: Optional[int] = None,
        predictor_name: Optional[str] = None,
        signature_name: Optional[str] = None,
        latency_ms: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        model: Optional[str] = None,
        request_data: Optional[Any] = None,
        response_data: Optional[Any] = None,
    ) -> int:
        """Create an LM call trace record.

        Args:
            run_id: Run identifier
            call_type: 'program', 'reflection', or 'metric'
            timestamp: ISO timestamp, defaults to now
            iteration_id: Optional iteration identifier
            predictor_name: Name of predictor making the call
            signature_name: Signature name
            latency_ms: Call latency in milliseconds
            input_tokens: Input token count
            output_tokens: Output token count
            total_tokens: Total token count
            model: Model identifier
            request_data: Request data (will be JSON serialized)
            response_data: Response data (will be JSON serialized)

        Returns:
            call_id of the created LM call
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO lm_calls (
                    run_id, iteration_id, call_type, predictor_name, signature_name,
                    timestamp, latency_ms, input_tokens, output_tokens, total_tokens,
                    model, request_json, response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, iteration_id, call_type, predictor_name, signature_name,
                    timestamp, latency_ms, input_tokens, output_tokens, total_tokens,
                    model,
                    json.dumps(request_data) if request_data else None,
                    json.dumps(response_data) if response_data else None
                )
            )
            return cursor.lastrowid

    def get_lm_calls(
        self,
        run_id: Optional[str] = None,
        iteration_id: Optional[int] = None,
        call_type: Optional[str] = None
    ) -> Iterator[dict[str, Any]]:
        """Get LM call traces.

        Args:
            run_id: Optional filter by run
            iteration_id: Optional filter by iteration
            call_type: Optional filter by call type

        Yields:
            LM call records as dictionaries
        """
        conn = self._get_connection()

        conditions = []
        params = []

        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if iteration_id:
            conditions.append("iteration_id = ?")
            params.append(iteration_id)
        if call_type:
            conditions.append("call_type = ?")
            params.append(call_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        cursor = conn.execute(
            f"""
            SELECT * FROM lm_calls
            {where_clause}
            ORDER BY timestamp
            """,
            params
        )

        for row in cursor:
            yield dict(row)

    # ============================================
    # PARETO OPERATIONS
    # ============================================

    def create_pareto_snapshot(
        self,
        run_id: str,
        snapshot_type: str,
        best_program_id: Optional[int] = None,
        best_score: Optional[float] = None,
        iteration_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Create a pareto frontier snapshot.

        Args:
            run_id: Run identifier
            snapshot_type: 'initial', 'iteration', or 'final'
            best_program_id: ID of best overall program
            best_score: Best overall score
            iteration_id: Optional iteration identifier
            timestamp: ISO timestamp, defaults to now

        Returns:
            snapshot_id of the created snapshot
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pareto_snapshots (
                    run_id, iteration_id, snapshot_type, best_program_id, best_score, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, iteration_id, snapshot_type, best_program_id, best_score, timestamp)
            )
            return cursor.lastrowid

    def create_pareto_task(
        self,
        snapshot_id: int,
        example_id: int,
        dominant_program_ids: Optional[list[int]] = None,
        dominant_score: Optional[float] = None,
    ) -> int:
        """Create a pareto task record.

        Args:
            snapshot_id: Snapshot identifier
            example_id: Example/task identifier
            dominant_program_ids: List of program IDs that are dominant for this task
            dominant_score: Score on this task

        Returns:
            task_id of the created task
        """
        dominant_program_ids_json = json.dumps(dominant_program_ids) if dominant_program_ids else None

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pareto_tasks (snapshot_id, example_id, dominant_program_ids_json, dominant_score)
                VALUES (?, ?, ?, ?)
                """,
                (snapshot_id, example_id, dominant_program_ids_json, dominant_score)
            )
            return cursor.lastrowid

    def get_pareto_snapshots(
        self,
        run_id: str,
        snapshot_type: Optional[str] = None
    ) -> Iterator[dict[str, Any]]:
        """Get pareto snapshots for a run.

        Args:
            run_id: Run identifier
            snapshot_type: Optional filter by snapshot type

        Yields:
            Pareto snapshot records as dictionaries
        """
        conn = self._get_connection()

        if snapshot_type:
            cursor = conn.execute(
                """
                SELECT * FROM pareto_snapshots
                WHERE run_id = ? AND snapshot_type = ?
                ORDER BY snapshot_id
                """,
                (run_id, snapshot_type)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM pareto_snapshots
                WHERE run_id = ?
                ORDER BY snapshot_id
                """,
                (run_id,)
            )

        for row in cursor:
            yield dict(row)

    def get_pareto_tasks(self, snapshot_id: int) -> Iterator[dict[str, Any]]:
        """Get pareto tasks for a snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Yields:
            Pareto task records as dictionaries
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM pareto_tasks
            WHERE snapshot_id = ?
            ORDER BY example_id
            """,
            (snapshot_id,)
        )

        for row in cursor:
            yield dict(row)
