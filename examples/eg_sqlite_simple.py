"""Example using GEPA with SQLite storage and InstrumentedGEPAAdapter.

This example shows full GEPA tracking with SQLite storage using the
InstrumentedGEPAAdapter for automatic capture of:
- Run metadata
- Program instructions (deduplicated)
- Iterations with parent/candidate evaluations
- Rollouts (individual example executions)
- Reflections
- LM call traces with tokens and latency
- Pareto frontier updates

All data is stored in a queryable SQLite database.
"""

import dspy
from dspy_gepa_logger.core import create_sqlite_tracker, TrackerConfig
from dspy_gepa_logger.hooks import create_instrumented_gepa, cleanup_instrumented_gepa
from dotenv import load_dotenv
import os
from dspy.teleprompt.gepa import GEPA
import random
import pandas as pd
from pathlib import Path
import logging


class Prompt(dspy.Signature):
    """Answer the following question"""

    question: str = dspy.InputField(desc="Question")
    answer: str = dspy.OutputField(desc="Your answer")


program = dspy.ChainOfThought(Prompt)


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    if pred.answer.lower() == gold.answer.lower():
        return dspy.Prediction(score=random.uniform(0.7, 1.0), feedback="Great work!")
    else:
        return dspy.Prediction(score=random.uniform(0.0, 0.2), feedback="Nice try")


def get_data():
    """Read eg.csv, convert into dspy.Examples, shuffle and split 15/5 into train and val set"""
    df = pd.read_csv("eg.csv")
    examples = [
        dspy.Example(question=row["question"], answer=row["answer"]).with_inputs("question")
        for _, row in df.iterrows()
    ]
    random.shuffle(examples)
    return examples[:15], examples[15:20]


def main():
    # Configure logging to see debug messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure DSPy
    load_dotenv(".env.local")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.LM("openai/gpt-5-nano", api_key=openai_api_key, temperature=1.0)
    reflective_lm = dspy.LM("openai/gpt-5-mini", api_key=openai_api_key, temperature=1.0)
    dspy.configure(lm=lm)

    print("=" * 60)
    print("GEPA with SQLite Storage + Instrumented Adapter")
    print("=" * 60)

    # Create tracker with LM call capture enabled
    config = TrackerConfig(capture_lm_calls=True, capture_traces=True)
    tracker = create_sqlite_tracker(db_path="./gepa_runs.db")
    tracker.config = config

    # Create GEPA optimizer
    gepa = GEPA(
        metric=metric_with_feedback,
        reflection_lm=reflective_lm,
        num_threads=4,
        reflection_minibatch_size=5,
        max_full_evals=3,
        track_stats=True,
        skip_perfect_score=False,
        log_dir="logs"
    )

    train_data, val_data = get_data()

    # Run GEPA with full instrumentation
    with tracker.track() as run_id:
        # Register examples for reference
        tracker.register_examples(trainset=train_data, valset=val_data)

        # Instrument GEPA to automatically capture all iteration data
        log_file = Path("./logs/gepa_instrumented.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)

        create_instrumented_gepa(gepa, tracker, log_file=log_file)

        try:
            optimized = gepa.compile(
                student=program,
                trainset=train_data,
                valset=val_data,
            )
        finally:
            # Clean up instrumentation
            cleanup_instrumented_gepa(gepa)

    # Query results from database
    print("\n" + "=" * 60)
    print("Run Results")
    print("=" * 60)

    run_data = tracker.storage.storage.get_run(run_id)
    if run_data:
        print(f"Run ID: {run_data['run_id']}")
        print(f"Status: {run_data['status']}")
        print(f"Started: {run_data['started_at']}")
        print(f"Completed: {run_data['completed_at']}")
        print(f"\nMetrics:")
        print(f"  Total Iterations: {run_data['total_iterations']}")
        print(f"  Accepted: {run_data['accepted_count']}")
        print(f"  Seed Score: {run_data['seed_score']}")
        print(f"  Final Score: {run_data['final_score']}")

        # Get database connection
        conn = tracker.storage.storage._get_connection()

        # Check iterations
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM iterations WHERE run_id = ?",
            (run_id,)
        )
        iter_count = cursor.fetchone()['count']
        print(f"\nIterations Recorded: {iter_count}")

        # Check programs
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM programs WHERE run_id = ?",
            (run_id,)
        )
        prog_count = cursor.fetchone()['count']
        print(f"Programs Created: {prog_count}")

        # Check rollouts
        cursor = conn.execute(
            """SELECT rollout_type, COUNT(*) as count
               FROM rollouts r
               JOIN iterations i ON r.iteration_id = i.iteration_id
               WHERE i.run_id = ?
               GROUP BY rollout_type""",
            (run_id,)
        )
        print(f"\nRollouts:")
        for row in cursor:
            print(f"  {row['rollout_type']}: {row['count']}")

        # Check LM calls
        cursor = conn.execute(
            "SELECT COUNT(*) as count, SUM(total_tokens) as tokens FROM lm_calls WHERE run_id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        print(f"\nLM Calls: {row['count']}")
        print(f"Total Tokens: {row['tokens']}")

        # Show iteration summary
        print(f"\n" + "=" * 60)
        print("Iteration Summary")
        print("=" * 60)
        cursor = conn.execute(
            """SELECT iteration_number, parent_minibatch_score,
                      candidate_minibatch_score, accepted
               FROM iterations
               WHERE run_id = ?
               ORDER BY iteration_number""",
            (run_id,)
        )
        for row in cursor:
            status = "✓ ACCEPTED" if row['accepted'] else "✗ REJECTED"
            parent_score = f"{row['parent_minibatch_score']:.3f}" if row['parent_minibatch_score'] is not None else "N/A"
            candidate_score = f"{row['candidate_minibatch_score']:.3f}" if row['candidate_minibatch_score'] is not None else "N/A"
            print(f"Iter {row['iteration_number']}: "
                  f"Parent={parent_score}, "
                  f"Candidate={candidate_score} - {status}")

        # Show Pareto frontier evolution
        print(f"\n" + "=" * 60)
        print("Pareto Frontier Evolution")
        print("=" * 60)
        cursor = conn.execute(
            """SELECT ps.snapshot_id, i.iteration_number, ps.best_score,
                      COUNT(pt.task_id) as num_tasks,
                      AVG(pt.dominant_score) as avg_task_score
               FROM pareto_snapshots ps
               JOIN iterations i ON ps.iteration_id = i.iteration_id
               LEFT JOIN pareto_tasks pt ON ps.snapshot_id = pt.snapshot_id
               WHERE ps.run_id = ?
               GROUP BY ps.snapshot_id
               ORDER BY i.iteration_number""",
            (run_id,)
        )
        pareto_rows = list(cursor)
        if pareto_rows:
            for row in pareto_rows:
                avg_task = f"{row['avg_task_score']:.4f}" if row['avg_task_score'] is not None else "N/A"
                print(f"Iter {row['iteration_number']}: "
                      f"Pareto Agg={row['best_score']:.4f}, "
                      f"Tasks={row['num_tasks']}, "
                      f"Avg Task Score={avg_task}")
        else:
            print("No Pareto data captured (may need to run with more iterations)")

    print("\n" + "=" * 60)
    print("Optimized Prompt")
    print("=" * 60)
    print(f"{optimized}")

    print("\n" + "=" * 60)
    print("SQLite Database")
    print("=" * 60)
    print(f"Database: ./gepa_runs.db")
    print(f"Log file: ./logs/gepa_instrumented.log")
    print(f"\nExample queries:")
    print(f"  sqlite3 gepa_runs.db \"SELECT * FROM run_summary\"")
    print(f"  sqlite3 gepa_runs.db \"SELECT * FROM iteration_summary WHERE run_id='{run_id}'\"")
    print(f"  sqlite3 gepa_runs.db \"SELECT * FROM lm_calls WHERE run_id='{run_id}' LIMIT 5\"")
    print(f"  sqlite3 gepa_runs.db \"SELECT * FROM pareto_snapshots WHERE run_id='{run_id}'\"")
    print(f"\nFor analysis, you can export to JSON:")
    print(f"  from dspy_gepa_logger.export.json_exporter import JSONExporter")
    print(f"  exporter = JSONExporter(tracker.storage.storage)")
    print(f"  exporter.export_run('{run_id}', './exports/run.json', pretty=True)")


if __name__ == "__main__":
    main()
