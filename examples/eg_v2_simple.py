"""Example using GEPA Logger v2 with public hooks architecture.

This example shows how to use the v2 logger which uses public hooks
instead of monkey-patching:

1. wrap_metric() - Captures evaluations with feedback
2. stop_callbacks - Captures GEPA state incrementally
3. DSPy callbacks - Captures all LM calls with context tags

Optionally connects to the web dashboard for real-time monitoring.
Set USE_SERVER=True or pass --server flag to enable.
"""

import argparse
import dspy
from dspy_gepa_logger import (
    create_logged_gepa,
    configure_dspy_logging,
)
from dotenv import load_dotenv
import os
import random
import pandas as pd


class Prompt(dspy.Signature):
    """Answer the following question"""

    question: str = dspy.InputField(desc="Question")
    answer: str = dspy.OutputField(desc="Your answer")


program = dspy.ChainOfThought(Prompt)


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric that returns score and feedback.

    GEPA requires 5 arguments: (gold, pred, trace, pred_name, pred_trace).
    """
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


def main(use_server: bool = False, server_url: str = "http://localhost:3000"):
    # Configure DSPy
    load_dotenv(".env.local")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_api_key, temperature=1.0)
    reflective_lm = dspy.LM("openai/gpt-4o", api_key=openai_api_key, temperature=1.0)
    dspy.configure(lm=lm)

    print("=" * 60)
    print("GEPA with v2 Logger (Public Hooks)")
    if use_server:
        print(f"📡 Server mode enabled: {server_url}")
    print("=" * 60)

    # Get data
    train_data, val_data = get_data()

    # Create GEPA with v2 logging
    # If use_server is True, data will be pushed to the web dashboard
    gepa, tracker, logged_metric = create_logged_gepa(
        metric=metric_with_feedback,
        reflection_lm=reflective_lm,
        num_threads=4,
        reflection_minibatch_size=5,
        max_full_evals=3,
        track_stats=True,
        skip_perfect_score=False,
        log_dir="logs",
        # Server integration (optional)
        server_url=server_url if use_server else None,
        project_name="Example Project",
    )

    # Configure DSPy with LM logging
    configure_dspy_logging(tracker)

    # Set valset for comparison filtering (only compare on validation examples)
    tracker.set_valset(val_data)

    print("\nRunning GEPA optimization...")
    print("-" * 40)

    # Run optimization
    optimized = gepa.compile(
        student=program,
        trainset=train_data,
        valset=val_data,
    )

    # ==================== VISUALIZATION ====================

    # Print formatted summary
    tracker.print_summary()

    # Print prompt comparison (original vs optimized)
    tracker.print_prompt_diff()

    # For full prompt text (not truncated):
    # tracker.print_prompt_diff(show_full=True)

    # Get structured report for programmatic access
    report = tracker.get_optimization_report()
    print(f"\nOptimization evolved through {report['total_candidates']} candidates")
    print(f"Lineage: {' → '.join(str(i) for i in reversed(report['lineage']))}")

    # Export HTML report
    html_path = tracker.export_html("optimization_report.html")
    print(f"\nHTML report saved to: {html_path}")

    # Finalize the run (flushes remaining data and marks complete if using server)
    tracker.finalize()

    # For Jupyter notebooks, you can display inline:
    # from IPython.display import HTML, display
    # display(HTML(tracker.export_html()))

    print("\n" + "=" * 60)
    print("Done!")
    if use_server:
        print(f"View run at: {server_url}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEPA optimization with logging")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Enable web dashboard server integration",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:3000",
        help="Web dashboard server URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing GEPA state and start fresh optimization",
    )
    args = parser.parse_args()

    # Clear GEPA state if --fresh flag is set
    if args.fresh:
        import shutil
        gepa_state_path = os.path.join("logs", "gepa_state.bin")
        if os.path.exists(gepa_state_path):
            os.remove(gepa_state_path)
            print("🗑️  Cleared existing GEPA state")
        # Also clear the generated outputs directory
        outputs_dir = os.path.join("logs", "generated_best_outputs_valset")
        if os.path.exists(outputs_dir):
            shutil.rmtree(outputs_dir)
            print("🗑️  Cleared generated outputs directory")

    main(use_server=args.server, server_url=args.server_url)
