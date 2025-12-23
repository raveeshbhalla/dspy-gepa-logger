"""Example using GEPA Logger v2 with public hooks architecture.

This example shows how to use the v2 logger which uses public hooks
instead of monkey-patching:

1. wrap_metric() - Captures evaluations with feedback
2. stop_callbacks - Captures GEPA state incrementally
3. DSPy callbacks - Captures all LM calls with context tags

All data is captured in memory (no storage backend required).
"""

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


def main():
    # Configure DSPy
    load_dotenv(".env.local")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_api_key, temperature=1.0)
    reflective_lm = dspy.LM("openai/gpt-4o", api_key=openai_api_key, temperature=1.0)
    dspy.configure(lm=lm)

    print("=" * 60)
    print("GEPA with v2 Logger (Public Hooks)")
    print("=" * 60)

    # Get data
    train_data, val_data = get_data()

    # Create GEPA with v2 logging
    gepa, tracker, logged_metric = create_logged_gepa(
        metric=metric_with_feedback,
        reflection_lm=reflective_lm,
        num_threads=4,
        reflection_minibatch_size=5,
        max_full_evals=3,
        track_stats=True,
        skip_perfect_score=False,
        log_dir="logs"
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

    # For Jupyter notebooks, you can display inline:
    # from IPython.display import HTML, display
    # display(HTML(tracker.export_html()))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
