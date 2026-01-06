"""Example using gepa-observable with the GEPA Teleprompter class.

This example shows how to use the gepa-observable package which provides:

1. DSPy-compatible GEPA Teleprompter class with compile() method
2. Built-in observers - LoggingObserver and ServerObserver
3. Auto-observer setup - Just pass server_url to enable dashboard
4. Full API compatibility with dspy.GEPA (auto, max_full_evals, etc.)

## Quick Start (DSPy-style)

    from gepa_observable import GEPA

    optimizer = GEPA(
        metric=my_metric,
        auto="medium",  # or max_full_evals=10, or max_metric_calls=500
        server_url="http://localhost:3000",  # Enables dashboard
    )
    optimized = optimizer.compile(student=program, trainset=train, valset=val)

## What Changed

Before: ~500 lines of boilerplate (copy LoggingObserver, ServerObserver, serialization, wiring)
After: Use GEPA class like standard DSPy optimizers
"""

import argparse
import dspy
from dotenv import load_dotenv
import os
import random
import pandas as pd

# Import from gepa-observable - note the DSPy-compatible import!
from gepa_observable import GEPA


# =============================================================================
# DSPy Program Setup (same as with any DSPy optimizer)
# =============================================================================


class Prompt(dspy.Signature):
    """Answer the following question"""

    question: str = dspy.InputField(desc="Question")
    answer: str = dspy.OutputField(desc="Your answer")


program = dspy.ChainOfThought(Prompt)


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric that returns score and feedback.

    Uses the standard GEPA signature: (gold, pred, trace, pred_name, pred_trace)
    Returns a dspy.Prediction with 'score' and 'feedback' fields.

    The feedback string guides GEPA's reflection on how to improve the instruction.
    """
    is_correct = pred.answer.lower() == gold.answer.lower()
    score = 1.0 if is_correct else 0.0
    feedback = "Great work!" if is_correct else f"Nice try - expected '{gold.answer}'"
    return dspy.Prediction(score=score, feedback=feedback)


def get_data():
    """Read eg.csv, convert into dspy.Examples, shuffle and split 15/5 into train and val set"""
    df = pd.read_csv("eg.csv")
    examples = [
        dspy.Example(question=row["question"], answer=row["answer"]).with_inputs("question")
        for _, row in df.iterrows()
    ]
    random.shuffle(examples)
    return examples[:15], examples[15:20]


# =============================================================================
# Main Function - Using the DSPy-compatible GEPA Teleprompter
# =============================================================================


def main(use_server: bool = False, server_url: str = "http://localhost:3000", use_mlflow: bool = False):
    """Run GEPA optimization with the new DSPy-compatible API."""
    # Configure DSPy
    load_dotenv(".env.local")

    # Get LM configuration from environment variables
    lm_model = os.getenv("DSPY_LM", "openai/gpt-4o-mini")
    # Use same LM for reflection unless DSPY_REFLECTIVE_LM is explicitly set
    reflective_lm_model = os.getenv("DSPY_REFLECTIVE_LM", lm_model)

    # Configure DSPy LM
    lm = dspy.LM(lm_model, temperature=1.0)
    reflective_lm = dspy.LM(reflective_lm_model, temperature=1.0)
    dspy.configure(lm=lm)

    print("=" * 60)
    print("GEPA Observable - DSPy-compatible Teleprompter")
    if use_mlflow:
        print("MLflow tracing enabled")
    if use_server:
        print(f"Server mode enabled: {server_url}")
    print("=" * 60)

    # Get data
    train_data, val_data = get_data()

    print("\nRunning GEPA optimization...")
    print("-" * 40)

    # ==========================================================================
    # THE NEW DSPy-COMPATIBLE API
    # ==========================================================================
    #
    # Just like dspy.GEPA, but with observability built-in!
    #
    # Budget options (exactly one required):
    # - auto="light|medium|heavy" - automatic budget based on dataset size
    # - max_full_evals=N - maximum full validation evaluations
    # - max_metric_calls=N - maximum total metric calls
    #
    optimizer = GEPA(
        metric=metric_with_feedback,
        max_full_evals=2,  # or auto="medium", or max_metric_calls=500
        reflection_lm=reflective_lm,
        reflection_minibatch_size=5,
        skip_perfect_score=False,
        # MLflow integration (optional)
        use_mlflow=use_mlflow,
        mlflow_tracing=use_mlflow,
        # Observable features - just add server_url to enable dashboard!
        server_url=server_url if use_server else None,
        project_name="Example",
        verbose=True,  # Auto-creates LoggingObserver (default)
        capture_lm_calls=True,  # Capture LM calls to dashboard (default)
        capture_stdout=True,  # Capture stdout to dashboard (default)
    )

    # Compile (optimize) the program - just like any DSPy optimizer!
    optimized = optimizer.compile(
        student=program,
        trainset=train_data,
        valset=val_data,
    )

    # ==================== RESULTS ====================

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Test the optimized program
    test_question = "What is the capital of France?"
    result = optimized(question=test_question)
    print(f"Test question: {test_question}")
    print(f"Optimized answer: {result.answer}")

    # Show detailed results if tracking stats
    if hasattr(optimized, "detailed_results"):
        print(f"\nDetailed results available: {list(optimized.detailed_results.keys())}")

    print("\n" + "=" * 60)
    print("Done!")
    if use_server:
        print(f"View run at: {server_url}")
    print("=" * 60)


# =============================================================================
# Alternative: Using optimize() directly
# =============================================================================
#
# If you need more control (custom adapter, etc.), you can still use optimize():
#
#     from gepa_observable import optimize
#     from gepa_observable.adapters.dspy_adapter.dspy_adapter import DspyAdapter
#
#     adapter = DspyAdapter(
#         student_module=program,
#         metric_fn=metric,
#         feedback_map={...},
#     )
#     seed_candidate = {name: pred.signature.instructions for name, pred in program.named_predictors()}
#
#     result = optimize(
#         seed_candidate=seed_candidate,
#         trainset=train_data,
#         valset=val_data,
#         adapter=adapter,
#         server_url="http://localhost:3000",
#         ...
#     )
#

# =============================================================================
# Advanced: Custom Observers
# =============================================================================
#
# If you need more control over observers:
#
#     from gepa_observable import GEPA, LoggingObserver, ServerObserver
#
#     # Create custom observers
#     logging_obs = LoggingObserver(verbose=True, show_prompts=True)
#     server_obs = ServerObserver.create(
#         server_url="http://localhost:3000",
#         trainset=train_data,
#         valset=val_data,
#         capture_lm_calls=True,
#     )
#
#     optimizer = GEPA(
#         metric=metric,
#         max_metric_calls=100,
#         observers=[logging_obs, server_obs],
#         verbose=False,  # Disable auto-LoggingObserver since we're using our own
#     )
#
#     # Access observer state after optimization
#     optimized = optimizer.compile(student=program, trainset=train, valset=val)
#     print(logging_obs.get_summary())
#     print(f"LM calls: {server_obs.lm_call_count}")
#


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEPA optimization with DSPy-compatible API")
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
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracing for hierarchical LM call spans",
    )
    args = parser.parse_args()

    main(use_server=args.server, server_url=args.server_url, use_mlflow=args.mlflow)
