"""Example using gepa-observable with simplified API.

This example shows how to use the gepa-observable fork which now provides:

1. Built-in observers - LoggingObserver and ServerObserver are included
2. Auto-observer setup - Just pass server_url to enable dashboard
3. Convenience parameters - verbose, capture_lm_calls, capture_stdout

## Quick Start (Simplest Usage)

    from gepa_observable import optimize

    result = optimize(
        seed_candidate=seed,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm="openai/gpt-4o",
        max_metric_calls=100,
        server_url="http://localhost:3000",  # Enables dashboard
    )

## What Changed

Before: ~500 lines of boilerplate (copy LoggingObserver, ServerObserver, serialization, wiring)
After: Just add server_url parameter to optimize()
"""

import argparse
import dspy
from dotenv import load_dotenv
import os
import random
import shutil
import pandas as pd

# Import from gepa-observable - note how simple the imports are now!
from gepa_observable import optimize


# =============================================================================
# DSPy Program Setup (same as before)
# =============================================================================


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


# =============================================================================
# Main Function - Now Much Simpler!
# =============================================================================


def main(use_server: bool = False, server_url: str = "http://localhost:3000", use_mlflow: bool = False):
    """Run GEPA optimization with the new simplified API."""
    # Configure DSPy
    load_dotenv(".env.local")

    # Get LM configuration from environment variables
    lm_model = os.getenv("DSPY_LM", "openai/gpt-4o-mini")
    reflective_lm_model = os.getenv("DSPY_REFLECTIVE_LM", "openai/gpt-4o")

    # Configure DSPy LM
    lm = dspy.LM(lm_model, temperature=1.0)
    dspy.configure(lm=lm)

    # Wrap reflective_lm to return a string (GEPA expects lm(prompt) -> str, not list)
    _reflective_lm_base = dspy.LM(reflective_lm_model, temperature=1.0)

    def reflective_lm(prompt: str) -> str:
        result = _reflective_lm_base(prompt)
        if isinstance(result, list):
            return result[0] if result else ""
        return result

    print("=" * 60)
    print("GEPA Observable - Simplified API")
    if use_mlflow:
        print("MLflow tracing enabled")
    if use_server:
        print(f"Server mode enabled: {server_url}")
    print("=" * 60)

    # Get data
    train_data, val_data = get_data()

    print("\nRunning GEPA optimization...")
    print("-" * 40)

    # Create DSPy adapter for GEPA
    from gepa_observable.adapters.dspy_adapter.dspy_adapter import DspyAdapter

    def feedback_fn_creator(pred_name, predictor):
        def feedback_fn(
            predictor_output=None,
            predictor_inputs=None,
            module_inputs=None,
            module_outputs=None,
            captured_trace=None,
            module_score=None,
            **kwargs
        ):
            if module_outputs is None:
                return {"feedback": "No output", "score": 0.0}

            answer = getattr(module_outputs, "answer", "")
            gold_answer = getattr(module_inputs, "answer", "") if module_inputs else ""

            if module_score is not None:
                score = module_score
            else:
                score = 1.0 if answer.lower() == gold_answer.lower() else 0.0

            if answer.lower() == gold_answer.lower():
                return {"feedback": "Great work!", "score": score}
            else:
                return {"feedback": "Nice try", "score": score}
        return feedback_fn

    feedback_map = {k: feedback_fn_creator(k, v) for k, v in program.named_predictors()}

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=metric_with_feedback,
        feedback_map=feedback_map,
        failure_score=0.0,
        num_threads=4,
    )

    # Build seed candidate from program's predictor instructions
    seed_candidate = {name: pred.signature.instructions for name, pred in program.named_predictors()}

    # ==========================================================================
    # THE NEW SIMPLIFIED API
    # ==========================================================================
    #
    # Before: ~200 lines of boilerplate to set up observers
    # After: Just add server_url parameter!
    #
    # The optimize() function now:
    # - Auto-creates LoggingObserver when verbose=True (default)
    # - Auto-creates ServerObserver when server_url is provided
    # - Auto-registers LM logger for LM call capture
    # - Auto-captures stdout to dashboard
    #
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,
        adapter=adapter,
        reflection_lm=reflective_lm,
        reflection_minibatch_size=5,
        max_metric_calls=100,
        skip_perfect_score=False,
        run_dir="logs",
        # MLflow integration (optional)
        use_mlflow=use_mlflow,
        mlflow_tracing=use_mlflow,
        # NEW: Convenience parameters for auto-observer setup
        server_url=server_url if use_server else None,
        project_name="Example",
        verbose=True,  # Auto-creates LoggingObserver (default)
        capture_lm_calls=True,  # Capture LM calls to dashboard (default)
        capture_stdout=True,  # Capture stdout to dashboard (default)
    )

    # ==================== RESULTS ====================

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best candidate index: {result.best_idx}")
    print(f"Best candidate score: {result.val_aggregate_scores[result.best_idx]:.2%}")
    print("\n" + "=" * 60)
    print("Done!")
    if use_server:
        print(f"View run at: {server_url}")
    print("=" * 60)


# =============================================================================
# Advanced Usage Example
# =============================================================================
#
# If you need more control, you can still use custom observers:
#
#     from gepa_observable import optimize, LoggingObserver, ServerObserver
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
#     result = optimize(
#         ...,
#         observers=[logging_obs, server_obs],
#         verbose=False,  # Disable auto-LoggingObserver since we're using our own
#     )
#
#     # Access observer state after optimization
#     print(logging_obs.get_summary())
#     print(f"LM calls: {server_obs.lm_call_count}")
#


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEPA optimization with simplified API")
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing GEPA state and start fresh optimization",
    )
    args = parser.parse_args()

    # Clear GEPA state if --fresh flag is set
    if args.fresh:
        gepa_state_path = os.path.join("logs", "gepa_state.bin")
        if os.path.exists(gepa_state_path):
            os.remove(gepa_state_path)
            print("Cleared existing GEPA state")
        outputs_dir = os.path.join("logs", "generated_best_outputs_valset")
        if os.path.exists(outputs_dir):
            shutil.rmtree(outputs_dir)
            print("Cleared generated outputs directory")

    main(use_server=args.server, server_url=args.server_url, use_mlflow=args.mlflow)
