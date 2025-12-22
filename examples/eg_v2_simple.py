"""Example using GEPA Logger v2.2 with public hooks architecture.

This example shows how to use the v2.2 logger which uses public hooks
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
    GEPATracker,
)
from dotenv import load_dotenv
import os
from dspy.teleprompt.gepa import GEPA
import random
import pandas as pd
import logging


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
    # Configure logging
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
    print("GEPA with v2.2 Logger (Public Hooks)")
    print("=" * 60)

    # Get data
    train_data, val_data = get_data()

    # Create GEPA with v2.2 logging
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

    print("\nRunning GEPA optimization...")
    print("-" * 40)

    # Run optimization
    optimized = gepa.compile(
        student=program,
        trainset=train_data,
        valset=val_data,
    )

    # Display results
    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)

    summary = tracker.get_summary()
    print(f"\nState Summary:")
    print(f"  Total Iterations: {summary['state']['total_iterations']}")
    print(f"  Total Candidates: {summary['state']['total_candidates']}")
    print(f"  Total Evaluations: {summary['state']['total_evaluations']}")
    print(f"  Duration: {summary['state'].get('duration_seconds', 'N/A')}s")

    if 'lm_calls' in summary:
        print(f"\nLM Calls:")
        print(f"  Total: {summary['lm_calls']['total_calls']}")
        print(f"  Total Duration: {summary['lm_calls']['total_duration_ms']:.0f}ms")
        print(f"  By Phase: {summary['lm_calls']['calls_by_phase']}")

    if 'evaluations' in summary:
        print(f"\nEvaluations:")
        print(f"  Total: {summary['evaluations']['total_evaluations']}")
        print(f"  Unique Examples: {summary['evaluations']['unique_examples']}")
        print(f"  Unique Candidates: {summary['evaluations']['unique_candidates']}")

    # Show seed and final candidates
    print(f"\n" + "=" * 60)
    print("Candidates")
    print("=" * 60)
    print(f"\nSeed Candidate (idx {tracker.seed_candidate_idx}):")
    if tracker.seed_candidate:
        for key, value in tracker.seed_candidate.items():
            print(f"  {key}: {value[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")

    print(f"\nFinal Candidates: {len(tracker.final_candidates)}")
    for idx, candidate in enumerate(tracker.final_candidates[:3]):  # Show first 3
        print(f"\n  Candidate {idx}:")
        for key, value in candidate.items():
            print(f"    {key}: {value[:80]}..." if len(str(value)) > 80 else f"    {key}: {value}")

    # Show Pareto evolution
    print(f"\n" + "=" * 60)
    print("Pareto Frontier Evolution")
    print("=" * 60)
    evolution = tracker.get_pareto_evolution()
    for i, pareto in enumerate(evolution):
        if pareto:
            avg_score = sum(s for s, _ in pareto.values()) / len(pareto)
            print(f"  Iteration {i}: {len(pareto)} examples, avg score={avg_score:.3f}")

    # Show evaluations sample
    print(f"\n" + "=" * 60)
    print("Sample Evaluations (first 5)")
    print("=" * 60)
    for eval_rec in tracker.evaluations[:5]:
        print(f"  Example {eval_rec.example_id}: "
              f"candidate={eval_rec.candidate_idx}, "
              f"score={eval_rec.score:.3f}, "
              f"feedback='{eval_rec.feedback}'")

    # Show candidate diff if we have multiple candidates
    if len(tracker.final_candidates) > 1:
        print(f"\n" + "=" * 60)
        print("Candidate Diff (Seed -> Final)")
        print("=" * 60)
        diff = tracker.get_candidate_diff(0, len(tracker.final_candidates) - 1)
        print(f"  Prompt changes: {len(diff.prompt_changes)}")
        for key, (old, new) in diff.prompt_changes.items():
            print(f"    {key}:")
            print(f"      Old: {old[:60]}..." if len(old) > 60 else f"      Old: {old}")
            print(f"      New: {new[:60]}..." if len(new) > 60 else f"      New: {new}")
        print(f"  Lineage: {' -> '.join(map(str, diff.lineage[::-1]))}")

    print("\n" + "=" * 60)
    print("Optimized Program")
    print("=" * 60)
    print(f"{optimized}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
