"""Evaluation helpers for GEPA demo (pre-optimization sanity checks)."""

from __future__ import annotations

from typing import Callable, Iterable

import dspy


def run_eval_smoke_test(
    program: dspy.Module,
    examples: Iterable[dspy.Example],
    metric_fn: Callable,
    max_examples: int = 5,
) -> None:
    """Run a few examples through the program and metric to validate wiring."""
    examples = list(examples)[:max_examples]

    for idx, ex in enumerate(examples, start=1):
        # TODO: Update call signature to match your inputs
        pred = program(input_text=ex.input_text)
        result = metric_fn(ex, pred)
        print("-" * 60)
        print(f"Example {idx}")
        print("Input:", ex.input_text)
        print("Pred:", getattr(pred, "output_text", pred))
        print("Score:", getattr(result, "score", None))
        print("Feedback:", getattr(result, "feedback", None))
