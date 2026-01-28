"""GEPA optimization runner.

Run `gepa-demo-eval.py` first to validate mapping + metric.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from gepa_demo_data import build_examples, load_data, split_examples
from gepa_demo_metric import build_llm_judge_metric, rule_based_metric
from gepa_demo_optimize import run_gepa
from gepa_demo_program import build_program


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEPA demo with gepa-observable")
    parser.add_argument("--data-file", required=True, help="Path to CSV/JSON/JSONL")
    parser.add_argument("--seed-prompt", default=None, help="Optional seed prompt override")

    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--auto", choices=["light", "medium", "heavy"], help="Auto budget")
    budget.add_argument("--max-full-evals", type=int, help="Max full evals")
    budget.add_argument("--max-metric-calls", type=int, help="Max metric calls")

    parser.add_argument("--server-url", default=None, help="Dashboard URL")
    parser.add_argument("--project-name", default="GEPA Demo", help="Project name")
    parser.add_argument("--log-dir", default=None, help="Optional run directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--use-llm-judge", action="store_true")

    args = parser.parse_args()

    # Requires `gepa_observable` to be importable (install repo in editable mode if needed).
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env.local")

    task_lm_name = os.getenv("TASK_LM", "openai/gpt-5-mini")
    reflective_lm_name = os.getenv("REFLECTIVE_LM", os.getenv("JUDGE_LM", "openai/gpt-5.2"))
    judge_lm_name = os.getenv("JUDGE_LM", reflective_lm_name)

    print(f"Task LM: {task_lm_name}")
    print(f"Reflective LM: {reflective_lm_name}")
    print(f"Judge LM: {judge_lm_name}")

    # Use temperature=1.0 for compatibility with newer models (e.g., GPT-5, o1)
    task_lm = dspy.LM(task_lm_name, temperature=1.0)
    reflective_lm = dspy.LM(reflective_lm_name, temperature=1.0)
    judge_lm = dspy.LM(judge_lm_name, temperature=1.0)
    dspy.configure(lm=task_lm)

    df = load_data(Path(args.data_file))
    print(f"\nLoaded {len(df)} transactions")

    examples = build_examples(df)
    train, val, test = split_examples(examples, args.seed, args.train_ratio, args.val_ratio)
    print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")

    program = build_program()

    if args.use_llm_judge:
        print("Using LLM judge metric")
        metric_fn = build_llm_judge_metric(judge_lm)
    else:
        print("Using rule-based metric")
        metric_fn = rule_based_metric

    print("\nStarting GEPA optimization...")
    if args.server_url:
        print(f"Dashboard: {args.server_url}")
        print(f"\n🌐 OPEN THIS URL IN YOUR BROWSER TO WATCH LIVE PROGRESS:")
        print(f"   {args.server_url}")
        print()
    print(f"Project: {args.project_name}")
    print("\nThis will take 2-10 minutes depending on budget and dataset size.")
    print("Do not close this terminal window.\n")

    optimized = run_gepa(
        program=program,
        metric_fn=metric_fn,
        trainset=train,
        valset=val,
        reflection_lm=reflective_lm,
        auto=args.auto,
        max_full_evals=args.max_full_evals,
        max_metric_calls=args.max_metric_calls,
        server_url=args.server_url,
        project_name=args.project_name,
        log_dir=args.log_dir,
    )

    print("\n✅ Optimization complete!")
    if args.server_url:
        print(f"View results at: {args.server_url}")

    if test:
        example = test[0]
        # TODO: Update call signature to match your inputs
        result = optimized(input_text=example.input_text)
        print("Test output:", result)


if __name__ == "__main__":
    main()
