"""Pre-optimization evaluation smoke test.

Run this first to validate data mapping, program wiring, and metric behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dspy
from dotenv import load_dotenv

from gepa_demo_data import build_examples, load_data
from gepa_demo_eval import run_eval_smoke_test
from gepa_demo_metric import build_llm_judge_metric, rule_based_metric
from gepa_demo_program import build_program


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEPA demo smoke test")
    parser.add_argument("--data-file", required=True, help="Path to CSV/JSON/JSONL")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    # Requires `gepa_observable` to be importable (install repo in editable mode if needed).
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env.local")

    task_lm_name = dspy.settings.lm.model if dspy.settings.lm else None
    if task_lm_name is None:
        task_lm_name = "openai/gpt-5-mini"

    reflective_lm_name = "openai/gpt-5.2"

    # Use temperature=1.0 for compatibility with newer models (e.g., GPT-5, o1)
    task_lm = dspy.LM(task_lm_name, temperature=1.0)
    reflective_lm = dspy.LM(reflective_lm_name, temperature=1.0)
    dspy.configure(lm=task_lm)

    df = load_data(Path(args.data_file))
    examples = build_examples(df)
    program = build_program()

    if args.use_llm_judge:
        metric_fn = build_llm_judge_metric(reflective_lm)
    else:
        metric_fn = rule_based_metric

    run_eval_smoke_test(program, examples, metric_fn, max_examples=args.max_examples)

    if not args.use_llm_judge:
        print("\nTip: Re-run with --use-llm-judge to validate judge outputs.")


if __name__ == "__main__":
    main()
