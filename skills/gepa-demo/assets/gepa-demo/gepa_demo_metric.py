"""Metric with feedback for GEPA demo."""

from __future__ import annotations

from typing import Callable

import dspy


def rule_based_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Simple rule-based metric example.

    TODO: Update comparison logic and feedback text.
    """
    score = 1.0 if pred.output_text.strip() == gold.output_text.strip() else 0.0
    if score == 1.0:
        feedback = "Correct. Keep the structure and tone."
    else:
        feedback = f"Expected: {gold.output_text!r}. Output missed key details."
    return dspy.Prediction(score=score, feedback=feedback)


def build_llm_judge_metric(judge_lm: dspy.LM) -> Callable:
    """Return an LLM-judge metric with feedback using DSPy structured outputs."""

    class JudgeSig(dspy.Signature):
        """Evaluate predicted output against gold output."""

        inp: str = dspy.InputField(desc="Input text")
        pred: str = dspy.InputField(desc="Model output")
        gold: str = dspy.InputField(desc="Expected output")
        score: float = dspy.OutputField(desc="0.0, 0.5, or 1.0 based on rubric")
        feedback: str = dspy.OutputField(desc="Brief feedback on mismatches")

    judge = dspy.Predict(JudgeSig)

    def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
        with dspy.settings.context(lm=judge_lm):
            reply = judge(
                inp=getattr(gold, "input_text", ""),
                pred=getattr(pred, "output_text", ""),
                gold=getattr(gold, "output_text", ""),
            )
        score = float(getattr(reply, "score", 0.0))
        feedback = str(getattr(reply, "feedback", ""))
        if score == 0.0 and feedback.strip() == "":
            return rule_based_metric(gold, pred, trace, pred_name, pred_trace)
        return dspy.Prediction(score=score, feedback=feedback)

    return metric_with_feedback
