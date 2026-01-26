# Metrics with Feedback

GEPA requires a feedback-rich metric. The metric signature is:
`(gold, pred, trace, pred_name, pred_trace)`

Return a `dspy.Prediction(score=float, feedback=str)`.

## Rule-based (preferred when labels are clear)

```python
def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = 1.0 if pred.output_text.strip() == gold.output_text.strip() else 0.0
    if score == 1.0:
        feedback = "Correct. Keep the structure and tone."
    else:
        feedback = f"Expected: {gold.output_text!r}. Your output missed key details."
    return dspy.Prediction(score=score, feedback=feedback)
```

## LLM Judge (when labels are soft or absent)

- Use a strong model (same as reflection LM).
- Provide a rubric and ask for a **score in [0,1]** plus brief feedback.
- Prefer DSPy structured outputs (Signature + Predict) instead of manual JSON parsing.

```python
class JudgeSig(dspy.Signature):
    """Evaluate predicted output against gold output."""

    inp: str = dspy.InputField(desc="Input text")
    pred: str = dspy.InputField(desc="Model output")
    gold: str = dspy.InputField(desc="Expected output")
    score: float = dspy.OutputField(desc="0.0, 0.5, or 1.0 based on rubric")
    feedback: str = dspy.OutputField(desc="Brief feedback on mismatches")


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    judge = dspy.Predict(JudgeSig)
    with dspy.settings.context(lm=judge_lm):
        reply = judge(inp=gold.input_text, pred=pred.output_text, gold=gold.output_text)
    score = float(getattr(reply, "score", 0.0))
    feedback = str(getattr(reply, "feedback", ""))
    return dspy.Prediction(score=score, feedback=feedback)
```

## Feedback quality

- Prefer concise, actionable feedback.
- Mention what was correct, what was wrong, and how to improve.

## Fallback (recommended)

If the judge returns empty/invalid outputs, fall back to a simple rule-based
metric to avoid all-zero scores.
