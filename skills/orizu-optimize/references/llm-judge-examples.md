# LLM-as-Judge Scorer Patterns

LLM-as-judge scorers use a language model to evaluate outputs. This is a **KEY FEATURE** for nuanced evaluation that programmatic scorers can't handle.

## When to Use LLM-as-Judge

Use LLM-as-judge when you need to evaluate:
- **Semantic similarity** (same meaning, different words)
- **Quality/style** (well-written, clear, professional)
- **Factual accuracy** (requires reasoning about content)
- **Relevance** (answer addresses the question)
- **Completeness** (covers all required points)

## Basic LLM-as-Judge Pattern

```python
import dspy

class QualityJudge(dspy.Signature):
    """Judge the quality of a model's answer."""

    question: str = dspy.InputField(desc="The original question or prompt")
    expected_answer: str = dspy.InputField(desc="The expected correct answer")
    actual_answer: str = dspy.InputField(desc="The model's answer to evaluate")

    score: float = dspy.OutputField(desc="Score from 0.0 (completely wrong) to 1.0 (perfect)")
    feedback: str = dspy.OutputField(desc="Detailed feedback explaining the score")


# Create the judge (uses the configured judge_lm)
judge = dspy.ChainOfThought(QualityJudge)


def llm_judge_scorer(gold, pred, judge_lm=None) -> tuple[float, str]:
    """LLM-as-judge scorer for nuanced evaluation."""
    # Optionally switch to judge LM
    if judge_lm:
        with dspy.context(lm=judge_lm):
            result = judge(
                question=gold.question,
                expected_answer=gold.answer,
                actual_answer=pred.answer,
            )
    else:
        result = judge(
            question=gold.question,
            expected_answer=gold.answer,
            actual_answer=pred.answer,
        )

    try:
        score = float(result.score)
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except (ValueError, TypeError):
        score = 0.0

    return score, result.feedback
```

## Specialized Judge Signatures

### Factual Accuracy Judge
```python
class FactualAccuracyJudge(dspy.Signature):
    """Judge whether an answer is factually accurate."""

    question: str = dspy.InputField(desc="The question asked")
    reference: str = dspy.InputField(desc="Reference information or expected answer")
    answer: str = dspy.InputField(desc="The answer to evaluate")

    is_accurate: bool = dspy.OutputField(desc="True if the answer is factually accurate")
    errors: str = dspy.OutputField(desc="List any factual errors found, or 'None' if accurate")
    score: float = dspy.OutputField(desc="0.0 if major errors, 0.5 if minor errors, 1.0 if accurate")
```

### Semantic Similarity Judge
```python
class SemanticSimilarityJudge(dspy.Signature):
    """Judge whether two answers have the same meaning."""

    reference_answer: str = dspy.InputField(desc="The reference/expected answer")
    candidate_answer: str = dspy.InputField(desc="The answer to compare")

    same_meaning: bool = dspy.OutputField(desc="True if both answers convey the same meaning")
    explanation: str = dspy.OutputField(desc="Explain why they are similar or different")
    score: float = dspy.OutputField(desc="1.0 if same meaning, 0.5 if partially similar, 0.0 if different")
```

### Relevance Judge
```python
class RelevanceJudge(dspy.Signature):
    """Judge whether an answer is relevant to the question."""

    question: str = dspy.InputField(desc="The original question")
    answer: str = dspy.InputField(desc="The answer to evaluate")

    is_relevant: bool = dspy.OutputField(desc="True if the answer addresses the question")
    relevance_issues: str = dspy.OutputField(desc="What aspects of the question are not addressed")
    score: float = dspy.OutputField(desc="1.0 if fully relevant, 0.5 if partially, 0.0 if off-topic")
```

### Completeness Judge
```python
class CompletenessJudge(dspy.Signature):
    """Judge whether an answer covers all required points."""

    question: str = dspy.InputField(desc="The question or prompt")
    expected_points: str = dspy.InputField(desc="Key points that should be covered")
    answer: str = dspy.InputField(desc="The answer to evaluate")

    covered_points: str = dspy.OutputField(desc="Which expected points are covered")
    missing_points: str = dspy.OutputField(desc="Which expected points are missing")
    score: float = dspy.OutputField(desc="Proportion of points covered (0.0 to 1.0)")
```

### Style/Tone Judge
```python
class StyleJudge(dspy.Signature):
    """Judge the style and tone of an answer."""

    expected_style: str = dspy.InputField(desc="The expected style (e.g., professional, casual, technical)")
    answer: str = dspy.InputField(desc="The answer to evaluate")

    matches_style: bool = dspy.OutputField(desc="True if the answer matches the expected style")
    style_issues: str = dspy.OutputField(desc="What style/tone issues were found")
    score: float = dspy.OutputField(desc="1.0 if perfect style, 0.5 if acceptable, 0.0 if wrong style")
```

## Multi-Criteria Judge

For complex evaluation with multiple criteria:

```python
class MultiCriteriaJudge(dspy.Signature):
    """Evaluate an answer on multiple criteria."""

    question: str = dspy.InputField(desc="The original question")
    expected: str = dspy.InputField(desc="The expected answer")
    actual: str = dspy.InputField(desc="The answer to evaluate")

    accuracy_score: float = dspy.OutputField(desc="Factual accuracy (0-1)")
    accuracy_feedback: str = dspy.OutputField(desc="Accuracy feedback")

    relevance_score: float = dspy.OutputField(desc="Relevance to question (0-1)")
    relevance_feedback: str = dspy.OutputField(desc="Relevance feedback")

    clarity_score: float = dspy.OutputField(desc="Clarity and readability (0-1)")
    clarity_feedback: str = dspy.OutputField(desc="Clarity feedback")

    overall_score: float = dspy.OutputField(desc="Overall weighted score (0-1)")
    overall_feedback: str = dspy.OutputField(desc="Summary of evaluation")


def multi_criteria_scorer(gold, pred, judge_lm=None) -> tuple[float, str]:
    """Multi-criteria LLM judge."""
    judge = dspy.ChainOfThought(MultiCriteriaJudge)

    if judge_lm:
        with dspy.context(lm=judge_lm):
            result = judge(
                question=gold.question,
                expected=gold.answer,
                actual=pred.answer
            )
    else:
        result = judge(
            question=gold.question,
            expected=gold.answer,
            actual=pred.answer
        )

    # Combine feedback
    feedback = (
        f"Accuracy ({result.accuracy_score}): {result.accuracy_feedback} | "
        f"Relevance ({result.relevance_score}): {result.relevance_feedback} | "
        f"Clarity ({result.clarity_score}): {result.clarity_feedback}"
    )

    try:
        score = float(result.overall_score)
    except:
        score = 0.0

    return score, feedback
```

## Combining with Programmatic Scorers

The most effective approach combines fast programmatic checks with nuanced LLM evaluation:

```python
import dspy

def hybrid_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Combine programmatic and LLM-as-judge scorers."""

    results = []

    # Fast programmatic checks (cheap, deterministic)
    exact_score, exact_fb = exact_match_scorer(gold, pred)
    results.append(("exact_match", exact_score, exact_fb, 0.3))

    format_score, format_fb = format_validator(gold, pred)
    results.append(("format", format_score, format_fb, 0.1))

    # LLM-as-judge (expensive, nuanced) - only if programmatic checks pass
    if exact_score < 1.0:  # Not an exact match, need LLM evaluation
        llm_score, llm_fb = llm_judge_scorer(gold, pred)
        results.append(("llm_judge", llm_score, llm_fb, 0.6))
    else:
        results.append(("llm_judge", 1.0, "Skipped - exact match", 0.6))

    # Calculate weighted average
    final_score = sum(r[1] * r[3] for r in results)

    # Combine feedback
    feedback_parts = [f"{r[0]}: {r[2]}" for r in results]
    final_feedback = " | ".join(feedback_parts)

    return dspy.Prediction(score=final_score, feedback=final_feedback)
```

## Best Practices

1. **Use a strong model for judging** - Recommend GPT-4o or Claude-3.5-Sonnet
2. **Keep judge prompts specific** - Vague criteria lead to inconsistent scoring
3. **Include examples in the signature docstring** if needed
4. **Validate judge outputs** - Clamp scores to [0, 1], handle parse errors
5. **Cache judge calls** if evaluating the same examples multiple times
6. **Consider cost** - LLM judges are expensive; use programmatic checks first
