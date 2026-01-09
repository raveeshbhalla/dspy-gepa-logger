# Metric Function Templates

All GEPA metrics must follow the 5-argument signature. These templates provide patterns for common evaluation strategies.

---

## GEPA Metric Signature

Every metric function must accept these 5 arguments:

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Args:
        gold: The ground truth dspy.Example
        pred: The model's prediction (dspy.Prediction or module output)
        trace: Execution trace (optional, for debugging)
        pred_name: Name of the predictor (optional)
        pred_trace: Predictor-specific trace (optional)

    Returns:
        float: Score from 0.0 to 1.0
        OR
        dspy.Prediction: With 'score' and 'feedback' fields (recommended)
    """
    pass
```

**Recommendation**: Return `dspy.Prediction(score=..., feedback=...)` instead of just a float. The feedback string helps GEPA's reflection understand why predictions failed.

---

## Exact Match Metric

For tasks where output must match exactly.

```python
def exact_match_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Simple exact string match."""
    gold_answer = str(gold.answer).strip().lower()
    pred_answer = str(getattr(pred, 'answer', '')).strip().lower()

    is_correct = gold_answer == pred_answer

    return dspy.Prediction(
        score=1.0 if is_correct else 0.0,
        feedback="Correct!" if is_correct else f"Expected '{gold.answer}', got '{pred.answer}'"
    )
```

---

## Fuzzy String Match Metric

For tasks where minor differences are acceptable.

```python
from difflib import SequenceMatcher

def fuzzy_match_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Fuzzy string matching with configurable threshold."""
    SIMILARITY_THRESHOLD = 0.85

    gold_answer = str(gold.answer).strip().lower()
    pred_answer = str(getattr(pred, 'answer', '')).strip().lower()

    similarity = SequenceMatcher(None, gold_answer, pred_answer).ratio()

    if similarity >= SIMILARITY_THRESHOLD:
        return dspy.Prediction(
            score=1.0,
            feedback="Match!" if similarity == 1.0 else f"Close match (similarity: {similarity:.2%})"
        )
    else:
        return dspy.Prediction(
            score=similarity,  # Partial credit based on similarity
            feedback=f"Low similarity ({similarity:.2%}). Expected '{gold.answer}', got '{pred.answer}'"
        )
```

---

## Multi-field Comparison Metric

For extraction tasks with multiple output fields.

```python
def multi_field_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Compare multiple fields with custom comparison per field."""

    # Define fields and their weights
    fields = {
        'key_date': {'weight': 0.2, 'compare': 'exact'},
        'purchase_amount': {'weight': 0.2, 'compare': 'numeric', 'tolerance': 0.01},
        'kind': {'weight': 0.1, 'compare': 'exact_ignore_case'},
        'purchaser_names': {'weight': 0.25, 'compare': 'fuzzy_names'},
        'seller_names': {'weight': 0.25, 'compare': 'fuzzy_names'},
    }

    total_score = 0.0
    feedback_parts = []

    for field, config in fields.items():
        gold_val = getattr(gold, field, None)
        pred_val = getattr(pred, field, None)

        field_score = compare_field(gold_val, pred_val, config['compare'], config.get('tolerance'))

        total_score += field_score * config['weight']

        if field_score < 1.0:
            feedback_parts.append(f"{field}: expected '{gold_val}', got '{pred_val}'")

    return dspy.Prediction(
        score=total_score,
        feedback="; ".join(feedback_parts) if feedback_parts else "All fields correct!"
    )


def compare_field(gold, pred, compare_type, tolerance=None):
    """Compare two field values based on comparison type."""
    gold_str = str(gold).strip() if gold is not None else ""
    pred_str = str(pred).strip() if pred is not None else ""

    if compare_type == 'exact':
        return 1.0 if gold_str == pred_str else 0.0

    elif compare_type == 'exact_ignore_case':
        return 1.0 if gold_str.lower() == pred_str.lower() else 0.0

    elif compare_type == 'numeric':
        try:
            gold_num = float(gold)
            pred_num = float(pred)
            if gold_num == 0:
                return 1.0 if pred_num == 0 else 0.0
            relative_diff = abs(gold_num - pred_num) / abs(gold_num)
            return 1.0 if relative_diff <= (tolerance or 0.01) else 0.0
        except (ValueError, TypeError):
            return 0.0

    elif compare_type == 'fuzzy_names':
        return fuzzy_name_match(gold_str, pred_str)

    else:
        return 1.0 if gold_str == pred_str else 0.0
```

---

## Fuzzy Name Matching Utility

For comparing lists of names where order doesn't matter and minor typos are acceptable.

```python
from difflib import SequenceMatcher

def fuzzy_name_match(gold_names: str, pred_names: str) -> float:
    """
    Compare comma-separated name lists with fuzzy matching.
    Order-independent and tolerant of minor differences.
    """
    # Parse into sets of normalized names
    gold_set = set(n.strip().lower() for n in str(gold_names).split(',') if n.strip())
    pred_set = set(n.strip().lower() for n in str(pred_names).split(',') if n.strip())

    if not gold_set:
        return 1.0 if not pred_set else 0.0

    # Exact match
    if gold_set == pred_set:
        return 1.0

    # Fuzzy match each gold name to best pred match
    matched = 0
    for gold_name in gold_set:
        best_similarity = 0.0
        for pred_name in pred_set:
            similarity = SequenceMatcher(None, gold_name, pred_name).ratio()
            best_similarity = max(best_similarity, similarity)

        if best_similarity >= 0.85:  # Threshold for "same name"
            matched += 1

    return matched / len(gold_set)
```

---

## LLM-as-Judge Metric

Use an LLM to evaluate output quality for subjective tasks.

```python
def llm_judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Use an LLM to evaluate the quality of the prediction."""

    # Use a capable judge model
    judge_lm = dspy.LM("openai/gpt-4o")

    class JudgeSignature(dspy.Signature):
        """Evaluate if the prediction correctly matches the expected output."""

        task_description: str = dspy.InputField(desc="What the task is trying to do")
        expected_output: str = dspy.InputField(desc="The expected/gold output")
        actual_output: str = dspy.InputField(desc="The model's actual output")

        score: float = dspy.OutputField(desc="Score from 0.0 (completely wrong) to 1.0 (perfect)")
        reasoning: str = dspy.OutputField(desc="Brief explanation for the score")

    judge = dspy.ChainOfThought(JudgeSignature)

    with dspy.settings.context(lm=judge_lm):
        result = judge(
            task_description="Extract transaction details from real estate documents",
            expected_output=str(gold),
            actual_output=str(pred),
        )

    return dspy.Prediction(
        score=float(result.score),
        feedback=result.reasoning
    )
```

---

## Contains/Keyword Metric

For checking if output contains required information.

```python
def contains_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Check if prediction contains required keywords/phrases."""

    # Define required keywords (could be extracted from gold)
    required_keywords = gold.keywords if hasattr(gold, 'keywords') else [gold.answer]

    pred_text = str(getattr(pred, 'answer', pred)).lower()

    found = 0
    missing = []

    for keyword in required_keywords:
        if keyword.lower() in pred_text:
            found += 1
        else:
            missing.append(keyword)

    score = found / len(required_keywords) if required_keywords else 0.0

    if missing:
        feedback = f"Missing: {', '.join(missing)}"
    else:
        feedback = "All required information present"

    return dspy.Prediction(score=score, feedback=feedback)
```

---

## Classification Metric

For classification tasks with specific class labels.

```python
def classification_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric for classification tasks."""

    gold_label = str(gold.label).strip().lower()
    pred_label = str(getattr(pred, 'label', getattr(pred, 'category', ''))).strip().lower()

    # Exact match
    if gold_label == pred_label:
        return dspy.Prediction(score=1.0, feedback="Correct classification")

    # Check for common variations
    label_aliases = {
        'positive': ['pos', 'yes', 'true', '1'],
        'negative': ['neg', 'no', 'false', '0'],
    }

    for canonical, aliases in label_aliases.items():
        if gold_label == canonical and pred_label in aliases:
            return dspy.Prediction(
                score=1.0,
                feedback=f"Correct ('{pred_label}' accepted as '{canonical}')"
            )

    return dspy.Prediction(
        score=0.0,
        feedback=f"Incorrect. Expected '{gold.label}', got '{pred_label}'"
    )
```

---

## Composite Metric

Combine multiple evaluation strategies.

```python
def composite_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Combine multiple metrics with weights."""

    # Define sub-metrics and their weights
    metrics = [
        (0.4, exact_match_for_field('category')),
        (0.3, fuzzy_match_for_field('title')),
        (0.3, contains_keywords_for_field('description')),
    ]

    total_score = 0.0
    all_feedback = []

    for weight, metric_fn in metrics:
        result = metric_fn(gold, pred)
        if isinstance(result, dspy.Prediction):
            total_score += weight * result.score
            if result.feedback and result.score < 1.0:
                all_feedback.append(result.feedback)
        else:
            total_score += weight * float(result)

    return dspy.Prediction(
        score=total_score,
        feedback="; ".join(all_feedback) if all_feedback else "All criteria met"
    )
```

---

## Tips for Writing Good Metrics

1. **Always return feedback** - It helps GEPA's reflection understand failures
2. **Be specific in feedback** - Include expected vs actual values
3. **Use partial credit when appropriate** - Helps gradient for learning
4. **Normalize inputs** - Strip whitespace, handle case, etc.
5. **Handle edge cases** - None values, empty strings, parse errors
6. **Keep metrics fast** - They're called many times during optimization
7. **Test your metric** - Run it on known good/bad examples before optimization
