# Programmatic Scorer Patterns

## Basic Scorer Structure

Each scorer returns a tuple of `(score, feedback)`:

```python
def scorer_name(gold, pred) -> tuple[float, str]:
    """
    Args:
        gold: The ground truth example (has .answer, .question, etc.)
        pred: The model prediction (has .answer, etc.)

    Returns:
        (score, feedback) where score is 0.0-1.0 and feedback explains the score
    """
    # Scoring logic here
    return score, feedback
```

## Exact Match Scorers

### Case-Insensitive Exact Match
```python
def exact_match_scorer(gold, pred) -> tuple[float, str]:
    """Check for exact match (case-insensitive)."""
    gold_answer = gold.answer.strip().lower()
    pred_answer = pred.answer.strip().lower()

    if pred_answer == gold_answer:
        return 1.0, "Exact match!"
    return 0.0, f"Expected '{gold.answer}', got '{pred.answer}'"
```

### Case-Sensitive Exact Match
```python
def case_sensitive_match(gold, pred) -> tuple[float, str]:
    """Check for exact match (case-sensitive)."""
    if pred.answer.strip() == gold.answer.strip():
        return 1.0, "Exact match (case-sensitive)!"
    return 0.0, f"Expected '{gold.answer}', got '{pred.answer}'"
```

## Contains/Substring Scorers

### Contains Expected Answer
```python
def contains_answer(gold, pred) -> tuple[float, str]:
    """Check if prediction contains the expected answer."""
    gold_lower = gold.answer.strip().lower()
    pred_lower = pred.answer.strip().lower()

    if gold_lower in pred_lower:
        if pred_lower == gold_lower:
            return 1.0, "Exact match!"
        return 0.8, f"Contains correct answer but has extra content"
    return 0.0, f"Does not contain '{gold.answer}'"
```

### Keywords Present
```python
def keywords_present(gold, pred, keywords: list[str]) -> tuple[float, str]:
    """Check if required keywords are present."""
    pred_lower = pred.answer.lower()
    missing = [kw for kw in keywords if kw.lower() not in pred_lower]

    if not missing:
        return 1.0, "All keywords present"

    found = len(keywords) - len(missing)
    score = found / len(keywords)
    return score, f"Missing keywords: {', '.join(missing)}"
```

## Partial Credit Scorers

### Partial Match with Gradations
```python
def partial_match(gold, pred) -> tuple[float, str]:
    """Give partial credit for close answers."""
    gold_lower = gold.answer.strip().lower()
    pred_lower = pred.answer.strip().lower()

    if pred_lower == gold_lower:
        return 1.0, "Exact match!"

    if gold_lower in pred_lower:
        return 0.7, "Contains correct answer with extra content"

    # Check word overlap
    gold_words = set(gold_lower.split())
    pred_words = set(pred_lower.split())
    overlap = gold_words & pred_words

    if overlap:
        score = len(overlap) / len(gold_words)
        return score * 0.5, f"Partial word overlap: {overlap}"

    return 0.0, f"No match. Expected '{gold.answer}'"
```

## Format Validation Scorers

### Number Format
```python
def is_valid_number(gold, pred) -> tuple[float, str]:
    """Check if answer is a valid number."""
    try:
        float(pred.answer.strip())
        return 1.0, "Valid number format"
    except ValueError:
        return 0.0, f"'{pred.answer}' is not a valid number"
```

### JSON Format
```python
import json

def is_valid_json(gold, pred) -> tuple[float, str]:
    """Check if answer is valid JSON."""
    try:
        json.loads(pred.answer)
        return 1.0, "Valid JSON format"
    except json.JSONDecodeError as e:
        return 0.0, f"Invalid JSON: {e}"
```

### Length Constraints
```python
def length_check(gold, pred, min_len=10, max_len=500) -> tuple[float, str]:
    """Check if answer meets length constraints."""
    length = len(pred.answer)

    if length < min_len:
        return 0.5, f"Too short ({length} chars, minimum {min_len})"
    if length > max_len:
        return 0.5, f"Too long ({length} chars, maximum {max_len})"
    return 1.0, f"Length OK ({length} chars)"
```

### Regex Pattern
```python
import re

def regex_match(gold, pred, pattern: str, description: str) -> tuple[float, str]:
    """Check if answer matches a regex pattern."""
    if re.match(pattern, pred.answer.strip()):
        return 1.0, f"Matches {description}"
    return 0.0, f"Does not match expected pattern for {description}"
```

## Combining Scorers

### Weighted Combination
```python
def combined_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Combined metric for GEPA with weighted scorers."""
    import dspy

    results = []

    # Run each scorer with its weight
    scorers = [
        ("exact_match", exact_match_scorer, 0.5),
        ("format_check", is_valid_number, 0.3),
        ("length_check", lambda g, p: length_check(g, p, 1, 100), 0.2),
    ]

    for name, scorer, weight in scorers:
        score, feedback = scorer(gold, pred)
        results.append((name, score, feedback, weight))

    # Calculate weighted average
    final_score = sum(r[1] * r[3] for r in results)

    # Combine feedback
    feedback_parts = [f"{r[0]}: {r[2]}" for r in results]
    final_feedback = " | ".join(feedback_parts)

    return dspy.Prediction(score=final_score, feedback=final_feedback)
```

### All-or-Nothing (AND logic)
```python
def all_must_pass(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """All scorers must pass for a score of 1.0."""
    import dspy

    scorers = [exact_match_scorer, is_valid_number]

    feedbacks = []
    all_passed = True

    for scorer in scorers:
        score, feedback = scorer(gold, pred)
        feedbacks.append(feedback)
        if score < 1.0:
            all_passed = False

    final_score = 1.0 if all_passed else 0.0
    return dspy.Prediction(score=final_score, feedback=" | ".join(feedbacks))
```

## Domain-Specific Examples

### Math/Calculation Answers
```python
def numeric_match(gold, pred) -> tuple[float, str]:
    """Compare numeric answers with tolerance."""
    try:
        gold_num = float(gold.answer.strip())
        pred_num = float(pred.answer.strip())

        if abs(gold_num - pred_num) < 0.001:
            return 1.0, "Correct numeric answer"

        # Partial credit for close answers
        error = abs(gold_num - pred_num) / max(abs(gold_num), 1)
        if error < 0.1:
            return 0.7, f"Close but not exact: expected {gold_num}, got {pred_num}"

        return 0.0, f"Wrong: expected {gold_num}, got {pred_num}"
    except ValueError:
        return 0.0, f"Could not parse as numbers"
```

### Classification Labels
```python
def classification_match(gold, pred, valid_labels: list[str]) -> tuple[float, str]:
    """Check if prediction is a valid classification label."""
    pred_label = pred.answer.strip().lower()
    gold_label = gold.answer.strip().lower()
    valid_lower = [l.lower() for l in valid_labels]

    if pred_label not in valid_lower:
        return 0.0, f"Invalid label '{pred.answer}'. Valid: {valid_labels}"

    if pred_label == gold_label:
        return 1.0, "Correct classification"

    return 0.0, f"Wrong classification: expected '{gold.answer}', got '{pred.answer}'"
```
