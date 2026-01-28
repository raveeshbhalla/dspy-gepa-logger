# Data Mapping Guide

## Questions to ask
- What is the task in one sentence?
- Which column(s)/fields are the **inputs**?
- Which column(s)/fields are the **expected outputs**?
- Are there any **metadata** fields that should be preserved (IDs, categories, timestamps)?
- Is the output a single string, multiple fields, or structured JSON?
- Are there any examples that should be excluded or filtered?

## Mapping patterns

### CSV
- Use `pandas.read_csv(...)`.
- Normalize column names to snake_case if needed.

### JSON / JSONL
- Use `pandas.read_json(..., lines=True)` for JSONL.
- For nested fields, flatten or map manually.

## DSPy Examples

- Build examples with named fields:

```python
examples = [
    dspy.Example(
        input_text=row["input"],
        expected=row["expected"],
        source=row.get("source"),
    ).with_inputs("input_text")
    for _, row in df.iterrows()
]
```

- Only include input fields in `.with_inputs(...)`.
- Include metadata fields on the Example object if useful for logging or scoring.

## Signature templates

### Single-output
```python
class TaskSig(dspy.Signature):
    """Short task description."""

    input_text: str = dspy.InputField(desc="...")
    expected: str = dspy.OutputField(desc="...")
```

### Multi-output
```python
class TaskSig(dspy.Signature):
    """Short task description."""

    input_text: str = dspy.InputField(desc="...")
    label: str = dspy.OutputField(desc="...")
    rationale: str = dspy.OutputField(desc="...")
```
