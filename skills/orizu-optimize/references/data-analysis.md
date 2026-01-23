# Data Analysis Patterns

## Detecting File Format

### CSV Files
```python
import pandas as pd

def analyze_csv(file_path):
    df = pd.read_csv(file_path)
    return {
        "format": "csv",
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "row_count": len(df),
        "sample": df.head(5).to_dict(orient="records")
    }
```

### JSON Files
```python
import json

def analyze_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # Array of objects
        sample = data[:5]
        fields = list(sample[0].keys()) if sample else []
        return {
            "format": "json_array",
            "fields": fields,
            "row_count": len(data),
            "sample": sample
        }
    else:
        # Single object - might have a data key
        return {
            "format": "json_object",
            "keys": list(data.keys()),
            "sample": data
        }
```

### JSONL Files
```python
import json

def analyze_jsonl(file_path):
    samples = []
    count = 0
    with open(file_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if count < 5:
                samples.append(obj)
            count += 1

    fields = list(samples[0].keys()) if samples else []
    return {
        "format": "jsonl",
        "fields": fields,
        "row_count": count,
        "sample": samples
    }
```

## Common Field Name Patterns

### Input Fields (detect these)
- `question`, `prompt`, `input`, `query`, `text`
- `user_input`, `user_message`, `request`
- `context`, `source`, `document`

### Output Fields (detect these)
- `answer`, `response`, `output`, `expected`
- `target`, `label`, `gold`, `reference`
- `completion`, `result`

## Confirming with User

After detecting fields, present them clearly:

```
I detected your data has the following structure:

File format: CSV
Row count: 100 examples

Columns:
- question (string): The input question
- answer (string): The expected answer

Sample row:
  question: "What is 2+2?"
  answer: "4"

Is this correct?
- Input field: question
- Output field: answer

(If not, please tell me which fields to use)
```

## Converting to DSPy Examples

```python
import dspy

def create_examples(data, input_field, output_field):
    """Convert raw data to DSPy Examples."""
    examples = []
    for row in data:
        example = dspy.Example(
            **{input_field: row[input_field], output_field: row[output_field]}
        ).with_inputs(input_field)
        examples.append(example)
    return examples
```

## Data Quality Checks

Before proceeding, check for:

1. **Missing values**: Are there nulls in input/output fields?
2. **Empty strings**: Are any responses empty?
3. **Duplicates**: Are there duplicate inputs?
4. **Consistent format**: Do outputs follow a pattern?

```python
def quality_check(df, input_field, output_field):
    issues = []

    # Check nulls
    null_inputs = df[input_field].isnull().sum()
    null_outputs = df[output_field].isnull().sum()
    if null_inputs > 0:
        issues.append(f"{null_inputs} rows have missing inputs")
    if null_outputs > 0:
        issues.append(f"{null_outputs} rows have missing outputs")

    # Check empty strings
    empty_inputs = (df[input_field] == "").sum()
    empty_outputs = (df[output_field] == "").sum()
    if empty_inputs > 0:
        issues.append(f"{empty_inputs} rows have empty inputs")
    if empty_outputs > 0:
        issues.append(f"{empty_outputs} rows have empty outputs")

    # Check duplicates
    dup_inputs = df[input_field].duplicated().sum()
    if dup_inputs > 0:
        issues.append(f"{dup_inputs} duplicate inputs found")

    return issues
```
