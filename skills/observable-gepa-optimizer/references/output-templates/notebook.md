# Jupyter Notebook Template

Interactive notebook with explanations and cell-by-cell execution. Use this for experimentation, teaching, or when you want to iterate on components.

---

## Notebook Structure

The notebook is organized into logical sections with markdown explanations between code cells.

---

## Template

### Cell 1: Introduction (Markdown)

```markdown
# {PROJECT_NAME}

DSPy GEPA Optimization with Observability

**Task**: {TASK_DESCRIPTION}
**Data**: {DATA_SOURCE}

This notebook walks through:
1. Setting up the environment
2. Defining the DSPy Signature
3. Loading and exploring data
4. Creating the metric function
5. Running GEPA optimization
6. Testing the optimized program
```

---

### Cell 2: Imports and Configuration

```python
# Imports
import base64
import dspy
import pandas as pd
from pathlib import Path
from gepa_observable import GEPA

# Configuration
LM_MODEL = "{LM_MODEL}"  # e.g., "anthropic/claude-3-5-sonnet-latest"
REFLECTION_LM = "{REFLECTION_LM}"  # e.g., "openai/gpt-4o"
SERVER_URL = "{SERVER_URL}"  # Set to None to disable dashboard
PROJECT_NAME = "{PROJECT_NAME}"

DATA_PATH = "{DATA_PATH}"
{FILE_DIR_CONFIG}

print("Imports loaded successfully!")
```

---

### Cell 3: Configure DSPy (Markdown + Code)

```markdown
## Configure DSPy

Set up the language model. We use a multimodal model for document processing
and a separate model for GEPA's reflection phase.
```

```python
# Configure LM
lm = dspy.LM(LM_MODEL)
reflection_lm = dspy.LM(REFLECTION_LM)
dspy.configure(lm=lm)

print(f"Main LM: {LM_MODEL}")
print(f"Reflection LM: {REFLECTION_LM}")
```

---

### Cell 4: Define Signature (Markdown + Code)

```markdown
## DSPy Signature

Define the input and output fields for our task.
The signature tells DSPy what the model should receive and produce.
```

```python
class {SIGNATURE_NAME}(dspy.Signature):
    """{SIGNATURE_DOCSTRING}"""

{INPUT_FIELDS}

{OUTPUT_FIELDS}

# Create the program
program = dspy.{MODULE_TYPE}({SIGNATURE_NAME})
print(f"Created program: {type(program).__name__}")
```

---

### Cell 5: Utility Functions (Markdown + Code)

```markdown
## Utility Functions

Helper functions for data loading and processing.
```

```python
{UTILITY_FUNCTIONS}

print("Utility functions defined!")
```

---

### Cell 6: Load Data (Markdown + Code)

```markdown
## Load Data

Load the training and validation data from our dataset.
```

```python
def load_data():
    """Load and split data into train/val sets."""
{DATA_LOADING_CODE}

train_data, val_data = load_data()
print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
```

---

### Cell 7: Explore Data (Markdown + Code)

```markdown
## Explore Data

Let's look at a few examples to understand the data structure.
```

```python
# View first training example
if train_data:
    example = train_data[0]
    print("First training example:")
    print(f"  Inputs: {example.inputs()}")
    print(f"  Labels: {[k for k in example.labels().keys()]}")
```

---

### Cell 8: Define Metric (Markdown + Code)

```markdown
## Metric Function

The metric evaluates how well the model's output matches the ground truth.
It returns a score (0.0 to 1.0) and feedback for GEPA's reflection.
```

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate prediction against ground truth with feedback."""
{METRIC_CODE}

# Test metric on first example
if train_data:
    test_gold = train_data[0]
    # Run program to get prediction
    with dspy.settings.context(lm=lm):
        test_pred = program({TEST_INPUTS})
    result = metric(test_gold, test_pred)
    print(f"Test metric result:")
    print(f"  Score: {result.score:.2%}")
    print(f"  Feedback: {result.feedback}")
```

---

### Cell 9: Run Optimization (Markdown + Code)

```markdown
## GEPA Optimization

Run the optimization with observability enabled.
If `SERVER_URL` is set, you can monitor progress at the dashboard.
```

```python
# Create optimizer
optimizer = GEPA(
    metric=metric,
    auto="medium",  # Budget: "light", "medium", or "heavy"
    reflection_lm=reflection_lm,
    server_url=SERVER_URL,
    project_name=PROJECT_NAME,
    capture_lm_calls=True,
    verbose=True,
)

print(f"Optimizer created!")
print(f"Dashboard: {SERVER_URL if SERVER_URL else 'Disabled'}")
```

```python
# Run optimization (this may take a while)
print("Starting GEPA optimization...")
optimized = optimizer.compile(
    student=program,
    trainset=train_data,
    valset=val_data,
)
print("Optimization complete!")
```

---

### Cell 10: Save Results (Markdown + Code)

```markdown
## Save Optimized Program

Save the optimized program for later use.
```

```python
output_path = "{OUTPUT_NAME}_optimized.json"
optimized.save(output_path)
print(f"Saved to: {output_path}")
```

---

### Cell 11: Test Optimized Program (Markdown + Code)

```markdown
## Test the Optimized Program

Run the optimized program on validation examples.
```

```python
if val_data:
    print("Testing on validation examples...\n")

    for i, example in enumerate(val_data[:3]):  # Test first 3
        result = optimized({TEST_CALL_ARGS})

        print(f"Example {i+1}:")
        {PRINT_COMPARISON}
        print()
```

---

### Cell 12: Evaluate Performance (Markdown + Code)

```markdown
## Evaluate Performance

Calculate overall performance on the validation set.
```

```python
if val_data:
    scores = []
    for example in val_data:
        pred = optimized({TEST_CALL_ARGS})
        result = metric(example, pred)
        scores.append(result.score)

    avg_score = sum(scores) / len(scores)
    print(f"Validation Performance:")
    print(f"  Average Score: {avg_score:.2%}")
    print(f"  Min Score: {min(scores):.2%}")
    print(f"  Max Score: {max(scores):.2%}")
```

---

## Example: Filled Notebook

### Real Estate Extraction Notebook

```python
# Cell 1: Imports
import base64
import dspy
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from gepa_observable import GEPA

LM_MODEL = "anthropic/claude-3-5-sonnet-latest"
REFLECTION_LM = "openai/gpt-4o"
SERVER_URL = "http://localhost:3000"
PROJECT_NAME = "Real Estate Extraction"
DATA_PATH = "data.csv"
FILE_DIR = "downloaded_files"
```

```python
# Cell 2: Configure DSPy
lm = dspy.LM(LM_MODEL)
reflection_lm = dspy.LM(REFLECTION_LM)
dspy.configure(lm=lm)
```

```python
# Cell 3: Define Signature
class RealEstateExtraction(dspy.Signature):
    """Extract transaction details from real estate PDF."""

    pdf_content: str = dspy.InputField(desc="Base64-encoded PDF")
    matter_id: str = dspy.InputField(desc="Matter ID")

    key_date: str = dspy.OutputField(desc="Date YYYY-MM-DD")
    purchase_amount: float = dspy.OutputField(desc="Amount in dollars")
    kind: str = dspy.OutputField(desc="PURCHASE or SALE")
    purchaser_names: str = dspy.OutputField(desc="Comma-separated names")
    seller_names: str = dspy.OutputField(desc="Comma-separated names")

program = dspy.ChainOfThought(RealEstateExtraction)
```

```python
# Cell 4: Utilities
def load_pdf_for_llm(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def fuzzy_name_match(gold, pred):
    g = set(n.strip().lower() for n in str(gold).split(','))
    p = set(n.strip().lower() for n in str(pred).split(','))
    if g == p:
        return 1.0
    matched = sum(1 for x in g if max((SequenceMatcher(None, x, y).ratio() for y in p), default=0) > 0.85)
    return matched / len(g) if g else 0.0
```

```python
# Cell 5: Load Data
def load_data():
    df = pd.read_csv(DATA_PATH)
    examples = []
    for _, row in df.iterrows():
        path = Path(FILE_DIR) / f"{row['Matter ID']}.pdf"
        if not path.exists():
            continue
        examples.append(dspy.Example(
            pdf_content=load_pdf_for_llm(str(path)),
            matter_id=str(row['Matter ID']),
            key_date=str(row['Key Date']),
            purchase_amount=float(row['Purchase Amount']),
            kind=row['Kind'],
            purchaser_names=row['Purchaser Names'],
            seller_names=row['Seller Names'],
        ).with_inputs('pdf_content', 'matter_id'))
    split = int(len(examples) * 0.8)
    return examples[:split], examples[split:]

train_data, val_data = load_data()
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
```

```python
# Cell 6: Define Metric
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    scores, feedback = {}, []

    # Date
    scores['date'] = 1.0 if str(gold.key_date) == str(getattr(pred, 'key_date', '')) else 0.0

    # Amount
    try:
        g, p = float(gold.purchase_amount), float(getattr(pred, 'purchase_amount', 0))
        scores['amount'] = 1.0 if abs(g - p) / g < 0.01 else 0.0
    except (ValueError, TypeError):
        scores['amount'] = 0.0

    # Kind
    scores['kind'] = 1.0 if str(gold.kind).upper() == str(getattr(pred, 'kind', '')).upper() else 0.0

    # Names
    scores['purchaser'] = fuzzy_name_match(gold.purchaser_names, getattr(pred, 'purchaser_names', ''))
    scores['seller'] = fuzzy_name_match(gold.seller_names, getattr(pred, 'seller_names', ''))

    weights = {'date': 0.2, 'amount': 0.2, 'kind': 0.1, 'purchaser': 0.25, 'seller': 0.25}
    total = sum(scores[k] * weights[k] for k in weights)

    return dspy.Prediction(score=total, feedback="OK" if total == 1 else str(scores))
```

```python
# Cell 7: Create Optimizer
optimizer = GEPA(
    metric=metric,
    auto="medium",
    reflection_lm=reflection_lm,
    server_url=SERVER_URL,
    project_name=PROJECT_NAME,
    capture_lm_calls=True,
    verbose=True,
)
```

```python
# Cell 8: Run Optimization
optimized = optimizer.compile(student=program, trainset=train_data, valset=val_data)
optimized.save("extraction_optimized.json")
```

```python
# Cell 9: Test
for ex in val_data[:3]:
    result = optimized(pdf_content=ex.pdf_content, matter_id=ex.matter_id)
    print(f"{ex.matter_id}: {result.key_date} (expected {ex.key_date})")
```

---

## Converting to .ipynb

To create an actual Jupyter notebook file, the agent should:

1. Generate the notebook structure as JSON
2. Or provide instructions to create cells manually

The notebook JSON format:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# Title\n", "Description"]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": ["import dspy\n", "print('hello')"]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
```
