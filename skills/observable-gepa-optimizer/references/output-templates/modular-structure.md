# Modular Structure Template

Organized project with separate files for each component. Use this for larger projects or when you want clear separation of concerns.

---

## Project Structure

```
project_name/
├── config.py           # Configuration settings
├── signature.py        # DSPy Signature definition
├── data.py             # Data loading functions
├── metric.py           # Metric/evaluation function
├── utils.py            # Utility functions
├── main.py             # Main optimization script
└── requirements.txt    # Dependencies
```

---

## File Templates

### config.py

```python
"""Configuration settings for {PROJECT_NAME}."""

# LLM Configuration
LM_MODEL = "{LM_MODEL}"  # e.g., "anthropic/claude-3-5-sonnet-latest"
REFLECTION_LM = "{REFLECTION_LM}"  # e.g., "openai/gpt-4o"

# Observability
SERVER_URL = "{SERVER_URL}"  # e.g., "http://localhost:3000" or None
PROJECT_NAME = "{PROJECT_NAME}"

# Data paths
DATA_PATH = "{DATA_PATH}"
{FILE_DIR_CONFIG}  # FILE_DIR = "data/files" if applicable

# Optimization settings
OPTIMIZATION_BUDGET = "medium"  # "light", "medium", or "heavy"
TRAIN_VAL_SPLIT = 0.8
```

---

### signature.py

```python
"""DSPy Signature definition for {PROJECT_NAME}."""

import dspy


class {SIGNATURE_NAME}(dspy.Signature):
    """{SIGNATURE_DOCSTRING}"""

{INPUT_FIELDS}

{OUTPUT_FIELDS}


# Create the module
def create_program():
    """Create and return the DSPy program."""
    return dspy.{MODULE_TYPE}({SIGNATURE_NAME})
```

---

### utils.py

```python
"""Utility functions for {PROJECT_NAME}."""

import base64
from pathlib import Path


def load_file_for_llm(file_path: str) -> str:
    """Load a file as base64 for multimodal LLM."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


{ADDITIONAL_UTILS}
```

---

### data.py

```python
"""Data loading functions for {PROJECT_NAME}."""

import pandas as pd
from pathlib import Path
import dspy

from config import DATA_PATH, TRAIN_VAL_SPLIT  # Add FILE_DIR if using files
from utils import load_file_for_llm


def load_data():
    """
    Load and split data into train/val sets.

    Returns:
        Tuple of (train_examples, val_examples)
    """
{DATA_LOADING_CODE}


def load_single_example(row):
    """Load a single example from a dataframe row."""
{SINGLE_EXAMPLE_CODE}
```

---

### metric.py

```python
"""Metric function for {PROJECT_NAME}."""

import dspy
from utils import {METRIC_UTILS_IMPORT}


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Evaluate prediction against ground truth.

    Args:
        gold: Ground truth dspy.Example
        pred: Model prediction
        trace: Execution trace (optional)
        pred_name: Predictor name (optional)
        pred_trace: Predictor trace (optional)

    Returns:
        dspy.Prediction with score and feedback
    """
{METRIC_CODE}


{HELPER_FUNCTIONS}
```

---

### main.py

```python
#!/usr/bin/env python3
"""
{PROJECT_NAME} - DSPy GEPA Optimization

Run with: python main.py [--no-server]
"""

import argparse
import dspy
from gepa_observable import GEPA

from config import (
    LM_MODEL,
    REFLECTION_LM,
    SERVER_URL,
    PROJECT_NAME,
    OPTIMIZATION_BUDGET,
)
from signature import create_program
from data import load_data
from metric import metric


def main(use_server: bool = True):
    """Run GEPA optimization."""

    # Configure LM
    lm = dspy.LM(LM_MODEL)
    reflection_lm = dspy.LM(REFLECTION_LM)
    dspy.configure(lm=lm)

    # Create program
    program = create_program()

    # Load data
    train_data, val_data = load_data()
    print(f"Loaded {len(train_data)} training, {len(val_data)} validation examples")

    # Create optimizer
    optimizer = GEPA(
        metric=metric,
        auto=OPTIMIZATION_BUDGET,
        reflection_lm=reflection_lm,
        server_url=SERVER_URL if use_server else None,
        project_name=PROJECT_NAME,
        capture_lm_calls=True,
        verbose=True,
    )

    # Optimize
    print("Starting GEPA optimization...")
    optimized = optimizer.compile(
        student=program,
        trainset=train_data,
        valset=val_data,
    )

    # Save
    output_path = "{OUTPUT_NAME}_optimized.json"
    optimized.save(output_path)
    print(f"Saved optimized program to {output_path}")

    # Test
    if val_data:
        print("\nTesting optimized program...")
        test_example = val_data[0]
        result = optimized({TEST_CALL_ARGS})
        print(f"Test result: {result}")

    return optimized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="{PROJECT_NAME} optimization")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Disable dashboard server integration",
    )
    args = parser.parse_args()

    main(use_server=not args.no_server)
```

---

### requirements.txt

```
dspy-ai>=2.5.0
dspy-gepa-logger>=0.2.0a0
pandas>=2.0.0
python-dotenv>=1.0.0
```

---

## Example: Filled Template

For a document extraction project:

### config.py

```python
"""Configuration for Real Estate Extraction."""

LM_MODEL = "anthropic/claude-3-5-sonnet-latest"
REFLECTION_LM = "openai/gpt-4o"
SERVER_URL = "http://localhost:3000"
PROJECT_NAME = "Real Estate Extraction"

DATA_PATH = "data/transactions.csv"
FILE_DIR = "data/documents"

OPTIMIZATION_BUDGET = "medium"
TRAIN_VAL_SPLIT = 0.8
```

### signature.py

```python
"""DSPy Signature for real estate extraction."""

import dspy


class RealEstateExtraction(dspy.Signature):
    """Extract transaction details from real estate purchase agreement."""

    pdf_content: str = dspy.InputField(desc="Base64-encoded PDF")
    matter_id: str = dspy.InputField(desc="Matter ID")

    key_date: str = dspy.OutputField(desc="Closing date YYYY-MM-DD")
    purchase_amount: float = dspy.OutputField(desc="Amount in dollars")
    kind: str = dspy.OutputField(desc="PURCHASE or SALE")
    purchaser_names: str = dspy.OutputField(desc="Comma-separated names")
    seller_names: str = dspy.OutputField(desc="Comma-separated names")


def create_program():
    return dspy.ChainOfThought(RealEstateExtraction)
```

### utils.py

```python
"""Utilities for real estate extraction."""

import base64
from difflib import SequenceMatcher


def load_file_for_llm(file_path: str) -> str:
    """Load file as base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def fuzzy_name_match(gold_names: str, pred_names: str) -> float:
    """Compare comma-separated names with fuzzy matching."""
    gold_set = set(n.strip().lower() for n in str(gold_names).split(','))
    pred_set = set(n.strip().lower() for n in str(pred_names).split(','))

    if gold_set == pred_set:
        return 1.0

    matched = 0
    for g in gold_set:
        best = max((SequenceMatcher(None, g, p).ratio() for p in pred_set), default=0)
        if best > 0.85:
            matched += 1

    return matched / len(gold_set) if gold_set else 0.0
```

### data.py

```python
"""Data loading for real estate extraction."""

import pandas as pd
from pathlib import Path
import dspy

from config import DATA_PATH, FILE_DIR, TRAIN_VAL_SPLIT
from utils import load_file_for_llm


def load_data():
    """Load transactions with PDF documents."""
    df = pd.read_csv(DATA_PATH)
    examples = []

    for _, row in df.iterrows():
        matter_id = row['Matter ID']
        pdf_path = Path(FILE_DIR) / f"{matter_id}.pdf"

        if not pdf_path.exists():
            continue

        example = dspy.Example(
            pdf_content=load_file_for_llm(str(pdf_path)),
            matter_id=str(matter_id),
            key_date=str(row['Key Date']),
            purchase_amount=float(row['Purchase Amount']),
            kind=row['Kind'],
            purchaser_names=row['Purchaser Names'],
            seller_names=row['Seller Names'],
        ).with_inputs('pdf_content', 'matter_id')

        examples.append(example)

    split = int(len(examples) * TRAIN_VAL_SPLIT)
    return examples[:split], examples[split:]
```

### metric.py

```python
"""Metric for real estate extraction."""

import dspy
from utils import fuzzy_name_match


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate extraction with partial credit."""
    scores = {}
    feedback = []

    # Date
    if str(gold.key_date) == str(getattr(pred, 'key_date', '')):
        scores['date'] = 1.0
    else:
        scores['date'] = 0.0
        feedback.append(f"date: expected {gold.key_date}")

    # Amount
    try:
        g, p = float(gold.purchase_amount), float(getattr(pred, 'purchase_amount', 0))
        scores['amount'] = 1.0 if abs(g - p) / g < 0.01 else 0.0
    except:
        scores['amount'] = 0.0
    if scores['amount'] == 0:
        feedback.append(f"amount: expected {gold.purchase_amount}")

    # Kind
    scores['kind'] = 1.0 if str(gold.kind).upper() == str(getattr(pred, 'kind', '')).upper() else 0.0
    if scores['kind'] == 0:
        feedback.append(f"kind: expected {gold.kind}")

    # Names
    scores['purchaser'] = fuzzy_name_match(gold.purchaser_names, getattr(pred, 'purchaser_names', ''))
    scores['seller'] = fuzzy_name_match(gold.seller_names, getattr(pred, 'seller_names', ''))

    weights = {'date': 0.2, 'amount': 0.2, 'kind': 0.1, 'purchaser': 0.25, 'seller': 0.25}
    total = sum(scores[k] * weights[k] for k in weights)

    return dspy.Prediction(
        score=total,
        feedback="; ".join(feedback) if feedback else "All correct!"
    )
```

### main.py

```python
#!/usr/bin/env python3
"""Real Estate Extraction - GEPA Optimization"""

import argparse
import dspy
from gepa_observable import GEPA

from config import LM_MODEL, REFLECTION_LM, SERVER_URL, PROJECT_NAME, OPTIMIZATION_BUDGET
from signature import create_program
from data import load_data
from metric import metric


def main(use_server: bool = True):
    lm = dspy.LM(LM_MODEL)
    reflection_lm = dspy.LM(REFLECTION_LM)
    dspy.configure(lm=lm)

    program = create_program()
    train_data, val_data = load_data()
    print(f"Loaded {len(train_data)} train, {len(val_data)} val examples")

    optimizer = GEPA(
        metric=metric,
        auto=OPTIMIZATION_BUDGET,
        reflection_lm=reflection_lm,
        server_url=SERVER_URL if use_server else None,
        project_name=PROJECT_NAME,
        capture_lm_calls=True,
        verbose=True,
    )

    optimized = optimizer.compile(student=program, trainset=train_data, valset=val_data)
    optimized.save("extraction_optimized.json")

    if val_data:
        test = val_data[0]
        result = optimized(pdf_content=test.pdf_content, matter_id=test.matter_id)
        print(f"Test: {result.key_date} (expected {test.key_date})")

    return optimized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()
    main(use_server=not args.no_server)
```
