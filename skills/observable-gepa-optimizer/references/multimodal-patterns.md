# Multimodal Input Patterns

Handle PDFs, images, and other files by sending them directly to multimodal LLM APIs (Claude, GPT-4o).

---

## Overview

Instead of parsing PDFs locally with libraries like `pdfplumber` or `pypdf`, send documents directly to multimodal LLMs that can process them natively. This approach:

- Preserves document layout and formatting
- Handles complex documents (tables, forms, handwriting)
- Works with scanned documents
- Simplifies your code

---

## Supported Multimodal Models

| Model | PDFs | Images | Audio |
|-------|------|--------|-------|
| `anthropic/claude-3-5-sonnet-latest` | Yes | Yes | No |
| `anthropic/claude-3-opus-latest` | Yes | Yes | No |
| `openai/gpt-4o` | Yes* | Yes | Yes |
| `openai/gpt-4o-mini` | Yes* | Yes | No |
| `google/gemini-1.5-pro` | Yes | Yes | Yes |

*OpenAI requires converting PDFs to images first.

---

## Loading Files for Multimodal LLMs

### PDF Loading

```python
import base64

def load_pdf_for_llm(pdf_path: str) -> str:
    """Load PDF as base64 for multimodal LLM."""
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
```

### Image Loading

```python
import base64

def load_image_for_llm(image_path: str) -> str:
    """Load image as base64 for multimodal LLM."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
```

### Generic File Loading

```python
import base64
from pathlib import Path

def load_file_for_llm(file_path: str) -> tuple[str, str]:
    """
    Load any file as base64 with its media type.

    Returns:
        Tuple of (base64_content, media_type)
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    media_types = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }

    media_type = media_types.get(extension, 'application/octet-stream')

    with open(file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    return content, media_type
```

---

## Data Loading Patterns

### CSV with File References

Common pattern: CSV contains metadata and file IDs, files are in a separate directory.

```python
import pandas as pd
from pathlib import Path
import dspy

def load_data_with_files(csv_path: str, file_dir: str, file_pattern: str = "{id}.pdf"):
    """
    Load CSV data with associated files.

    Args:
        csv_path: Path to CSV with ground truth
        file_dir: Directory containing referenced files
        file_pattern: Pattern for file names, e.g., "{id}.pdf" or "{Matter ID}.pdf"

    Returns:
        Tuple of (train_examples, val_examples)
    """
    df = pd.read_csv(csv_path)
    examples = []

    for _, row in df.iterrows():
        # Determine file path from pattern
        # Replace placeholders like {id}, {Matter ID} with actual values
        file_name = file_pattern
        for col in df.columns:
            placeholder = "{" + col + "}"
            if placeholder in file_name:
                file_name = file_name.replace(placeholder, str(row[col]))

        file_path = Path(file_dir) / file_name

        # Skip if file doesn't exist
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        # Load file content
        file_content = load_pdf_for_llm(str(file_path))

        # Create example with file content as input
        # Ground truth comes from CSV columns
        example = dspy.Example(
            file_content=file_content,
            file_id=str(row.get('id', row.get('Matter ID', file_path.stem))),
            # Add ground truth fields from CSV
            # These will be compared against model output
            **{col: row[col] for col in df.columns if col not in ['id', 'Matter ID']}
        ).with_inputs('file_content', 'file_id')

        examples.append(example)

    # Split 80/20 for train/val
    split_idx = int(len(examples) * 0.8)
    return examples[:split_idx], examples[split_idx:]
```

### OwnRight-style Real Estate Data

Specific pattern for the OwnRight customer example:

```python
import pandas as pd
from pathlib import Path
import dspy

DATA_PATH = "examples/customers/ownright/data.csv"
PDF_DIR = "examples/customers/ownright/downloaded_files"

def load_ownright_data():
    """Load OwnRight real estate data with PDF documents."""
    df = pd.read_csv(DATA_PATH)
    examples = []

    for _, row in df.iterrows():
        matter_id = row['Matter ID']

        # Try PDF first, then PNG
        pdf_path = Path(PDF_DIR) / f"{matter_id}.pdf"
        if not pdf_path.exists():
            pdf_path = Path(PDF_DIR) / f"{matter_id}.png"
            if not pdf_path.exists():
                continue

        file_content = load_pdf_for_llm(str(pdf_path))

        example = dspy.Example(
            pdf_content=file_content,
            matter_id=matter_id,
            # Ground truth from CSV
            key_date=str(row['Key Date']),
            purchase_amount=float(row['Purchase Amount']),
            kind=row['Kind'],
            purchaser_names=row['Purchaser Names'],
            seller_names=row['Seller Names'],
        ).with_inputs('pdf_content', 'matter_id')

        examples.append(example)

    split_idx = int(len(examples) * 0.8)
    return examples[:split_idx], examples[split_idx:]
```

---

## Configuring Multimodal LMs in DSPy

### Claude (Anthropic)

```python
import dspy

# Claude 3.5 Sonnet - best for document understanding
lm = dspy.LM("anthropic/claude-3-5-sonnet-latest")

# Claude 3 Opus - most capable
lm = dspy.LM("anthropic/claude-3-opus-latest")

dspy.configure(lm=lm)
```

### GPT-4o (OpenAI)

```python
import dspy

# GPT-4o - multimodal capable
lm = dspy.LM("openai/gpt-4o")

# GPT-4o mini - faster, cheaper
lm = dspy.LM("openai/gpt-4o-mini")

dspy.configure(lm=lm)
```

### Using Different Models for Different Purposes

```python
import dspy
from gepa_observable import GEPA

# Main model for document processing (needs vision)
main_lm = dspy.LM("anthropic/claude-3-5-sonnet-latest")

# Reflection model for prompt optimization (text-only is fine)
reflection_lm = dspy.LM("openai/gpt-4o")

dspy.configure(lm=main_lm)

optimizer = GEPA(
    metric=metric,
    auto="medium",
    reflection_lm=reflection_lm,  # Use different model for reflection
    server_url="http://localhost:3000",
)
```

---

## DSPy Signature for Multimodal Tasks

### PDF Extraction Signature

```python
class PDFExtraction(dspy.Signature):
    """Extract structured information from a PDF document."""

    pdf_content: str = dspy.InputField(
        desc="Base64-encoded PDF content. The document will be processed visually."
    )
    document_id: str = dspy.InputField(
        desc="Document identifier for reference"
    )

    # Output fields - customize based on what you're extracting
    key_date: str = dspy.OutputField(desc="Primary date in YYYY-MM-DD format")
    amount: float = dspy.OutputField(desc="Main amount in dollars")
    parties: list[str] = dspy.OutputField(desc="Names of all parties involved")
```

### Image Analysis Signature

```python
class ImageAnalysis(dspy.Signature):
    """Analyze an image and extract information."""

    image_content: str = dspy.InputField(
        desc="Base64-encoded image content"
    )
    analysis_type: str = dspy.InputField(
        desc="Type of analysis: 'describe', 'extract_text', 'classify'",
        default="describe"
    )

    description: str = dspy.OutputField(desc="Description or analysis result")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0 to 1.0")
```

---

## Complete Example: PDF Extraction Pipeline

```python
#!/usr/bin/env python3
"""Complete PDF extraction pipeline with GEPA optimization."""

import base64
import dspy
import pandas as pd
from pathlib import Path
from gepa_observable import GEPA


# Configuration
LM_MODEL = "anthropic/claude-3-5-sonnet-latest"
REFLECTION_LM = "openai/gpt-4o"
DATA_PATH = "data.csv"
FILE_DIR = "files"


def load_pdf_for_llm(pdf_path: str) -> str:
    """Load PDF as base64."""
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class DocumentExtraction(dspy.Signature):
    """Extract key information from document."""

    document_content: str = dspy.InputField(desc="Base64-encoded document")
    document_id: str = dspy.InputField(desc="Document ID")

    extracted_date: str = dspy.OutputField(desc="Date in YYYY-MM-DD")
    extracted_amount: float = dspy.OutputField(desc="Amount in dollars")
    extracted_names: str = dspy.OutputField(desc="Comma-separated names")


program = dspy.ChainOfThought(DocumentExtraction)


def load_data():
    """Load data with documents."""
    df = pd.read_csv(DATA_PATH)
    examples = []

    for _, row in df.iterrows():
        doc_id = row['id']
        doc_path = Path(FILE_DIR) / f"{doc_id}.pdf"

        if not doc_path.exists():
            continue

        content = load_pdf_for_llm(str(doc_path))

        example = dspy.Example(
            document_content=content,
            document_id=str(doc_id),
            # Ground truth
            extracted_date=str(row['date']),
            extracted_amount=float(row['amount']),
            extracted_names=row['names'],
        ).with_inputs('document_content', 'document_id')

        examples.append(example)

    split = int(len(examples) * 0.8)
    return examples[:split], examples[split:]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate extraction accuracy."""
    score = 0.0
    feedback = []

    # Check date
    if str(gold.extracted_date) == str(getattr(pred, 'extracted_date', '')):
        score += 0.33
    else:
        feedback.append(f"date: expected {gold.extracted_date}")

    # Check amount
    try:
        gold_amt = float(gold.extracted_amount)
        pred_amt = float(getattr(pred, 'extracted_amount', 0))
        if abs(gold_amt - pred_amt) / gold_amt < 0.01:
            score += 0.33
        else:
            feedback.append(f"amount: expected {gold_amt}, got {pred_amt}")
    except:
        feedback.append("amount: parse error")

    # Check names (fuzzy)
    gold_names = set(n.strip().lower() for n in gold.extracted_names.split(','))
    pred_names = set(n.strip().lower() for n in str(getattr(pred, 'extracted_names', '')).split(','))
    if gold_names == pred_names:
        score += 0.34
    else:
        feedback.append(f"names: expected {gold.extracted_names}")

    return dspy.Prediction(
        score=score,
        feedback="; ".join(feedback) if feedback else "All correct!"
    )


def main():
    # Configure LM
    lm = dspy.LM(LM_MODEL)
    reflection_lm = dspy.LM(REFLECTION_LM)
    dspy.configure(lm=lm)

    # Load data
    train, val = load_data()
    print(f"Loaded {len(train)} train, {len(val)} val examples")

    # Optimize
    optimizer = GEPA(
        metric=metric,
        auto="medium",
        reflection_lm=reflection_lm,
        server_url="http://localhost:3000",
        project_name="Document Extraction",
        capture_lm_calls=True,
        verbose=True,
    )

    optimized = optimizer.compile(student=program, trainset=train, valset=val)
    optimized.save("optimized_extractor.json")

    return optimized


if __name__ == "__main__":
    main()
```

---

## Tips for Multimodal Tasks

1. **Use vision-capable models** - Claude 3.5 Sonnet or GPT-4o for best results
2. **Keep files reasonable size** - Very large PDFs may hit token limits
3. **Be specific in signatures** - Describe expected format in OutputField descriptions
4. **Handle missing files gracefully** - Skip or warn when files don't exist
5. **Test with a few examples first** - Multimodal calls can be expensive
6. **Use text-only models for reflection** - Saves cost, reflection doesn't need vision
