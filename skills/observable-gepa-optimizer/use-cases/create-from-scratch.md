---
name: create-from-scratch
description: Interactive workflow to create optimized DSPy programs from labelled datasets
---

# Create Optimized DSPy Programs from Scratch

This workflow guides you through creating a complete, runnable DSPy program with GEPA optimization from your labelled dataset.

## Overview

**What you provide:**
- A labelled dataset (CSV, JSON, etc.) with ground truth
- Optionally: Files referenced by the data (PDFs, images, etc.)

**What you get:**
- Complete DSPy Signature definition
- DSPy Module (ChainOfThought, ReAct, etc.)
- Data loading code for your specific format
- Metric/feedback function for evaluation
- GEPA optimizer setup with gepa-observable
- requirements.txt
- Your choice of: single .py file, modular structure, or Jupyter notebook

---

## Interactive Workflow

The agent will guide you through 5 phases to understand your requirements and generate the appropriate code.

### Phase 1: Data Understanding

**Questions the agent will ask:**

1. **"What is the path to your labelled dataset?"**
   - Accepts: CSV, JSON, or directory path
   - Example: `./data/training.csv` or `./examples/customers/ownright/data.csv`

2. **[Agent reads and analyzes the file]**
   - "I see your data has these columns: [list]. Let me understand the structure..."
   - Agent examines column types, sample values, and patterns

3. **"Which column(s) contain the INPUT that the model will process?"**
   - These are the fields the LLM will receive
   - Example: For document extraction, this might be a file path or ID column

4. **"Which column(s) contain the EXPECTED OUTPUT (ground truth)?"**
   - These are what the model should produce
   - Example: extracted fields like `key_date`, `amount`, `names`

5. **"Do any columns reference external files (PDFs, images)?"**
   - If yes: "What is the directory containing these files?"
   - If yes: "What is the naming pattern?" (e.g., `{Matter ID}.pdf`)
   - Files will be sent directly to multimodal LLM APIs (Claude, GPT-4o)

### Phase 2: Task Definition

**Questions the agent will ask:**

6. **"What type of task is this?"**
   - **Extraction** - Pull structured data from unstructured input (documents, PDFs)
   - **Classification** - Assign labels or categories to input
   - **Question Answering** - Answer questions based on context
   - **Summarization** - Condense content while preserving key information
   - **Generation** - Create new content based on specifications
   - **Custom** - Describe your specific task

7. **[Agent proposes a DSPy Signature based on your data and task]**
   ```python
   class YourTask(dspy.Signature):
       """Task description based on your input."""
       input_field: str = dspy.InputField(desc="...")
       output_field: str = dspy.OutputField(desc="...")
   ```
   "Does this look correct? Any fields to add, remove, or rename?"

8. **"What DSPy module should we use?"**
   - **ChainOfThought** (default) - Good for reasoning tasks, shows thinking process
   - **ReAct** - For tasks requiring tool use or multi-step actions
   - **Predict** - Simple, direct prediction without chain of thought

### Phase 3: Evaluation Strategy

**Questions the agent will ask:**

9. **"How should we evaluate if the model's output is correct?"**
   - **Exact match** - String must match exactly (good for categorical outputs)
   - **Fuzzy match** - Allow minor differences (typos, formatting variations)
   - **Field-by-field comparison** - Compare each output field separately with custom logic
   - **LLM-as-judge** - Use an LLM to score output quality
   - **Custom metric** - You provide the evaluation logic

10. **[If field-by-field comparison]**
    "How should each field be compared?"
    - Dates: Exact match or format-normalized?
    - Numbers: Exact or within tolerance (e.g., 1%)?
    - Names: Fuzzy match? Order-independent?
    - Text: Semantic similarity? Contains keywords?

11. **"Should incorrect predictions receive partial credit?"**
    - **Binary** - 0 or 1 only
    - **Partial credit** - Score based on how many fields/criteria match (0.0 to 1.0)

### Phase 4: Output Preferences

**Questions the agent will ask:**

12. **"What output format do you prefer?"**
    - **Single .py file** - Everything in one file, easiest to run
    - **Modular structure** - Separate files (signature.py, metric.py, data.py, main.py)
    - **Jupyter notebook** - Interactive with explanations and cell-by-cell execution

13. **"Should I generate a requirements.txt?"**
    - Default: Yes
    - Will include: `dspy-ai`, `dspy-gepa-logger`, `pandas`, and any task-specific packages

14. **"Where should I create the output files?"**
    - Default: Current directory or same location as your data
    - Agent confirms the path before writing

### Phase 5: Code Generation

Based on your answers, the agent generates:

1. **Complete, runnable code** with all components:
   - Configuration section (LM model, server URL, paths)
   - DSPy Signature definition
   - DSPy Module instantiation
   - Data loading function
   - Metric function with feedback
   - Main function with GEPA optimization
   - Command-line interface for parameters

2. **requirements.txt** with all dependencies

3. **Instructions** for running the optimization

---

## What the Generated Code Includes

### Configuration

```python
# User-configurable settings
LM_MODEL = "anthropic/claude-3-5-sonnet-latest"  # Multimodal for PDFs
REFLECTION_LM = "openai/gpt-4o"
SERVER_URL = "http://localhost:3000"  # Set to None to disable dashboard
PROJECT_NAME = "Your Project"

DATA_PATH = "path/to/your/data.csv"
FILE_DIR = "path/to/your/files"  # If applicable
```

### DSPy Signature

Tailored to your task and output fields:

```python
class YourTaskSignature(dspy.Signature):
    """Description of what this task does."""

    # Input fields based on your data
    document: str = dspy.InputField(desc="Input document content")

    # Output fields matching your ground truth columns
    extracted_field_1: str = dspy.OutputField(desc="...")
    extracted_field_2: float = dspy.OutputField(desc="...")
```

### Data Loading

Handles your specific data format:

```python
def load_data():
    """Load and split data into train/val sets."""
    df = pd.read_csv(DATA_PATH)
    examples = []

    for _, row in df.iterrows():
        # Load any referenced files (PDFs, images)
        # Create dspy.Example with inputs and ground truth
        example = dspy.Example(...).with_inputs(...)
        examples.append(example)

    # Split 80/20
    split_idx = int(len(examples) * 0.8)
    return examples[:split_idx], examples[split_idx:]
```

### Metric Function

With feedback for GEPA reflection:

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate prediction against ground truth with feedback."""
    # Compare fields based on your evaluation strategy
    # Calculate score (0.0 to 1.0)
    # Generate feedback string for incorrect predictions

    return dspy.Prediction(
        score=total_score,
        feedback=feedback_string
    )
```

### GEPA Optimization

With full observability:

```python
optimizer = GEPA(
    metric=metric,
    auto="medium",
    reflection_lm=dspy.LM(REFLECTION_LM),
    server_url=SERVER_URL,
    project_name=PROJECT_NAME,
    capture_lm_calls=True,
    verbose=True,
)

optimized = optimizer.compile(
    student=program,
    trainset=train_data,
    valset=val_data,
)
```

---

## References

For detailed patterns and templates, see:

- `../references/task-types.md` - Task-specific signature patterns
- `../references/metric-templates.md` - Metric function examples
- `../references/multimodal-patterns.md` - PDF/image handling with multimodal LLMs
- `../references/output-templates/single-file.md` - Single file template
- `../references/output-templates/modular-structure.md` - Modular project template
- `../references/output-templates/notebook.md` - Jupyter notebook template
