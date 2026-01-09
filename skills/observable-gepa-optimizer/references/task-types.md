# Task Types and Signature Patterns

Common task types with DSPy Signature patterns. Use these as templates when defining your task.

---

## Extraction

Extract structured data from unstructured input (documents, text, files).

### Basic Text Extraction

```python
class ExtractInfo(dspy.Signature):
    """Extract structured information from the input text."""

    text: str = dspy.InputField(desc="Input text to extract from")

    # Define output fields for each piece of data to extract
    name: str = dspy.OutputField(desc="Extracted name")
    date: str = dspy.OutputField(desc="Extracted date in YYYY-MM-DD format")
    amount: float = dspy.OutputField(desc="Extracted amount in dollars")
```

### Multi-field Document Extraction

For extracting multiple structured fields from documents (like the OwnRight real estate example):

```python
class DocumentExtraction(dspy.Signature):
    """Extract structured data from a document."""

    document_content: str = dspy.InputField(desc="Document content (text or base64 for PDFs)")
    document_id: str = dspy.InputField(desc="Document identifier for reference")

    # Multiple output fields matching your ground truth columns
    key_date: str = dspy.OutputField(desc="Primary date in YYYY-MM-DD format")
    transaction_amount: float = dspy.OutputField(desc="Amount in dollars")
    transaction_type: str = dspy.OutputField(desc="Type: PURCHASE, SALE, etc.")
    party_a_names: str = dspy.OutputField(desc="Comma-separated names of first party")
    party_b_names: str = dspy.OutputField(desc="Comma-separated names of second party")
```

### Multimodal Extraction (PDFs/Images)

For documents sent directly to multimodal LLM APIs:

```python
class PDFExtraction(dspy.Signature):
    """Extract structured data from a PDF document using vision capabilities."""

    pdf_content: str = dspy.InputField(desc="Base64-encoded PDF content")
    extraction_instructions: str = dspy.InputField(
        desc="Specific fields to extract",
        default="Extract all relevant transaction details"
    )

    # Output fields
    extracted_data: dict = dspy.OutputField(desc="Dictionary of extracted fields")
```

---

## Classification

Assign labels or categories to input.

### Binary Classification

```python
class BinaryClassify(dspy.Signature):
    """Classify the input as positive or negative."""

    text: str = dspy.InputField(desc="Text to classify")

    label: str = dspy.OutputField(desc="Classification: 'positive' or 'negative'")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0 to 1.0")
```

### Multi-class Classification

```python
class MultiClassify(dspy.Signature):
    """Classify the input into one of the defined categories."""

    text: str = dspy.InputField(desc="Text to classify")
    categories: str = dspy.InputField(
        desc="Available categories",
        default="tech, sports, politics, entertainment, business"
    )

    category: str = dspy.OutputField(desc="One of the available categories")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")
```

### Intent Classification

```python
class IntentClassify(dspy.Signature):
    """Classify user intent from their message."""

    user_message: str = dspy.InputField(desc="User's input message")

    intent: str = dspy.OutputField(
        desc="User intent: 'question', 'request', 'complaint', 'feedback', 'other'"
    )
    entities: list[str] = dspy.OutputField(desc="Key entities mentioned")
```

---

## Question Answering

Answer questions based on context.

### Extractive QA

```python
class ExtractiveQA(dspy.Signature):
    """Answer the question by extracting relevant information from the context."""

    context: str = dspy.InputField(desc="Context containing the answer")
    question: str = dspy.InputField(desc="Question to answer")

    answer: str = dspy.OutputField(desc="Answer extracted from context")
    evidence: str = dspy.OutputField(desc="Quote from context supporting the answer")
```

### Open-domain QA

```python
class OpenQA(dspy.Signature):
    """Answer the question using general knowledge."""

    question: str = dspy.InputField(desc="Question to answer")

    answer: str = dspy.OutputField(desc="Concise, accurate answer")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
```

### Multi-hop QA

```python
class MultiHopQA(dspy.Signature):
    """Answer complex questions requiring multiple reasoning steps."""

    context: str = dspy.InputField(desc="Context with relevant information")
    question: str = dspy.InputField(desc="Complex question requiring multiple steps")

    reasoning_steps: list[str] = dspy.OutputField(desc="List of reasoning steps taken")
    answer: str = dspy.OutputField(desc="Final answer")
```

---

## Summarization

Condense content while preserving key information.

### Basic Summarization

```python
class Summarize(dspy.Signature):
    """Summarize the input text concisely."""

    text: str = dspy.InputField(desc="Text to summarize")
    max_length: int = dspy.InputField(desc="Maximum summary length in words", default=100)

    summary: str = dspy.OutputField(desc="Concise summary")
```

### Structured Summarization

```python
class StructuredSummary(dspy.Signature):
    """Create a structured summary with key points."""

    document: str = dspy.InputField(desc="Document to summarize")

    main_topic: str = dspy.OutputField(desc="Main topic in one sentence")
    key_points: list[str] = dspy.OutputField(desc="3-5 key points")
    conclusions: str = dspy.OutputField(desc="Main conclusions or takeaways")
```

---

## Generation

Create new content based on specifications.

### Content Generation

```python
class GenerateContent(dspy.Signature):
    """Generate content based on the given specifications."""

    topic: str = dspy.InputField(desc="Topic to write about")
    style: str = dspy.InputField(desc="Writing style: formal, casual, technical")
    length: str = dspy.InputField(desc="Desired length: short, medium, long")

    content: str = dspy.OutputField(desc="Generated content")
```

### Code Generation

```python
class GenerateCode(dspy.Signature):
    """Generate code based on the description."""

    description: str = dspy.InputField(desc="What the code should do")
    language: str = dspy.InputField(desc="Programming language")

    code: str = dspy.OutputField(desc="Generated code")
    explanation: str = dspy.OutputField(desc="Brief explanation of the code")
```

---

## Choosing a DSPy Module

After defining your Signature, choose the appropriate module:

| Module | Use When | Example |
|--------|----------|---------|
| `dspy.ChainOfThought(Sig)` | Task benefits from step-by-step reasoning | Extraction, QA, Classification |
| `dspy.Predict(Sig)` | Simple, direct output needed | Binary classification, simple extraction |
| `dspy.ReAct(Sig, tools=[...])` | Task requires tool use or external actions | Search, calculation, multi-step tasks |
| `dspy.ProgramOfThought(Sig)` | Need to generate and execute code | Math problems, data analysis |

### Example Usage

```python
# For extraction with reasoning
program = dspy.ChainOfThought(DocumentExtraction)

# For simple classification
program = dspy.Predict(BinaryClassify)

# For tasks needing tools
program = dspy.ReAct(
    OpenQA,
    tools=[dspy.Tool.from_function(search_web)]
)
```
