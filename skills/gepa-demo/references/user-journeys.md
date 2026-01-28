# User Journeys

1) CSV with clean input/expected
- User provides CSV with obvious columns.
- Map `input` + `expected`, build Examples, rule-based metric, split, run GEPA.

2) JSONL with messy keys
- Normalize keys, confirm mapping, create Signature and metric, run GEPA.

3) No labels, rubric-only
- Use LLM judge with rubric, provide rich feedback, run GEPA.

4) Multi-field outputs + metadata
- Define Signature with multiple OutputFields, metric returns composite score + feedback.
