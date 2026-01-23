# Budget Configuration Guide

GEPA optimization budget controls how much exploration and evaluation happens. Choose based on your time, cost, and quality requirements.

## Quick Options (auto)

Use the `auto` parameter for preset budgets:

### `auto="light"`
- **Candidates explored**: ~6
- **Time**: 5-15 minutes (depends on model speed)
- **Cost**: $ (lowest)
- **Best for**: Quick iterations, testing setup, small datasets
- **When to use**: You want fast feedback, already have a good prompt

### `auto="medium"` (Recommended)
- **Candidates explored**: ~12
- **Time**: 15-45 minutes
- **Cost**: $$
- **Best for**: Most use cases, balanced exploration
- **When to use**: Default choice, good tradeoff between quality and cost

### `auto="heavy"`
- **Candidates explored**: ~18
- **Time**: 45+ minutes
- **Cost**: $$$
- **Best for**: Final optimization, complex tasks, large budgets
- **When to use**: You want maximum quality, have time and budget

## Custom Options

For fine-grained control, use one of these parameters:

### `max_full_evals`
Maximum number of full validation set evaluations.

```python
optimizer = GEPA(
    metric=metric,
    max_full_evals=5,  # At most 5 full valset evaluations
)
```

**Guidance**:
- Each full eval runs metric on entire validation set
- More evals = more exploration = higher cost
- Typical range: 2-20

### `max_metric_calls`
Maximum total metric function calls across all evaluations.

```python
optimizer = GEPA(
    metric=metric,
    max_metric_calls=500,  # At most 500 metric calls total
)
```

**Guidance**:
- Includes both minibatch and full valset evals
- More granular control than max_full_evals
- Calculate: ~(trainset_size + valset_size) per full iteration

## Budget Calculation Formula

GEPA uses this formula for `auto` budgets:

```python
num_candidates = {
    "light": 6,
    "medium": 12,
    "heavy": 18,
}[auto]

# Approximate metric calls
calls_per_candidate = len(trainset) + len(valset)
total_calls = num_candidates * calls_per_candidate
```

## Cost Estimation

### LLM Costs (approximate)

For a dataset with 50 train + 20 val examples:

| Budget | Candidates | Metric Calls | ~GPT-4o-mini Cost | ~GPT-4o Cost |
|--------|------------|--------------|-------------------|--------------|
| light  | 6          | ~420         | $1-2              | $5-10        |
| medium | 12         | ~840         | $2-5              | $10-25       |
| heavy  | 18         | ~1260        | $5-10             | $25-50       |

**Note**: Actual costs vary based on prompt length, output length, and model used.

### Additional Costs

Remember to account for:
- **Reflection LM calls** - GEPA uses a (usually stronger) model for reflection
- **LLM-as-judge calls** - If using LLM-based scorers, each metric call = more LLM calls

## Recommendations by Use Case

### Prototyping / Testing Setup
```python
auto="light"
```
Quick feedback, verify everything works.

### Regular Optimization
```python
auto="medium"
```
Good balance for most tasks.

### Production / Final Optimization
```python
auto="heavy"
# or for even more exploration:
max_full_evals=25
```
Maximum quality when cost isn't the primary concern.

### Limited Budget
```python
max_metric_calls=200
```
Hard cap on total API calls.

### Large Validation Set
```python
# If valset is large, limit evals
max_full_evals=5
```
Avoid excessive costs from large validation sets.

## Monitoring Your Budget

The dashboard shows:
- Current iteration count
- Total evaluations completed
- Estimated remaining budget
- Cost tracking (if configured)

Watch the dashboard during optimization to understand your spend.

## Tips

1. **Start with `auto="light"`** to verify setup, then run `auto="medium"` or `auto="heavy"`
2. **Use smaller validation sets** (10-20 examples) to reduce cost per iteration
3. **LLM-as-judge multiplies cost** - consider using it only for final validation
4. **Strong reflection LM is worth it** - better reflection = fewer iterations needed
