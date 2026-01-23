---
name: ask-about-run
description: Answer questions about completed GEPA optimization runs. Analyze prompt evolution, score improvements, iteration details, and lineage. Use when users want to understand their optimization results or compare runs.
---

# Ask About Optimization Run

Help users understand their completed GEPA optimization runs by answering questions about:
- Score improvements and trends
- Prompt evolution across iterations
- Lineage and ancestry of candidates
- Why certain iterations succeeded or failed
- Comparison between runs

## Getting Run Data

### From Dashboard API

If the dashboard is running, fetch run data:

```python
import requests

def get_runs(server_url="http://localhost:3000"):
    """List all optimization runs."""
    response = requests.get(f"{server_url}/api/runs")
    return response.json()

def get_run_details(run_id, server_url="http://localhost:3000"):
    """Get full details for a specific run."""
    response = requests.get(f"{server_url}/api/runs/{run_id}")
    return response.json()
```

### From Saved Files

If the user has saved run data, look for:
- `optimized_program.json` - The final optimized DSPy program
- `run_history.json` - If tracking was enabled

## Common Questions

### "What improved?"

1. Compare seed validation score to final best score
2. Show the score progression across iterations
3. Highlight the biggest jumps

```python
def analyze_improvement(run_data):
    seed_score = run_data["seed_validation"]["avg_score"]
    best_score = run_data["best_candidate"]["score"]

    improvement = best_score - seed_score
    improvement_pct = (improvement / seed_score) * 100 if seed_score > 0 else 0

    return {
        "seed_score": seed_score,
        "final_score": best_score,
        "absolute_improvement": improvement,
        "relative_improvement_pct": improvement_pct,
    }
```

### "Show me the prompt evolution"

1. Get the seed candidate text
2. Get the best candidate text
3. Show a diff or comparison

```python
def show_prompt_evolution(run_data):
    seed = run_data["seed_candidate"]
    best = run_data["best_candidate"]["instructions"]

    print("=== SEED PROMPT ===")
    for component, text in seed.items():
        print(f"\n{component}:")
        print(text)

    print("\n=== OPTIMIZED PROMPT ===")
    for component, text in best.items():
        print(f"\n{component}:")
        print(text)
```

### "What changed between iteration X and Y?"

1. Find both iterations in the data
2. Compare the candidate instructions
3. Show what was added/removed/modified

### "Why did iteration X improve?"

1. Look at the reflection event for that iteration
2. Show what feedback was given
3. Show what changes were proposed

```python
def explain_iteration(run_data, iteration_num):
    iteration = run_data["iterations"][iteration_num]

    return {
        "parent_score": iteration["parent_score"],
        "new_score": iteration["new_score"],
        "accepted": iteration["accepted"],
        "reflection_feedback": iteration.get("reflection", {}).get("feedback"),
        "proposed_changes": iteration.get("reflection", {}).get("proposed_texts"),
    }
```

### "Show me the worst examples"

1. Look at validation set evaluations
2. Sort by score (ascending)
3. Show the examples with lowest scores

```python
def find_worst_examples(run_data, n=5):
    evals = run_data["final_evaluation"]["per_example"]
    sorted_evals = sorted(evals, key=lambda x: x["score"])
    return sorted_evals[:n]
```

### "Compare with my previous run"

1. Get data from both runs
2. Compare:
   - Starting scores
   - Final scores
   - Number of iterations
   - Improvement rates

## Data Structure Reference

### Run Object
```json
{
  "id": "run_abc123",
  "project_name": "Math QA",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:45:00Z",

  "seed_candidate": {
    "predictor_name": "Original instruction text..."
  },

  "seed_validation": {
    "avg_score": 0.65,
    "per_example": [...]
  },

  "iterations": [...],

  "best_candidate": {
    "idx": 3,
    "score": 0.85,
    "instructions": {
      "predictor_name": "Optimized instruction text..."
    }
  },

  "final_evaluation": {
    "avg_score": 0.85,
    "per_example": [...]
  }
}
```

### Iteration Object
```json
{
  "iteration": 0,
  "parent_idx": 0,
  "parent_score": 0.65,

  "reflection": {
    "components_updated": ["predictor_name"],
    "proposed_texts": {
      "predictor_name": "New instruction..."
    },
    "feedback_used": "..."
  },

  "minibatch_eval": {
    "parent_scores": [0.6, 0.7, 0.65],
    "new_scores": [0.8, 0.7, 0.75]
  },

  "accepted": true,
  "proceed_to_valset": true,

  "valset_eval": {
    "score": 0.72,
    "is_new_best": true
  }
}
```

## Tips for Answering Questions

1. **Be specific with numbers** - Show exact scores, not just "it improved"
2. **Quote the actual prompts** - Let users see what changed
3. **Reference iterations by number** - "In iteration 3, the score jumped from..."
4. **Explain the why** - Use reflection feedback to explain improvements
5. **Visualize when helpful** - Tables and comparisons are clearer than paragraphs
