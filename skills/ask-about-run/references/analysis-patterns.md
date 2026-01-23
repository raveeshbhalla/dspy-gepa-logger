# Analysis Patterns for Optimization Runs

## Score Trend Analysis

```python
def plot_score_trend(iterations):
    """Show how scores evolved over iterations."""
    scores = []
    for it in iterations:
        if it.get("valset_eval"):
            scores.append((it["iteration"], it["valset_eval"]["score"]))

    print("Iteration | Score")
    print("-" * 20)
    for it_num, score in scores:
        bar = "#" * int(score * 20)
        print(f"{it_num:9d} | {score:.2%} {bar}")
```

## Candidate Lineage

```python
def trace_lineage(run_data, candidate_idx):
    """Trace the ancestry of a candidate."""
    lineage = []
    current = candidate_idx

    for it in reversed(run_data["iterations"]):
        if it.get("new_candidate_idx") == current:
            lineage.append({
                "iteration": it["iteration"],
                "parent": it["parent_idx"],
                "score_change": it["valset_eval"]["score"] - it["parent_score"]
                    if it.get("valset_eval") else None
            })
            current = it["parent_idx"]

    return list(reversed(lineage))
```

## Feedback Analysis

```python
def extract_common_feedback(run_data):
    """Find common themes in feedback across iterations."""
    feedbacks = []
    for it in run_data["iterations"]:
        if it.get("minibatch_eval", {}).get("feedbacks"):
            feedbacks.extend(it["minibatch_eval"]["feedbacks"])

    # Count common words/phrases
    from collections import Counter
    words = []
    for fb in feedbacks:
        if fb:
            words.extend(fb.lower().split())

    return Counter(words).most_common(20)
```

## Example Difficulty Analysis

```python
def categorize_examples(run_data):
    """Categorize examples by difficulty."""
    final_evals = run_data["final_evaluation"]["per_example"]

    easy = [e for e in final_evals if e["score"] >= 0.9]
    medium = [e for e in final_evals if 0.5 <= e["score"] < 0.9]
    hard = [e for e in final_evals if e["score"] < 0.5]

    return {
        "easy": len(easy),
        "medium": len(medium),
        "hard": len(hard),
        "hard_examples": hard
    }
```

## Prompt Diff

```python
def diff_prompts(old_text, new_text):
    """Show what changed between two prompt versions."""
    import difflib

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile="old", tofile="new"
    )
    return "".join(diff)
```

## Acceptance Rate

```python
def calculate_acceptance_rate(run_data):
    """Calculate what percentage of proposed changes were accepted."""
    iterations = run_data["iterations"]
    total = len(iterations)
    accepted = sum(1 for it in iterations if it.get("accepted"))

    return {
        "total_proposals": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_rate": accepted / total if total > 0 else 0
    }
```
