# Interactive Prompting with AskUserQuestion

When the agent has access to the AskUserQuestion tool, use it to create a more guided, interactive experience.

## When to Use AskUserQuestion

### Step 3: Task Schema Definition
Ask user to identify input and output fields:

```
Question: "Which fields should be used as inputs for the model?"
Options:
- All fields except the target
- Specific subset (let me choose)
- I'm not sure (recommend based on data)

Question: "What should the model predict (outputs)?"
Options:
- Single field (classification or prediction)
- Multiple fields (multi-output task)
- Structured output (JSON/complex)
```

### Step 4: Model Selection
Present latest models with descriptions:

```
Question: "Which model should we use for the task (cheaper, faster)?"
Options:
- gpt-5-mini (Fast, cost-effective, good for most tasks)
- claude-4-5-haiku (Fast, Anthropic's efficient model)
- Other (specify custom model)

Question: "Which model for reflection/judging (stronger, more accurate)?"
Options:
- gpt-5.2 (Recommended - powerful, reliable)
- claude-4-5-opus (Excellent reasoning, Anthropic)
- Same as task model (saves cost)
```

### Step 4: Budget Selection
```
Question: "How much optimization budget do you want?"
Options:
- Light (Quick demo, ~20-30 evaluations, 2-5 min)
- Medium (Balanced, ~50-100 evaluations, 5-15 min) (Recommended)
- Heavy (Thorough, ~200+ evaluations, 15-30 min)
- Custom (I'll specify exact numbers)
```

### Step 5: API Key Confirmation
```
Question: "Have you added your API key to .env.local?"
Options:
- Yes, I've added my API key (Recommended)
- No, I need to do that now
- I need help finding my API key
```

### Step 12: Dashboard Readiness
```
Question: "Ready to start optimization?"
Options:
- Yes, I have the dashboard open at http://localhost:3000 (Recommended)
- Let me open it now
- Skip dashboard (run without visualization)
```

## Benefits

1. **Clearer choices**: Users see options rather than free-form questions
2. **Guided experience**: Recommendations help users make good decisions
3. **Validation**: Ensures critical steps (like API key) are completed
4. **Better UX**: More interactive than text-only prompts

## Fallback

If AskUserQuestion is not available:
- Use regular text prompts
- List options clearly with numbers
- Ask for user confirmation explicitly
- Be more verbose in explanations
