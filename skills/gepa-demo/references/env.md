# optimizer-script/.env.local

Place the env file in `<user-folder>/optimizer-script/.env.local` (next to `gepa-demo-script.py`).

## Standard API keys

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`

## Model choices

- `TASK_LM` (task LM, smaller/cheaper for production use)
- `REFLECTIVE_LM` (strong model for reflection and optimization)
- `JUDGE_LM` (optional; defaults to reflective LM, used for evaluation)

## Example

```
OPENAI_API_KEY=sk-...
TASK_LM=openai/gpt-5-mini
REFLECTIVE_LM=openai/gpt-5.2
JUDGE_LM=openai/gpt-5.2
```

## Important Notes

- **CRITICAL**: Replace `sk-...` or `your_openai_api_key_here` with your actual API key
- Add keys for other providers if using non-OpenAI models
- Keep REFLECTIVE_LM and JUDGE_LM the same unless you have a specific reason to split them
