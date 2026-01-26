---
name: gepa-demo
description: Guide new or experienced users to run a GEPA prompt-optimization demo with gepa-observable and the web dashboard. Use when a user is interested in determining the value of prompt optimization, and has CSV/JSON/JSONL data. We map their data to DSPy Examples, define a DSPy Signatures & Module, help them establish a metric (rule-based or LLM judge with feedback), and finally split their dataset and create an optimizationscript for them.
---

# GEPA Demo Onboarding

Deliver a working end-to-end demo for a user dataset with minimal DSPy knowledge.

## Workflow (follow in order)

0) Validate prerequisites
- Check that required tools are installed:
  - Python 3.8+: `python --version`
  - Node.js 20.19+: `node --version` (Prisma requirement)
  - npm: `npm --version`
  - git: `git --version`
- If any are missing, provide installation instructions for the user's OS.

1) Ensure repo is cloned and installed
- Confirm if it's ok to clone `dspy-gepa-logger` repo to the current folder (if not, ask where to clone it).
- Clone it from https://github.com/raveeshbhalla/dspy-gepa-logger.git
- Install in editable mode so `gepa_observable` imports work from any folder.
- See `references/repo-setup.md`.

2) Confirm data location
- Look in the current folder for csv, json or jsonl files.
  - Confirm with the user that if that's the file they want to use for optimization.
  - If not, ask where the data file lives and move it to the current folder.

3) Inspect data and define the task schema
- Load a small sample (first ~5–10 rows).
- Ask the user to map which fields are **inputs** and which are **expected outputs**.
  - Show them the fields and some sample values so they have an idea of what data they're working with.
- Identify optional **metadata** fields (IDs, timestamps, categories, etc.).
- Decide the output type (single field, multi-field, structured JSON, etc.).
- For messy columns, propose a normalized mapping and confirm.
- See `references/data-mapping.md` for question prompts and mapping patterns.

4) Collect model + budget preferences
- **IMPORTANT**: Use AskUserQuestion tool if available for better interactivity
- Ask for:
  - **Task LM** (what they'll use in production - can be smaller/cheaper like gpt-5-mini, claude-4-5-haiku)
  - **Reflective/Judge LM** (recommend stronger model like gpt-5.2, claude-4-5-sonnet)
  - **Budget**: `auto=light|medium|heavy` OR `max_full_evals` OR `max_metric_calls`. Default to `auto=light`
  - Whether to attach the web dashboard (default: yes)
  - Project/run name
  - Seed prompt (user can provide some text, rubrics or point to a file to extract from)
- If unclear, propose sensible defaults based on your web search and confirm.

5) Create `.env.local` and verify API key
- Create `<user-folder>/optimizer-script/.env.local` with template:
  ```
  OPENAI_API_KEY=sk-...  # User must replace with actual key
  TASK_LM=openai/gpt-5-mini
  REFLECTIVE_LM=openai/gpt-5.2
  JUDGE_LM=openai/gpt-5.2
  ```
- **⚠️ CRITICAL**: Inform the user they MUST manually edit this file and add their actual API key
- The placeholder `your_openai_api_key_here` will cause authentication errors
- If using other providers (Anthropic, etc.), add those keys too (see `references/env.md`)
- Keep the reflective LM and judge LM the same unless the user explicitly splits them
- **WAIT** for user confirmation that they've added their API key before proceeding to verify that it can be found

6) Build DSPy objects
- Create `dspy.Example` list from the dataset and call `.with_inputs(...)`.
- Define a `dspy.Signature` from input/output fields.
- Choose a simple `dspy.Module` (e.g., `ChainOfThought` or `Predict`).
- See `references/data-mapping.md` for patterns.

7) Create the metric with feedback
- Prefer rule-based metrics when ground truth is clean.
- Otherwise use an LLM judge with strong feedback text.
- Metric must return `dspy.Prediction(score=float, feedback=str)`.
- For LLM judges, define a DSPy `Signature` (e.g., `JudgeSig`) and call
  `dspy.ChainOfThought(JudgeSig)` to get structured outputs
- Keep a rule-based fallback if judge output is empty/unusable.
- See `references/metrics.md` for templates.

8) Split dataset (train/val/test)
- If dataset <100, skip test set.
- Ensure valset is representative but not too large for budget.
- Use a fixed seed for reproducibility.

9) Generate demo scripts (split files)
- Create `<user-folder>/optimizer-script/` and copy **all files** from `assets/gepa-demo/` into it.
- Edit `gepa_demo_data.py`, `gepa_demo_program.py`, `gepa_demo_metric.py` to match the dataset.
- Ensure the runner `gepa-demo-script.py` keeps CLI args for:
  - `--seed-prompt`
  - `--auto` or `--max-full-evals` or `--max-metric-calls`
  - `--server-url`
  - `--project-name`
  - `--log-dir`
  - `--train-ratio`/`--val-ratio`
- Keep all paths relative to `optimizer-script/`.

10) Run a pre-optimization smoke test
- Run `python <user-folder>/optimizer-script/gepa-demo-eval.py --data-file <path>`.
- Confirm the program runs, metric returns `score` + `feedback`, and outputs look sane.
- This validates data mapping, program wiring, and metric behavior before full optimization.
- Also run with `--use-llm-judge`. If scores are all zero, the judge output
  is likely malformed or not being parsed.

11) Start the web dashboard
- Follow `references/web-dashboard.md` to start the server from the **cloned repo** `web/` directory.
- Ensure `npx prisma generate` is run before `npx prisma migrate deploy`
- Keep the terminal window open - the server must stay running
- Verify server is running at http://localhost:3000
- If server stops, restart with: `cd web && npm run dev`

12) Run the optimizer
- **BEFORE starting**: Ask user to open http://localhost:3000 in their browser to watch live progress
- Run `python <user-folder>/optimizer-script/gepa-demo-script.py ...` with the server URL
- Inform user this will take 2-10 minutes depending on budget and dataset size
- Progress will appear in both:
  - Terminal output (text logs)
  - Web dashboard (visual charts and tables)
- Don't close terminal while optimization is running
- After completion, verify runs appear in the dashboard
- If the CLI times out, the run may still be active. Check the dashboard
  before restarting.

13) Sanity check + handoff
- Run a small test input through the optimized program.
- Show where outputs/logs live.
- Verify final accuracy improvement in the database or dashboard.
- Provide next steps:
  - Tune budgets (`auto=medium` or `auto=heavy` for better results)
  - Refine judge rubric in the metric
  - Add more training data
  - Test on held-out test set if available
  - Deploy optimized prompt to production

## Troubleshooting

**To start completely fresh:**
```bash
# Stop web server
pkill -f "next dev"

# Remove directories
rm -rf optimizer-script dspy-gepa-logger

# Clear database (if needed)
rm dspy-gepa-logger/web/dev.db

# Start from step 0
```

**Common issues:**

1. **"Incorrect API key" error**
   - Edit `optimizer-script/.env.local` with your actual API key
   - Replace `your_openai_api_key_here` with `sk-...`

2. **"Cannot find module '.prisma/client'" error**
   - Run `npx prisma generate` in the `web/` directory
   - Then run `npx prisma migrate deploy`

3. **"Port 3000 already in use" error**
   - Stop existing server: `lsof -i :3000` then kill the process
   - Or use a different port in the web server config

4. **Web server stops unexpectedly**
   - Check the background process: `lsof -i :3000`
   - Restart: `cd web && npm run dev`
   - Keep terminal window open while optimization runs

5. **"temperature not supported" error**
   - Update model name or check if model requires `temperature=1.0`
   - Edit the generated scripts to adjust temperature

6. **No data showing in dashboard**
   - Verify server is running: http://localhost:3000
   - Check database has records: `sqlite3 web/dev.db "SELECT COUNT(*) FROM Run;"`
   - Ensure `--server-url http://localhost:3000` was passed to optimizer

## Using AskUserQuestion Tool

When available, use the AskUserQuestion tool for better interactivity:
- Step 3: Ask about input/output field mapping
- Step 4: Present model choices with descriptions
- Step 5: Confirm API key is added
- Step 12: Confirm user opened browser before starting optimization

This creates a more guided, interactive experience.

## References

- `references/data-mapping.md`
- `references/metrics.md`
- `references/env.md`
- `references/repo-setup.md`
- `references/web-dashboard.md`
- `references/interactive-prompting.md`
- `references/user-journeys.md`
- `assets/gepa-demo/`
7. **"listen EPERM" / port bind errors**
   - Some sandboxed shells cannot bind to ports without elevated permissions.
   - Retry starting the server with an escalated command or run it in a user shell.
