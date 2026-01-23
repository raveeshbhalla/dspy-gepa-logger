# Orizu Demo Setup Guide

This guide walks you through setting up and running the Orizu prompt optimization demo using Claude Code and observable GEPA.

## What You'll Experience

The demo showcases Orizu's two core capabilities:

1. **Intelligent Scorer Creation** - Convert your evaluation data into sophisticated scoring functions (both programmatic and LLM-as-judge)
2. **Observable Prompt Optimization** - Run GEPA optimization with a real-time dashboard showing iterations, scores, and prompt evolution

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+** - Check with `python --version`
- **Node.js 18+** - Check with `node --version`
- **Claude Code** - The AI coding assistant
- **API Keys** - At least one of: OpenAI, Anthropic, or Google

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/raveeshbhalla/dspy-gepa-logger.git
cd dspy-gepa-logger
```

### 2. Install Python Package

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Install python-dotenv for .env.local support
pip install python-dotenv
```

### 3. Set Up the Web Dashboard

```bash
cd web

# Install dependencies
npm install

# Set up the database
echo 'DATABASE_URL="file:./dev.db"' > .env
npx prisma generate
npx prisma migrate deploy

cd ..
```

### 4. Create Your API Key File

Create a `.env.local` file in the repository root with your API keys:

```bash
# Create .env.local
cat > .env.local << 'EOF'
# Add the API keys you have (you need at least one)

# OpenAI (for gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=your-openai-key-here

# Anthropic (for claude-3-opus, claude-3-sonnet, etc.)
ANTHROPIC_API_KEY=your-anthropic-key-here

# Google (for gemini models)
GOOGLE_API_KEY=your-google-key-here
EOF
```

**Important**: The skill will tell you exactly which keys you need based on the models you choose.

### 5. Verify Setup

```bash
# Check Python package is installed
python -c "from gepa_observable import GEPA; print('OK')"

# Check web dashboard can start
cd web && npm run dev &
# Wait a few seconds, then visit http://localhost:3000
# Press Ctrl+C to stop for now
```

## Running the Demo

### Start Claude Code

Open Claude Code in the repository directory:

```bash
cd /path/to/dspy-gepa-logger
claude
```

### Available Skills

The demo includes two main skills:

#### `/orizu-optimize` - Main Demo Skill

This is the primary skill that guides you through the entire workflow:

```
/orizu-optimize path/to/your/data.csv
```

Or use the included example data:

```
/orizu-optimize examples/eg.csv
```

**What it does:**
1. Analyzes your evaluation data
2. Asks clarifying questions about your task
3. Creates scorer functions (programmatic + LLM-as-judge)
4. Helps you configure models and API keys
5. Runs GEPA optimization with the dashboard
6. Answers questions about results

#### `/ask-about-run` - Results Analysis

After running optimization, use this to explore your results:

```
/ask-about-run
```

**What it does:**
- Lists your optimization runs
- Analyzes score improvements
- Shows prompt evolution
- Explains why iterations succeeded or failed

### Demo Flow

1. **Start the demo with your data:**
   ```
   /orizu-optimize examples/eg.csv
   ```

2. **Answer the skill's questions:**
   - Confirm data format and field mappings
   - Describe what makes a good vs bad output
   - Choose your models (program LM, reflective LM, judge LM)

3. **Set up environment:**
   - The skill will tell you which API keys you need
   - Create/update `.env.local` with the required keys

4. **Configure optimization:**
   - Choose budget: `light`, `medium`, or `heavy`
   - The skill will start the dashboard automatically

5. **Watch optimization:**
   - Open http://localhost:3000 in your browser
   - See real-time iteration progress
   - Watch scores improve and prompts evolve

6. **Explore results:**
   - Ask questions about the run
   - See what changed in the optimized prompt
   - Understand why certain iterations improved

## Example Data

The repository includes example data you can use:

- `examples/eg.csv` - Math and string manipulation questions (20 examples)

Or bring your own data in CSV, JSON, or JSONL format with:
- An input field (question, prompt, input, etc.)
- An expected output field (answer, response, expected, etc.)

## Model Recommendations

### For the Demo

| Role | Recommended Model | Why |
|------|------------------|-----|
| Program LM | `openai/gpt-4o-mini` | Fast, cheap, good baseline |
| Reflective LM | `openai/gpt-4o` | Strong reasoning for reflection |
| Judge LM | `openai/gpt-4o` | Accurate evaluation |

### Budget Recommendations

| Option | Time | Cost | Use Case |
|--------|------|------|----------|
| `light` | 5-15 min | $ | Quick demo, testing setup |
| `medium` | 15-45 min | $$ | Standard demo, good results |
| `heavy` | 45+ min | $$$ | Best results, more time |

## Troubleshooting

### Dashboard won't start

```bash
# Make sure you're in the web directory
cd web

# Check for port conflicts
lsof -i :3000

# Try a fresh install
rm -rf node_modules
npm install
npm run dev
```

### API key errors

```bash
# Verify your .env.local file exists and has keys
cat .env.local

# Test with the validation script
python skills/orizu-optimize/scripts/validate-env.py openai/gpt-4o-mini
```

### Import errors

```bash
# Reinstall the package
pip install -e .

# Or reinstall with all dependencies
pip install -e ".[dev]"
```

## What's Next

After the demo, you can:

1. **Try with your own data** - Bring your evaluation dataset
2. **Customize scorers** - Modify the generated scorer functions
3. **Integrate into your workflow** - Use the optimized prompts in production
4. **Explore the dashboard** - Browse past runs and compare results

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the skill documentation in `skills/orizu-optimize/`
3. Contact the Orizu team at support@orizu.ai
