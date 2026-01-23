#!/usr/bin/env python3
"""
Validate that required API keys are set in .env.local

Usage:
    python validate-env.py [model1] [model2] ...

Example:
    python validate-env.py openai/gpt-4o-mini openai/gpt-4o
"""

import os
import sys
from pathlib import Path

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Look for .env.local in current directory and parent directories
    env_file = Path(".env.local")
    if not env_file.exists():
        env_file = Path("../.env.local")
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass


def get_required_keys(models: list[str]) -> set[str]:
    """Determine which API keys are needed based on model names."""
    required = set()

    for model in models:
        model_lower = model.lower()
        if model_lower.startswith("openai/") or "gpt" in model_lower:
            required.add("OPENAI_API_KEY")
        elif model_lower.startswith("anthropic/") or "claude" in model_lower:
            required.add("ANTHROPIC_API_KEY")
        elif model_lower.startswith("google/") or "gemini" in model_lower:
            required.add("GOOGLE_API_KEY")
        elif model_lower.startswith("together/"):
            required.add("TOGETHER_API_KEY")
        elif model_lower.startswith("groq/"):
            required.add("GROQ_API_KEY")
        elif model_lower.startswith("mistral/"):
            required.add("MISTRAL_API_KEY")

    return required


def validate_keys(required_keys: set[str]) -> tuple[list[str], list[str]]:
    """Check which required keys are present vs missing."""
    present = []
    missing = []

    for key in sorted(required_keys):
        value = os.getenv(key)
        if value and len(value) > 0:
            present.append(key)
        else:
            missing.append(key)

    return present, missing


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else []

    if not models:
        print("Usage: python validate-env.py [model1] [model2] ...")
        print("Example: python validate-env.py openai/gpt-4o-mini openai/gpt-4o")
        sys.exit(1)

    print(f"Checking API keys for models: {', '.join(models)}")
    print()

    required_keys = get_required_keys(models)

    if not required_keys:
        print("No API keys detected as required for these models.")
        sys.exit(0)

    present, missing = validate_keys(required_keys)

    if present:
        print("Found API keys:")
        for key in present:
            print(f"  [OK] {key}")

    if missing:
        print()
        print("MISSING API keys:")
        for key in missing:
            print(f"  [MISSING] {key}")

        print()
        print("Please add the missing keys to your .env.local file:")
        print()
        for key in missing:
            print(f"  {key}=your-key-here")
        print()
        sys.exit(1)

    print()
    print("All required API keys are present!")
    sys.exit(0)


if __name__ == "__main__":
    main()
