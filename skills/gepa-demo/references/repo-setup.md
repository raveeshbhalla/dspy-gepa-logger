# Repo Setup (Required)

This skill assumes the user has cloned the `dspy-gepa-logger` repo locally so the web dashboard can run.

Repository: https://github.com/raveeshbhalla/dspy-gepa-logger.git

Steps:
1) Ask the user where to clone the repo (or use a default in their workspace).
2) Clone the repo:
```bash
git clone https://github.com/raveeshbhalla/dspy-gepa-logger.git
```
3) Install the Python package in editable mode so `gepa_observable` is importable from any folder:

```bash
pip install -e <path-to-cloned-repo>
```

If the user cannot use editable installs, set `PYTHONPATH` to `<repo>/src` for the session.
