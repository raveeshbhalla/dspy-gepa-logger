"""Unified context management for GEPA logging.

This module provides a single shared contextvar that all logging hooks use
to track the current iteration, phase, and candidate being processed.

Key design decisions:
1. Single contextvar (not multiple) to avoid mismatch bugs
2. Thread-safe via Python's contextvars module
3. set_ctx(key=None) clears that key
4. with_ctx() restores previous state on exit
"""

import contextvars
from typing import Any


# THE context - used by ALL hooks (state logger, LM logger, metric wrapper, proposer)
_gepa_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "gepa_context",
    default={}
)


def set_ctx(**kwargs: Any) -> None:
    """Update context with new values.

    Preserves existing values not overwritten. Pass key=None to clear that key.

    Args:
        iteration: Current GEPA iteration number
        phase: Current phase ('eval', 'reflection', 'proposal', 'selection', None)
        candidate_idx: Index of candidate being processed
        Any other keys you want to track

    Example:
        set_ctx(iteration=5, phase="eval", candidate_idx=3)
        set_ctx(phase=None)  # Clears phase key
    """
    current = dict(_gepa_context.get() or {})

    for key, value in kwargs.items():
        if value is not None:
            current[key] = value
        elif key in current:
            # Explicit None clears the key
            del current[key]

    _gepa_context.set(current)


def get_ctx() -> dict[str, Any]:
    """Get current context dict.

    Returns a copy of the context to prevent accidental mutation.

    Returns:
        dict with keys like: iteration, phase, candidate_idx, etc.
        Returns empty dict if no context set.

    Example:
        ctx = get_ctx()
        iteration = ctx.get('iteration')
        phase = ctx.get('phase')
    """
    return dict(_gepa_context.get() or {})


def clear_ctx() -> None:
    """Clear all context.

    Call this at the start of a new optimization run to ensure clean state.
    """
    _gepa_context.set({})


class _ContextManager:
    """Context manager for temporary context changes."""

    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs
        self._old_ctx: dict[str, Any] | None = None

    def __enter__(self) -> "_ContextManager":
        # Save current context
        self._old_ctx = get_ctx()
        # Apply new context values
        set_ctx(**self._kwargs)
        return self

    def __exit__(self, *args: Any) -> None:
        # Restore previous context
        if self._old_ctx is not None:
            _gepa_context.set(self._old_ctx)


def with_ctx(**kwargs: Any) -> _ContextManager:
    """Context manager for temporary context changes.

    On enter, applies the given context values. On exit, restores the
    previous context state completely.

    Args:
        **kwargs: Context values to set temporarily

    Returns:
        Context manager

    Example:
        with with_ctx(phase="reflection"):
            # LM calls here will be tagged with phase="reflection"
            do_reflection()
        # Context restored to previous state after block
    """
    return _ContextManager(**kwargs)


# Convenience functions for common operations


def set_iteration(iteration: int) -> None:
    """Set the current iteration number."""
    set_ctx(iteration=iteration)


def set_phase(phase: str | None) -> None:
    """Set the current phase."""
    set_ctx(phase=phase)


def set_candidate(candidate_idx: int | None) -> None:
    """Set the current candidate index."""
    set_ctx(candidate_idx=candidate_idx)


def get_iteration() -> int | None:
    """Get the current iteration number."""
    return get_ctx().get("iteration")


def get_phase() -> str | None:
    """Get the current phase."""
    return get_ctx().get("phase")


def get_candidate() -> int | None:
    """Get the current candidate index."""
    return get_ctx().get("candidate_idx")
