"""Core utilities for GEPA Observable.

This module provides shared utilities for context management, LM logging,
and serialization used by the observable GEPA optimizer.
"""

from .context import (
    clear_ctx,
    get_candidate,
    get_ctx,
    get_iteration,
    get_phase,
    set_candidate,
    set_ctx,
    set_iteration,
    set_phase,
    with_ctx,
)
from .lm_logger import DSPyLMLogger, LMCall
from .serialization import serialize_example_inputs, serialize_output, serialize_value

__all__ = [
    # Context management
    "set_ctx",
    "get_ctx",
    "clear_ctx",
    "with_ctx",
    "set_iteration",
    "set_phase",
    "set_candidate",
    "get_iteration",
    "get_phase",
    "get_candidate",
    # LM logging
    "DSPyLMLogger",
    "LMCall",
    # Serialization
    "serialize_value",
    "serialize_output",
    "serialize_example_inputs",
]
