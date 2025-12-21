"""Hooks for integrating with DSPy and GEPA."""

from dspy_gepa_logger.hooks.callback_handler import GEPALoggingCallback
from dspy_gepa_logger.hooks.gepa_adapter import (
    InstrumentedGEPAAdapter,
    create_instrumented_gepa,
    cleanup_instrumented_gepa,
)

__all__ = [
    "GEPALoggingCallback",
    "InstrumentedGEPAAdapter",
    "create_instrumented_gepa",
    "cleanup_instrumented_gepa",
]
