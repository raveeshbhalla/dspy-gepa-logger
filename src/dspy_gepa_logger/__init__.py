"""GEPA Run Tracking Library for DSPy.

This library provides comprehensive logging and visualization infrastructure
for DSPy GEPA optimization runs using public hooks (no monkey-patching).

Usage:
    from dspy_gepa_logger import create_logged_gepa, configure_dspy_logging

    gepa, tracker, logged_metric = create_logged_gepa(metric=my_metric)
    configure_dspy_logging(tracker)

    result = gepa.compile(student=MyProgram(), trainset=train, valset=val)

    # View results
    print(tracker.get_summary())
    tracker.print_prompt_diff()  # Compare original vs optimized prompts
"""

# Public API
from dspy_gepa_logger.api import (
    create_logged_gepa,
    configure_dspy_logging,
    create_tracker,
    wrap_metric,
    wrap_proposer,
    wrap_selector,
)

# Tracker
from dspy_gepa_logger.core.tracker_v2 import GEPATracker, CandidateDiff, LoggedLM

# Components
from dspy_gepa_logger.core.logged_metric import LoggedMetric, EvaluationRecord
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger, LMCall
from dspy_gepa_logger.core.state_logger import GEPAStateLogger, IterationDelta, IterationMetadata
from dspy_gepa_logger.core.logged_proposer import (
    LoggedInstructionProposer,
    LoggedSelector,
    ReflectionCall,
    ProposalCall,
)

# Context utilities
from dspy_gepa_logger.core.context import set_ctx, get_ctx, clear_ctx, with_ctx

# Serialization utilities
from dspy_gepa_logger.core.serialization import (
    serialize_value,
    serialize_output,
    serialize_example_inputs,
)

# Server integration (optional)
from dspy_gepa_logger.server.client import ServerClient

try:
    from dspy_gepa_logger._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without build

__all__ = [
    # API functions
    "create_logged_gepa",
    "configure_dspy_logging",
    "create_tracker",
    "wrap_metric",
    "wrap_proposer",
    "wrap_selector",
    # Tracker
    "GEPATracker",
    "CandidateDiff",
    "LoggedLM",
    # Components
    "LoggedMetric",
    "EvaluationRecord",
    "DSPyLMLogger",
    "LMCall",
    "GEPAStateLogger",
    "IterationDelta",
    "IterationMetadata",
    "LoggedInstructionProposer",
    "LoggedSelector",
    "ReflectionCall",
    "ProposalCall",
    # Context
    "set_ctx",
    "get_ctx",
    "clear_ctx",
    "with_ctx",
    # Serialization
    "serialize_value",
    "serialize_output",
    "serialize_example_inputs",
    # Server
    "ServerClient",
]
