"""Core tracking infrastructure for GEPA logging."""

# Context utilities
from dspy_gepa_logger.core.context import set_ctx, get_ctx, clear_ctx, with_ctx

# Metric logging
from dspy_gepa_logger.core.logged_metric import LoggedMetric, EvaluationRecord

# LM call logging
from dspy_gepa_logger.core.lm_logger import DSPyLMLogger, LMCall

# State logging
from dspy_gepa_logger.core.state_logger import GEPAStateLogger, IterationDelta, IterationMetadata

# Proposer/selector logging
from dspy_gepa_logger.core.logged_proposer import (
    LoggedInstructionProposer,
    LoggedSelector,
    ReflectionCall,
    ProposalCall,
)

# Unified tracker
from dspy_gepa_logger.core.tracker_v2 import GEPATracker, CandidateDiff

# Serialization utilities
from dspy_gepa_logger.core.serialization import (
    serialize_value,
    serialize_output,
    serialize_example_inputs,
)


__all__ = [
    # Context
    "set_ctx",
    "get_ctx",
    "clear_ctx",
    "with_ctx",
    # Metric
    "LoggedMetric",
    "EvaluationRecord",
    # LM logger
    "DSPyLMLogger",
    "LMCall",
    # State logger
    "GEPAStateLogger",
    "IterationDelta",
    "IterationMetadata",
    # Proposer
    "LoggedInstructionProposer",
    "LoggedSelector",
    "ReflectionCall",
    "ProposalCall",
    # Tracker
    "GEPATracker",
    "CandidateDiff",
    # Serialization
    "serialize_value",
    "serialize_output",
    "serialize_example_inputs",
]
