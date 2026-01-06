# Copyright (c) 2025 - GEPA Observable Fork
# A fork of GEPA with first-class observer support for full optimization observability
#
# This package provides an observable GEPA optimizer with DSPy-compatible API.
# Use the GEPA class for a Teleprompter-style interface, or optimize() for direct access.

from gepa_observable.api import optimize
from gepa_observable.gepa import GEPA
from gepa_observable.observers import (
    # Event dataclasses
    AcceptanceDecisionEvent,
    IterationStartEvent,
    MergeEvent,
    MiniBatchEvalEvent,
    OptimizationCompleteEvent,
    ReflectionEvent,
    SeedValidationEvent,
    ValsetEvalEvent,
    # Protocol and manager
    GEPAObserver,
    ObserverManager,
    # Built-in observers
    LoggingObserver,
    ServerObserver,
)

__all__ = [
    # Main API - DSPy-compatible Teleprompter
    "GEPA",
    # Direct API - for advanced usage
    "optimize",
    # Observer protocol and manager
    "GEPAObserver",
    "ObserverManager",
    # Built-in observers
    "LoggingObserver",
    "ServerObserver",
    # Event dataclasses
    "SeedValidationEvent",
    "IterationStartEvent",
    "MiniBatchEvalEvent",
    "ReflectionEvent",
    "AcceptanceDecisionEvent",
    "ValsetEvalEvent",
    "MergeEvent",
    "OptimizationCompleteEvent",
]

__version__ = "0.1.0"
