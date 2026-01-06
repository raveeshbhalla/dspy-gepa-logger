# Copyright (c) 2025 - GEPA Observable Fork
# A fork of GEPA with first-class observer support for full optimization observability

from gepa_observable.api import optimize
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
    # Main API
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
