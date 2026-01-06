# Copyright (c) 2025 - GEPA Observable Fork
# Helpers for DSPy LM integration and auto-registration

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gepa_observable.core.lm_logger import DSPyLMLogger


def auto_register_lm_logger(lm_logger: DSPyLMLogger) -> bool:
    """Attempt to auto-register LM logger with the current DSPy LM.

    This function tries to add the LM logger to the callbacks of the
    currently configured DSPy LM. This allows automatic capture of
    LM calls without requiring manual registration.

    Args:
        lm_logger: The DSPyLMLogger instance to register.

    Returns:
        True if registration was successful, False otherwise.

    Example:
        >>> from gepa_observable.core.lm_logger import DSPyLMLogger
        >>> from gepa_observable._lm_integration import auto_register_lm_logger
        >>> lm_logger = DSPyLMLogger()
        >>> if auto_register_lm_logger(lm_logger):
        ...     print("LM logger registered automatically")
        ... else:
        ...     print("Manual registration required")
    """
    try:
        import dspy

        # Check if dspy.settings has an LM configured
        current_lm = getattr(dspy.settings, "lm", None)
        if current_lm is None:
            return False

        # DSPy LMs have a callbacks attribute
        if hasattr(current_lm, "callbacks"):
            callbacks = current_lm.callbacks
            if callbacks is None:
                current_lm.callbacks = [lm_logger]
                return True

            # Normalize callbacks to a list - handle iterables (set, tuple) and non-iterables
            if not isinstance(callbacks, (list, tuple)):
                try:
                    callbacks = list(callbacks)
                except TypeError:
                    # Non-iterable single callback
                    callbacks = [callbacks]

            if lm_logger not in callbacks:
                # Reassign to handle both mutable and immutable callback collections
                current_lm.callbacks = list(callbacks) + [lm_logger]
                return True
            else:
                # Already registered
                return True

        return False

    except ImportError:
        return False
    except Exception:
        return False


def warn_lm_logger_not_registered() -> None:
    """Emit a warning that the LM logger could not be auto-registered."""
    warnings.warn(
        "Could not auto-register LM logger with DSPy. "
        "For LM call capture, pass callbacks=[server_observer.get_lm_logger()] "
        "to your dspy.LM() constructor.",
        UserWarning,
        stacklevel=3,
    )
