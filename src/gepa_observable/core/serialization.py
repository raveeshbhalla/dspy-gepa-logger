"""Serialization utilities for GEPA logging.

Handles conversion of DSPy objects (like Video, Image, etc.) and other
non-JSON-serializable types to JSON-safe formats for server transmission.
"""

import json
from typing import Any


def serialize_value(value: Any) -> Any:
    """Serialize a single value to JSON-safe format.

    Handles special DSPy types like Video, Image, Audio, and other
    non-JSON-serializable objects by converting them to string placeholders.

    Args:
        value: Any Python value to serialize

    Returns:
        A JSON-serializable representation of the value
    """
    if value is None:
        return None

    # Get class name for type checking (avoids import dependencies)
    class_name = getattr(value, "__class__", type(value)).__name__

    # Handle DSPy media types
    if class_name == "Video":
        if hasattr(value, "url") and value.url:
            return f"[Video: {value.url}]"
        return "[Video]"

    if class_name == "Image":
        if hasattr(value, "url") and value.url:
            return f"[Image: {value.url}]"
        return "[Image]"

    if class_name == "Audio":
        if hasattr(value, "url") and value.url:
            return f"[Audio: {value.url}]"
        return "[Audio]"

    # Handle bytes
    if isinstance(value, bytes):
        return f"[bytes: {len(value)} bytes]"

    # Handle common serializable types directly
    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]

    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}

    # Try JSON serialization to check if it's already serializable
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        # Fall back to string representation
        return str(value)


def serialize_output(output: Any) -> str | None:
    """Serialize an output object to a JSON string.

    For DSPy predictions, extracts the actual fields (reasoning, answer, etc.)
    instead of internal attributes like _completions.

    Args:
        output: The output object to serialize (typically a DSPy Prediction)

    Returns:
        A JSON string representation, or None if output is None
    """
    if output is None:
        return None

    try:
        if hasattr(output, "__dict__"):
            # For DSPy predictions, skip private fields
            fields = {
                k: serialize_value(v)
                for k, v in output.__dict__.items()
                if not k.startswith("_")
            }
            if fields:
                return json.dumps(fields, default=str)
            # Fallback to all fields if no public fields
            return json.dumps(
                {k: serialize_value(v) for k, v in output.__dict__.items()},
                default=str,
            )
        elif hasattr(output, "toDict") and callable(output.toDict):
            raw_dict = output.toDict()
            serialized = {k: serialize_value(v) for k, v in raw_dict.items()}
            return json.dumps(serialized, default=str)
        else:
            return json.dumps(serialize_value(output), default=str)
    except (TypeError, ValueError):
        return str(output)


def serialize_example_inputs(example: Any) -> dict[str, Any] | None:
    """Serialize example inputs to a JSON-safe dict for display.

    Handles non-serializable types like dspy.Video by converting to
    string placeholders.

    Args:
        example: A DSPy Example or similar object

    Returns:
        A dict of input field names to serialized values, or None on error
    """
    if example is None:
        return None

    try:
        # DSPy Example with .inputs() method - get only input fields
        if hasattr(example, "inputs") and callable(example.inputs):
            input_keys = example.inputs()
            raw_dict = {k: getattr(example, k, None) for k in input_keys}
            return {k: serialize_value(v) for k, v in raw_dict.items()}

        # Object with toDict method
        if hasattr(example, "toDict") and callable(example.toDict):
            raw_dict = example.toDict()
            return {k: serialize_value(v) for k, v in raw_dict.items()}

        # Generic object with __dict__
        if hasattr(example, "__dict__"):
            raw_dict = {
                k: v for k, v in example.__dict__.items() if not k.startswith("_")
            }
            return {k: serialize_value(v) for k, v in raw_dict.items()}

        # Fallback
        return {"value": str(example)}

    except Exception:
        return None
