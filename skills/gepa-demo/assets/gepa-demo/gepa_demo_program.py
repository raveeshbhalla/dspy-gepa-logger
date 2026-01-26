"""DSPy Signature and program setup for GEPA demo."""

import dspy


# TODO: Define Signature
class TaskSig(dspy.Signature):
    """Short task description."""

    input_text: str = dspy.InputField(desc="...")
    output_text: str = dspy.OutputField(desc="...")


def build_program() -> dspy.Module:
    """Return the DSPy program to optimize."""
    return dspy.ChainOfThought(TaskSig)
