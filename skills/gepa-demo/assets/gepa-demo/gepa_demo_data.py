"""Data loading and example construction for GEPA demo."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

import dspy
import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(data_path, lines=(suffix == ".jsonl"))
    raise ValueError(f"Unsupported data format: {suffix}")


def build_examples(df: pd.DataFrame) -> list[dspy.Example]:
    """Map raw rows to dspy.Example objects.

    TODO: Update field names and inputs to match your dataset.
    """
    examples: list[dspy.Example] = []
    for _, row in df.iterrows():
        ex = dspy.Example(
            # TODO: map your fields
            input_text=row["input"],
            output_text=row["expected"],
            # metadata=row.get("metadata"),
        ).with_inputs("input_text")
        examples.append(ex)
    return examples


def split_examples(
    examples: Iterable[dspy.Example],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    examples = list(examples)
    rng = random.Random(seed)
    rng.shuffle(examples)

    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = examples[:n_train]
    val = examples[n_train : n_train + n_val]
    test = examples[n_train + n_val :]

    if n < 100:
        return train, val, []
    return train, val, test
