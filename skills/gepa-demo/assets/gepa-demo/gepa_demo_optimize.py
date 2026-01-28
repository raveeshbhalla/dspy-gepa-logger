"""Optimization runner for GEPA demo."""

from __future__ import annotations

import dspy

from gepa_observable import GEPA


def run_gepa(
    program: dspy.Module,
    metric_fn,
    trainset,
    valset,
    reflection_lm: dspy.LM,
    auto: str | None,
    max_full_evals: int | None,
    max_metric_calls: int | None,
    server_url: str | None,
    project_name: str,
    log_dir: str | None,
):
    optimizer = GEPA(
        metric=metric_fn,
        auto=auto,
        max_full_evals=max_full_evals,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        server_url=server_url,
        project_name=project_name,
        verbose=True,
    )

    return optimizer.compile(student=program, trainset=trainset, valset=valset)
