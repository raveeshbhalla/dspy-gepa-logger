# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
# Modified for gepa-observable with observer support

import os
import random
from collections.abc import Sequence
from typing import Any, Literal, cast

# Use local imports for modified modules, gepa imports for unchanged ones
from gepa.adapters.default_adapter.default_adapter import ChatCompletionCallable, DefaultAdapter
from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import LoggerProtocol, StdOutLogger
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.base import CandidateSelector, LanguageModel, ReflectionComponentSelector
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
)
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.utils import FileStopper, StopperProtocol
from gepa.core.result import GEPAResult

# Local imports for modified modules
from gepa_observable.core.engine import GEPAEngine
from gepa_observable.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa_observable.observers import GEPAObserver, ObserverManager


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[DataInst] | DataLoader[DataId, DataInst],
    valset: list[DataInst] | DataLoader[DataId, DataInst] | None = None,
    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None,
    task_lm: str | ChatCompletionCallable | None = None,
    # Reflection-based configuration
    reflection_lm: LanguageModel | str | None = None,
    candidate_selection_strategy: CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"] = "pareto",
    skip_perfect_score: bool = True,
    batch_sampler: BatchSampler | Literal["epoch_shuffled"] = "epoch_shuffled",
    reflection_minibatch_size: int | None = None,
    perfect_score: float = 1.0,
    reflection_prompt_template: str | None = None,
    # Component selection configuration
    module_selector: ReflectionComponentSelector | str = "round_robin",
    # Merge-based configuration
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    # Budget and Stop Condition
    max_metric_calls: int | None = None,
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None,
    # Logging
    logger: LoggerProtocol | None = None,
    run_dir: str | None = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    track_best_outputs: bool = False,
    display_progress_bar: bool = False,
    use_cloudpickle: bool = False,
    # Reproducibility
    seed: int = 0,
    raise_on_exception: bool = True,
    val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | Literal["full_eval"] | None = None,
    # GEPA Observable additions
    observers: list[GEPAObserver] | None = None,
    mlflow_tracing: bool = False,
) -> GEPAResult[RolloutOutput, DataId]:
    """
    GEPA is an evolutionary optimizer that evolves (multiple) text components of a complex system to optimize them towards a given metric.

    This is the gepa-observable fork with first-class observer support for full optimization observability.

    ## Observer Support (gepa-observable addition)

    - observers: List of GEPAObserver instances to receive callbacks during optimization.
      Each observer can implement any subset of:
        - on_seed_validation: Called after seed candidate evaluation
        - on_iteration_start: Called at start of each iteration
        - on_minibatch_eval: Called after mini-batch evaluations
        - on_reflection: Called after reflection produces new texts
        - on_acceptance_decision: Called after acceptance decision
        - on_valset_eval: Called after validation set evaluation
        - on_merge: Called when merge operations are attempted
        - on_optimization_complete: Called when optimization finishes

    - mlflow_tracing: If True and use_mlflow=True, wraps optimization phases in MLflow spans
      for hierarchical tracing of LM calls with GEPA context attributes.

    ## Original GEPA Parameters

    Concepts:
    - System: A harness that uses text components to perform a task.
    - Candidate: A mapping from component names to component text.
    - DataInst: An (uninterpreted) data type over which the system operates.
    - RolloutOutput: The output of the system on a DataInst.

    Parameters:
    - seed_candidate: The initial candidate to start with.
    - trainset: Training data for reflective updates.
    - valset: Validation data for tracking Pareto scores.
    - adapter: A GEPAAdapter instance for system evaluation.
    - task_lm: Model for task execution (if no adapter provided).
    - reflection_lm: Language model for reflection.
    - candidate_selection_strategy: Strategy for selecting candidates ('pareto', 'current_best', 'epsilon_greedy').
    - skip_perfect_score: Skip updating if perfect score achieved.
    - batch_sampler: Strategy for selecting training examples.
    - reflection_minibatch_size: Examples per reflection step.
    - perfect_score: The perfect score value.
    - reflection_prompt_template: Custom reflection prompt.
    - module_selector: Component selection strategy ('round_robin', 'all').
    - use_merge: Enable merge-based optimization.
    - max_merge_invocations: Maximum merge attempts.
    - merge_val_overlap_floor: Minimum validation overlap for merge.
    - max_metric_calls: Maximum metric calls budget.
    - stop_callbacks: Custom stopping conditions.
    - logger: Logger instance.
    - run_dir: Directory for saving state.
    - use_wandb: Enable Weights & Biases logging.
    - use_mlflow: Enable MLflow logging.
    - track_best_outputs: Track best outputs per validation task.
    - display_progress_bar: Show tqdm progress bar.
    - use_cloudpickle: Use cloudpickle for serialization.
    - seed: Random seed.
    - raise_on_exception: Propagate exceptions vs graceful stop.
    - val_evaluation_policy: Validation evaluation strategy.
    """
    if adapter is None:
        assert task_lm is not None, (
            "Since no adapter is provided, GEPA requires a task LM to be provided. Please set the `task_lm` parameter."
        )
        active_adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] = cast(
            GEPAAdapter[DataInst, Trajectory, RolloutOutput], DefaultAdapter(model=task_lm)
        )
    else:
        assert task_lm is None, (
            "Since an adapter is provided, GEPA does not require a task LM to be provided. Please set the `task_lm` parameter to None."
        )
        active_adapter = adapter

    # Normalize datasets to DataLoader instances
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset) if valset is not None else train_loader

    # Comprehensive stop_callback logic
    stop_callbacks_list: list[StopperProtocol] = []
    if stop_callbacks is not None:
        if isinstance(stop_callbacks, Sequence):
            stop_callbacks_list.extend(stop_callbacks)
        else:
            stop_callbacks_list.append(stop_callbacks)

    # Add file stopper if run_dir is provided
    if run_dir is not None:
        stop_file_path = os.path.join(run_dir, "gepa.stop")
        file_stopper = FileStopper(stop_file_path)
        stop_callbacks_list.append(file_stopper)

    # Add max_metric_calls stopper if provided
    if max_metric_calls is not None:
        from gepa.utils import MaxMetricCallsStopper

        max_calls_stopper = MaxMetricCallsStopper(max_metric_calls)
        stop_callbacks_list.append(max_calls_stopper)

    # Assert that at least one stopping condition is provided
    if not stop_callbacks_list:
        raise ValueError(
            "The user must provide at least one of stop_callbacks or max_metric_calls to specify a stopping condition."
        )

    # Create composite stopper if multiple stoppers, or use single stopper
    stop_callback: StopperProtocol
    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        from gepa.utils import CompositeStopper

        stop_callback = CompositeStopper(*stop_callbacks_list)

    if not hasattr(active_adapter, "propose_new_texts"):
        assert reflection_lm is not None, (
            f"reflection_lm was not provided. The adapter used '{active_adapter!s}' does not provide a propose_new_texts method, "
            + "and hence, GEPA will use the default proposer, which requires a reflection_lm to be specified."
        )

    if isinstance(reflection_lm, str):
        import litellm

        reflection_lm_name = reflection_lm

        def _reflection_lm(prompt: str) -> str:
            completion = litellm.completion(model=reflection_lm_name, messages=[{"role": "user", "content": prompt}])
            return completion.choices[0].message.content  # type: ignore

        reflection_lm = _reflection_lm

    if logger is None:
        logger = StdOutLogger()

    rng = random.Random(seed)

    candidate_selector: CandidateSelector
    if isinstance(candidate_selection_strategy, str):
        factories = {
            "pareto": lambda: ParetoCandidateSelector(rng=rng),
            "current_best": lambda: CurrentBestCandidateSelector(),
            "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng),
        }

        try:
            candidate_selector = factories[candidate_selection_strategy]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown candidate_selector strategy: {candidate_selection_strategy}. "
                "Supported strategies: 'pareto', 'current_best', 'epsilon_greedy'"
            ) from exc
    elif isinstance(candidate_selection_strategy, CandidateSelector):
        candidate_selector = candidate_selection_strategy
    else:
        raise TypeError(
            "candidate_selection_strategy must be a supported string strategy or an instance of CandidateSelector."
        )

    if val_evaluation_policy is None or val_evaluation_policy == "full_eval":
        val_evaluation_policy = FullEvaluationPolicy()
    elif not isinstance(val_evaluation_policy, EvaluationPolicy):
        raise ValueError(
            f"val_evaluation_policy should be one of 'full_eval' or an instance of EvaluationPolicy, but got {type(val_evaluation_policy)}"
        )

    if isinstance(module_selector, str):
        module_selector_cls = {
            "round_robin": RoundRobinReflectionComponentSelector,
            "all": AllReflectionComponentSelector,
        }.get(module_selector)

        assert module_selector_cls is not None, (
            f"Unknown module_selector strategy: {module_selector}. Supported strategies: 'round_robin', 'all'"
        )

        module_selector_instance: ReflectionComponentSelector = module_selector_cls()
    else:
        module_selector_instance = module_selector

    if batch_sampler == "epoch_shuffled":
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=reflection_minibatch_size or 3, rng=rng)
    else:
        assert reflection_minibatch_size is None, (
            "reflection_minibatch_size only accepted if batch_sampler is 'epoch_shuffled'"
        )

    experiment_tracker = create_experiment_tracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    if reflection_prompt_template is not None:
        assert not (adapter is not None and getattr(adapter, "propose_new_texts", None) is not None), (
            f"Adapter {adapter!s} provides its own propose_new_texts method; reflection_prompt_template will be ignored. "
            "Set reflection_prompt_template to None."
        )

    # Create observer manager
    observer_manager = ObserverManager(
        observers=observers or [],
        mlflow_tracing=mlflow_tracing,
    )

    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=train_loader,
        adapter=active_adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=batch_sampler,
        perfect_score=perfect_score,
        skip_perfect_score=skip_perfect_score,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm,
        reflection_prompt_template=reflection_prompt_template,
        observer_manager=observer_manager,  # Pass observer manager
    )

    def evaluator(inputs: list[DataInst], prog: dict[str, str]) -> tuple[list[RolloutOutput], list[float], list[str | None] | None]:
        eval_out = active_adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.feedbacks

    merge_proposer: MergeProposer | None = None
    if use_merge:
        merge_proposer = MergeProposer(
            logger=logger,
            valset=val_loader,
            evaluator=evaluator,
            use_merge=use_merge,
            max_merge_invocations=max_merge_invocations,
            rng=rng,
            val_overlap_floor=merge_val_overlap_floor,
        )

    engine = GEPAEngine(
        run_dir=run_dir,
        evaluator=evaluator,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=perfect_score,
        seed=seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        logger=logger,
        experiment_tracker=experiment_tracker,
        track_best_outputs=track_best_outputs,
        display_progress_bar=display_progress_bar,
        raise_on_exception=raise_on_exception,
        stop_callback=stop_callback,
        val_evaluation_policy=val_evaluation_policy,
        use_cloudpickle=use_cloudpickle,
        observer_manager=observer_manager,  # Pass observer manager
    )

    with experiment_tracker:
        state = engine.run()

    return GEPAResult.from_state(state, run_dir=run_dir, seed=seed)
