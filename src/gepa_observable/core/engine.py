# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
# Modified for gepa-observable with observer support

import traceback
from contextlib import contextmanager
from typing import Any, Generic

from gepa.core.adapter import DataInst, EvaluatorFn, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState, ProgramIdx, initialize_gepa_state
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.merge import MergeProposer
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.utils import StopperProtocol

from gepa_observable.observers import (
    AcceptanceDecisionEvent,
    IterationStartEvent,
    MergeEvent,
    ObserverManager,
    OptimizationCompleteEvent,
    SeedValidationEvent,
    ValsetEvalEvent,
)

# Import local reflective mutation proposer
from gepa_observable.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Orchestrates the optimization loop using pluggable candidate proposers.

    Modified for gepa-observable with observer support and optional MLflow tracing.
    """

    def __init__(
        self,
        run_dir: str | None,
        evaluator: EvaluatorFn,
        valset: list[DataInst] | DataLoader[DataId, DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        # Logging
        logger: LoggerProtocol,
        experiment_tracker: ExperimentTracker,
        # Optional parameters
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        # Budget and Stop Condition
        stop_callback: StopperProtocol | None = None,
        val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | None = None,
        # GEPA Observable additions
        observer_manager: ObserverManager | None = None,
    ):
        self.logger = logger
        self.run_dir = run_dir

        # Graceful stopping mechanism
        self._stop_requested = False

        # Set up stopping mechanism
        self.stop_callback = stop_callback
        self.evaluator = evaluator
        self.valset = ensure_loader(valset) if valset is not None else None
        self.seed_candidate = seed_candidate

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar
        self.use_cloudpickle = use_cloudpickle

        self.raise_on_exception = raise_on_exception
        self.val_evaluation_policy: EvaluationPolicy[DataId, DataInst] = (
            val_evaluation_policy if val_evaluation_policy is not None else FullEvaluationPolicy()
        )

        # GEPA Observable additions
        self.observer_manager = observer_manager or ObserverManager()

    @contextmanager
    def _mlflow_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Context manager for optional MLflow tracing spans."""
        if not self.observer_manager.mlflow_tracing:
            yield None
            return

        try:
            import mlflow
            with mlflow.start_span(name=name) as span:
                if attributes:
                    span.set_attributes({f"gepa.{k}": v for k, v in attributes.items()})
                yield span
        except ImportError:
            yield None
        except Exception as e:
            import logging
            logging.warning(f"MLflow span creation failed: {e}")
            yield None

    def _evaluate_on_valset(
        self,
        program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
    ) -> tuple[dict[DataId, RolloutOutput], dict[DataId, float], dict[DataId, str | None] | None]:
        valset = self.valset
        assert valset is not None

        val_ids = self.val_evaluation_policy.get_eval_batch(valset, state)
        batch = valset.fetch(val_ids)
        outputs, scores, feedbacks = self.evaluator(batch, program)
        assert len(outputs) == len(val_ids), "Eval outputs should match length of selected validation indices"

        outputs_by_val_idx = dict(zip(val_ids, outputs, strict=False))
        scores_by_val_idx = dict(zip(val_ids, scores, strict=False))
        feedbacks_by_val_idx = dict(zip(val_ids, feedbacks, strict=False)) if feedbacks else None
        return outputs_by_val_idx, scores_by_val_idx, feedbacks_by_val_idx

    def _get_pareto_front_programs(self, state: GEPAState[RolloutOutput, DataId]) -> dict[DataId, set[ProgramIdx]]:
        return state.program_at_pareto_front_valset

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        # Wrap valset evaluation in MLflow span
        with self._mlflow_span("valset_eval", {"iteration": state.i, "phase": "valset_eval"}):
            valset_outputs, valset_subscores, valset_feedbacks = self._evaluate_on_valset(new_program, state)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )
        state.full_program_trace[-1]["new_program_idx"] = new_program_idx
        state.full_program_trace[-1]["evaluated_val_indices"] = sorted(valset_subscores.keys())

        valset_score = self.val_evaluation_policy.get_valset_score(new_program_idx, state)

        linear_pareto_front_program_idx = self.val_evaluation_policy.get_best_program(state)
        is_new_best = new_program_idx == linear_pareto_front_program_idx

        if is_new_best:
            self.logger.log(f"Iteration {state.i + 1}: Found a better program on the valset with score {valset_score}.")

        valset = self.valset
        assert valset is not None

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            valset_size=len(valset),
            val_evaluation_policy=self.val_evaluation_policy,
        )

        # Notify observers of valset evaluation
        self.observer_manager.notify_valset_eval(ValsetEvalEvent(
            iteration=state.i,
            candidate_idx=new_program_idx,
            candidate=new_program,
            val_ids=list(valset_subscores.keys()),
            scores=valset_subscores,
            outputs=valset_outputs,
            is_new_best=is_new_best,
            valset_score=valset_score,
            feedbacks=valset_feedbacks,
        ))

        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        # Check tqdm availability if progress bar is enabled
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError("tqdm must be installed when display_progress_bar is enabled")

            # Check if stop_callback contains MaxMetricCallsStopper
            total_calls: int | None = None
            stop_cb = self.stop_callback
            if stop_cb is not None:
                max_calls_attr = getattr(stop_cb, "max_metric_calls", None)
                if isinstance(max_calls_attr, int):
                    # Direct MaxMetricCallsStopper
                    total_calls = max_calls_attr
                else:
                    stoppers = getattr(stop_cb, "stoppers", None)
                    if stoppers is not None:
                        # CompositeStopper - iterate to find MaxMetricCallsStopper
                        for stopper in stoppers:
                            stopper_max = getattr(stopper, "max_metric_calls", None)
                            if isinstance(stopper_max, int):
                                total_calls = stopper_max
                                break

            if total_calls is not None:
                progress_bar = tqdm(total=total_calls, desc="GEPA Optimization", unit="rollouts")
            else:
                progress_bar = tqdm(desc="GEPA Optimization", unit="rollouts")
            progress_bar.update(0)

        # Prepare valset
        valset = self.valset
        if valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        # Store feedbacks from seed validation for observer notification
        seed_feedbacks_cache: dict[DataId, str | None] | None = None
        seed_outputs_cache: dict[DataId, RolloutOutput] = {}

        def valset_evaluator(program: dict[str, str]) -> tuple[dict[DataId, RolloutOutput], dict[DataId, float]]:
            nonlocal seed_feedbacks_cache, seed_outputs_cache
            all_ids = list(valset.all_ids())
            all_outputs, all_scores, all_feedbacks = self.evaluator(valset.fetch(all_ids), program)
            # Cache feedbacks and outputs for seed validation observer notification
            if seed_feedbacks_cache is None:
                seed_feedbacks_cache = dict(zip(all_ids, all_feedbacks, strict=False)) if all_feedbacks else None
                seed_outputs_cache = dict(zip(all_ids, all_outputs, strict=False))
            return (
                dict(zip(all_ids, all_outputs, strict=False)),
                dict(zip(all_ids, all_scores, strict=False)),
            )

        # Initialize state with seed validation
        with self._mlflow_span("seed_validation", {"phase": "seed_validation"}):
            state = initialize_gepa_state(
                run_dir=self.run_dir,
                logger=self.logger,
                seed_candidate=self.seed_candidate,
                valset_evaluator=valset_evaluator,
                track_best_outputs=self.track_best_outputs,
            )

        # Notify observers of seed validation
        seed_scores = state.prog_candidate_val_subscores[0]
        self.observer_manager.notify_seed_validation(SeedValidationEvent(
            seed_candidate=self.seed_candidate,
            valset_scores=seed_scores,
            valset_outputs=seed_outputs_cache,
            total_evals=state.total_num_evals,
            valset_feedbacks=seed_feedbacks_cache,
        ))

        # Log base program score
        base_val_avg, base_val_coverage = state.get_program_average_val_subset(0)
        self.experiment_tracker.log_metrics(
            {
                "base_program_full_valset_score": base_val_avg,
                "base_program_val_coverage": base_val_coverage,
                "iteration": state.i + 1,
            },
            step=state.i + 1,
        )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {base_val_avg} "
            f"over {base_val_coverage} / {len(valset)} examples"
        )

        # Merge scheduling
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        # Main loop
        last_pbar_val = 0
        while not self._should_stop(state):
            if self.display_progress_bar and progress_bar is not None:
                delta = state.total_num_evals - last_pbar_val
                progress_bar.update(delta)
                last_pbar_val = state.total_num_evals

            assert state.is_consistent()
            try:
                state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
                state.i += 1
                state.full_program_trace.append({"i": state.i})

                # Wrap iteration in MLflow span
                with self._mlflow_span(f"iteration_{state.i}", {"iteration": state.i}):
                    # 1) Attempt merge first if scheduled and last iter found new program
                    if self.merge_proposer is not None and self.merge_proposer.use_merge:
                        if self.merge_proposer.merges_due > 0 and self.merge_proposer.last_iter_found_new_program:
                            proposal = self.merge_proposer.propose(state)
                            self.merge_proposer.last_iter_found_new_program = False  # old behavior

                            if proposal is not None and proposal.tag == "merge":
                                parent_sums = proposal.subsample_scores_before or [float("-inf"), float("-inf")]
                                new_after = proposal.subsample_scores_after
                                new_sum = sum(new_after) if new_after else float("-inf")
                                accepted = new_sum >= max(parent_sums)

                                # Notify observers of merge attempt
                                self.observer_manager.notify_merge(MergeEvent(
                                    iteration=state.i,
                                    parent_candidate_ids=proposal.parent_program_ids,
                                    merged_candidate=proposal.candidate,
                                    subsample_scores_before=proposal.subsample_scores_before,
                                    subsample_scores_after=proposal.subsample_scores_after,
                                    accepted=accepted,
                                ))

                                if accepted:
                                    # ACCEPTED: consume one merge attempt and record it
                                    self._run_full_eval_and_add(
                                        new_program=proposal.candidate,
                                        state=state,
                                        parent_program_idx=proposal.parent_program_ids,
                                    )
                                    self.merge_proposer.merges_due -= 1
                                    self.merge_proposer.total_merges_tested += 1
                                    continue  # skip reflective this iteration
                                else:
                                    # REJECTED: do NOT consume merges_due or total_merges_tested
                                    self.logger.log(
                                        f"Iteration {state.i + 1}: New program subsample score {new_sum} "
                                        f"is worse than both parents {parent_sums}, skipping merge"
                                    )
                                    # Skip reflective this iteration (old behavior)
                                    continue

                        # Old behavior: regardless of whether we attempted, clear the flag before reflective
                        self.merge_proposer.last_iter_found_new_program = False

                    # 2) Reflective mutation proposer
                    # Note: The proposer now handles its own observer notifications for minibatch eval and reflection
                    proposal = self.reflective_proposer.propose(state)
                    if proposal is None:
                        self.logger.log(f"Iteration {state.i + 1}: Reflective mutation did not propose a new candidate")
                        continue

                    # Acceptance: require strict improvement on subsample
                    old_sum = sum(proposal.subsample_scores_before or [])
                    new_sum = sum(proposal.subsample_scores_after or [])
                    accepted = new_sum > old_sum
                    proceed_to_valset = accepted

                    # Notify observers of acceptance decision
                    self.observer_manager.notify_acceptance_decision(AcceptanceDecisionEvent(
                        iteration=state.i,
                        parent_score_sum=old_sum,
                        new_score_sum=new_sum,
                        accepted=accepted,
                        proceed_to_valset=proceed_to_valset,
                    ))

                    if not accepted:
                        self.logger.log(
                            f"Iteration {state.i + 1}: New subsample score {new_sum} is not better than old score {old_sum}, skipping"
                        )
                        continue
                    else:
                        self.logger.log(
                            f"Iteration {state.i + 1}: New subsample score {new_sum} is better than old score {old_sum}. Continue to full eval and add to candidate pool."
                        )

                    # Accept: full eval + add
                    self._run_full_eval_and_add(
                        new_program=proposal.candidate,
                        state=state,
                        parent_program_idx=proposal.parent_program_ids,
                    )

                    # Schedule merge attempts like original behavior
                    if self.merge_proposer is not None:
                        self.merge_proposer.last_iter_found_new_program = True
                        if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
                            self.merge_proposer.merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {state.i + 1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                if self.raise_on_exception:
                    raise e
                else:
                    continue

        # Close progress bar if it exists
        if self.display_progress_bar and progress_bar is not None:
            progress_bar.close()

        state.save(self.run_dir)

        # Notify observers of optimization complete
        best_idx = max(range(len(state.program_full_scores_val_set)),
                       key=lambda i: state.program_full_scores_val_set[i])
        self.observer_manager.notify_optimization_complete(OptimizationCompleteEvent(
            total_iterations=state.i + 1,
            total_evals=state.total_num_evals,
            best_candidate_idx=best_idx,
            best_score=state.program_full_scores_val_set[best_idx],
            best_candidate=state.program_candidates[best_idx],
        ))

        return state

    def _should_stop(self, state: GEPAState[RolloutOutput, DataId]) -> bool:
        """Check if the optimization should stop."""
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback(state):
            return True
        return False

    def request_stop(self) -> None:
        """Manually request the optimization to stop gracefully."""
        self.logger.log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
