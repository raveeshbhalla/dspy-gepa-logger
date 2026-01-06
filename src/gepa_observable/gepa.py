# Copyright (c) 2025 - GEPA Observable Fork
# DSPy-compatible GEPA Teleprompter with full observability support

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal

from dspy.teleprompt import Teleprompter

# DSPy GEPA auto run settings - number of candidates for each preset
AUTO_RUN_SETTINGS = {
    "light": 6,
    "medium": 12,
    "heavy": 18,
}

from gepa_observable.observers import GEPAObserver, LoggingObserver, ServerObserver

if TYPE_CHECKING:
    import dspy
    from dspy import Example, LM, Module


# Type alias for GEPA feedback metric
GEPAFeedbackMetric = Callable[
    [Any, Any, Any, str | None, Any | None],  # gold, pred, trace, pred_name, pred_trace
    Any  # Returns Prediction with score and feedback, or float
]


class GEPA(Teleprompter):
    """Observable GEPA optimizer with DSPy-compatible API.

    This is a reflective evolutionary optimizer that evolves prompt instructions
    to improve performance on a given metric. It's designed to be a drop-in
    replacement for dspy.GEPA with added observability features for the
    GEPA web dashboard.

    Usage:
        >>> from gepa_observable import GEPA
        >>> optimizer = GEPA(
        ...     metric=my_metric,
        ...     auto="medium",  # or max_full_evals=10, or max_metric_calls=500
        ...     server_url="http://localhost:3000",  # Enable dashboard
        ... )
        >>> optimized = optimizer.compile(student=program, trainset=train, valset=val)

    The metric function must accept 5 arguments:
        - gold: The ground truth example
        - pred: The model prediction
        - trace: Execution trace (may be None)
        - pred_name: Name of the predictor (for predictor-level feedback)
        - pred_trace: Predictor-specific trace

    And return either:
        - A float score
        - A dspy.Prediction with 'score' and optionally 'feedback' fields

    Budget Control (exactly one required):
        - auto: "light" (6 candidates), "medium" (12), or "heavy" (18) preset
        - max_full_evals: Maximum full validation evaluations
        - max_metric_calls: Maximum total metric calls

    Observable Features:
        - server_url: URL of GEPA web dashboard for real-time monitoring
        - observers: List of GEPAObserver instances for custom callbacks
        - capture_lm_calls: Capture all LM calls during optimization
        - capture_stdout: Capture stdout to dashboard
    """

    def __init__(
        self,
        metric: GEPAFeedbackMetric,
        *,
        # Budget (exactly one required)
        auto: Literal["light", "medium", "heavy"] | None = None,
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        # DSPy GEPA params
        reflection_lm: "LM | str | None" = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        skip_perfect_score: bool = True,
        add_format_failure_as_feedback: bool = False,
        component_selector: Literal["round_robin", "all"] | Any = "round_robin",
        use_merge: bool = True,
        max_merge_invocations: int = 5,
        num_threads: int | None = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        log_dir: str | None = None,
        track_stats: bool = False,
        track_best_outputs: bool = False,
        warn_on_score_mismatch: bool = True,
        enable_tool_optimization: bool = False,
        seed: int = 0,
        # Logging
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
        # Observable additions
        observers: list[GEPAObserver] | None = None,
        server_url: str | None = None,
        project_name: str = "GEPA Run",
        run_name: str | None = None,
        verbose: bool = True,
        capture_lm_calls: bool = True,
        capture_stdout: bool = True,
        mlflow_tracing: bool = False,
        # Passthrough for advanced GEPA options
        gepa_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the Observable GEPA optimizer.

        Args:
            metric: Evaluation metric function (5-argument GEPA feedback metric)

            Budget (exactly one required):
                auto: Preset budget - "light" (6 candidates), "medium" (12), or "heavy" (18)
                max_full_evals: Maximum number of full validation evaluations
                max_metric_calls: Maximum total metric calls

            Reflection Configuration:
                reflection_lm: Language model for reflection (recommend strong model)
                reflection_minibatch_size: Examples per reflection step (default: 3)
                candidate_selection_strategy: "pareto" or "current_best"
                skip_perfect_score: Skip reflection if perfect score achieved
                add_format_failure_as_feedback: Add format failures as feedback
                component_selector: "round_robin" or "all"

            Merge Configuration:
                use_merge: Enable merge-based optimization (default: True)
                max_merge_invocations: Maximum merge attempts (default: 5)

            Execution:
                num_threads: Threads for parallel evaluation
                failure_score: Score for failed evaluations (default: 0.0)
                perfect_score: Perfect score value (default: 1.0)
                seed: Random seed for reproducibility

            Logging:
                log_dir: Directory for saving state (alias: run_dir)
                track_stats: Track optimization statistics
                track_best_outputs: Track best outputs per validation task
                warn_on_score_mismatch: Warn on score/feedback mismatch
                use_wandb: Enable Weights & Biases logging
                use_mlflow: Enable MLflow logging

            Observable Features:
                observers: Custom GEPAObserver instances
                server_url: GEPA web dashboard URL
                project_name: Project name for dashboard
                run_name: Run name for dashboard
                verbose: Enable console logging
                capture_lm_calls: Capture LM calls to dashboard
                capture_stdout: Capture stdout to dashboard
                mlflow_tracing: Enable MLflow tracing spans

            Advanced:
                enable_tool_optimization: Jointly optimize ReAct tool descriptions
                gepa_kwargs: Additional kwargs passed to gepa.optimize
        """
        # Validate budget options
        budget_options = [auto, max_full_evals, max_metric_calls]
        budget_set = sum(1 for opt in budget_options if opt is not None)
        if budget_set == 0:
            raise ValueError(
                "Exactly one of 'auto', 'max_full_evals', or 'max_metric_calls' must be set."
            )
        if budget_set > 1:
            raise ValueError(
                "Only one of 'auto', 'max_full_evals', or 'max_metric_calls' can be set, "
                f"but got: auto={auto}, max_full_evals={max_full_evals}, max_metric_calls={max_metric_calls}"
            )

        # Store configuration
        self.metric = metric
        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls

        # Reflection config
        self.reflection_lm = reflection_lm
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        self.skip_perfect_score = skip_perfect_score
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.component_selector = component_selector

        # Merge config
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations

        # Execution config
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.seed = seed

        # Logging config (log_dir is an alias for run_dir)
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.track_best_outputs = track_best_outputs
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.enable_tool_optimization = enable_tool_optimization

        # External logging
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs
        self.use_mlflow = use_mlflow
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name

        # Observable config
        self.observers = observers or []
        self.server_url = server_url
        self.project_name = project_name
        self.run_name = run_name
        self.verbose = verbose
        self.capture_lm_calls = capture_lm_calls
        self.capture_stdout = capture_stdout
        self.mlflow_tracing = mlflow_tracing

        # Advanced passthrough
        self.gepa_kwargs = gepa_kwargs or {}

        # Track stats requirement
        if track_best_outputs and not track_stats:
            raise ValueError("track_best_outputs requires track_stats=True")

    def _build_seed_candidate(self, student: "Module") -> dict[str, str]:
        """Build seed candidate dict from student module's predictor instructions.

        This extracts the current instruction text from each predictor in the
        student module to form the initial candidate for optimization.

        Args:
            student: The DSPy module to extract instructions from

        Returns:
            Dictionary mapping predictor names to their instruction text
        """
        seed_candidate: dict[str, str] = {}

        # Handle ReAct modules if tool optimization is enabled
        if self.enable_tool_optimization:
            import json
            from dspy import ReAct

            TOOL_MODULE_PREFIX = "tool_module"

            # Track which predictors are claimed by ReAct modules
            claimed_predictors: set[str] = set()

            for name, module in student.named_sub_modules():
                if isinstance(module, ReAct):
                    # Get the react and extract predictors
                    react_predictor = getattr(module, "react", None)
                    extract_predictor = getattr(module, "extract", None)

                    if react_predictor is not None and extract_predictor is not None:
                        # Get predictor names
                        react_name = f"{name}.react"
                        extract_name = f"{name}.extract"
                        claimed_predictors.add(react_name)
                        claimed_predictors.add(extract_name)

                        # Build tool config
                        tool_config = {
                            "react_instructions": str(react_predictor.signature.instructions),
                            "extract_instructions": str(extract_predictor.signature.instructions),
                            "tools": [],
                        }

                        # Add tool descriptions
                        for tool in getattr(module, "tools", []):
                            tool_info = {
                                "name": tool.name,
                                "desc": tool.desc,
                                "args": {},
                            }
                            for arg_name, arg_field in tool.args.items():
                                # json_schema_extra may be None if no extra metadata was provided
                                extra = arg_field.json_schema_extra
                                tool_info["args"][arg_name] = (
                                    extra.get("desc", "") if extra else ""
                                )
                            tool_config["tools"].append(tool_info)

                        seed_candidate[f"{TOOL_MODULE_PREFIX}:{extract_name}"] = json.dumps(
                            tool_config
                        )

            # Add remaining predictors not claimed by ReAct
            for pred_name, predictor in student.named_predictors():
                if pred_name not in claimed_predictors:
                    seed_candidate[pred_name] = str(predictor.signature.instructions)
        else:
            # Simple case: just extract predictor instructions
            for pred_name, predictor in student.named_predictors():
                seed_candidate[pred_name] = str(predictor.signature.instructions)

        return seed_candidate

    def auto_budget(
        self,
        num_preds: int,
        valset_size: int,
        minibatch_size: int = 25,
        full_eval_steps: int = 5,
    ) -> int:
        """Calculate max_metric_calls from auto preset using DSPy's exact formula.

        This matches DSPy's GEPA.auto_budget() implementation. The budget accounts for:
        1. Initial full evaluation on the default program
        2. Up to 5 bootstrap trials per candidate
        3. Minibatch evaluations during optimization
        4. Periodic full evaluations

        Auto presets determine num_candidates (n):
            - light: n=6 candidates
            - medium: n=12 candidates
            - heavy: n=18 candidates

        The num_trials is calculated as:
            num_trials = max(2 * num_vars * log2(num_candidates), 1.5 * num_candidates)

        Where num_vars = num_preds * 2 (for both instruction and few-shot variables).

        Args:
            num_preds: Number of predictors in the program
            valset_size: Size of the validation set
            minibatch_size: Size of minibatch for evaluations (default: 25)
            full_eval_steps: Steps between full evaluations (default: 5)

        Returns:
            Calculated max_metric_calls budget
        """
        if self.auto is None:
            raise ValueError("auto_budget requires auto preset to be set")

        num_candidates = AUTO_RUN_SETTINGS.get(self.auto, 12)

        # DSPy's formula for num_trials (matches MIPROv2._set_num_trials_from_num_candidates)
        # num_vars accounts for both instruction and few-shot example variables
        num_vars = num_preds * 2
        num_trials = int(
            max(2 * num_vars * math.log2(num_candidates), 1.5 * num_candidates)
        )

        # Budget components (matching DSPy's GEPA.auto_budget):
        # 1. Initial full evaluation on default program: V (valset_size)
        # 2. Up to 5 trials for bootstrapping each candidate: num_candidates * 5
        # 3. N minibatch evaluations: num_trials * minibatch_size
        # 4. Periodic full evaluations: (num_trials + 1) // full_eval_steps + 1
        # 5. Extra final full eval if num_trials < full_eval_steps: 1

        budget = valset_size  # Initial full eval
        budget += num_candidates * 5  # Bootstrap trials
        budget += num_trials * minibatch_size  # Minibatch evals
        budget += ((num_trials + 1) // full_eval_steps + 1) * valset_size  # Periodic full evals

        if num_trials < full_eval_steps:
            budget += valset_size  # Extra final full eval

        return budget

    def compile(
        self,
        student: "Module",
        *,
        trainset: list["Example"],
        teacher: "Module | None" = None,
        valset: list["Example"] | None = None,
    ) -> "Module":
        """Compile (optimize) the student module using GEPA.

        Args:
            student: The DSPy module to optimize
            trainset: Training examples for reflective updates
            teacher: Not supported in GEPA (included for API compatibility)
            valset: Validation examples (uses trainset if None)

        Returns:
            The optimized DSPy module. If track_stats=True, the returned
            module will have a 'detailed_results' attribute with optimization
            metadata.
        """
        if teacher is not None:
            warnings.warn(
                "GEPA does not support the 'teacher' parameter. It will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Import here to avoid circular imports
        import random

        import dspy
        # Use our forked DspyAdapter which has feedbacks support in EvaluationBatch
        from gepa_observable.adapters.dspy_adapter.dspy_adapter import DspyAdapter

        from gepa_observable.api import optimize

        # Validate trainset
        if not trainset:
            raise ValueError("trainset must be provided and non-empty")

        valset = valset or trainset

        # Get predictors from student module
        predictors = list(student.named_predictors())
        num_predictors = len(predictors)

        # Calculate budget if using auto preset
        effective_max_metric_calls = self.max_metric_calls
        if self.auto is not None:
            # Use DSPy's exact auto_budget formula
            effective_max_metric_calls = self.auto_budget(
                num_preds=num_predictors,
                valset_size=len(valset),
            )

        elif self.max_full_evals is not None:
            # Convert max_full_evals to max_metric_calls
            # Each full eval = len(valset) metric calls
            effective_max_metric_calls = self.max_full_evals * len(valset)

        # Create feedback map for predictors using feedback_fn_creator pattern
        # This wraps the user's metric to translate PredictorFeedbackFn signature
        # to the standard GEPA metric signature (gold, pred, trace, pred_name, pred_trace)
        def feedback_fn_creator(pred_name: str, predictor) -> Callable:
            """Create a feedback function that wraps the user's metric."""
            def feedback_fn(predictor_output, predictor_inputs, module_inputs,
                            module_outputs, captured_trace):
                # Build pred_trace for this specific predictor
                # Shape: [(predictor, predictor_inputs, predictor_output)]
                pred_trace = [(predictor, predictor_inputs, predictor_output)]

                # Call metric with standard GEPA signature
                result = self.metric(
                    module_inputs,      # gold (Example)
                    module_outputs,     # pred (Prediction)
                    captured_trace,     # trace (full execution trace)
                    pred_name,          # pred_name (predictor name)
                    pred_trace,         # pred_trace (predictor-specific trace)
                )

                # Normalize to ScoreWithFeedback format (dict with score and feedback)
                if hasattr(result, 'score'):
                    return {'score': result.score, 'feedback': getattr(result, 'feedback', '')}
                else:
                    # Metric returned just a float score
                    return {'score': result, 'feedback': ''}

            return feedback_fn

        feedback_map = {name: feedback_fn_creator(name, pred) for name, pred in predictors}

        # Create RNG with seed
        rng = random.Random(self.seed)

        # Resolve reflection_lm - handle string model names by converting to dspy.LM
        # This must happen before creating the adapter so it gets the resolved LM
        resolved_reflection_lm: dspy.LM | None = None
        if isinstance(self.reflection_lm, str):
            # Convert string model name to dspy.LM instance
            resolved_reflection_lm = dspy.LM(self.reflection_lm)
        elif isinstance(self.reflection_lm, dspy.LM):
            resolved_reflection_lm = self.reflection_lm
        elif self.reflection_lm is None and getattr(dspy.settings, "lm", None) is not None:
            # Use configured LM as fallback
            resolved_reflection_lm = dspy.settings.lm

        # Create the DspyAdapter with the resolved LM
        adapter = DspyAdapter(
            student_module=student,
            metric_fn=self.metric,
            feedback_map=feedback_map,
            failure_score=self.failure_score,
            num_threads=self.num_threads,
            add_format_failure_as_feedback=self.add_format_failure_as_feedback,
            rng=rng,
            reflection_lm=resolved_reflection_lm,
            custom_instruction_proposer=None,  # Could add instruction_proposer param later
            warn_on_score_mismatch=self.warn_on_score_mismatch,
            enable_tool_optimization=self.enable_tool_optimization,
            reflection_minibatch_size=self.reflection_minibatch_size,
        )

        # Build seed candidate from predictor instructions
        seed_candidate = self._build_seed_candidate(student)

        # Merge gepa_kwargs with explicit params
        # Explicit params take precedence
        merged_kwargs = dict(self.gepa_kwargs)

        # Call our observable optimize function
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            task_lm=None,  # Adapter handles this
            reflection_lm=resolved_reflection_lm,
            candidate_selection_strategy=self.candidate_selection_strategy,
            skip_perfect_score=self.skip_perfect_score,
            reflection_minibatch_size=self.reflection_minibatch_size,
            perfect_score=self.perfect_score,
            module_selector=self.component_selector,
            use_merge=self.use_merge,
            max_merge_invocations=self.max_merge_invocations,
            max_metric_calls=effective_max_metric_calls,
            run_dir=self.log_dir,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            wandb_init_kwargs=self.wandb_init_kwargs,
            use_mlflow=self.use_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            mlflow_experiment_name=self.mlflow_experiment_name,
            track_best_outputs=self.track_best_outputs,
            seed=self.seed,
            # Observable params
            observers=self.observers,
            mlflow_tracing=self.mlflow_tracing,
            server_url=self.server_url,
            project_name=self.project_name,
            run_name=self.run_name,
            verbose=self.verbose,
            capture_lm_calls=self.capture_lm_calls,
            capture_stdout=self.capture_stdout,
            # Pass through any additional gepa_kwargs
            **merged_kwargs,
        )

        # Build the optimized program
        optimized = adapter.build_program(result.best_candidate)

        # Attach detailed results if tracking stats
        if self.track_stats:
            optimized.detailed_results = {
                "best_candidate": result.best_candidate,
                "best_score": result.best_score,
                "candidates": getattr(result, "candidates", None),
                "scores": getattr(result, "scores", None),
                "seed_candidate": seed_candidate,
                "total_metric_calls": getattr(result, "total_metric_calls", None),
            }

        return optimized
