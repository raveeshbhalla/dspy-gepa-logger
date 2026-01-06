"""
This file provides an example adapter allowing GEPA to optimize text components of DSPy programs (instructions and prompts).
The most up-to-date version of this file is in the DSPy repository: https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa_utils.py
"""

import logging
import random
from typing import Any, Callable, Protocol

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import TraceData

# Use our forked EvaluationBatch which has the 'feedbacks' field
from gepa_observable.core.adapter import EvaluationBatch, GEPAAdapter

logger = logging.getLogger(__name__)


class LoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log(self, x: str):
        self.logger.info(x)


DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]


class ScoreWithFeedback(Prediction):
    score: float
    feedback: str


class PredictorFeedbackFn(Protocol):
    def __call__(
        self,
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Example,
        module_outputs: Prediction,
        captured_trace: DSPyTrace,
    ) -> ScoreWithFeedback:
        """
        This function is used to provide feedback to a specific predictor.
        The function is called with the following arguments:
        - predictor_output: The output of the predictor.
        - predictor_inputs: The inputs to the predictor.
        - module_inputs: The inputs to the whole program --- `Example`.
        - module_outputs: The outputs of the whole program --- `Prediction`.
        - captured_trace: The trace of the module's execution.
        # Shape of trace is: [predictor_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)

        The function should return a `ScoreWithFeedback` object.
        The feedback is a string that is used to guide the evolution of the predictor.
        """
        ...


class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    """DSPy adapter with feedback extraction support.

    This is a fork of DSPy's DspyAdapter that uses gepa_observable's EvaluationBatch
    which includes a 'feedbacks' field for capturing per-example feedback from metrics.
    """

    def __init__(
        self,
        student_module,
        metric_fn: Callable,
        feedback_map: dict[str, Callable],
        failure_score=0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
        # Additional DSPy parameters for API compatibility
        reflection_lm=None,
        custom_instruction_proposer=None,
        warn_on_score_mismatch: bool = True,
        enable_tool_optimization: bool = False,
        reflection_minibatch_size: int | None = None,
    ):
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)

        # Additional DSPy parameters (stored for API compatibility)
        self.reflection_lm = reflection_lm
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.enable_tool_optimization = enable_tool_optimization
        self.reflection_minibatch_size = reflection_minibatch_size

        # Cache predictor names/signatures
        self.named_predictors = list(self.student.named_predictors())

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])
        return new_prog

    def evaluate(self, batch, candidate, capture_traces=False):
        program = self.build_program(candidate)

        if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

            trajs = bootstrap_trace_data(
                program=program,
                dataset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
            )
            scores = []
            outputs = []
            feedbacks = []  # Capture feedback from metric
            for t in trajs:
                outputs.append(t["prediction"])
                if hasattr(t["prediction"], "__class__") and t.get("score") is None:
                    scores.append(self.failure_score)
                    feedbacks.append(None)
                else:
                    score_result = t["score"]
                    if hasattr(score_result, "score"):
                        scores.append(score_result["score"])
                        # Extract feedback if present
                        feedbacks.append(getattr(score_result, "feedback", None) or score_result.get("feedback"))
                    else:
                        scores.append(score_result)
                        feedbacks.append(None)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs, feedbacks=feedbacks)
        else:
            evaluator = Evaluate(
                devset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(batch) * 100,
            )
            res = evaluator(program)
            # DSPy EvaluationResult.results is a list of (example, prediction, score) tuples
            outputs = [r[1] for r in res.results]
            score_results = [r[2] for r in res.results]
            scores = []
            feedbacks = []
            for s in score_results:
                if hasattr(s, "score"):
                    scores.append(s["score"])
                    feedbacks.append(getattr(s, "feedback", None) or s.get("feedback"))
                else:
                    scores.append(s)
                    feedbacks.append(None)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None, feedbacks=feedbacks)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        from dspy.teleprompt.bootstrap_trace import FailedPrediction

        program = self.build_program(candidate)

        ret_d: dict[str, list[dict[str, Any]]] = {}
        for pred_name in components_to_update:
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            assert module is not None

            if pred_name not in self.feedback_map:
                continue

            items: list[dict[str, Any]] = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]
                module_score = data["score"]
                if hasattr(module_score, "score"):
                    module_score = module_score["score"]

                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    continue

                selected = None
                for t in trace_instances:
                    if isinstance(t[2], FailedPrediction):
                        selected = t
                        break

                if selected is None:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)

                inputs = selected[1]
                outputs = selected[2]

                new_inputs = {}
                new_outputs = {}

                contains_history = False
                history_key_name = None
                for input_key, input_val in inputs.items():
                    if isinstance(input_val, History):
                        contains_history = True
                        assert history_key_name is None
                        history_key_name = input_key

                if contains_history:
                    s = "```json\n"
                    for i, message in enumerate(inputs[history_key_name].messages):
                        s += f"  {i}: {message}\n"
                    s += "```"
                    new_inputs["Context"] = s

                for input_key, input_val in inputs.items():
                    if contains_history and input_key == history_key_name:
                        continue
                    new_inputs[input_key] = str(input_val)

                if isinstance(outputs, FailedPrediction):
                    s = "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                    s += "```\n"
                    s += outputs.completion_text + "\n"
                    s += "```\n\n"
                    new_outputs = s
                else:
                    for output_key, output_val in outputs.items():
                        new_outputs[output_key] = str(output_val)

                d = {"Inputs": new_inputs, "Generated Outputs": new_outputs}
                if isinstance(outputs, FailedPrediction):
                    adapter = ChatAdapter()
                    structure_instruction = ""
                    for dd in adapter.format(module.signature, [], {}):
                        structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                    d["Feedback"] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                    # d['score'] = self.failure_score
                else:
                    feedback_fn = self.feedback_map[pred_name]
                    fb = feedback_fn(
                        predictor_output=outputs,
                        predictor_inputs=inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=trace,
                    )
                    d["Feedback"] = fb["feedback"]
                    # Warn if scores mismatch (DSPy doesn't support predictor-level scoring)
                    if fb["score"] != module_score and self.warn_on_score_mismatch:
                        logger.warning(
                            f"Score mismatch: feedback returned {fb['score']}, module score was {module_score}. "
                            "GEPA does not support predictor-level scoring; using module score."
                        )
                    # d['score'] = fb.score
                items.append(d)

            if len(items) == 0:
                # raise Exception(f"No valid predictions found for module {module.signature}.")
                continue
            ret_d[pred_name] = items

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose new instruction texts based on reflection.

        This method wraps the reflection LM to handle DSPy's LM interface
        which returns a list instead of a string.
        """
        import dspy
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        reflection_lm = self.reflection_lm or dspy.settings.lm
        if reflection_lm is None:
            raise ValueError("reflection_lm must be provided for instruction proposal")

        # If custom proposer provided, use it
        if self.custom_instruction_proposer:
            with dspy.context(lm=reflection_lm):
                return self.custom_instruction_proposer(
                    candidate=candidate,
                    reflective_dataset=reflective_dataset,
                    components_to_update=components_to_update,
                )

        results: dict[str, str] = {}

        with dspy.context(lm=reflection_lm):
            for name in components_to_update:
                if name not in reflective_dataset or not reflective_dataset.get(name):
                    continue

                base_instruction = candidate[name]
                dataset_with_feedback = reflective_dataset[name]

                # Wrap DSPy LM to return string (DSPy LMs return list)
                def lm_wrapper(prompt: str) -> str:
                    result = reflection_lm(prompt)
                    if isinstance(result, list):
                        return result[0] if result else ""
                    return str(result)

                results[name] = InstructionProposalSignature.run(
                    lm=lm_wrapper,
                    input_dict={
                        "current_instruction_doc": base_instruction,
                        "dataset_with_feedback": dataset_with_feedback,
                    },
                )["new_instruction"]

        return results
