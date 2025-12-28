"""HTTP client for pushing run data to the GEPA Logger web server.

This module provides the ServerClient class that handles all communication
between the Python GEPATracker and the NextJS web server.

Usage:
    client = ServerClient("http://localhost:3000", project_name="My Project")
    run_id = client.start_run(config={}, seed_prompt={"role": "You are helpful"})

    # Push data during optimization
    client.push_iteration(iteration_data)
    client.push_evaluations(evaluation_batch)
    client.push_candidates(candidate_batch)
    client.push_lm_calls(lm_call_batch)

    # Complete the run
    client.complete_run(status="COMPLETED", best_prompt={...}, best_score=0.95)
"""

import json
import logging
from dataclasses import asdict
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable objects to strings.

    DSPy LM calls may contain non-JSON-serializable objects like
    ModelMetaclass or custom types. This function converts them to strings.

    Args:
        obj: Any object to convert

    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert anything else to string
        return str(obj)


class ServerClient:
    """HTTP client for communicating with the GEPA Logger web server.

    This client handles batched pushes and graceful error handling.
    All operations are non-blocking and will not fail the optimization
    if the server is unavailable.

    Attributes:
        server_url: Base URL of the web server (e.g., "http://localhost:3000")
        project_name: Name of the project to associate runs with
        run_id: ID of the current run (set after start_run)
    """

    def __init__(
        self,
        server_url: str,
        project_name: str = "Default",
        timeout: float = 10.0,
    ):
        """Initialize the server client.

        Args:
            server_url: Base URL of the web server
            project_name: Project name for organizing runs (default: "Default")
            timeout: HTTP request timeout in seconds (default: 10.0)
        """
        self.server_url = server_url.rstrip("/")
        self.project_name = project_name
        self.timeout = timeout
        self.run_id: str | None = None
        self._connected = False

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Make an HTTP request to the server.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/api/runs")
            data: JSON data to send (for POST/PUT)

        Returns:
            Response JSON or None if request failed
        """
        url = f"{self.server_url}{endpoint}"
        try:
            response = requests.request(
                method,
                url,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            # Restore connection status on successful request
            # This allows recovery after transient network issues
            if not self._connected and self.run_id:
                logger.info(f"Reconnected to server at {self.server_url}")
                self._connected = True
            return response.json()
        except requests.exceptions.ConnectionError:
            if self._connected:
                logger.warning(f"Lost connection to server at {self.server_url}")
                self._connected = False
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Request to {url} timed out")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error from {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error making request to {url}: {e}")
            return None

    def start_run(
        self,
        config: dict[str, Any] | None = None,
        seed_prompt: dict[str, str] | None = None,
        name: str | None = None,
        valset_example_ids: list[str] | None = None,
    ) -> str | None:
        """Start a new run on the server.

        Args:
            config: Optional configuration dict for the run
            seed_prompt: Optional seed prompt (candidate 0)
            name: Optional run name
            valset_example_ids: Optional list of validation set example IDs

        Returns:
            Run ID if successful, None otherwise
        """
        result = self._request(
            "POST",
            "/api/runs",
            {
                "projectName": self.project_name,
                "name": name,
                "config": config,
                "seedPrompt": seed_prompt,
                "valsetExampleIds": valset_example_ids,
            },
        )

        if result and "runId" in result:
            self.run_id = result["runId"]
            self._connected = True
            logger.info(f"Started run {self.run_id} on server")
            return self.run_id

        return None

    def update_run(
        self,
        seed_prompt: dict[str, str] | None = None,
        valset_example_ids: list[str] | None = None,
    ) -> bool:
        """Update run metadata (seed_prompt, valset, etc.).

        This is used to update the run with information that wasn't available
        when start_run was called (e.g., seed_prompt from first iteration).

        Args:
            seed_prompt: Optional seed prompt (candidate 0)
            valset_example_ids: Optional list of validation set example IDs

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id:
            return False

        data: dict[str, Any] = {}
        if seed_prompt is not None:
            data["seedPrompt"] = seed_prompt
        if valset_example_ids is not None:
            data["valsetExampleIds"] = valset_example_ids

        if not data:
            return True  # Nothing to update

        result = self._request(
            "PATCH",
            f"/api/runs/{self.run_id}",
            data,
        )

        return result is not None

    def push_iteration(
        self,
        iteration_number: int,
        timestamp: float,
        total_evals: int,
        num_candidates: int = 0,
        pareto_size: int = 0,
        pareto_frontier: dict[str, float] | None = None,
        pareto_programs: dict[str, int] | None = None,
        reflection_input: str | None = None,
        reflection_output: str | None = None,
        proposed_changes: list[dict[str, str]] | None = None,
        parent_candidate_idx: int | None = None,
        child_candidate_idxs: list[int] | None = None,
    ) -> bool:
        """Push iteration data to the server.

        Args:
            iteration_number: Current iteration number
            timestamp: Timestamp of the iteration
            total_evals: Total evaluations so far
            num_candidates: Number of candidates in this iteration
            pareto_size: Size of pareto frontier
            pareto_frontier: Pareto frontier dict (example_id -> score)
            pareto_programs: Programs at pareto front (example_id -> candidate_idx)
            reflection_input: LM prompt for reflection (JSON string)
            reflection_output: LM response from reflection (JSON string)
            proposed_changes: List of proposed prompt changes
            parent_candidate_idx: Which candidate was mutated
            child_candidate_idxs: New candidate indices created in this iteration

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id:
            return False

        data: dict[str, Any] = {
            "iterationNumber": iteration_number,
            "timestamp": timestamp,
            "totalEvals": total_evals,
            "numCandidates": num_candidates,
            "paretoSize": pareto_size,
            "paretoFrontier": pareto_frontier,
            "paretoPrograms": pareto_programs,
        }

        # Add optional reflection/proposal data
        if reflection_input is not None:
            data["reflectionInput"] = reflection_input
        if reflection_output is not None:
            data["reflectionOutput"] = reflection_output
        if proposed_changes is not None:
            data["proposedChanges"] = json.dumps(proposed_changes)
        if parent_candidate_idx is not None:
            data["parentCandidateIdx"] = parent_candidate_idx
        if child_candidate_idxs is not None:
            data["childCandidateIdxs"] = json.dumps(child_candidate_idxs)

        result = self._request(
            "POST",
            f"/api/runs/{self.run_id}/iterations",
            data,
        )

        return result is not None

    def push_evaluations(
        self,
        evaluations: list[dict[str, Any] | Any],
    ) -> bool:
        """Push a batch of evaluations to the server.

        Args:
            evaluations: List of evaluation dicts or EvaluationRecord dataclasses

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id or not evaluations:
            return False

        # Convert dataclasses to dicts if needed
        eval_dicts = []
        for ev in evaluations:
            if hasattr(ev, "__dataclass_fields__"):
                # It's a dataclass, convert to dict
                ev_dict = asdict(ev)
                # Map field names to API format
                eval_dicts.append({
                    "evalId": ev_dict.get("eval_id"),
                    "exampleId": ev_dict.get("example_id"),
                    "candidateIdx": ev_dict.get("candidate_idx"),
                    "iteration": ev_dict.get("iteration"),
                    "phase": ev_dict.get("phase"),
                    "score": ev_dict.get("score"),
                    "feedback": ev_dict.get("feedback"),
                    "exampleInputs": ev_dict.get("example_inputs"),
                    "predictionPreview": ev_dict.get("prediction_preview"),
                    "predictionRef": ev_dict.get("prediction_ref"),
                    "timestamp": ev_dict.get("timestamp"),
                })
            else:
                eval_dicts.append(ev)

        result = self._request(
            "POST",
            f"/api/runs/{self.run_id}/evaluations",
            {"evaluations": eval_dicts},
        )

        return result is not None

    def push_candidates(
        self,
        candidates: list[tuple[int, dict[str, str], int | None, int | None]]
        | list[dict[str, Any]],
    ) -> bool:
        """Push a batch of candidates to the server.

        Args:
            candidates: List of (idx, content, parent_idx, created_at_iter) tuples
                       or list of dicts with candidateIdx, content, parentIdx, createdAtIter

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id or not candidates:
            return False

        # Convert tuples to dicts if needed
        cand_dicts = []
        for cand in candidates:
            if isinstance(cand, tuple):
                idx, content, parent_idx, created_at_iter = cand
                cand_dicts.append({
                    "candidateIdx": idx,
                    "content": content,
                    "parentIdx": parent_idx,
                    "createdAtIter": created_at_iter,
                })
            else:
                cand_dicts.append(cand)

        result = self._request(
            "POST",
            f"/api/runs/{self.run_id}/candidates",
            {"candidates": cand_dicts},
        )

        return result is not None

    def push_lm_calls(
        self,
        lm_calls: list[dict[str, Any] | Any],
    ) -> bool:
        """Push a batch of LM calls to the server.

        Args:
            lm_calls: List of LM call dicts or LMCall dataclasses

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id or not lm_calls:
            return False

        # Convert dataclasses to dicts if needed
        lm_dicts = []
        for lm in lm_calls:
            if hasattr(lm, "__dataclass_fields__"):
                # It's a dataclass, convert to dict
                lm_dict = asdict(lm)
                # Map field names to API format
                # Use _make_serializable for inputs/outputs to handle
                # non-JSON-serializable types like ModelMetaclass
                lm_dicts.append({
                    "callId": lm_dict.get("call_id"),
                    "model": lm_dict.get("model"),
                    "startTime": lm_dict.get("start_time"),
                    "endTime": lm_dict.get("end_time"),
                    "durationMs": lm_dict.get("duration_ms"),
                    "iteration": lm_dict.get("iteration"),
                    "phase": lm_dict.get("phase"),
                    "candidateIdx": lm_dict.get("candidate_idx"),
                    "inputs": _make_serializable(lm_dict.get("inputs")),
                    "outputs": _make_serializable(lm_dict.get("outputs")),
                })
            else:
                # For dict-type entries, also make serializable
                lm["inputs"] = _make_serializable(lm.get("inputs"))
                lm["outputs"] = _make_serializable(lm.get("outputs"))
                lm_dicts.append(lm)

        result = self._request(
            "POST",
            f"/api/runs/{self.run_id}/lm-calls",
            {"lmCalls": lm_dicts},
        )

        return result is not None

    def push_logs(
        self,
        logs: list[dict[str, Any]],
    ) -> bool:
        """Push a batch of log entries to the server.

        Args:
            logs: List of log entry dicts with keys:
                - logType: "stdout", "stderr", "lm_call", "info"
                - content: Log content (plain text or JSON string)
                - timestamp: Unix timestamp
                - iteration: Optional iteration number
                - phase: Optional phase name

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id or not logs:
            return False

        result = self._request(
            "POST",
            f"/api/runs/{self.run_id}/logs",
            {"logs": logs},
        )

        return result is not None

    def update_evaluation_feedback(
        self,
        feedback_updates: list[dict[str, Any]],
    ) -> bool:
        """Update feedback for existing evaluations.

        This is used to add feedback from reflection events to evaluations
        that were already pushed (since feedback is generated after evaluation).

        Args:
            feedback_updates: List of dicts with keys:
                - exampleId: Example ID to match
                - candidateIdx: Candidate index to match
                - iteration: Iteration number to match
                - feedback: New feedback string to set

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id or not feedback_updates:
            return False

        result = self._request(
            "PATCH",
            f"/api/runs/{self.run_id}/evaluations/feedback",
            {"updates": feedback_updates},
        )

        return result is not None

    def complete_run(
        self,
        status: str = "COMPLETED",
        best_prompt: dict[str, str] | None = None,
        best_candidate_idx: int | None = None,
        best_score: float | None = None,
        seed_score: float | None = None,
    ) -> bool:
        """Mark the run as completed or failed.

        Args:
            status: "COMPLETED" or "FAILED"
            best_prompt: The best optimized prompt
            best_candidate_idx: Index of the best candidate
            best_score: Score of the best candidate
            seed_score: Score of the seed candidate

        Returns:
            True if successful, False otherwise
        """
        if not self.run_id:
            return False

        result = self._request(
            "PUT",
            f"/api/runs/{self.run_id}/status",
            {
                "status": status,
                "bestPrompt": best_prompt,
                "bestCandidateIdx": best_candidate_idx,
                "bestScore": best_score,
                "seedScore": seed_score,
            },
        )

        if result:
            logger.info(f"Run {self.run_id} marked as {status}")

        return result is not None

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to the server."""
        return self._connected and self.run_id is not None
