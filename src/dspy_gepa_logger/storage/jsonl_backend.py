"""JSONL file-based storage backend."""

import json
import shutil
from pathlib import Path
from typing import Any, Iterator

from dspy_gepa_logger.models.run import GEPARunRecord
from dspy_gepa_logger.models.iteration import IterationRecord
from dspy_gepa_logger.models.candidate import CandidateRecord


class JSONLStorageBackend:
    """File-based storage using JSONL format.

    Directory structure:
        {base_dir}/{run_id}/
            metadata.json      - Run config, seed, timestamps, summary
            iterations.jsonl   - One iteration per line
            candidates.jsonl   - All candidates
            traces/            - Optional separate trace files
                iter_{n}.json
    """

    def __init__(
        self,
        base_dir: str | Path,
        save_traces_separately: bool = False,
    ):
        """Initialize the JSONL storage backend.

        Args:
            base_dir: Base directory for all run data
            save_traces_separately: If True, save detailed traces in separate files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.save_traces_separately = save_traces_separately

    def _run_dir(self, run_id: str) -> Path:
        """Get the directory for a specific run."""
        return self.base_dir / run_id

    def _metadata_path(self, run_id: str) -> Path:
        """Get the metadata file path for a run."""
        return self._run_dir(run_id) / "metadata.json"

    def _iterations_path(self, run_id: str) -> Path:
        """Get the iterations file path for a run."""
        return self._run_dir(run_id) / "iterations.jsonl"

    def _candidates_path(self, run_id: str) -> Path:
        """Get the candidates file path for a run."""
        return self._run_dir(run_id) / "candidates.jsonl"

    def _traces_dir(self, run_id: str) -> Path:
        """Get the traces directory for a run."""
        return self._run_dir(run_id) / "traces"

    def save_run_start(self, run: GEPARunRecord) -> None:
        """Persist initial run metadata when a run begins."""
        run_dir = self._run_dir(run.run_id)
        run_dir.mkdir(exist_ok=True)

        # Write metadata (without iterations/candidates which are separate)
        metadata = {
            "run_id": run.run_id,
            "started_at": run.started_at.isoformat(),
            "completed_at": None,
            "config": run.config.to_dict(),
            "seed_candidate": run.seed_candidate,
            "seed_val_scores": {str(k): v for k, v in run.seed_val_scores.items()},
            "seed_aggregate_score": run.seed_aggregate_score,
            "status": run.status,
        }

        with open(self._metadata_path(run.run_id), "w") as f:
            json.dump(metadata, f, indent=2)

        # Create empty iterations and candidates files
        self._iterations_path(run.run_id).touch()
        self._candidates_path(run.run_id).touch()

        # Create traces directory if needed
        if self.save_traces_separately:
            self._traces_dir(run.run_id).mkdir(exist_ok=True)

    def save_run_end(self, run: GEPARunRecord) -> None:
        """Persist final run state when a run completes."""
        # Read existing metadata
        metadata_path = self._metadata_path(run.run_id)
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update with final state
        metadata.update({
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "status": run.status,
            "error_message": run.error_message,
            "total_iterations": run.total_iterations,
            "total_metric_calls": run.total_metric_calls,
            "best_candidate_idx": run.best_candidate_idx,
            "best_aggregate_score": run.best_aggregate_score,
            "final_pareto_frontier": {str(k): list(v) for k, v in run.final_pareto_frontier.items()},
        })

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_iteration(self, run_id: str, iteration: IterationRecord) -> None:
        """Persist a single iteration record."""
        iteration_dict = iteration.to_dict()

        # Optionally save traces separately
        if self.save_traces_separately and iteration.minibatch_traces:
            traces_dir = self._traces_dir(run_id)
            traces_dir.mkdir(exist_ok=True)

            trace_file = traces_dir / f"iter_{iteration.iteration_number}.json"
            with open(trace_file, "w") as f:
                json.dump(iteration_dict.pop("minibatch_traces", []), f)

        # Append to iterations file
        with open(self._iterations_path(run_id), "a") as f:
            f.write(json.dumps(iteration_dict, default=str) + "\n")

    def save_candidate(self, run_id: str, candidate: CandidateRecord) -> None:
        """Persist a candidate record."""
        with open(self._candidates_path(run_id), "a") as f:
            f.write(json.dumps(candidate.to_dict(), default=str) + "\n")

    def load_run(self, run_id: str) -> GEPARunRecord | None:
        """Load a complete run by ID."""
        run = self.load_run_metadata(run_id)
        if run is None:
            return None

        # Load iterations
        run.iterations = list(self.load_iterations(run_id))

        # Load candidates
        run.candidates = list(self.load_candidates(run_id))

        return run

    def load_run_metadata(self, run_id: str) -> GEPARunRecord | None:
        """Load just the run metadata (without iterations/candidates)."""
        metadata_path = self._metadata_path(run_id)
        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = json.load(f)

        # Build a minimal run record from metadata
        from dspy_gepa_logger.models.run import GEPARunConfig
        from datetime import datetime

        config = GEPARunConfig.from_dict(data.get("config", {}))

        return GEPARunRecord(
            run_id=data["run_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            config=config,
            seed_candidate=data.get("seed_candidate", {}),
            seed_val_scores={int(k): v for k, v in data.get("seed_val_scores", {}).items()},
            seed_aggregate_score=data.get("seed_aggregate_score", 0.0),
            total_iterations=data.get("total_iterations", 0),
            total_metric_calls=data.get("total_metric_calls", 0),
            best_candidate_idx=data.get("best_candidate_idx", 0),
            best_aggregate_score=data.get("best_aggregate_score", 0.0),
            final_pareto_frontier={int(k): set(v) for k, v in data.get("final_pareto_frontier", {}).items()},
            status=data.get("status", "unknown"),
            error_message=data.get("error_message"),
        )

    def load_iterations(self, run_id: str) -> Iterator[IterationRecord]:
        """Stream iterations for a run."""
        iterations_path = self._iterations_path(run_id)
        if not iterations_path.exists():
            return

        with open(iterations_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                # Load traces from separate file if needed
                if self.save_traces_separately and "minibatch_traces" not in data:
                    trace_file = self._traces_dir(run_id) / f"iter_{data['iteration_number']}.json"
                    if trace_file.exists():
                        with open(trace_file) as tf:
                            data["minibatch_traces"] = json.load(tf)

                yield IterationRecord.from_dict(data)

    def load_candidates(self, run_id: str) -> Iterator[CandidateRecord]:
        """Stream candidates for a run."""
        candidates_path = self._candidates_path(run_id)
        if not candidates_path.exists():
            return

        with open(candidates_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield CandidateRecord.from_dict(data)

    def list_runs(self) -> list[str]:
        """List all available run IDs."""
        runs = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                runs.append(path.name)
        return sorted(runs)

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its data."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return False
        shutil.rmtree(run_dir)
        return True
