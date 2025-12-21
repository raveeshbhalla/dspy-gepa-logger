"""Parser for extracting data from GEPA's native log directory.

GEPA writes validation scores and other data to its log_dir during optimization.
This parser extracts that data and correlates it with our tracked iterations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GEPALogParser:
    """Parser for GEPA's log directory structure.

    GEPA creates a log directory with structure:
    - generated_best_outputs_valset/task_{id}/iter_{iteration}_prog_{program}.json
    - gepa_state.bin (pickled state)

    This parser extracts:
    - Validation scores per task per iteration
    - Pareto frontier updates
    """

    def __init__(self, log_dir: str | Path):
        """Initialize the parser.

        Args:
            log_dir: Path to GEPA's log directory
        """
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            raise ValueError(f"Log directory does not exist: {log_dir}")

    def parse_validation_scores(self) -> dict[int, dict[int, dict[int, float]]]:
        """Parse validation scores from generated_best_outputs_valset.

        Returns:
            Nested dictionary: {iteration: {task_id: {program_id: score}}}
        """
        validation_scores = {}

        valset_dir = self.log_dir / "generated_best_outputs_valset"
        if not valset_dir.exists():
            logger.warning(f"Validation directory not found: {valset_dir}")
            return validation_scores

        # Pattern: iter_{iteration}_prog_{program}.json
        file_pattern = re.compile(r"iter_(\d+)_prog_(\d+)\.json")

        # Walk through task directories
        for task_dir in sorted(valset_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
                continue

            # Extract task ID
            task_match = re.match(r"task_(\d+)", task_dir.name)
            if not task_match:
                continue
            task_id = int(task_match.group(1))

            # Parse each validation file
            for val_file in task_dir.iterdir():
                if not val_file.is_file() or not val_file.suffix == ".json":
                    continue

                # Extract iteration and program ID from filename
                match = file_pattern.match(val_file.name)
                if not match:
                    logger.warning(f"Could not parse filename: {val_file.name}")
                    continue

                iteration = int(match.group(1))
                program_id = int(match.group(2))

                # Read score from file
                try:
                    with val_file.open() as f:
                        score = json.load(f)
                        if not isinstance(score, (int, float)):
                            logger.warning(f"Invalid score format in {val_file}: {score}")
                            continue
                except Exception as e:
                    logger.warning(f"Error reading {val_file}: {e}")
                    continue

                # Store in nested dict
                if iteration not in validation_scores:
                    validation_scores[iteration] = {}
                if task_id not in validation_scores[iteration]:
                    validation_scores[iteration][task_id] = {}
                validation_scores[iteration][task_id][program_id] = score

        return validation_scores

    def compute_aggregate_scores(
        self,
        validation_scores: dict[int, dict[int, dict[int, float]]]
    ) -> dict[int, dict[int, float]]:
        """Compute aggregate validation scores per iteration/program.

        Args:
            validation_scores: Nested dict from parse_validation_scores

        Returns:
            Dictionary: {iteration: {program_id: aggregate_score}}
        """
        aggregate_scores = {}

        for iteration, tasks in validation_scores.items():
            aggregate_scores[iteration] = {}

            # Group by program_id across tasks
            program_scores = {}
            for task_id, programs in tasks.items():
                for program_id, score in programs.items():
                    if program_id not in program_scores:
                        program_scores[program_id] = []
                    program_scores[program_id].append(score)

            # Compute mean score for each program
            for program_id, scores in program_scores.items():
                aggregate_scores[iteration][program_id] = sum(scores) / len(scores)

        return aggregate_scores

    def extract_pareto_frontier(
        self,
        validation_scores: dict[int, dict[int, dict[int, float]]]
    ) -> dict[int, dict]:
        """Extract pareto frontier for each iteration.

        Args:
            validation_scores: Nested dict from parse_validation_scores

        Returns:
            Dictionary: {iteration: {
                "best_program_id": int,
                "best_score": float,
                "tasks": {task_id: {"program_id": int, "score": float}}
            }}
        """
        pareto_frontiers = {}

        for iteration, tasks in validation_scores.items():
            # Find best program per task
            task_bests = {}
            all_aggregate_scores = {}

            for task_id, programs in tasks.items():
                # Find best program for this task
                best_program_id = max(programs.keys(), key=lambda p: programs[p])
                best_score = programs[best_program_id]
                task_bests[task_id] = {
                    "program_id": best_program_id,
                    "score": best_score
                }

                # Track scores for overall aggregation
                for program_id, score in programs.items():
                    if program_id not in all_aggregate_scores:
                        all_aggregate_scores[program_id] = []
                    all_aggregate_scores[program_id].append(score)

            # Find overall best program (highest average score)
            best_program_id = None
            best_score = None
            if all_aggregate_scores:
                best_program_id = max(
                    all_aggregate_scores.keys(),
                    key=lambda p: sum(all_aggregate_scores[p]) / len(all_aggregate_scores[p])
                )
                scores = all_aggregate_scores[best_program_id]
                best_score = sum(scores) / len(scores)

            pareto_frontiers[iteration] = {
                "best_program_id": best_program_id,
                "best_score": best_score,
                "tasks": task_bests
            }

        return pareto_frontiers

    def parse_all(self) -> dict:
        """Parse all available data from GEPA logs.

        Returns:
            Dictionary containing:
                - validation_scores: Raw validation scores
                - aggregate_scores: Aggregated scores per iteration/program
                - pareto_frontiers: Pareto frontier per iteration
        """
        validation_scores = self.parse_validation_scores()
        aggregate_scores = self.compute_aggregate_scores(validation_scores)
        pareto_frontiers = self.extract_pareto_frontier(validation_scores)

        return {
            "validation_scores": validation_scores,
            "aggregate_scores": aggregate_scores,
            "pareto_frontiers": pareto_frontiers
        }

    def get_validation_rollouts(
        self,
        iteration: int,
        program_id: int
    ) -> list[dict]:
        """Get validation rollouts for a specific iteration/program.

        Args:
            iteration: Iteration number
            program_id: Program ID

        Returns:
            List of rollout dictionaries with {task_id, score}
        """
        validation_scores = self.parse_validation_scores()

        if iteration not in validation_scores:
            return []

        rollouts = []
        for task_id, programs in validation_scores[iteration].items():
            if program_id in programs:
                rollouts.append({
                    "task_id": task_id,
                    "score": programs[program_id]
                })

        return rollouts
