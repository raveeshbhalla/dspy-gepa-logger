"""Basic tests for dspy-gepa-logger."""

import uuid
from datetime import datetime

import pytest

from dspy_gepa_logger import track_gepa_run, MemoryStorageBackend, GEPARunTracker
from dspy_gepa_logger.models import (
    GEPARunRecord,
    IterationRecord,
    CandidateRecord,
    ReflectionRecord,
)


def test_track_gepa_run_creates_tracker():
    """Test that track_gepa_run creates a valid tracker."""
    tracker = track_gepa_run(log_dir="./test_logs")
    assert isinstance(tracker, GEPARunTracker)
    assert tracker.storage is not None
    assert tracker.config is not None


def test_memory_backend_basic_operations():
    """Test basic operations with memory backend."""
    storage = MemoryStorageBackend()

    # Create a run
    run = GEPARunRecord(
        run_id="test-run",
        seed_candidate={"pred": "test"},
        seed_aggregate_score=0.5,
    )

    # Save and load
    storage.save_run_start(run)
    loaded = storage.load_run_metadata("test-run")

    assert loaded is not None
    assert loaded.run_id == "test-run"
    assert loaded.seed_aggregate_score == 0.5

    # List runs
    runs = storage.list_runs()
    assert "test-run" in runs


def test_tracker_lifecycle():
    """Test tracker start/end lifecycle."""
    storage = MemoryStorageBackend()
    tracker = GEPARunTracker(storage=storage)

    # Start run
    run_id = tracker.start_run(
        seed_candidate={"pred": "test instruction"},
        seed_val_scores={0: 0.5, 1: 0.6},
    )

    assert tracker.is_tracking
    assert tracker.current_run_id == run_id

    # End run
    tracker.end_run(status="completed")

    # Load and verify
    run = storage.load_run_metadata(run_id)
    assert run.status == "completed"
    assert run.seed_aggregate_score == 0.55  # (0.5 + 0.6) / 2


def test_iteration_recording():
    """Test recording a complete iteration."""
    storage = MemoryStorageBackend()
    tracker = GEPARunTracker(storage=storage)

    run_id = tracker.start_run(
        seed_candidate={"pred": "initial"},
        seed_val_scores={0: 0.5},
    )

    # Start iteration
    iter_num = tracker.start_iteration(
        parent_candidate_idx=0,
        parent_prompt={"pred": "parent instruction"},
        parent_val_score=0.5,
    )

    assert iter_num == 1

    # Record parent evaluation
    tracker.record_parent_evaluation(
        minibatch_ids=[0, 1],
        minibatch_inputs=[{"q": "q1"}, {"q": "q2"}],
        minibatch_outputs=[{"a": "a1"}, {"a": "a2"}],
        minibatch_scores=[0.8, 0.9],
        minibatch_feedback=["good", "great"],
    )

    # Record reflection
    tracker.record_reflection(
        components_to_update=["pred"],
        reflective_datasets={"pred": [{"Inputs": {}, "Generated_Outputs": {}, "Feedback": "test"}]},
        proposed_instructions={"pred": "improved instruction"},
        duration_ms=100.0,
    )

    # Record candidate evaluation
    tracker.record_candidate_evaluation(
        minibatch_outputs=[{"a": "a1"}, {"a": "a2"}],
        minibatch_scores=[0.9, 1.0],
    )

    # Accept candidate
    tracker.record_acceptance(
        accepted=True,
        reason="score_improved",
        val_aggregate_score=0.95,
        new_candidate_idx=1,
    )

    # End iteration
    tracker.end_iteration(duration_ms=500.0)

    tracker.end_run(status="completed")

    # Verify
    run = storage.load_run(run_id)
    assert len(run.iterations) == 1

    iteration = run.iterations[0]
    assert iteration.iteration_number == 1
    assert iteration.accepted == True
    assert iteration.val_aggregate_score == 0.95
    assert len(iteration.minibatch_ids) == 2
    assert iteration.reflection is not None
    assert iteration.reflection.proposed_instructions == {"pred": "improved instruction"}


def test_candidate_recording():
    """Test recording candidates."""
    storage = MemoryStorageBackend()
    tracker = GEPARunTracker(storage=storage)

    run_id = tracker.start_run()

    # Record a candidate
    tracker.record_candidate(
        candidate_idx=1,
        instructions={"pred": "test instruction"},
        parent_indices=[0],
        creation_iteration=1,
        creation_type="reflective_mutation",
        val_subscores={0: 0.8, 1: 0.9},
        val_aggregate_score=0.85,
        metric_calls_at_discovery=10,
    )

    tracker.end_run(status="completed")

    # Verify
    run = storage.load_run(run_id)
    assert len(run.candidates) == 1

    candidate = run.candidates[0]
    assert candidate.candidate_idx == 1
    assert candidate.val_aggregate_score == 0.85
    assert candidate.creation_type == "reflective_mutation"


def test_context_manager():
    """Test using tracker with context manager."""
    storage = MemoryStorageBackend()
    tracker = GEPARunTracker(storage=storage)

    with tracker.track() as run_id:
        assert tracker.is_tracking
        assert tracker.current_run_id == run_id

    # After context exits, run should be completed
    run = storage.load_run_metadata(run_id)
    assert run.status == "completed"


def test_data_model_serialization():
    """Test that data models can serialize/deserialize."""
    # Test IterationRecord
    iteration = IterationRecord(
        run_id="test",
        iteration_number=1,
        parent_candidate_idx=0,
        parent_prompt={"pred": "test"},
        parent_val_score=0.5,
        minibatch_ids=[0, 1],
        minibatch_inputs=[{"q": "q1"}],
        minibatch_outputs=[{"a": "a1"}],
        minibatch_scores=[0.8],
        minibatch_feedback=["good"],
        accepted=True,
    )

    # Serialize and deserialize
    data = iteration.to_dict()
    restored = IterationRecord.from_dict(data)

    assert restored.iteration_number == 1
    assert restored.accepted == True
    assert len(restored.minibatch_ids) == 2


def test_run_metrics():
    """Test computed metrics on GEPARunRecord."""
    run = GEPARunRecord(
        run_id="test",
        seed_aggregate_score=0.5,
        best_aggregate_score=0.8,
    )

    # Add some iterations
    run.iterations = [
        IterationRecord(
            run_id="test",
            iteration_number=1,
            parent_candidate_idx=0,
            parent_prompt={},
            parent_val_score=0.5,
            accepted=True,
        ),
        IterationRecord(
            run_id="test",
            iteration_number=2,
            parent_candidate_idx=1,
            parent_prompt={},
            parent_val_score=0.6,
            accepted=False,
        ),
        IterationRecord(
            run_id="test",
            iteration_number=3,
            parent_candidate_idx=1,
            parent_prompt={},
            parent_val_score=0.6,
            accepted=True,
        ),
    ]

    assert abs(run.improvement - 0.3) < 0.0001  # 0.8 - 0.5 (with floating point tolerance)
    assert len(run.accepted_iterations) == 2
    assert abs(run.acceptance_rate - 2 / 3) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
