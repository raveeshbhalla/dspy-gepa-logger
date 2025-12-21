"""Tests for SQLite storage integration."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime

from dspy_gepa_logger.storage.sqlite_backend import SQLiteStorage
from dspy_gepa_logger.storage.sqlite_adapter import SQLiteStorageAdapter
from dspy_gepa_logger.core.tracker import GEPARunTracker
from dspy_gepa_logger.core.config import TrackerConfig
from dspy_gepa_logger.models.run import GEPARunRecord, GEPARunConfig
from dspy_gepa_logger.models.iteration import IterationRecord
from dspy_gepa_logger.models.candidate import CandidateRecord


class TestSQLiteStorage:
    """Test low-level SQLite storage operations."""

    @pytest.fixture
    def storage(self):
        """Create a temporary SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path)
            yield storage
            storage.close()

    def test_create_run(self, storage):
        """Test creating a run."""
        storage.create_run(
            run_id="test-run-1",
            config={"param1": "value1"}
        )

        # Verify it's in the database
        run = storage.get_run("test-run-1")
        assert run is not None
        assert run["run_id"] == "test-run-1"
        assert run["status"] == "running"

    def test_create_program(self, storage):
        """Test creating a program."""
        storage.create_run(run_id="test-run-1", config={})

        program_id = storage.create_program(
            run_id="test-run-1",
            signature="MySignature",
            instructions={"predict": "Answer the question"},
        )

        assert program_id is not None

        # Verify it's in the database
        program = storage.get_program(program_id)
        assert program is not None
        assert program["signature"] == "MySignature"

    def test_create_iteration(self, storage):
        """Test creating an iteration."""
        storage.create_run(run_id="test-run-1", config={})

        prog1 = storage.create_program(
            run_id="test-run-1",
            signature="Sig1",
            instructions={"predict": "v1"},
        )

        prog2 = storage.create_program(
            run_id="test-run-1",
            signature="Sig1",
            instructions={"predict": "v2"},
        )

        iteration_id = storage.create_iteration(
            run_id="test-run-1",
            iteration_number=1,
            iteration_type="reflective_mutation",
            parent_program_id=prog1,
            candidate_program_id=prog2,
            parent_minibatch_score=0.5,
            candidate_minibatch_score=0.7,
            accepted=True,
        )

        assert iteration_id is not None

        # Verify it's in the database
        iterations = list(storage.get_iterations("test-run-1"))
        assert len(iterations) == 1
        assert iterations[0]["iteration_number"] == 1


class TestSQLiteStorageAdapter:
    """Test the adapter that bridges GEPARunTracker to SQLiteStorage."""

    @pytest.fixture
    def adapter(self):
        """Create a temporary adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            adapter = SQLiteStorageAdapter(db_path)
            yield adapter
            adapter.close()

    def test_save_run_start(self, adapter):
        """Test saving run start."""
        run = GEPARunRecord(
            run_id="test-run-1",
            started_at=datetime.utcnow(),
            config=GEPARunConfig(),
            seed_candidate={"predict": "Answer the question"},
            seed_val_scores={0: 0.8, 1: 0.7},
            seed_aggregate_score=0.75,
            status="running"
        )

        adapter.save_run_start(run)

        # Verify in database
        run_data = adapter.storage.get_run("test-run-1")
        assert run_data is not None
        assert run_data["run_id"] == "test-run-1"
        assert run_data["status"] == "running"

        print(f"✓ Run created: {run_data['run_id']}")

    def test_save_iteration(self, adapter):
        """Test saving an iteration."""
        # First create a run
        run = GEPARunRecord(
            run_id="test-run-1",
            started_at=datetime.utcnow(),
            config=GEPARunConfig(),
            seed_candidate={"predict": "v1"},
            seed_val_scores={},
            seed_aggregate_score=0.0,
            status="running"
        )
        adapter.save_run_start(run)

        # Create iteration using the correct structure
        iteration = IterationRecord(
            run_id="test-run-1",
            iteration_number=1,
            parent_candidate_idx=0,
            parent_prompt={"predict": "v1"},
            parent_val_score=0.5,
            selection_strategy="pareto",
            iteration_type="reflective_mutation",
            # Parent minibatch evaluation
            minibatch_ids=[0],
            minibatch_inputs=[{"question": "What is 2+2?"}],
            minibatch_outputs=[{"answer": "4"}],
            minibatch_scores=[0.5],
            minibatch_feedback=["Good"],
            # Candidate proposal
            new_candidate_prompt={"predict": "v2"},
            new_candidate_minibatch_outputs=[{"answer": "Four"}],
            new_candidate_minibatch_scores=[0.7],
            # Acceptance
            accepted=True,
            acceptance_reason="improved",
            val_aggregate_score=0.7,
        )

        adapter.save_iteration("test-run-1", iteration)

        # Verify in database
        iterations = list(adapter.storage.get_iterations("test-run-1"))
        assert len(iterations) == 1
        assert iterations[0]["iteration_number"] == 1

        # Verify rollouts
        rollouts = list(adapter.storage.get_rollouts(iterations[0]["iteration_id"], "parent_minibatch"))
        assert len(rollouts) >= 1

        print(f"✓ Iteration saved with {len(rollouts)} rollouts")

    def test_save_run_end(self, adapter):
        """Test saving run end."""
        # Create run
        run = GEPARunRecord(
            run_id="test-run-1",
            started_at=datetime.utcnow(),
            config=GEPARunConfig(),
            seed_candidate={},
            seed_val_scores={},
            seed_aggregate_score=0.0,
            status="running"
        )
        adapter.save_run_start(run)

        # End run
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        adapter.save_run_end(run)

        # Verify
        run_data = adapter.storage.get_run("test-run-1")
        assert run_data["status"] == "completed"
        assert run_data["completed_at"] is not None

        print(f"✓ Run ended: status={run_data['status']}")


class TestGEPARunTrackerWithSQLite:
    """Test GEPARunTracker with SQLite storage."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with temporary SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            adapter = SQLiteStorageAdapter(db_path)
            config = TrackerConfig(capture_lm_calls=True)
            tracker = GEPARunTracker(storage=adapter, config=config)
            yield tracker
            adapter.close()

    def test_track_context_manager(self, tracker):
        """Test the track() context manager."""
        with tracker.track() as run_id:
            assert run_id is not None
            assert tracker.is_tracking
            assert tracker.current_run_id == run_id

        # After context exits
        assert not tracker.is_tracking

        # Verify in database
        run_data = tracker.storage.storage.get_run(run_id)
        assert run_data is not None
        assert run_data["status"] == "completed"

        print(f"✓ Track context manager works: run_id={run_id}")

    def test_start_and_end_iteration(self, tracker):
        """Test recording an iteration."""
        with tracker.track() as run_id:
            # Start iteration
            iteration_idx = tracker.start_iteration(
                parent_candidate_idx=0,
                parent_prompt={"predict": "v1"},
                parent_val_score=0.5,
            )

            assert iteration_idx == 1
            assert tracker._current_iteration is not None

            # Record parent evaluation on minibatch
            tracker.record_parent_evaluation(
                minibatch_ids=[0],
                minibatch_inputs=[{"question": "What?"}],
                minibatch_outputs=[{"answer": "Because"}],
                minibatch_scores=[0.5],
                minibatch_feedback=["OK"],
            )

            # Record reflection
            tracker.record_reflection(
                components_to_update=["predict"],
                reflective_datasets={"predict": [{"question": "What?", "answer": "Because", "score": 0.5}]},
                proposed_instructions={"predict": "v2"},
            )

            # Record candidate evaluation
            tracker.record_candidate_evaluation(
                minibatch_outputs=[{"answer": "Because!"}],
                minibatch_scores=[0.7],
            )

            # Record acceptance
            tracker.record_acceptance(
                accepted=True,
                reason="improved",
                val_scores={0: 0.7},
                val_aggregate_score=0.7,
                new_candidate_idx=1,
            )

            # End iteration
            tracker.end_iteration()

        # Verify in database
        iterations = list(tracker.storage.storage.get_iterations(run_id))
        assert len(iterations) == 1
        assert iterations[0]["iteration_number"] == 1
        assert iterations[0]["accepted"] == 1

        # Check rollouts
        iteration_id = iterations[0]["iteration_id"]
        parent_rollouts = list(tracker.storage.storage.get_rollouts(iteration_id, "parent_minibatch"))
        candidate_rollouts = list(tracker.storage.storage.get_rollouts(iteration_id, "candidate_minibatch"))

        assert len(parent_rollouts) == 1
        assert len(candidate_rollouts) == 1

        print(f"✓ Iteration recorded with {len(parent_rollouts)} parent + {len(candidate_rollouts)} candidate rollouts")


def run_all_tests():
    """Run all tests manually."""
    print("=" * 60)
    print("SQLite Integration Tests")
    print("=" * 60)

    # Test 1: SQLite Storage
    print("\n1. Testing SQLiteStorage...")
    test_storage = TestSQLiteStorage()

    # Test create_run
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(db_path)
        test_storage.test_create_run(storage)
        storage.close()

    # Test create_program
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(db_path)
        test_storage.test_create_program(storage)
        storage.close()

    # Test create_iteration
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(db_path)
        test_storage.test_create_iteration(storage)
        storage.close()

    # Test 2: SQLiteStorageAdapter
    print("\n2. Testing SQLiteStorageAdapter...")
    test_adapter = TestSQLiteStorageAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        adapter = SQLiteStorageAdapter(db_path)
        test_adapter.test_save_run_start(adapter)
        adapter.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        adapter = SQLiteStorageAdapter(db_path)
        test_adapter.test_save_iteration(adapter)
        adapter.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        adapter = SQLiteStorageAdapter(db_path)
        test_adapter.test_save_run_end(adapter)
        adapter.close()

    # Test 3: GEPARunTracker
    print("\n3. Testing GEPARunTracker with SQLite...")
    test_tracker = TestGEPARunTrackerWithSQLite()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        adapter = SQLiteStorageAdapter(db_path)
        config = TrackerConfig(capture_lm_calls=True)
        tracker = GEPARunTracker(storage=adapter, config=config)
        test_tracker.test_track_context_manager(tracker)
        adapter.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        adapter = SQLiteStorageAdapter(db_path)
        config = TrackerConfig(capture_lm_calls=True)
        tracker = GEPARunTracker(storage=adapter, config=config)
        test_tracker.test_start_and_end_iteration(tracker)
        adapter.close()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
