"""Tests for the unified context system.

Tests cover:
1. Basic set/get/clear operations
2. Key clearing with explicit None
3. Context manager (with_ctx) behavior
4. Thread-safety with concurrent access
5. Convenience functions
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dspy_gepa_logger.core.context import (
    set_ctx,
    get_ctx,
    clear_ctx,
    with_ctx,
    set_iteration,
    set_phase,
    set_candidate,
    get_iteration,
    get_phase,
    get_candidate,
)


class TestBasicOperations:
    """Test basic set/get/clear operations."""

    def setup_method(self):
        """Clear context before each test."""
        clear_ctx()

    def test_empty_context(self):
        """Context should be empty after clear."""
        clear_ctx()
        ctx = get_ctx()
        assert ctx == {}

    def test_set_single_value(self):
        """Should set a single value."""
        set_ctx(iteration=5)
        ctx = get_ctx()
        assert ctx == {"iteration": 5}

    def test_set_multiple_values(self):
        """Should set multiple values at once."""
        set_ctx(iteration=5, phase="eval", candidate_idx=3)
        ctx = get_ctx()
        assert ctx == {"iteration": 5, "phase": "eval", "candidate_idx": 3}

    def test_set_preserves_existing(self):
        """Should preserve existing values not overwritten."""
        set_ctx(iteration=5, phase="eval")
        set_ctx(candidate_idx=3)  # Only set candidate_idx
        ctx = get_ctx()
        assert ctx == {"iteration": 5, "phase": "eval", "candidate_idx": 3}

    def test_set_overwrites_existing(self):
        """Should overwrite existing values when specified."""
        set_ctx(iteration=5, phase="eval")
        set_ctx(iteration=6)  # Overwrite iteration
        ctx = get_ctx()
        assert ctx == {"iteration": 6, "phase": "eval"}

    def test_clear_all(self):
        """clear_ctx should remove all values."""
        set_ctx(iteration=5, phase="eval", candidate_idx=3)
        clear_ctx()
        ctx = get_ctx()
        assert ctx == {}


class TestKeyClearing:
    """Test clearing individual keys with None."""

    def setup_method(self):
        clear_ctx()

    def test_set_none_clears_key(self):
        """Setting a key to None should remove it."""
        set_ctx(iteration=5, phase="eval")
        set_ctx(phase=None)  # Clear phase
        ctx = get_ctx()
        assert ctx == {"iteration": 5}
        assert "phase" not in ctx

    def test_set_none_nonexistent_key(self):
        """Setting None for non-existent key should be no-op."""
        set_ctx(iteration=5)
        set_ctx(phase=None)  # phase doesn't exist
        ctx = get_ctx()
        assert ctx == {"iteration": 5}

    def test_clear_multiple_keys(self):
        """Should be able to clear multiple keys."""
        set_ctx(iteration=5, phase="eval", candidate_idx=3)
        set_ctx(phase=None, candidate_idx=None)
        ctx = get_ctx()
        assert ctx == {"iteration": 5}


class TestContextManager:
    """Test with_ctx() context manager."""

    def setup_method(self):
        clear_ctx()

    def test_with_ctx_sets_values(self):
        """Context manager should set values inside block."""
        with with_ctx(phase="reflection"):
            ctx = get_ctx()
            assert ctx.get("phase") == "reflection"

    def test_with_ctx_restores_empty(self):
        """Context manager should restore empty context."""
        with with_ctx(phase="reflection"):
            pass
        ctx = get_ctx()
        assert ctx == {}

    def test_with_ctx_restores_previous(self):
        """Context manager should restore previous context."""
        set_ctx(iteration=5, phase="eval")
        with with_ctx(phase="reflection"):
            ctx = get_ctx()
            assert ctx.get("phase") == "reflection"
            assert ctx.get("iteration") == 5
        ctx = get_ctx()
        assert ctx.get("phase") == "eval"  # Restored
        assert ctx.get("iteration") == 5

    def test_nested_context_managers(self):
        """Nested context managers should work correctly."""
        set_ctx(iteration=5)
        with with_ctx(phase="eval"):
            assert get_phase() == "eval"
            with with_ctx(phase="reflection", candidate_idx=3):
                assert get_phase() == "reflection"
                assert get_candidate() == 3
            # After inner block exits
            assert get_phase() == "eval"
            assert get_candidate() is None
        # After outer block exits
        assert get_phase() is None
        assert get_iteration() == 5

    def test_with_ctx_on_exception(self):
        """Context manager should restore even on exception."""
        set_ctx(iteration=5, phase="eval")
        with pytest.raises(ValueError):
            with with_ctx(phase="reflection"):
                assert get_phase() == "reflection"
                raise ValueError("test error")
        # Should be restored despite exception
        ctx = get_ctx()
        assert ctx.get("phase") == "eval"


class TestConvenienceFunctions:
    """Test convenience getter/setter functions."""

    def setup_method(self):
        clear_ctx()

    def test_set_get_iteration(self):
        """Test iteration convenience functions."""
        assert get_iteration() is None
        set_iteration(5)
        assert get_iteration() == 5

    def test_set_get_phase(self):
        """Test phase convenience functions."""
        assert get_phase() is None
        set_phase("eval")
        assert get_phase() == "eval"
        set_phase(None)
        assert get_phase() is None

    def test_set_get_candidate(self):
        """Test candidate convenience functions."""
        assert get_candidate() is None
        set_candidate(3)
        assert get_candidate() == 3
        set_candidate(None)
        assert get_candidate() is None


class TestGetCtxReturnsCopy:
    """Test that get_ctx returns a copy, not the internal dict."""

    def setup_method(self):
        clear_ctx()

    def test_mutation_does_not_affect_internal(self):
        """Mutating returned dict should not affect context."""
        set_ctx(iteration=5)
        ctx = get_ctx()
        ctx["iteration"] = 999  # Mutate returned dict
        ctx["new_key"] = "value"

        # Internal context should be unchanged
        actual = get_ctx()
        assert actual == {"iteration": 5}


class TestThreadSafety:
    """Test thread-safety of context operations.

    contextvars are designed to be thread-safe, but we verify it here.
    Each thread should have its own isolated context.
    """

    def setup_method(self):
        clear_ctx()

    def test_threads_have_isolated_context(self):
        """Each thread should have its own context."""
        results = {}
        errors = []

        def worker(thread_id: int):
            try:
                # Each thread sets its own iteration
                set_ctx(iteration=thread_id, phase=f"phase_{thread_id}")
                time.sleep(0.01)  # Small delay to allow interleaving

                # Verify thread sees its own context
                ctx = get_ctx()
                if ctx.get("iteration") != thread_id:
                    errors.append(f"Thread {thread_id} saw wrong iteration: {ctx.get('iteration')}")
                if ctx.get("phase") != f"phase_{thread_id}":
                    errors.append(f"Thread {thread_id} saw wrong phase: {ctx.get('phase')}")

                results[thread_id] = ctx
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # Run multiple threads concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check for errors
        assert not errors, f"Thread errors: {errors}"

        # Each thread should have recorded its own context
        assert len(results) == 10

    def test_thread_pool_context_isolation(self):
        """ThreadPoolExecutor workers should have isolated context."""
        results = []
        errors = []

        def task(task_id: int) -> dict:
            try:
                # Set context for this task
                clear_ctx()  # Start fresh in each task
                set_ctx(iteration=task_id, candidate_idx=task_id * 10)
                time.sleep(0.005)

                # Read and return context
                ctx = get_ctx()
                if ctx.get("iteration") != task_id:
                    errors.append(f"Task {task_id} saw wrong iteration")
                return {"task_id": task_id, "ctx": ctx}
            except Exception as e:
                errors.append(str(e))
                return {"task_id": task_id, "error": str(e)}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(task, i) for i in range(20)]
            for future in as_completed(futures):
                results.append(future.result())

        assert not errors, f"Task errors: {errors}"
        assert len(results) == 20


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        clear_ctx()

    def test_custom_keys(self):
        """Should support arbitrary custom keys."""
        set_ctx(custom_key="custom_value", another_key=123)
        ctx = get_ctx()
        assert ctx == {"custom_key": "custom_value", "another_key": 123}

    def test_empty_set_ctx(self):
        """Calling set_ctx with no args should be no-op."""
        set_ctx(iteration=5)
        set_ctx()  # No args
        ctx = get_ctx()
        assert ctx == {"iteration": 5}

    def test_various_value_types(self):
        """Should handle various value types."""
        set_ctx(
            int_val=42,
            float_val=3.14,
            str_val="hello",
            list_val=[1, 2, 3],
            dict_val={"a": 1},
            bool_val=True,
        )
        ctx = get_ctx()
        assert ctx["int_val"] == 42
        assert ctx["float_val"] == 3.14
        assert ctx["str_val"] == "hello"
        assert ctx["list_val"] == [1, 2, 3]
        assert ctx["dict_val"] == {"a": 1}
        assert ctx["bool_val"] is True
