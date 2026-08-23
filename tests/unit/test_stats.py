"""Unit tests for memory_reuse.stats."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st

from memory_reuse.stats import CacheStats, StatsTracker


class TestCacheStats:
    def test_hit_rate_zero_when_no_requests(self) -> None:
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self) -> None:
        stats = CacheStats(hits=5, misses=0, total_requests=5)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self) -> None:
        stats = CacheStats(hits=0, misses=10, total_requests=10)
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self) -> None:
        stats = CacheStats(hits=3, misses=1, total_requests=4)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_to_dict_keys(self) -> None:
        stats = CacheStats(hits=2, misses=1, errors=0, total_requests=3)
        d = stats.to_dict()
        assert set(d.keys()) == {
            "hits",
            "exact_hits",
            "semantic_hits",
            "misses",
            "errors",
            "total_requests",
            "hit_rate",
        }

    def test_to_dict_values(self) -> None:
        stats = CacheStats(
            hits=2, exact_hits=1, semantic_hits=1, misses=1, errors=0, total_requests=3
        )
        d = stats.to_dict()
        assert d["hits"] == 2
        assert d["exact_hits"] == 1
        assert d["semantic_hits"] == 1
        assert d["misses"] == 1
        assert d["hit_rate"] == pytest.approx(2 / 3)

    def test_semantic_counters_default_zero(self) -> None:
        stats = CacheStats()
        assert stats.exact_hits == 0
        assert stats.semantic_hits == 0


class TestStatsTracker:
    def test_initial_state(self) -> None:
        tracker = StatsTracker()
        stats = tracker.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.errors == 0
        assert stats.total_requests == 0

    def test_record_hit(self) -> None:
        tracker = StatsTracker()
        tracker.record_hit()
        stats = tracker.get_stats()
        assert stats.hits == 1
        assert stats.total_requests == 1

    def test_record_hit_is_alias_for_exact_hit(self) -> None:
        tracker = StatsTracker()
        tracker.record_hit()
        stats = tracker.get_stats()
        assert stats.exact_hits == 1
        assert stats.semantic_hits == 0
        assert stats.hits == 1

    def test_record_exact_hit(self) -> None:
        tracker = StatsTracker()
        tracker.record_exact_hit()
        stats = tracker.get_stats()
        assert stats.exact_hits == 1
        assert stats.semantic_hits == 0
        assert stats.hits == 1
        assert stats.total_requests == 1

    def test_record_semantic_hit(self) -> None:
        tracker = StatsTracker()
        tracker.record_semantic_hit()
        stats = tracker.get_stats()
        assert stats.semantic_hits == 1
        assert stats.exact_hits == 0
        assert stats.hits == 1
        assert stats.total_requests == 1

    def test_hits_equals_exact_plus_semantic(self) -> None:
        tracker = StatsTracker()
        for _ in range(3):
            tracker.record_exact_hit()
        for _ in range(2):
            tracker.record_semantic_hit()
        stats = tracker.get_stats()
        assert stats.exact_hits == 3
        assert stats.semantic_hits == 2
        assert stats.hits == stats.exact_hits + stats.semantic_hits
        assert stats.hits == 5
        assert stats.total_requests == 5

    def test_record_miss(self) -> None:
        tracker = StatsTracker()
        tracker.record_miss()
        stats = tracker.get_stats()
        assert stats.misses == 1
        assert stats.total_requests == 1

    def test_record_error_does_not_increment_total(self) -> None:
        tracker = StatsTracker()
        tracker.record_error()
        stats = tracker.get_stats()
        assert stats.errors == 1
        assert stats.total_requests == 0

    def test_reset(self) -> None:
        tracker = StatsTracker()
        tracker.record_exact_hit()
        tracker.record_semantic_hit()
        tracker.record_miss()
        tracker.reset()
        stats = tracker.get_stats()
        assert stats.hits == 0
        assert stats.exact_hits == 0
        assert stats.semantic_hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0

    def test_recording_never_raises(self) -> None:
        """Recording is best-effort: a broken counter must not surface (Req 8.5)."""
        tracker = StatsTracker()

        # Force every increment to raise by making the backing attributes objects
        # that reject in-place addition.
        class _Boom:
            def __iadd__(self, other: object) -> _Boom:
                raise RuntimeError("counter is broken")

        tracker._hits = _Boom()  # type: ignore[assignment]
        tracker._exact_hits = _Boom()  # type: ignore[assignment]
        tracker._semantic_hits = _Boom()  # type: ignore[assignment]
        tracker._misses = _Boom()  # type: ignore[assignment]
        tracker._errors = _Boom()  # type: ignore[assignment]
        tracker._total = _Boom()  # type: ignore[assignment]

        # None of these should raise despite the broken counters.
        tracker.record_hit()
        tracker.record_exact_hit()
        tracker.record_semantic_hit()
        tracker.record_miss()
        tracker.record_error()

    def test_hit_rate_after_operations(self) -> None:
        tracker = StatsTracker()
        for _ in range(3):
            tracker.record_hit()
        tracker.record_miss()
        stats = tracker.get_stats()
        assert stats.hit_rate == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_concurrent_increments(self) -> None:
        """Ensure no lost updates under concurrent async tasks."""
        tracker = StatsTracker()
        n = 1000

        async def bump() -> None:
            tracker.record_hit()

        await asyncio.gather(*[bump() for _ in range(n)])
        assert tracker.get_stats().hits == n


class TestStatsProperties:
    """Property-based tests for the statistics tracker.

    **Validates: Requirements 8.5** (Property 9: Non-fatal stats).
    """

    # An operation is one of the recordable events.
    _ops = st.sampled_from(["exact_hit", "semantic_hit", "hit", "miss", "error"])

    @staticmethod
    def _apply(tracker: StatsTracker, op: str) -> None:
        if op == "exact_hit":
            tracker.record_exact_hit()
        elif op == "semantic_hit":
            tracker.record_semantic_hit()
        elif op == "hit":
            tracker.record_hit()
        elif op == "miss":
            tracker.record_miss()
        else:
            tracker.record_error()

    @given(ops=st.lists(_ops, max_size=100))
    def test_hits_equal_exact_plus_semantic(self, ops: list[str]) -> None:
        """`hits` always equals `exact_hits + semantic_hits` for any sequence."""
        tracker = StatsTracker()
        for op in ops:
            self._apply(tracker, op)
        stats = tracker.get_stats()
        assert stats.hits == stats.exact_hits + stats.semantic_hits

    @given(ops=st.lists(_ops, max_size=100))
    def test_counter_correctness(self, ops: list[str]) -> None:
        """Each counter reflects exactly the operations applied."""
        tracker = StatsTracker()
        for op in ops:
            self._apply(tracker, op)
        stats = tracker.get_stats()

        expected_exact = ops.count("exact_hit") + ops.count("hit")
        expected_semantic = ops.count("semantic_hit")
        expected_misses = ops.count("miss")
        expected_errors = ops.count("error")

        assert stats.exact_hits == expected_exact
        assert stats.semantic_hits == expected_semantic
        assert stats.misses == expected_misses
        assert stats.errors == expected_errors
        assert stats.total_requests == stats.hits + stats.misses

    @given(ops=st.lists(_ops, max_size=100))
    def test_recording_never_raises(self, ops: list[str]) -> None:
        """No recording call raises even when the backing counters are broken."""

        class _Boom:
            def __iadd__(self, other: object) -> _Boom:
                raise RuntimeError("counter is broken")

        tracker = StatsTracker()
        tracker._hits = _Boom()  # type: ignore[assignment]
        tracker._exact_hits = _Boom()  # type: ignore[assignment]
        tracker._semantic_hits = _Boom()  # type: ignore[assignment]
        tracker._misses = _Boom()  # type: ignore[assignment]
        tracker._errors = _Boom()  # type: ignore[assignment]
        tracker._total = _Boom()  # type: ignore[assignment]

        for op in ops:
            # Must not raise regardless of the operation.
            self._apply(tracker, op)
