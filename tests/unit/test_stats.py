"""Unit tests for memory_reuse.stats."""

from __future__ import annotations

import asyncio

import pytest

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
        assert set(d.keys()) == {"hits", "misses", "errors", "total_requests", "hit_rate"}

    def test_to_dict_values(self) -> None:
        stats = CacheStats(hits=2, misses=1, errors=0, total_requests=3)
        d = stats.to_dict()
        assert d["hits"] == 2
        assert d["misses"] == 1
        assert d["hit_rate"] == pytest.approx(2 / 3)


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
        tracker.record_hit()
        tracker.record_miss()
        tracker.reset()
        stats = tracker.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0

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
