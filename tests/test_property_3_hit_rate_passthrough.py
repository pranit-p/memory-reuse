"""Property 3: Hit rate is passed through unchanged from CacheStats.

Feature: observability-and-cost-analytics, Property 3.

*For any* sequence of underlying hit/miss records on the shared ``StatsTracker``,
the ``AnalyticsSnapshot.hit_rate`` equals ``CacheStats.hit_rate`` for the same
counters, and the analytics layer never alters ``hits``, ``misses``, or
``total_requests``.

Validates: Requirements 1.1, 8.4.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics import AnalyticsTracker, CacheHitEvent
from memory_reuse.stats import StatsTracker

# A record op on the shared StatsTracker: exact hit, semantic hit, or miss.
_ops = st.lists(st.sampled_from(["exact", "semantic", "miss"]), max_size=50)


class TestProperty3HitRatePassthrough:
    """Feature: observability-and-cost-analytics, Property 3.

    Hit rate is passed through unchanged from CacheStats.

    Validates: Requirements 1.1, 8.4
    """

    @settings(max_examples=100)
    @given(ops=_ops)
    def test_hit_rate_matches_and_counters_untouched(self, ops: list[str]) -> None:
        """Snapshot hit rate equals CacheStats.hit_rate; counters are untouched."""
        stats = StatsTracker()
        tracker = AnalyticsTracker(stats)

        for op in ops:
            if op == "exact":
                stats.record_exact_hit()
            elif op == "semantic":
                stats.record_semantic_hit()
            else:
                stats.record_miss()
            # Record an analytics event alongside each op to prove the analytics
            # layer never writes back to the underlying counters.
            tracker.record_hit_event(CacheHitEvent(tokens_in=1, latency_saved=0.1))

        core = stats.get_stats()
        snap = tracker.snapshot()

        assert snap.hit_rate == core.hit_rate
        # The analytics layer must not have altered the core counters.
        assert core.hits == sum(1 for o in ops if o in ("exact", "semantic"))
        assert core.misses == sum(1 for o in ops if o == "miss")
        assert core.total_requests == len(ops)
        assert core.hits == core.exact_hits + core.semantic_hits
