"""Example tests for the MemoryCache analytics accessor (Phase 5).

Covers the snapshot accessor behaviour required by Requirement 3:

* a fresh cache returns a zeroed snapshot without raising (3.2),
* recorded hit events are reflected in the snapshot (3.1),
* ``enable_stats=False`` zeroes analytics regardless of recorded events (3.4),
* adding the accessor does not change ``stats`` / ``CacheStats`` (3.5).
"""

from __future__ import annotations

from decimal import Decimal

from memory_reuse import CacheConfig, CacheHitEvent, MemoryCache, PricingConfig
from memory_reuse.analytics import AnalyticsSnapshot


def test_fresh_cache_returns_zeroed_snapshot() -> None:
    """A brand-new cache reports a fully zeroed analytics snapshot (Req 3.2)."""
    cache = MemoryCache(CacheConfig(backend="memory"))
    snap = cache.analytics
    assert isinstance(snap, AnalyticsSnapshot)
    assert snap.hit_rate == 0.0
    assert snap.tokens_saved == 0
    assert snap.cost_saved == Decimal("0")
    assert snap.latency_saved == 0.0


def test_recorded_events_are_reflected() -> None:
    """Recorded hit events accumulate into the snapshot (Req 3.1)."""
    cache = MemoryCache(
        CacheConfig(
            backend="memory",
            pricing=PricingConfig(input_token_price=1e-3, output_token_price=2e-3),
        )
    )
    cache.record_hit_event(CacheHitEvent(tokens_in=1000, tokens_out=500, latency_saved=0.4))
    cache.record_hit_event(CacheHitEvent(tokens_in=200, tokens_out=100, latency_saved=0.1))

    snap = cache.analytics
    assert snap.tokens_saved == 1000 + 500 + 200 + 100
    # (1200 * 1e-3) + (600 * 2e-3) = 1.2 + 1.2 = 2.40
    assert snap.cost_saved == Decimal("2.40")
    assert snap.currency == "USD"
    assert abs(snap.latency_saved - 0.5) < 1e-9


def test_disabled_stats_zeroes_analytics() -> None:
    """With enable_stats=False the snapshot is zeroed regardless of events (Req 3.4)."""
    cache = MemoryCache(
        CacheConfig(
            backend="memory",
            enable_stats=False,
            pricing=PricingConfig(input_token_price=1e-3, output_token_price=1e-3),
        )
    )
    cache.record_hit_event(CacheHitEvent(tokens_in=1000, tokens_out=1000, latency_saved=1.0))

    snap = cache.analytics
    assert snap.hit_rate == 0.0
    assert snap.tokens_saved == 0
    assert snap.cost_saved == Decimal("0")
    assert snap.latency_saved == 0.0


def test_accessor_does_not_change_stats() -> None:
    """The analytics accessor leaves stats / CacheStats untouched (Req 3.5)."""
    cache = MemoryCache(CacheConfig(backend="memory"))
    before = cache.stats.to_dict()
    cache.record_hit_event(CacheHitEvent(tokens_in=1000, tokens_out=1000))
    after = cache.stats.to_dict()
    # Recording analytics events must not touch the core hit/miss counters.
    assert before == after
    assert cache.stats.hits == cache.stats.exact_hits + cache.stats.semantic_hits
