"""Example tests: worked recommendation scenarios (Phase 6, Req 6).

Reproduces the two scenarios from the product brief through the public
``MemoryCache`` surface — a highly-repetitive tool recommended for caching, and a
volatile tool recommended against semantic caching — plus the insufficient-data
case that emits nothing.
"""

from __future__ import annotations

from memory_reuse import CacheConfig, MemoryCache


def _cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory"))


def test_high_repeat_tool_recommended_for_caching() -> None:
    """search_confluence: many repeated inputs, not cached → enable_cache (Req 6.2)."""
    cache = _cache()
    for i in range(40):
        # Only two distinct inputs across 40 calls → high repeat ratio.
        cache.record_operation("search_confluence", hit=False, input_hash="q1" if i % 2 else "q2")

    report = cache.analyze()
    recs = {r.operation: r for r in report.recommendations}
    assert "search_confluence" in recs
    rec = recs["search_confluence"]
    assert rec.action == "enable_cache"
    assert "repeat" in rec.reason.lower() or "%" in rec.reason


def test_volatile_tool_recommended_against_semantic() -> None:
    """get_stock_price: volatile data → avoid_semantic (Req 6.4)."""
    cache = _cache()
    for i in range(30):
        cache.record_operation("get_stock_price", hit=False, input_hash=f"sym{i}", volatile=True)

    report = cache.analyze()
    recs = {r.operation: r for r in report.recommendations}
    assert "get_stock_price" in recs
    assert recs["get_stock_price"].action == "avoid_semantic"


def test_low_hit_rate_cached_tool_recommended_ttl_bump() -> None:
    """A cached tool with a poor hit rate → increase_ttl (Req 6.3)."""
    cache = _cache()
    for i in range(40):
        # Mostly misses, flagged as already cached, with churny inputs so the
        # repeat-ratio rule does not pre-empt the low-hit-rate rule.
        cache.record_operation("weather_lookup", hit=False, input_hash=f"city{i}", cached=True)

    report = cache.analyze()
    recs = {r.operation: r for r in report.recommendations}
    assert "weather_lookup" in recs
    assert recs["weather_lookup"].action == "increase_ttl"


def test_insufficient_data_emits_no_recommendation() -> None:
    """Below min_calls, no recommendation is emitted (Req 6.6)."""
    cache = _cache()
    for _ in range(3):  # far below the default min_calls
        cache.record_operation("rarely_used", hit=False, input_hash="x")

    report = cache.analyze()
    assert report.recommendations == []
    # The stats are still recorded, just not enough to recommend on.
    assert report.stats["rarely_used"].calls == 3
