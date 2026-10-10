"""Regression tests for false-like cached values (Fix #4).

A cached value can legitimately be ``None``, ``False``, ``0``, ``""`` or ``[]``.
These were previously indistinguishable from a cache miss (both surfaced as
``None``), so a stored false-like value was recomputed on every lookup. The
internal lookup path now uses a ``MISS`` sentinel, so a stored false-like value
is served as a genuine hit and the underlying work runs only once.
"""

from __future__ import annotations

import pytest

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.integrations.langgraph import cached_node, cached_tool

FALSE_LIKE = [None, False, 0, 0.0, "", [], {}]


@pytest.fixture
def cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


class TestLookupStoreFalseLike:
    @pytest.mark.parametrize("value", FALSE_LIKE)
    async def test_stored_false_like_value_is_a_hit(
        self, cache: MemoryCache, value: object
    ) -> None:
        """A stored false-like value round-trips through lookup as a hit."""
        await cache.store(["k"], "", value, scope="global", scope_id=None)
        # Public lookup returns the stored value (``None`` for a stored ``None``).
        assert await cache.lookup(["k"], "", scope="global", scope_id=None) == value

    async def test_internal_lookup_distinguishes_miss_from_cached_none(
        self, cache: MemoryCache
    ) -> None:
        from memory_reuse._utils import MISS

        # A genuine miss returns the sentinel, not ``None``.
        miss = await cache._lookup(["absent"], "", scope="global", scope_id=None)
        assert miss is MISS

        # A stored ``None`` returns ``None`` (a hit), never the sentinel.
        await cache.store(["present"], "", None, scope="global", scope_id=None)
        hit = await cache._lookup(["present"], "", scope="global", scope_id=None)
        assert hit is None
        assert hit is not MISS


class TestCachedToolFalseLike:
    @pytest.mark.parametrize("value", FALSE_LIKE)
    async def test_tool_body_runs_once_for_false_like_result(
        self, cache: MemoryCache, value: object
    ) -> None:
        """A tool returning a false-like value is cached, not re-executed."""
        calls = {"n": 0}

        @cached_tool(cache, scope="global", ttl=300)
        async def produce() -> object:
            calls["n"] += 1
            return value

        first = await produce()
        second = await produce()
        assert first == value
        assert second == value
        assert calls["n"] == 1, "false-like result should be served from cache"


class TestCachedNodeFalseLike:
    @pytest.mark.parametrize("value", FALSE_LIKE)
    async def test_node_body_runs_once_for_false_like_result(
        self, cache: MemoryCache, value: object
    ) -> None:
        calls = {"n": 0}

        @cached_node(cache, scope="global")
        async def node(state: dict) -> object:
            calls["n"] += 1
            return value

        first = await node({"a": 1})
        second = await node({"a": 1})
        assert first == value
        assert second == value
        assert calls["n"] == 1


class TestGetOrComputeFalseLike:
    @pytest.mark.parametrize("value", FALSE_LIKE)
    async def test_get_or_compute_caches_false_like(self, value: object) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", single_flight=True))
        calls = {"n": 0}

        async def compute() -> object:
            calls["n"] += 1
            return value

        a = await cache.get_or_compute(["k"], compute, scope="global", scope_id=None, ttl=300)
        b = await cache.get_or_compute(["k"], compute, scope="global", scope_id=None, ttl=300)
        assert a == value
        assert b == value
        assert calls["n"] == 1
