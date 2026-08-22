"""Unit tests for ExactCache."""

from __future__ import annotations

import pytest

from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.exact import ExactCache
from memory_reuse.config import CacheConfig
from memory_reuse.exceptions import ScopeViolationError
from memory_reuse.stats import StatsTracker


def make_cache(ttl: int | None = 3600) -> ExactCache:
    backend = InMemoryBackend()
    config = CacheConfig(default_ttl=ttl)
    stats = StatsTracker()
    return ExactCache(backend=backend, config=config, stats=stats)


class TestExactCacheHitMiss:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self) -> None:
        cache = make_cache()
        result = await cache.get(["prompt", "v1"], scope="global", scope_id=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get_hit(self) -> None:
        cache = make_cache()
        await cache.set(["prompt", "v1"], {"answer": 42}, scope="global", scope_id=None, ttl=None)
        result = await cache.get(["prompt", "v1"], scope="global", scope_id=None)
        assert result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_stats_hit_recorded(self) -> None:
        backend = InMemoryBackend()
        stats = StatsTracker()
        cache = ExactCache(backend=backend, config=CacheConfig(), stats=stats)
        await cache.set(["k"], "val", scope="global", scope_id=None, ttl=None)
        await cache.get(["k"], scope="global", scope_id=None)
        assert stats.get_stats().hits == 1
        assert stats.get_stats().total_requests == 1

    @pytest.mark.asyncio
    async def test_stats_miss_recorded(self) -> None:
        backend = InMemoryBackend()
        stats = StatsTracker()
        cache = ExactCache(backend=backend, config=CacheConfig(), stats=stats)
        await cache.get(["missing"], scope="global", scope_id=None)
        assert stats.get_stats().misses == 1

    @pytest.mark.asyncio
    async def test_different_key_parts_different_entries(self) -> None:
        cache = make_cache()
        await cache.set(["prompt", "a"], "result-a", scope="global", scope_id=None, ttl=None)
        await cache.set(["prompt", "b"], "result-b", scope="global", scope_id=None, ttl=None)
        assert await cache.get(["prompt", "a"], scope="global", scope_id=None) == "result-a"
        assert await cache.get(["prompt", "b"], scope="global", scope_id=None) == "result-b"


class TestExactCacheScopeIsolation:
    @pytest.mark.asyncio
    async def test_user_scope_requires_scope_id(self) -> None:
        cache = make_cache()
        with pytest.raises(ScopeViolationError):
            await cache.get(["k"], scope="user", scope_id=None)

    @pytest.mark.asyncio
    async def test_session_scope_requires_scope_id(self) -> None:
        cache = make_cache()
        with pytest.raises(ScopeViolationError):
            await cache.set(["k"], "v", scope="session", scope_id=None, ttl=None)

    @pytest.mark.asyncio
    async def test_user_a_cannot_see_user_b_data(self) -> None:
        cache = make_cache()
        await cache.set(["key"], "alice-data", scope="user", scope_id="alice", ttl=None)
        result = await cache.get(["key"], scope="user", scope_id="bob")
        assert result is None

    @pytest.mark.asyncio
    async def test_global_and_user_scopes_are_isolated(self) -> None:
        cache = make_cache()
        await cache.set(["key"], "global-data", scope="global", scope_id=None, ttl=None)
        result = await cache.get(["key"], scope="user", scope_id="alice")
        assert result is None

    @pytest.mark.asyncio
    async def test_user_scoped_cache(self) -> None:
        cache = make_cache()
        await cache.set(["q"], "answer", scope="user", scope_id="user-1", ttl=None)
        assert await cache.get(["q"], scope="user", scope_id="user-1") == "answer"
        assert await cache.get(["q"], scope="user", scope_id="user-2") is None


class TestExactCacheInvalidate:
    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self) -> None:
        cache = make_cache()
        await cache.set(["k"], "v", scope="global", scope_id=None, ttl=None)
        await cache.invalidate(["k"], scope="global", scope_id=None)
        assert await cache.get(["k"], scope="global", scope_id=None) is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_no_error(self) -> None:
        cache = make_cache()
        await cache.invalidate(["nonexistent"], scope="global", scope_id=None)
