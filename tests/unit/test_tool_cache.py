"""Unit tests for ToolCache."""

from __future__ import annotations

import pytest

from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.tool import ToolCache
from memory_reuse.config import CacheConfig
from memory_reuse.exceptions import InvalidTTLError, ScopeViolationError
from memory_reuse.stats import StatsTracker


def make_tool_cache() -> ToolCache:
    backend = InMemoryBackend()
    config = CacheConfig()
    stats = StatsTracker()
    return ToolCache(backend=backend, config=config, stats=stats)


class TestToolCacheBasics:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self) -> None:
        tc = make_tool_cache()
        result = await tc.get("search", {"q": "AI"}, scope="global", scope_id=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get_hit(self) -> None:
        tc = make_tool_cache()
        await tc.set(
            "search", {"q": "AI"}, ["result1", "result2"], scope="global", scope_id=None, ttl=300
        )
        result = await tc.get("search", {"q": "AI"}, scope="global", scope_id=None)
        assert result == ["result1", "result2"]

    @pytest.mark.asyncio
    async def test_different_args_are_different_entries(self) -> None:
        tc = make_tool_cache()
        await tc.set("search", {"q": "AI"}, "ai-result", scope="global", scope_id=None, ttl=60)
        await tc.set("search", {"q": "ML"}, "ml-result", scope="global", scope_id=None, ttl=60)
        assert await tc.get("search", {"q": "AI"}, scope="global", scope_id=None) == "ai-result"
        assert await tc.get("search", {"q": "ML"}, scope="global", scope_id=None) == "ml-result"

    @pytest.mark.asyncio
    async def test_different_tool_names_isolated(self) -> None:
        tc = make_tool_cache()
        await tc.set("toolA", {}, "a-result", scope="global", scope_id=None, ttl=60)
        await tc.set("toolB", {}, "b-result", scope="global", scope_id=None, ttl=60)
        assert await tc.get("toolA", {}, scope="global", scope_id=None) == "a-result"
        assert await tc.get("toolB", {}, scope="global", scope_id=None) == "b-result"


class TestToolCacheTTL:
    @pytest.mark.asyncio
    async def test_invalid_ttl_zero_raises(self) -> None:
        tc = make_tool_cache()
        with pytest.raises(InvalidTTLError):
            await tc.set("t", {}, "v", scope="global", scope_id=None, ttl=0)

    @pytest.mark.asyncio
    async def test_invalid_ttl_negative_raises(self) -> None:
        tc = make_tool_cache()
        with pytest.raises(InvalidTTLError):
            await tc.set("t", {}, "v", scope="global", scope_id=None, ttl=-10)

    @pytest.mark.asyncio
    async def test_invalid_ttl_string_raises(self) -> None:
        tc = make_tool_cache()
        with pytest.raises(InvalidTTLError):
            await tc.set("t", {}, "v", scope="global", scope_id=None, ttl="300")  # type: ignore

    @pytest.mark.asyncio
    async def test_ttl_expiry(self) -> None:
        backend = InMemoryBackend()
        tc = ToolCache(backend=backend, config=CacheConfig(), stats=StatsTracker())
        await tc.set("tool", {"x": 1}, "result", scope="global", scope_id=None, ttl=60)
        # Expire the backend entry manually
        key = list(backend._store.keys())[0]
        backend._store[key].expires_at = 0.0
        result = await tc.get("tool", {"x": 1}, scope="global", scope_id=None)
        assert result is None


class TestToolCacheScopeValidation:
    @pytest.mark.asyncio
    async def test_user_scope_without_id_raises_on_get(self) -> None:
        tc = make_tool_cache()
        with pytest.raises(ScopeViolationError):
            await tc.get("t", {}, scope="user", scope_id=None)

    @pytest.mark.asyncio
    async def test_user_scope_without_id_raises_on_set(self) -> None:
        tc = make_tool_cache()
        with pytest.raises(ScopeViolationError):
            await tc.set("t", {}, "v", scope="user", scope_id=None, ttl=60)

    @pytest.mark.asyncio
    async def test_user_scoped_entries_isolated(self) -> None:
        tc = make_tool_cache()
        await tc.set("t", {"a": 1}, "user1-val", scope="user", scope_id="user1", ttl=60)
        assert await tc.get("t", {"a": 1}, scope="user", scope_id="user2") is None
        assert await tc.get("t", {"a": 1}, scope="user", scope_id="user1") == "user1-val"
