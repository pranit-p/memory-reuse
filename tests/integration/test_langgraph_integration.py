"""Integration tests for LangGraph cached_node and cached_tool decorators."""

from __future__ import annotations

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.exceptions import ScopeViolationError
from memory_reuse.integrations.langgraph import cached_node, cached_tool


def make_cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


class TestCachedTool:
    @pytest.mark.asyncio
    async def test_async_function_cached(self) -> None:
        cache = make_cache()
        call_count = 0

        @cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result:{query}"

        result1 = await fetch("hello")
        result2 = await fetch("hello")
        assert result1 == result2 == "result:hello"
        assert call_count == 1  # Second call was cached

    @pytest.mark.asyncio
    async def test_async_function_different_args_different_cache(self) -> None:
        cache = make_cache()
        call_count = 0

        @cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result:{query}"

        await fetch("a")
        await fetch("b")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_sync_function_cached(self) -> None:
        """Sync functions wrapped by cached_tool are also awaitable."""
        cache = make_cache()
        call_count = 0

        @cached_tool(cache, scope="global", ttl=300)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # cached_tool always returns an async wrapper — await it directly
        result1 = await compute(5)
        result2 = await compute(5)
        assert result1 == result2 == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_user_scope_with_context(self) -> None:
        cache = make_cache()
        cache.set_context(user_id="alice")
        call_count = 0

        @cached_tool(cache, scope="user", ttl=60)
        async def get_data(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"data:{key}"

        result1 = await get_data("x")
        result2 = await get_data("x")
        assert result1 == result2
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_user_scope_without_context_raises(self) -> None:
        cache = make_cache()

        @cached_tool(cache, scope="user", ttl=60)
        async def get_data(key: str) -> str:
            return "data"

        with pytest.raises(ScopeViolationError):
            await get_data("x")

    @pytest.mark.asyncio
    async def test_user_scope_isolation(self) -> None:
        """Alice and Bob should have separate caches."""
        cache_alice = make_cache()
        cache_alice.set_context(user_id="alice")
        cache_bob = make_cache()
        cache_bob.set_context(user_id="bob")

        call_count = 0

        @cached_tool(cache_alice, scope="user", ttl=60)
        async def get_data_alice(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return "alice-data"

        @cached_tool(cache_bob, scope="user", ttl=60)
        async def get_data_bob(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return "bob-data"

        r_alice = await get_data_alice("k")
        r_bob = await get_data_bob("k")
        assert r_alice == "alice-data"
        assert r_bob == "bob-data"

    @pytest.mark.asyncio
    async def test_stats_updated(self) -> None:
        cache = make_cache()

        @cached_tool(cache, scope="global", ttl=300)
        async def fn(x: int) -> int:
            return x

        await fn(1)  # miss
        await fn(1)  # hit
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1


class TestCachedNode:
    @pytest.mark.asyncio
    async def test_node_output_cached(self) -> None:
        cache = make_cache()
        call_count = 0

        @cached_node(cache, scope="global")
        async def process(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"output": state["input"]}

        state = {"input": "hello"}
        r1 = await process(state)
        r2 = await process(state)
        assert r1 == r2 == {"output": "hello"}
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_node_key_fields_subset(self) -> None:
        cache = make_cache()
        call_count = 0

        @cached_node(cache, scope="global", key_fields=["messages"])
        async def process(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"out": "done"}

        # These two states differ only in a field not in key_fields
        state1 = {"messages": ["hi"], "ephemeral": "abc"}
        state2 = {"messages": ["hi"], "ephemeral": "xyz"}
        r1 = await process(state1)
        r2 = await process(state2)
        assert r1 == r2
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_node_user_scope_from_state(self) -> None:
        cache = make_cache()
        call_count = 0

        @cached_node(cache, scope="user")
        async def process(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"done": True}

        state = {"user_id": "alice", "messages": ["hello"]}
        r1 = await process(state)
        r2 = await process(state)
        assert r1 == r2
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_node_user_scope_missing_raises(self) -> None:
        cache = make_cache()

        @cached_node(cache, scope="user")
        async def process(state: dict) -> dict:
            return {}

        with pytest.raises(ScopeViolationError):
            await process({"messages": ["hi"]})

    @pytest.mark.asyncio
    async def test_sync_node_wrapped(self) -> None:
        """Sync functions wrapped by cached_node are always awaitable."""
        cache = make_cache()
        call_count = 0

        @cached_node(cache, scope="global")
        def process(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": "ok"}

        # cached_node always returns an async wrapper
        r = await process({"input": "x"})
        assert r == {"result": "ok"}
        # Second call — same state, cache hit
        r2 = await process({"input": "x"})
        assert r2 == {"result": "ok"}
        assert call_count == 1
