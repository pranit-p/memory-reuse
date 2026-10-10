"""Regression tests for backend-failure handling (Fix #6).

A cache backend failure (timeout, connection drop, transient error) must never
turn into a valid-looking cache hit or an unrelated result. Instead it is
recorded as an error and treated as a miss, so the application re-runs the real
computation and stays correct. These use a fault-injecting in-memory backend so
the behaviour is deterministic and offline.
"""

from __future__ import annotations

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse._utils import MISS
from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.integrations.langgraph import cached_tool


class _FaultyBackend(InMemoryBackend):
    """In-memory backend that can be told to raise on get/set."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_get = False
        self.fail_set = False

    async def get(self, key: str) -> bytes | None:
        if self.fail_get:
            raise RuntimeError("backend get boom")
        return await super().get(key)

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        if self.fail_set:
            raise RuntimeError("backend set boom")
        await super().set(key, value, ttl=ttl)


def _cache_with(backend: InMemoryBackend) -> MemoryCache:
    cache = MemoryCache(CacheConfig(backend="memory"))
    cache._backend = backend
    # Rebuild the cache layers against the injected backend.
    from memory_reuse.cache.exact import ExactCache
    from memory_reuse.cache.tool import ToolCache

    cache.exact = ExactCache(backend, cache._config, cache._stats_tracker)
    cache.tool = ToolCache(backend, cache._config, cache._stats_tracker)
    return cache


class TestBackendGetFailure:
    async def test_exact_get_failure_is_a_miss(self) -> None:
        backend = _FaultyBackend()
        cache = _cache_with(backend)
        await cache.exact.set(["k"], "real-value", scope="global", scope_id=None, ttl=300)

        backend.fail_get = True
        # A backend GET error must not surface the (now unreadable) value or raise.
        assert await cache.exact.get(["k"], scope="global", scope_id=None) is None
        assert await cache.exact._get(["k"], "global", None) is MISS
        assert cache.stats.errors >= 1

    async def test_tool_get_failure_is_a_miss(self) -> None:
        backend = _FaultyBackend()
        cache = _cache_with(backend)
        await cache.tool.set("t", {"a": 1}, "v", scope="global", scope_id=None, ttl=300)

        backend.fail_get = True
        assert await cache.tool.get("t", {"a": 1}, scope="global", scope_id=None) is None

    async def test_lookup_failure_falls_through_to_miss(self) -> None:
        backend = _FaultyBackend()
        cache = _cache_with(backend)
        backend.fail_get = True
        assert await cache.lookup(["k"], "", scope="global", scope_id=None) is None


class TestBackendSetFailure:
    async def test_set_failure_does_not_crash_caller(self) -> None:
        backend = _FaultyBackend()
        cache = _cache_with(backend)
        backend.fail_set = True
        # A failed store must not raise; the next lookup is simply a miss.
        await cache.exact.set(["k"], "v", scope="global", scope_id=None, ttl=300)
        backend.fail_set = False
        assert await cache.exact.get(["k"], scope="global", scope_id=None) is None


class TestDecoratorWithFailingBackend:
    async def test_tool_recomputes_when_backend_fails(self) -> None:
        """With a failing backend every call is a miss → body runs, result correct."""
        backend = _FaultyBackend()
        cache = _cache_with(backend)
        backend.fail_get = True
        backend.fail_set = True
        calls = {"n": 0}

        @cached_tool(cache, scope="global", ttl=300)
        async def tool() -> str:
            calls["n"] += 1
            return "computed"

        # The result is always the real computed value, never a stale/wrong one.
        assert await tool() == "computed"
        assert await tool() == "computed"
        assert calls["n"] == 2
