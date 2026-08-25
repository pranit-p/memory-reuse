"""TTL behaviour tests for the Phase 3 graph-level and node-level caches.

These are example (not property-based) tests using the in-memory backend. They
assert two things (Reqs 8.4, 8.5, 13.5):

* A stored whole-run / node entry with a short/zero TTL is treated as a **miss**
  after expiry, so the wrapped graph / node body executes again.
* The **effective TTL** forwarded to ``store`` matches the wrapper ``ttl`` when
  supplied, and falls back to ``config.default_ttl`` otherwise.
"""

from __future__ import annotations

import time

import pytest

from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.integrations.langgraph import cached_node
from tests.conftest import StubGraph


def _spy_backend_ttls(cache: MemoryCache) -> list[int | None]:
    """Record every ``ttl`` passed to the underlying backend's ``set``.

    Returns the list that will be populated as ``set`` is called, so a test can
    assert the effective TTL forwarded down to the backend.
    """
    backend = cache.exact._backend  # type: ignore[attr-defined]
    recorded: list[int | None] = []
    original_set = backend.set

    async def spy_set(key: str, value: bytes, ttl: int | None = None) -> None:
        recorded.append(ttl)
        await original_set(key, value, ttl=ttl)

    backend.set = spy_set  # type: ignore[method-assign]
    return recorded


class TestGraphLevelTTL:
    @pytest.mark.asyncio
    async def test_zero_ttl_entry_is_a_miss_after_expiry(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, ttl=0)

        state = {"question": "hi", "n": 1}
        await wrapped.ainvoke(state)
        assert graph.node_calls == 1

        # A zero TTL expires immediately; the next access must re-run the graph.
        time.sleep(0.01)
        await wrapped.ainvoke(dict(state))
        assert graph.node_calls == 2

    @pytest.mark.asyncio
    async def test_short_ttl_entry_is_a_miss_after_expiry(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, ttl=1)

        state = {"q": "x"}
        await wrapped.ainvoke(state)
        assert graph.node_calls == 1

        # Still fresh -> hit, no further node runs.
        await wrapped.ainvoke(dict(state))
        assert graph.node_calls == 1

        # Force expiry by advancing the entry's deadline into the past.
        _expire_all(cache)
        await wrapped.ainvoke(dict(state))
        assert graph.node_calls == 2

    @pytest.mark.asyncio
    async def test_effective_ttl_uses_wrapper_ttl(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        recorded = _spy_backend_ttls(cache)
        wrapped = cache.wrap_graph(StubGraph(), ttl=42)

        await wrapped.ainvoke({"q": "x"})

        assert recorded == [42]

    @pytest.mark.asyncio
    async def test_effective_ttl_falls_back_to_config_default(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=1234))
        recorded = _spy_backend_ttls(cache)
        wrapped = cache.wrap_graph(StubGraph())  # no wrapper ttl

        await wrapped.ainvoke({"q": "x"})

        assert recorded == [1234]


class TestNodeLevelTTL:
    @pytest.mark.asyncio
    async def test_zero_ttl_node_entry_is_a_miss_after_expiry(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        calls = 0

        @cached_node(cache, ttl=0)
        async def node(state: dict) -> dict:
            nonlocal calls
            calls += 1
            return {"out": state["q"]}

        await node({"q": "x"})
        assert calls == 1

        time.sleep(0.01)
        await node({"q": "x"})
        assert calls == 2

    @pytest.mark.asyncio
    async def test_short_ttl_node_entry_is_a_miss_after_expiry(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        calls = 0

        @cached_node(cache, ttl=1)
        async def node(state: dict) -> dict:
            nonlocal calls
            calls += 1
            return {"out": state["q"]}

        await node({"q": "x"})
        assert calls == 1

        # Fresh -> hit, body skipped.
        await node({"q": "x"})
        assert calls == 1

        _expire_all(cache)
        await node({"q": "x"})
        assert calls == 2

    @pytest.mark.asyncio
    async def test_effective_ttl_uses_node_ttl(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        recorded = _spy_backend_ttls(cache)

        @cached_node(cache, ttl=77)
        async def node(state: dict) -> dict:
            return {"out": state["q"]}

        await node({"q": "x"})

        assert recorded == [77]

    @pytest.mark.asyncio
    async def test_effective_ttl_falls_back_to_config_default(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=555))
        recorded = _spy_backend_ttls(cache)

        @cached_node(cache)  # no node ttl
        async def node(state: dict) -> dict:
            return {"out": state["q"]}

        await node({"q": "x"})

        assert recorded == [555]


def _expire_all(cache: MemoryCache) -> None:
    """Force every in-memory entry's TTL deadline into the past.

    The in-memory backend expires entries lazily on access by comparing
    ``time.monotonic()`` against a stored ``expires_at``; rewinding the deadline
    guarantees the next access sees the entry as expired without a real sleep.
    """
    backend = cache.exact._backend  # type: ignore[attr-defined]
    assert isinstance(backend, InMemoryBackend)
    store = backend._store  # type: ignore[attr-defined]
    for entry in store.values():
        if entry.expires_at is not None:
            entry.expires_at = 0.0
