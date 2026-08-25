"""Tests for the Phase 3 graph-level execution cache (``wrap_graph``)."""

from __future__ import annotations

import builtins

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.exceptions import BackendNotAvailableError, ScopeViolationError
from tests.conftest import RaisingGraph, StubGraph


@pytest.fixture
def cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


class TestWrapGraphAPI:
    def test_returns_object_with_invoke_and_ainvoke(self, cache: MemoryCache) -> None:
        wrapped = cache.wrap_graph(StubGraph())
        assert hasattr(wrapped, "invoke")
        assert hasattr(wrapped, "ainvoke")

    def test_scope_falls_back_to_config_default(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory", default_scope="global"))
        wrapped = cache.wrap_graph(StubGraph())
        assert wrapped._scope == "global"

    def test_missing_langgraph_raises_named_error(
        self, cache: MemoryCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "langgraph":
                raise ImportError("no langgraph")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(BackendNotAvailableError, match="langgraph"):
            cache.wrap_graph(StubGraph())


class TestAsyncCachePath:
    @pytest.mark.asyncio
    async def test_store_and_replay_round_trip(self, cache: MemoryCache) -> None:
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)
        state = {"question": "hi", "n": 1}

        first = await wrapped.ainvoke(state)
        assert graph.node_calls == 1

        second = await wrapped.ainvoke(dict(state))
        assert graph.node_calls == 1  # zero further nodes run on hit
        assert second == first

    @pytest.mark.asyncio
    async def test_miss_returns_real_run_result(self, cache: MemoryCache) -> None:
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)
        state = {"q": "x"}
        result = await wrapped.ainvoke(state)
        assert result == {**state, "Final_Result": result["Final_Result"]}
        assert graph.node_calls == 1

    @pytest.mark.asyncio
    async def test_error_propagates_and_stores_nothing(self, cache: MemoryCache) -> None:
        graph = RaisingGraph()
        wrapped = cache.wrap_graph(graph)
        with pytest.raises(RuntimeError, match="boom"):
            await wrapped.ainvoke({"q": "x"})
        # A subsequent call still misses and re-raises (nothing stored).
        with pytest.raises(RuntimeError, match="boom"):
            await wrapped.ainvoke({"q": "x"})
        assert graph.node_calls == 2

    @pytest.mark.asyncio
    async def test_bypass_always_runs(self, cache: MemoryCache) -> None:
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)
        await wrapped.ainvoke({"q": "x"})
        await wrapped.ainvoke({"q": "x"}, bypass_cache=True)
        assert graph.node_calls == 2

    @pytest.mark.asyncio
    async def test_no_store_leaves_cache_unchanged(self, cache: MemoryCache) -> None:
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)
        await wrapped.ainvoke({"q": "x"}, no_store=True)
        await wrapped.ainvoke({"q": "x"})  # still a miss
        assert graph.node_calls == 2

    @pytest.mark.asyncio
    async def test_distinct_graph_ids_never_collide(self, cache: MemoryCache) -> None:
        g1, g2 = StubGraph(), StubGraph()
        w1 = cache.wrap_graph(g1, graph_id="a")
        w2 = cache.wrap_graph(g2, graph_id="b")
        await w1.ainvoke({"q": "x"})
        await w2.ainvoke({"q": "x"})  # different key -> miss, runs
        assert g2.node_calls == 1


class TestScope:
    @pytest.mark.asyncio
    async def test_scope_isolation(self, cache: MemoryCache) -> None:
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, scope="user")
        await wrapped.ainvoke({"q": "x", "user_id": "alice"})
        await wrapped.ainvoke({"q": "x", "user_id": "bob"})  # different scope -> miss
        assert graph.node_calls == 2

    @pytest.mark.asyncio
    async def test_state_scope_id_before_context(self, cache: MemoryCache) -> None:
        cache.set_context(user_id="ctx-user")
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, scope="user")
        # state-supplied id wins; second call with same state hits
        await wrapped.ainvoke({"q": "x", "user_id": "alice"})
        await wrapped.ainvoke({"q": "x", "user_id": "alice"})
        assert graph.node_calls == 1

    @pytest.mark.asyncio
    async def test_unresolvable_scope_raises(self, cache: MemoryCache) -> None:
        wrapped = cache.wrap_graph(StubGraph(), scope="user")
        with pytest.raises(ScopeViolationError):
            await wrapped.ainvoke({"q": "x"})


class TestSerialisability:
    @pytest.mark.asyncio
    async def test_unserialisable_result_raises(self, cache: MemoryCache) -> None:
        # A self-referential container triggers json's circular-reference
        # error (default=str cannot break a cycle inside a list/dict), so the
        # pre-store check must raise.
        cyclic: list = []
        cyclic.append(cyclic)

        class CyclicGraph:
            async def ainvoke(self, state: object, *a: object, **k: object) -> object:
                return {"cycle": cyclic}

        wrapped = cache.wrap_graph(CyclicGraph())
        with pytest.raises(ValueError, match="serialisable"):
            await wrapped.ainvoke({"q": "x"})


class TestSyncBridge:
    def test_invoke_matches_ainvoke_behaviour(self, cache: MemoryCache) -> None:
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)
        first = wrapped.invoke({"q": "x"})
        assert graph.node_calls == 1
        second = wrapped.invoke({"q": "x"})
        assert graph.node_calls == 1  # hit
        assert second == first


class TestInvalidateNode:
    @pytest.mark.asyncio
    async def test_invalidate_missing_is_safe(self, cache: MemoryCache) -> None:
        # Idempotent / safe when nothing exists (Req 13.6).
        await cache.invalidate_node("some.node", {"q": "x"})
        await cache.invalidate_node("some.node", {"q": "x"})

    @pytest.mark.asyncio
    async def test_invalidate_forces_recompute(self, cache: MemoryCache) -> None:
        # Seed an exact entry under the cached_node key shape then invalidate.
        key_parts = ["my.node", {"q": "x"}]
        await cache.exact.set(key_parts, {"cached": True}, scope="global", scope_id=None)
        assert await cache.exact.get(key_parts, scope="global", scope_id=None) is not None

        await cache.invalidate_node("my.node", {"q": "x"})
        assert await cache.exact.get(key_parts, scope="global", scope_id=None) is None
