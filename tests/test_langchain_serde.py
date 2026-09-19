"""Example tests for the LangChain (de)serialization codec (Phase 6, Req 9).

Covers:
* the codec round-trips LangChain message objects faithfully through the cache
  store/load path — a hit returns objects equal to the stored ones (Req 9.6);
* a wrapped graph replays real message objects on a cache hit (Req 9.6);
* constructing the codec without langchain-core raises the named error (Req 9.8).
"""

from __future__ import annotations

import builtins

import pytest

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.exceptions import BackendNotAvailableError
from memory_reuse.integrations import langchain_serde


def test_codec_missing_dependency_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """With langchain-core absent, the codec factory raises a named error (Req 9.8)."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "langchain_core" or name.startswith("langchain_core."):
            raise ImportError("simulated missing langchain-core")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendNotAvailableError, match=r"memory-reuse\[langchain\]"):
        langchain_serde.langchain_message_codec()


# The round-trip tests need a real langchain-core; skip cleanly when absent.
langchain_core = pytest.importorskip("langchain_core")


def _codec_cache() -> MemoryCache:
    serializer, deserializer = langchain_serde.langchain_message_codec()
    return MemoryCache(
        CacheConfig(backend="memory", serializer=serializer, deserializer=deserializer)
    )


@pytest.mark.asyncio
async def test_message_objects_round_trip_through_cache() -> None:
    """A cached state's message objects survive store/load as objects (Req 9.6)."""
    from langchain_core.messages import AIMessage, HumanMessage

    cache = _codec_cache()
    state = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="hello there")],
        "meta": {"turn": 1},  # plain JSON content passes through untouched
    }
    await cache.exact.set(["state"], state, scope="global", scope_id=None, ttl=60)
    loaded = await cache.exact.get(["state"], scope="global", scope_id=None)

    assert isinstance(loaded["messages"][0], HumanMessage)
    assert isinstance(loaded["messages"][1], AIMessage)
    assert loaded["messages"][-1].content == "hello there"
    assert loaded["meta"] == {"turn": 1}


@pytest.mark.asyncio
async def test_wrap_graph_hit_replays_message_objects() -> None:
    """A wrap_graph cache hit returns message objects, matching the miss (Req 9.6).

    Uses a stub graph whose final state contains a LangChain ``AIMessage``. On the
    first invoke (miss) the graph runs and the result is stored; on the second
    (hit) the stored state is replayed — and with the codec the replayed message
    is a real ``AIMessage`` with the same content, not a stringified object.
    """
    from langchain_core.messages import AIMessage

    class _StubGraph:
        def __init__(self) -> None:
            self.node_calls = 0

        async def ainvoke(self, state: object, *a: object, **k: object) -> object:
            self.node_calls += 1
            return {"messages": [AIMessage(content="final answer")]}

    cache = _codec_cache()
    graph = _StubGraph()
    wrapped = cache.wrap_graph(graph, scope="global", key_fields=["q"], ttl=60)

    miss = await wrapped.ainvoke({"q": "question"})
    assert graph.node_calls == 1
    assert miss["messages"][-1].content == "final answer"

    hit = await wrapped.ainvoke({"q": "question"})
    assert graph.node_calls == 1  # served from cache, graph not re-run
    assert isinstance(hit["messages"][-1], AIMessage)
    assert hit["messages"][-1].content == "final answer"
