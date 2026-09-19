"""Example tests: zero-config faithful serialization for wrapped graphs (Req 10).

The flagship UX: ``wrap_graph`` auto-applies the LangChain message codec when
langchain-core is present, so a cache HIT replays real message objects with no
serializer wiring from the developer. These tests exercise that plug-and-play
path plus the opt-out and the config-wins precedence.
"""

from __future__ import annotations

import pytest

from memory_reuse import CacheConfig, MemoryCache

pytest.importorskip("langchain_core")


class _StubGraph:
    """A stub compiled graph whose final state carries a LangChain message."""

    def __init__(self) -> None:
        self.node_calls = 0

    async def ainvoke(self, state: object, *a: object, **k: object) -> object:
        from langchain_core.messages import AIMessage

        self.node_calls += 1
        return {"messages": [AIMessage(content="the answer")]}


@pytest.mark.asyncio
async def test_zero_config_wrap_graph_replays_message_objects() -> None:
    """No serializer config: a hit still returns real message objects (Req 10.1, 10.2)."""
    from langchain_core.messages import AIMessage

    cache = MemoryCache(CacheConfig(backend="memory"))  # nothing configured
    graph = _StubGraph()
    wrapped = cache.wrap_graph(graph, scope="global", key_fields=["q"], ttl=60)

    miss = await wrapped.ainvoke({"q": "question"})
    assert graph.node_calls == 1
    assert miss["messages"][-1].content == "the answer"

    hit = await wrapped.ainvoke({"q": "question"})
    assert graph.node_calls == 1  # served from cache
    assert isinstance(hit["messages"][-1], AIMessage)
    assert hit["messages"][-1].content == "the answer"


@pytest.mark.asyncio
async def test_serialize_messages_false_falls_back_to_default() -> None:
    """serialize_messages=False disables the auto codec (Req 10.5).

    Without the codec, a hit replays the JSON-serialised state — the message
    becomes a string, so it lacks ``.content``. This asserts the opt-out is
    honoured (the object is *not* reconstructed).
    """
    cache = MemoryCache(CacheConfig(backend="memory"))
    graph = _StubGraph()
    wrapped = cache.wrap_graph(
        graph, scope="global", key_fields=["q"], ttl=60, serialize_messages=False
    )

    await wrapped.ainvoke({"q": "q2"})
    hit = await wrapped.ainvoke({"q": "q2"})
    # Codec disabled → the replayed message is a plain string, not an object.
    assert not hasattr(hit["messages"][-1], "content")


@pytest.mark.asyncio
async def test_configured_serializer_wins_over_auto() -> None:
    """An explicit cache serializer takes precedence over the auto codec (Req 10.6).

    Here we configure a sentinel serializer/deserializer that tags the payload;
    the wrapped graph must use it rather than the LangChain codec. We assert the
    graph round-trips through the configured pair (the sentinel survives),
    proving the auto codec did not override the explicit choice.
    """
    marker = {"seen": 0}

    def serializer(v: object) -> object:
        marker["seen"] += 1
        return {"__explicit__": v}

    def deserializer(v: object) -> object:
        if isinstance(v, dict) and set(v) == {"__explicit__"}:
            return v["__explicit__"]
        return v

    cache = MemoryCache(
        CacheConfig(backend="memory", serializer=serializer, deserializer=deserializer)
    )

    class _PlainGraph:
        def __init__(self) -> None:
            self.node_calls = 0

        async def ainvoke(self, state: object, *a: object, **k: object) -> object:
            self.node_calls += 1
            return {"answer": "plain"}

    graph = _PlainGraph()
    wrapped = cache.wrap_graph(graph, scope="global", key_fields=["q"], ttl=60)

    await wrapped.ainvoke({"q": "q3"})
    # The explicit serializer was used (not the auto codec).
    assert marker["seen"] >= 1
    hit = await wrapped.ainvoke({"q": "q3"})
    assert hit == {"answer": "plain"}
