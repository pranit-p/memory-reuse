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
async def test_message_with_non_loadable_object_does_not_crash_on_load() -> None:
    """A message embedding a non-LangChain-loadable object round-trips without crashing.

    Regression for the tool-calling case: a real agent's cached ``AIMessage`` can
    carry a provider tool-call object (e.g. a LiteLLM
    ``ChatCompletionMessageToolCall``) in ``additional_kwargs``. ``dumpd`` marks
    such objects ``not_implemented`` and ``load`` refuses them — so the codec
    must NOT wrap a non-round-trippable message, letting it degrade to the
    default serialization rather than crashing the cache read.
    """
    from langchain_core.messages import AIMessage

    class _NotSerializable:
        """A plain object LangChain cannot serialise/deserialise."""

        def __repr__(self) -> str:
            return "_NotSerializable()"

    cache = _codec_cache()
    # An AIMessage whose additional_kwargs holds a non-loadable object, mirroring
    # a tool-call payload that isn't a LangChain Serializable.
    msg = AIMessage(content="calling a tool", additional_kwargs={"tool_obj": _NotSerializable()})
    state = {"messages": [msg], "ok": True}

    # Store must succeed and, crucially, a subsequent load must not raise.
    await cache.exact.set(["s"], state, scope="global", scope_id=None, ttl=60)
    loaded = await cache.exact.get(["s"], scope="global", scope_id=None)

    # The read completed (no NotImplementedError) and JSON-native content is intact.
    assert loaded is not None
    assert loaded["ok"] is True
    # The message still rebuilds as a real AIMessage with its content — the
    # un-loadable embedded sub-object is skipped rather than crashing or
    # degrading the whole message to a dict.
    restored = loaded["messages"][-1]
    assert isinstance(restored, AIMessage)
    assert restored.content == "calling a tool"


@pytest.mark.asyncio
async def test_semantic_path_round_trips_message_objects() -> None:
    """The codec applies to the semantic cache layer too (store + hit).

    Regression: a wrap_graph with ``semantic=True`` stores/loads through the
    semantic cache, which is a different path than the exact cache. The per-call
    codec must reach that path so a semantic hit reconstructs real message
    objects rather than strings.
    """
    from langchain_core.messages import AIMessage

    from tests.conftest import StubEmbedder, make_semantic_cache

    serializer, deserializer = langchain_serde.langchain_message_codec()
    cache = make_semantic_cache(StubEmbedder(), serializer=serializer, deserializer=deserializer)

    state = {"messages": [AIMessage(content="semantic answer")]}
    # Store via the combined flow (writes the semantic layer too).
    await cache.store(["k"], "how do I reset my password?", state, scope="global", scope_id=None)

    # A reworded query embeds to the same stub vector → semantic hit. The result
    # must contain a real AIMessage, not a stringified one.
    got = await cache.lookup(
        ["different-key"],
        "how do I reset my password?",
        scope="global",
        scope_id=None,
    )
    assert got is not None
    restored = got["messages"][-1]
    assert isinstance(restored, AIMessage)
    assert restored.content == "semantic answer"


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
