"""End-to-end graph-level cache test against a real compiled LangGraph graph.

Skipped when LangGraph is not installed, keeping the core suite LangGraph-free.
"""

from __future__ import annotations

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache

pytest.importorskip("langgraph")
from langgraph.graph import END, START, StateGraph  # noqa: E402


def _build_graph(counter: dict) -> object:
    def answer(state: dict) -> dict:
        counter["answer"] += 1
        return {"answer": f"answer for {state['question']}"}

    builder = StateGraph(dict)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


@pytest.mark.asyncio
async def test_graph_level_hit_runs_zero_nodes() -> None:
    counter = {"answer": 0}
    graph = _build_graph(counter)
    cache = MemoryCache(CacheConfig(backend="memory"))
    wrapped = cache.wrap_graph(graph, key_fields=["question"])

    first = await wrapped.ainvoke({"question": "hello"})
    assert counter["answer"] == 1

    second = await wrapped.ainvoke({"question": "hello"})
    assert counter["answer"] == 1  # graph-level hit: no node executed
    assert second == first


@pytest.mark.asyncio
async def test_graph_level_miss_runs_nodes() -> None:
    counter = {"answer": 0}
    graph = _build_graph(counter)
    cache = MemoryCache(CacheConfig(backend="memory"))
    wrapped = cache.wrap_graph(graph, key_fields=["question"])

    await wrapped.ainvoke({"question": "a"})
    await wrapped.ainvoke({"question": "b"})  # different key -> miss
    assert counter["answer"] == 2
