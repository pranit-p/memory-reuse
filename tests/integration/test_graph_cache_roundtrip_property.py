"""Property-based test for ``CachedGraph.ainvoke`` store-and-replay round trip.

Exercises the async whole-run cache path in
``memory_reuse.integrations.langgraph`` (Task 4.2 of the graph-level-cache
spec). Uses the deterministic stub compiled graph (with a per-node execution
counter) from ``tests/conftest.py``; no real LLM or network is involved.

* **Property 1 — Whole-run store-and-replay round trip:** for any serialisable
  initial input state, invoking a ``CachedGraph`` twice with the same state
  executes the wrapped graph on the first (miss) call, executes zero nodes on
  the second (hit) call, and returns a ``Final_Result`` on the second call
  equal to the one produced and stored on the first.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from tests.conftest import StubGraph

# ---------------------------------------------------------------------------
# Hypothesis strategies: JSON-serialisable dict states plus non-dict states.
# ---------------------------------------------------------------------------

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=20),
)

_dict_states = st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=_json_scalars,
    max_size=5,
)

_non_dict_states = st.one_of(
    st.text(max_size=20),
    st.integers(min_value=-1000, max_value=1000),
    st.lists(_json_scalars, max_size=5),
)

_states = st.one_of(_dict_states, _non_dict_states)


# ---------------------------------------------------------------------------
# Property 1: Whole-run store-and-replay round trip
# ---------------------------------------------------------------------------


class TestProperty1StoreAndReplayRoundTrip:
    """Feature: graph-level-cache, Property 1.

    Whole-run store-and-replay round trip.

    Validates: Requirements 2.1, 2.2, 2.4, 8.1, 8.2
    """

    @settings(max_examples=100)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_store_then_replay_round_trip(self, state: Any) -> None:
        """Two identical invocations run one node then zero, returning the same.

        The first call misses: it looks up before any node (Req 2.1), runs the
        wrapped graph exactly once, and stores the ``Final_Result`` (Req 2.4,
        8.1). The second call with an equal state hits: it returns the stored
        result with zero nodes run (Req 2.2), and that returned value equals the
        one produced and stored on the first run (Req 8.2).
        """
        # Fresh cache per example so the first invocation is a guaranteed miss.
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)

        # First call: miss -> runs the real graph exactly once and stores.
        first = await wrapped.ainvoke(state)
        assert graph.node_calls == 1

        # Second call with an equal-but-distinct copy: hit -> zero nodes run.
        second = await wrapped.ainvoke(_copy_state(state))
        assert graph.node_calls == 1

        # The replayed result equals the one produced and stored on the miss.
        assert second == first


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_state(state: Any) -> Any:
    """Return an equal-but-distinct copy of a state (dicts/lists copied)."""
    if isinstance(state, dict):
        return dict(state)
    if isinstance(state, list):
        return list(state)
    return state
