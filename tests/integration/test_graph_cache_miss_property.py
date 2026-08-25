"""Property-based test for ``CachedGraph.ainvoke`` on a cache miss.

Exercises the async whole-run cache path in
``memory_reuse.integrations.langgraph`` (Task 4 of the graph-level-cache spec).
Uses the deterministic stub compiled graph (with a per-node execution counter)
from ``tests/conftest.py``; no real LLM or network is involved.

* **Property 2 — Miss returns the real run result:** for any initial input
  state with an empty cache, the first invocation returns a ``Final_Result``
  equal to the result of executing the wrapped graph directly on that state.
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
# Property 2: Miss returns the real run result
# ---------------------------------------------------------------------------


class TestProperty2MissReturnsRealRunResult:
    """Feature: graph-level-cache, Property 2.

    Miss returns the real run result.

    Validates: Requirements 2.3
    """

    @settings(max_examples=100)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_miss_returns_real_run_result(self, state: Any) -> None:
        """A first (miss) invocation returns the wrapped graph's real result.

        The wrapped ``StubGraph`` is deterministic, so an independent reference
        graph run on the same state yields the exact ``Final_Result`` the miss
        must return, and exactly one node runs on the miss.
        """
        # Fresh cache per example so every invocation is a guaranteed miss.
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)

        result = await wrapped.ainvoke(state)

        # Reference: executing the wrapped graph directly on the same state.
        expected = StubGraph()._run(state)
        assert result == expected

        # The miss ran the real graph exactly once (no cached replay).
        assert graph.node_calls == 1
