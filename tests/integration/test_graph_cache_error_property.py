"""Property-based test for ``CachedGraph.ainvoke`` error propagation on a miss.

Exercises the async whole-run cache path in
``memory_reuse.integrations.langgraph`` (Task 4.4 of the graph-level-cache
spec). Uses the ``RaisingGraph`` stub (whose nodes always raise) from
``tests/conftest.py``; no real LLM or network is involved.

* **Property 3 — Errors on a miss propagate and store nothing:** for any
  initial input state, if the wrapped graph raises during a miss run, the
  ``Cached_Graph`` propagates that error and stores no entry, so a subsequent
  invocation with the same state still misses (and re-raises).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from tests.conftest import RaisingGraph

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
# Property 3: Errors on a miss propagate and store nothing
# ---------------------------------------------------------------------------


class TestProperty3ErrorsPropagateAndStoreNothing:
    """Feature: graph-level-cache, Property 3.

    Errors on a miss propagate and store nothing.

    Validates: Requirements 2.5
    """

    @settings(max_examples=100)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_errors_propagate_and_store_nothing(self, state: Any) -> None:
        """A raising graph propagates the error and leaves the cache empty.

        The wrapped ``RaisingGraph`` always raises on execution, so the first
        invocation (a miss) must re-raise the error rather than swallow it, and
        must not store anything. Consequently a second invocation with the same
        state is still a miss: it runs the graph again (incrementing the node
        counter) and re-raises.
        """
        # Fresh cache per example so the first invocation is a guaranteed miss.
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        graph = RaisingGraph()
        wrapped = cache.wrap_graph(graph)

        # First invocation (miss) must propagate the graph's error unchanged.
        with pytest.raises(RuntimeError, match="boom"):
            await wrapped.ainvoke(copy.deepcopy(state))
        assert graph.node_calls == 1

        # Nothing was stored: a second invocation with the same state still
        # misses, so the graph runs again and the error propagates again.
        with pytest.raises(RuntimeError, match="boom"):
            await wrapped.ainvoke(copy.deepcopy(state))
        assert graph.node_calls == 2
