"""Property-based tests for scope isolation and resolution on the graph-level
whole-run cache (``CachedGraph.ainvoke``).

Exercises the async whole-run cache path in
``memory_reuse.integrations.langgraph`` (Task 4.6 of the graph-level-cache
spec). Uses the deterministic stub compiled graph (with a per-node execution
counter) from ``tests/conftest.py``; no real LLM or network is involved.

* **Property 11 — Scope isolation of graph-level entries:** a ``Final_Result``
  stored under one scope id is a miss under a different scope id, and a global
  lookup never matches a user- or session-scoped entry.
* **Property 12 — Scope id resolves from state before context:** the resolved
  scope id equals the state-supplied id when present, otherwise the
  context-supplied id.
* **Property 13 — Unresolvable non-global scope raises ScopeViolationError:**
  a non-global scope with no scope id in the state or the MemoryCache context
  raises :exc:`~memory_reuse.exceptions.ScopeViolationError`.
"""

from __future__ import annotations

from typing import Literal

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.exceptions import ScopeViolationError
from tests.conftest import StubGraph

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=20),
)

# Base state fields that never include a scope id (``user_id`` / ``session_id``
# are added explicitly by the tests) so scope resolution stays under control.
_base_state = st.dictionaries(
    keys=st.text(min_size=1, max_size=10).filter(lambda k: k not in ("user_id", "session_id")),
    values=_json_scalars,
    max_size=4,
)

_scope_ids = st.text(min_size=1, max_size=12)

# A non-global scope paired with the matching state field name.
_NonGlobalScope = tuple[Literal["user", "session"], str]
_non_global_scopes: st.SearchStrategy[_NonGlobalScope] = st.sampled_from(
    [("user", "user_id"), ("session", "session_id")]
)


def _fresh_cache() -> MemoryCache:
    """A fresh in-memory MemoryCache so every example starts empty."""
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


# ---------------------------------------------------------------------------
# Property 11: Scope isolation of graph-level entries
# ---------------------------------------------------------------------------


class TestProperty11ScopeIsolation:
    """Feature: graph-level-cache, Property 11.

    Scope isolation of graph-level entries.

    Validates: Requirements 5.1, 5.2, 5.3, 11.7
    """

    @settings(max_examples=100)
    @given(base=_base_state, id_a=_scope_ids, id_b=_scope_ids, sc=_non_global_scopes)
    @pytest.mark.asyncio
    async def test_distinct_scope_ids_never_share_entry(
        self, base: dict, id_a: str, id_b: str, sc: _NonGlobalScope
    ) -> None:
        """A run stored under one scope id misses under a different scope id."""
        assume(id_a != id_b)
        scope, field = sc
        cache = _fresh_cache()
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, scope=scope)

        # Store under scope id A.
        await wrapped.ainvoke({**base, field: id_a})
        assert graph.node_calls == 1

        # A run with the same state but a different scope id is a miss.
        await wrapped.ainvoke({**base, field: id_b})
        assert graph.node_calls == 2

        # And re-running under id A hits the original entry (zero further runs).
        await wrapped.ainvoke({**base, field: id_a})
        assert graph.node_calls == 2

    @settings(max_examples=100)
    @given(base=_base_state, scope_id=_scope_ids, sc=_non_global_scopes)
    @pytest.mark.asyncio
    async def test_global_never_matches_scoped_entry(
        self, base: dict, scope_id: str, sc: _NonGlobalScope
    ) -> None:
        """A global lookup never matches a user/session-scoped entry."""
        scope, field = sc
        cache = _fresh_cache()

        scoped_graph = StubGraph()
        scoped = cache.wrap_graph(scoped_graph, scope=scope)
        await scoped.ainvoke({**base, field: scope_id})
        assert scoped_graph.node_calls == 1

        # A global wrapper over the identical state must miss (scoped entry is
        # invisible to the global scope) and run its own graph.
        global_graph = StubGraph()
        wrapped_global = cache.wrap_graph(global_graph, scope="global")
        await wrapped_global.ainvoke({**base, field: scope_id})
        assert global_graph.node_calls == 1


# ---------------------------------------------------------------------------
# Property 12: Scope id resolves from state before context
# ---------------------------------------------------------------------------


class TestProperty12ScopeIdResolution:
    """Feature: graph-level-cache, Property 12.

    Scope id resolves from state before context.

    Validates: Requirements 5.4
    """

    @settings(max_examples=100)
    @given(
        base=_base_state,
        state_id=_scope_ids,
        context_id=_scope_ids,
        sc=_non_global_scopes,
    )
    @pytest.mark.asyncio
    async def test_state_scope_id_wins_over_context(
        self, base: dict, state_id: str, context_id: str, sc: _NonGlobalScope
    ) -> None:
        """When both are present the state-supplied scope id is used."""
        assume(state_id != context_id)
        scope, field = sc
        cache = _fresh_cache()
        cache.set_context(**{field: context_id})

        # A wrapper scoped by the *state* id stores under ``state_id``.
        state_graph = StubGraph()
        state_wrapped = cache.wrap_graph(state_graph, scope=scope)
        await state_wrapped.ainvoke({**base, field: state_id})
        assert state_graph.node_calls == 1

        # A wrapper relying on the *context* id (no scope field in the state)
        # resolves to ``context_id`` != ``state_id``, so it is a distinct entry
        # and must run its own graph rather than replaying the state entry.
        context_graph = StubGraph()
        context_wrapped = cache.wrap_graph(context_graph, scope=scope)
        await context_wrapped.ainvoke(dict(base))
        assert context_graph.node_calls == 1

    @settings(max_examples=100)
    @given(base=_base_state, context_id=_scope_ids, sc=_non_global_scopes)
    @pytest.mark.asyncio
    async def test_context_scope_id_used_when_state_missing(
        self, base: dict, context_id: str, sc: _NonGlobalScope
    ) -> None:
        """With no state-supplied id the context-supplied id resolves the scope."""
        scope, field = sc
        cache = _fresh_cache()
        cache.set_context(**{field: context_id})
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, scope=scope)

        # First run stores under the context-resolved scope id.
        await wrapped.ainvoke(dict(base))
        assert graph.node_calls == 1

        # A second run with the same context resolves to the same scope id and
        # replays from cache (zero further nodes run).
        await wrapped.ainvoke(dict(base))
        assert graph.node_calls == 1


# ---------------------------------------------------------------------------
# Property 13: Unresolvable non-global scope raises ScopeViolationError
# ---------------------------------------------------------------------------


class TestProperty13UnresolvableScopeRaises:
    """Feature: graph-level-cache, Property 13.

    Unresolvable non-global scope raises ScopeViolationError.

    Validates: Requirements 5.5, 11.7
    """

    @settings(max_examples=100)
    @given(base=_base_state, sc=_non_global_scopes)
    @pytest.mark.asyncio
    async def test_missing_scope_id_raises(self, base: dict, sc: _NonGlobalScope) -> None:
        """A non-global scope with no id in state or context raises."""
        scope, _field = sc
        cache = _fresh_cache()  # empty context
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, scope=scope)

        with pytest.raises(ScopeViolationError):
            await wrapped.ainvoke(dict(base))

        # The graph body never ran because resolution failed before lookup.
        assert graph.node_calls == 0
