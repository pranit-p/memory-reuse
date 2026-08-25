"""Property-based tests for node-level cache invalidation on
:meth:`memory_reuse.core.MemoryCache.invalidate_node` and its interaction with
the ``cached_node`` decorator (Task 9.4 of the graph-level-cache spec).

Each decorated node counts its real executions, so "the body ran" versus "the
body was skipped (a cache hit)" is directly assertable. No real LLM or network
is involved.

* **Property 15 — Node invalidation is scope-isolated:** invalidating a cached
  node output in one scope leaves another scope's entry for the same node and
  input intact (still a hit).
* **Property 23 — Invalidation forces the next run to execute:** after a node's
  cached output is invalidated, the next call with the same input is a miss and
  executes the node body.
* **Property 24 — Invalidation is safe and idempotent:** invalidating an entry
  that does not exist completes without raising, and invalidating the same entry
  twice has the same effect as invalidating it once.
"""

from __future__ import annotations

from typing import Literal

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.integrations.langgraph import cached_node

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=20),
)

# Input state fields that never include a scope id, so scope resolution stays
# under the tests' explicit control (``user_id`` / ``session_id`` are injected).
_input_state = st.dictionaries(
    keys=st.text(min_size=1, max_size=10).filter(lambda k: k not in ("user_id", "session_id")),
    values=_json_scalars,
    max_size=4,
)

_scope_ids = st.text(min_size=1, max_size=12)

_NonGlobalScope = tuple[Literal["user", "session"], str]
_non_global_scopes: st.SearchStrategy[_NonGlobalScope] = st.sampled_from(
    [("user", "user_id"), ("session", "session_id")]
)


def _fresh_cache() -> MemoryCache:
    """A fresh in-memory MemoryCache so every example starts empty."""
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


def _make_counting_node(cache: MemoryCache, *, scope: str) -> tuple:
    """Build a ``cached_node``-decorated node plus a per-node call counter.

    Returns ``(node, calls)`` where ``calls`` is a single-element list holding
    the number of times the node body actually executed (a cache hit skips the
    body, so the counter does not advance).
    """
    calls = [0]

    @cached_node(cache, scope=scope)  # type: ignore[arg-type]
    async def node(state: dict) -> dict:
        calls[0] += 1
        return {"out": f"result::{sorted((k, v) for k, v in state.items())}"}

    return node, calls


# ---------------------------------------------------------------------------
# Property 15: Node invalidation is scope-isolated
# ---------------------------------------------------------------------------


class TestProperty15NodeInvalidationScopeIsolated:
    """Feature: graph-level-cache, Property 15.

    Node invalidation is scope-isolated.

    Validates: Requirements 13.3
    """

    @settings(max_examples=100)
    @given(
        state=_input_state,
        id_a=_scope_ids,
        id_b=_scope_ids,
        sc=_non_global_scopes,
    )
    @pytest.mark.asyncio
    async def test_invalidating_one_scope_leaves_other_intact(
        self, state: dict, id_a: str, id_b: str, sc: _NonGlobalScope
    ) -> None:
        """Invalidating scope A's entry leaves scope B's entry a hit."""
        assume(id_a != id_b)
        scope, field = sc
        cache = _fresh_cache()
        node, calls = _make_counting_node(cache, scope=scope)

        state_a = {**state, field: id_a}
        state_b = {**state, field: id_b}

        # Seed both scopes (each a miss that executes the body once).
        await node(state_a)
        await node(state_b)
        assert calls[0] == 2

        # Invalidate only scope A's entry.
        await cache.invalidate_node(node, state_a, scope=scope, scope_id=id_a)

        # Scope B is untouched: replaying it is a hit (body does not run).
        await node(state_b)
        assert calls[0] == 2

        # Scope A now misses and re-executes the body.
        await node(state_a)
        assert calls[0] == 3


# ---------------------------------------------------------------------------
# Property 23: Invalidation forces the next run to execute
# ---------------------------------------------------------------------------


class TestProperty23InvalidationForcesExecute:
    """Feature: graph-level-cache, Property 23.

    Invalidation forces the next run to execute.

    Validates: Requirements 13.2
    """

    @settings(max_examples=100)
    @given(state=_input_state)
    @pytest.mark.asyncio
    async def test_next_run_after_invalidation_is_a_miss(self, state: dict) -> None:
        """After invalidation the same input misses and runs the body again."""
        cache = _fresh_cache()
        node, calls = _make_counting_node(cache, scope="global")

        # First call misses and executes; second call hits and skips.
        await node(dict(state))
        assert calls[0] == 1
        await node(dict(state))
        assert calls[0] == 1

        # Invalidate the cached output for this input.
        await cache.invalidate_node(node, dict(state))

        # The next call with the same input is a miss and executes again.
        await node(dict(state))
        assert calls[0] == 2


# ---------------------------------------------------------------------------
# Property 24: Invalidation is safe and idempotent
# ---------------------------------------------------------------------------


class TestProperty24InvalidationSafeAndIdempotent:
    """Feature: graph-level-cache, Property 24.

    Invalidation is safe and idempotent.

    Validates: Requirements 13.6
    """

    @settings(max_examples=100)
    @given(node_id=st.text(min_size=1, max_size=20), state=_input_state)
    @pytest.mark.asyncio
    async def test_invalidating_missing_entry_is_safe(self, node_id: str, state: dict) -> None:
        """Invalidating an entry that does not exist completes without raising."""
        cache = _fresh_cache()
        # No entry was ever stored: both calls must complete without error.
        await cache.invalidate_node(node_id, dict(state))
        await cache.invalidate_node(node_id, dict(state))

    @settings(max_examples=100)
    @given(state=_input_state)
    @pytest.mark.asyncio
    async def test_double_invalidation_matches_single(self, state: dict) -> None:
        """Invalidating twice has the same effect as invalidating once."""
        cache = _fresh_cache()
        node, calls = _make_counting_node(cache, scope="global")

        # Seed an entry (one execution) and confirm a subsequent hit.
        await node(dict(state))
        assert calls[0] == 1
        await node(dict(state))
        assert calls[0] == 1

        # Invalidate the same entry twice.
        await cache.invalidate_node(node, dict(state))
        await cache.invalidate_node(node, dict(state))

        # The entry is gone exactly once: the next call misses and re-executes,
        # then the following call hits again (state restored by the miss store).
        await node(dict(state))
        assert calls[0] == 2
        await node(dict(state))
        assert calls[0] == 2
