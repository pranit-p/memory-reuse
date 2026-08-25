"""Property-based tests for ``CachedGraph.ainvoke`` bypass/no-store and stats.

Exercises the async whole-run cache path in
``memory_reuse.integrations.langgraph`` (Task 4.7 of the graph-level-cache
spec). Uses the deterministic stub compiled graph (with a per-node execution
counter) and the deterministic stub embedder (with an embed-call counter) from
``tests/conftest.py``; no real LLM or network is involved.

* **Property 14 — Bypass always runs and no-store never stores:** a
  bypass-lookup invocation executes the wrapped graph and does not return the
  cached ``Final_Result``; a no-store invocation executes the graph and leaves
  the cache unchanged, so a following normal invocation still misses.
* **Property 17 — Statistics account hits and misses exactly:** a graph-level
  exact hit increments ``exact_hits`` and ``hits`` by exactly one, a semantic
  hit increments ``semantic_hits`` and ``hits`` by exactly one (distinct from
  exact), and a miss increments ``misses`` by exactly one.
* **Property 18 — Statistics failures never fail the invocation:** if
  statistics recording raises internally, the ``CachedGraph`` still returns the
  correct ``Final_Result``.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from tests.conftest import StubEmbedder, StubGraph, make_semantic_cache

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


def _fresh_cache() -> MemoryCache:
    """A fresh in-memory MemoryCache so every example starts empty."""
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


def _copy_state(state: Any) -> Any:
    """Return an equal-but-distinct copy of a state (dicts/lists copied)."""
    if isinstance(state, dict):
        return dict(state)
    if isinstance(state, list):
        return list(state)
    return state


# ---------------------------------------------------------------------------
# Property 14: Bypass always runs and no-store never stores
# ---------------------------------------------------------------------------


class TestProperty14BypassAndNoStore:
    """Feature: graph-level-cache, Property 14.

    Bypass always runs and no-store never stores.

    Validates: Requirements 6.2, 6.3
    """

    @settings(max_examples=100, deadline=None)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_bypass_runs_and_ignores_cached_result(self, state: Any) -> None:
        """A bypass-lookup invocation runs the graph over a seeded cache (Req 6.2).

        With the cache pre-seeded for the state (a normal invocation stored the
        result), a subsequent ``bypass_cache=True`` invocation must execute the
        wrapped graph again rather than returning the cached ``Final_Result``.
        """
        cache = _fresh_cache()
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)

        # Seed the cache with a normal (miss) invocation.
        await wrapped.ainvoke(state)
        assert graph.node_calls == 1

        # A bypass-lookup invocation runs the graph again despite the seed.
        await wrapped.ainvoke(_copy_state(state), bypass_cache=True)
        assert graph.node_calls == 2

    @settings(max_examples=100, deadline=None)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_no_store_leaves_cache_unchanged(self, state: Any) -> None:
        """A no-store invocation runs but stores nothing (Req 6.3).

        Starting from an empty cache, a ``no_store=True`` invocation executes
        the graph but writes no entry, so a following *normal* invocation with
        the same state still misses and runs the graph a second time.
        """
        cache = _fresh_cache()
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)

        # no-store miss run: executes the graph, stores nothing.
        first = await wrapped.ainvoke(state, no_store=True)
        assert graph.node_calls == 1

        # A subsequent normal invocation still misses (nothing was stored).
        second = await wrapped.ainvoke(_copy_state(state))
        assert graph.node_calls == 2
        assert second == first

        # And that normal miss *did* store, so a third normal call now hits.
        await wrapped.ainvoke(_copy_state(state))
        assert graph.node_calls == 2


# ---------------------------------------------------------------------------
# Property 17: Statistics account hits and misses exactly
# ---------------------------------------------------------------------------


class TestProperty17StatsExact:
    """Feature: graph-level-cache, Property 17.

    Statistics account hits and misses exactly.

    Validates: Requirements 9.1, 9.2, 9.3, 12.7
    """

    @settings(max_examples=100, deadline=None)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_exact_hit_increments_exact_and_hits_by_one(self, state: Any) -> None:
        """A graph-level exact hit bumps ``exact_hits`` and ``hits`` by one (Req 9.1)."""
        cache = _fresh_cache()  # exact-only (semantic disabled)
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)

        # Seed the entry with a miss run so the next call is an exact hit.
        await wrapped.ainvoke(state)

        before = cache.stats
        await wrapped.ainvoke(_copy_state(state))  # exact hit
        after = cache.stats

        assert after.exact_hits - before.exact_hits == 1
        assert after.hits - before.hits == 1
        # An exact hit records neither a semantic hit nor a miss.
        assert after.semantic_hits - before.semantic_hits == 0
        assert after.misses - before.misses == 0

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    @pytest.mark.asyncio
    async def test_semantic_hit_increments_semantic_and_hits_by_one(
        self, data: st.DataObject
    ) -> None:
        """A graph-level semantic hit bumps ``semantic_hits`` and ``hits`` by one (Req 9.2).

        The stored and variant states differ in their ``q`` value, so the exact
        keys never collide and the variant lookup takes the semantic path. With
        a permissive effective threshold (0.0) the stored vector matches, so the
        variant is a semantic hit that is distinct from an exact hit.
        """
        stored_text = data.draw(st.text(min_size=1, max_size=20))
        variant_text = data.draw(
            st.text(min_size=1, max_size=20).filter(lambda t: t != stored_text)
        )

        embedder = StubEmbedder()
        cache = make_semantic_cache(embedder, similarity_threshold=0.0)
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, semantic=True, key_fields=["q"])

        # Seed the stored run.
        await wrapped.ainvoke({"q": stored_text})

        before = cache.stats
        await wrapped.ainvoke({"q": variant_text})  # semantic hit
        after = cache.stats

        assert after.semantic_hits - before.semantic_hits == 1
        assert after.hits - before.hits == 1
        # A semantic hit is distinct from an exact hit.
        assert after.exact_hits - before.exact_hits == 0

    @settings(max_examples=100, deadline=None)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_miss_increments_misses_by_one(self, state: Any) -> None:
        """A graph-level miss bumps ``misses`` by exactly one (Req 9.3).

        Uses an exact-only (semantic-disabled) wrapper so the miss consults only
        the exact cache: a single ``record_miss`` and no hit of any kind.
        """
        cache = _fresh_cache()  # exact-only (semantic disabled)
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)

        before = cache.stats
        await wrapped.ainvoke(state)  # first-time miss
        after = cache.stats

        assert after.misses - before.misses == 1
        assert after.hits - before.hits == 0
        assert after.exact_hits - before.exact_hits == 0
        assert after.semantic_hits - before.semantic_hits == 0


# ---------------------------------------------------------------------------
# Property 18: Statistics failures never fail the invocation
# ---------------------------------------------------------------------------


class TestProperty18StatsFailuresNeverFail:
    """Feature: graph-level-cache, Property 18.

    Statistics failures never fail the invocation.

    Validates: Requirements 9.4
    """

    @settings(max_examples=100, deadline=None)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_recording_error_does_not_fail_invocation(self, state: Any) -> None:
        """An internal stats-recording failure never fails the invocation (Req 9.4).

        The tracker's counter mutations are forced to raise from *inside* the
        ``record_*`` methods (the point at which the tracker's best-effort
        suppression is meant to protect the caller). Both the miss run and the
        subsequent (would-be hit) invocation must still return the correct
        ``Final_Result``.
        """
        cache = _fresh_cache()  # fresh per example, so patching is self-contained
        tracker = cache._stats_tracker

        # Replace each internal counter with an object that raises on ``+= 1``.
        # The mutation happens inside ``contextlib.suppress(Exception)`` in every
        # ``record_*`` method, so a real recording failure is exercised end to
        # end while the tracker's suppression keeps the caller unaffected.
        class _ExplodingCounter:
            def __add__(self, _other: object) -> _ExplodingCounter:
                raise RuntimeError("stats exploded")

        for attr in ("_hits", "_exact_hits", "_semantic_hits", "_misses", "_errors", "_total"):
            setattr(tracker, attr, _ExplodingCounter())

        reference = StubGraph()._run(_copy_state(state))

        graph = StubGraph()
        wrapped = cache.wrap_graph(graph)

        # Miss run: stats recording raises internally but is swallowed.
        first = await wrapped.ainvoke(state)
        assert first == reference

        # Would-be exact hit: recording raises again but the stored value is
        # still returned correctly.
        second = await wrapped.ainvoke(_copy_state(state))
        assert second == reference
