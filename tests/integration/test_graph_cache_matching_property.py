"""Property-based tests for ``CachedGraph.ainvoke`` exact/semantic matching.

Exercises the async whole-run cache path in
``memory_reuse.integrations.langgraph`` (Task 4.5 of the graph-level-cache
spec). Uses the deterministic stub compiled graph (with a per-node execution
counter) and the deterministic stub embedder (with an embed-call counter) from
``tests/conftest.py``; no real LLM or network is involved.

* **Property 7 — Exact hits and exact-only lookups never embed:** an exact hit
  returns without invoking the embedder, and an ``exact_only`` wrapper never
  invokes the embedder even on a miss.
* **Property 8 — Semantic matches reuse a stored run:** a stored run is served
  as a hit for a semantically-close variant whose similarity meets the
  effective threshold.
* **Property 9 — Semantic-disabled behaves exactly like the exact path:** a
  ``semantic=False`` wrapper produces the same hit/miss outcomes and returned
  values as a reference model that uses only the exact cache.
* **Property 10 — Wrapper threshold overrides the config default:** the
  ``similarity_threshold`` supplied to ``wrap_graph`` is the effective
  threshold for the wrapper's semantic lookups; when none is supplied the
  ``CacheConfig`` default applies.
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


# ---------------------------------------------------------------------------
# Property 7: Exact hits and exact-only lookups never embed
# ---------------------------------------------------------------------------


class TestProperty7ExactNeverEmbeds:
    """Feature: graph-level-cache, Property 7.

    Exact hits and exact-only lookups never embed.

    Validates: Requirements 4.1, 4.2, 6.5
    """

    @settings(max_examples=100, deadline=None)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_exact_hit_never_embeds(self, state: Any) -> None:
        """A second (exact-hit) invocation computes no embedding (Req 4.1, 4.2).

        Even with ``semantic=True``, the exact-first path must short-circuit the
        embedder on a hit: the embed counter is unchanged across the hit.
        """
        embedder = StubEmbedder()
        cache = make_semantic_cache(embedder, similarity_threshold=0.0)
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, semantic=True)

        # First call is a miss: runs the graph and stores.
        first = await wrapped.ainvoke(state)
        assert graph.node_calls == 1
        embed_after_miss = embedder.embed_calls

        # Second call with an equal state is an exact hit: zero nodes run and
        # no further embedding is computed.
        second = await wrapped.ainvoke(_copy_state(state))
        assert graph.node_calls == 1
        assert second == first
        assert embedder.embed_calls == embed_after_miss

    @settings(max_examples=100, deadline=None)
    @given(state=_states)
    @pytest.mark.asyncio
    async def test_exact_only_never_embeds_even_on_miss(self, state: Any) -> None:
        """An ``exact_only`` wrapper never embeds, even on a miss (Req 6.5).

        Semantic matching is enabled on the cache, but the wrapper forces
        exact-only lookups/stores, so the embedder is never touched across a
        miss followed by an exact hit.
        """
        embedder = StubEmbedder()
        cache = make_semantic_cache(embedder, similarity_threshold=0.0)
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, semantic=True, exact_only=True)

        await wrapped.ainvoke(state)  # miss
        await wrapped.ainvoke(_copy_state(state))  # exact hit

        assert graph.node_calls == 1  # exactly one real run
        assert embedder.embed_calls == 0  # never embedded


# ---------------------------------------------------------------------------
# Property 8: Semantic matches reuse a stored run
# ---------------------------------------------------------------------------


class TestProperty8SemanticReuse:
    """Feature: graph-level-cache, Property 8.

    Semantic matches reuse a stored run.

    Validates: Requirements 4.3, 4.4
    """

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    @pytest.mark.asyncio
    async def test_semantic_variant_reuses_stored_run(self, data: st.DataObject) -> None:
        """A semantically-close variant hits the stored run (Req 4.3, 4.4).

        The stored and variant states have *distinct* ``q`` values, so their
        exact keys differ and the second lookup must take the semantic path.
        With a permissive effective threshold (0.0) any stored vector qualifies
        as a match, so the variant returns the stored ``Final_Result`` as a
        semantic hit with zero further nodes run.
        """
        stored_text = data.draw(st.text(min_size=1, max_size=20))
        # Draw a variant guaranteed to differ, so the exact keys never collide
        # (an equal ``q`` would be an exact hit rather than a semantic one).
        variant_text = data.draw(
            st.text(min_size=1, max_size=20).filter(lambda t: t != stored_text)
        )

        embedder = StubEmbedder()
        cache = make_semantic_cache(embedder, similarity_threshold=0.0)
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, semantic=True, key_fields=["q"])

        first = await wrapped.ainvoke({"q": stored_text})
        assert graph.node_calls == 1

        # Distinct exact key -> exact miss -> semantic path -> match at 0.0.
        second = await wrapped.ainvoke({"q": variant_text})

        # Reused the stored run: no new node executed, the stored result came
        # back, and the hit was recorded as semantic (not exact).
        assert graph.node_calls == 1
        assert second == first
        assert cache.stats.semantic_hits >= 1


# ---------------------------------------------------------------------------
# Property 9: Semantic-disabled behaves exactly like the exact path
# ---------------------------------------------------------------------------


class TestProperty9DisabledEqualsExact:
    """Feature: graph-level-cache, Property 9.

    Semantic-disabled behaves exactly like the exact path.

    Validates: Requirements 4.5, 14.5
    """

    @settings(max_examples=100)
    @given(states=st.lists(_states, min_size=1, max_size=6))
    @pytest.mark.asyncio
    async def test_disabled_matches_exact_reference_model(self, states: list[Any]) -> None:
        """A ``semantic=False`` wrapper matches an exact-only reference model.

        For each state in the sequence the wrapper's hit/miss outcome (observed
        via the node-execution counter) and returned value must equal a
        reference model driven purely by ``cache.exact`` on the same derived
        key parts.
        """
        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        assert cache.semantic is None  # semantic machinery not even built
        graph = StubGraph()
        wrapped = cache.wrap_graph(graph, semantic=False)

        # Reference model: an independent exact cache keyed identically.
        ref_cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))
        ref_graph = StubGraph()

        for state in states:
            calls_before = graph.node_calls
            result = await wrapped.ainvoke(state)

            key_parts, _ = _derive(state, ref_graph)
            ref_hit = await ref_cache.exact.get(key_parts, scope="global", scope_id=None)
            if ref_hit is None:
                # Reference miss: run and store, then compare outcome.
                ref_result = ref_graph._run(state)
                await ref_cache.exact.set(key_parts, ref_result, scope="global", scope_id=None)
                # Wrapper must also have run exactly one node (a miss).
                assert graph.node_calls == calls_before + 1
                assert result == ref_result
            else:
                # Reference hit: wrapper must have run zero nodes and returned
                # the same stored value.
                assert graph.node_calls == calls_before
                assert result == ref_hit


# ---------------------------------------------------------------------------
# Property 10: Wrapper threshold overrides the config default
# ---------------------------------------------------------------------------


class TestProperty10ThresholdOverride:
    """Feature: graph-level-cache, Property 10.

    Wrapper threshold overrides the config default.

    Validates: Requirements 4.6
    """

    @settings(max_examples=100, deadline=None)
    @given(override=st.floats(min_value=0.0, max_value=1.0))
    @pytest.mark.asyncio
    async def test_wrapper_threshold_is_effective(self, override: float) -> None:
        """The wrapper threshold reaches the semantic lookup (Req 4.6).

        A spying ``lookup`` captures the ``threshold`` forwarded by the wrapper.
        When a ``similarity_threshold`` is supplied to ``wrap_graph`` the
        forwarded value equals that override; the config default (0.95) is used
        only when no override is supplied.
        """
        embedder = StubEmbedder()
        cache = make_semantic_cache(embedder, similarity_threshold=0.95)

        captured: dict[str, float | None] = {}
        real_lookup = cache.lookup

        async def spy_lookup(*args: Any, **kwargs: Any) -> Any:
            captured["threshold"] = kwargs.get("threshold")
            return await real_lookup(*args, **kwargs)

        cache.lookup = spy_lookup  # type: ignore[method-assign]

        # With an explicit override the forwarded threshold is the override.
        with_override = cache.wrap_graph(StubGraph(), semantic=True, similarity_threshold=override)
        await with_override.ainvoke({"q": "x"})
        assert captured["threshold"] == override

        # Without an override the wrapper forwards ``None`` so the config
        # default (0.95) applies inside the semantic cache.
        no_override = cache.wrap_graph(StubGraph(), semantic=True)
        await no_override.ainvoke({"q": "y"})
        assert captured["threshold"] is None


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


def _derive(state: Any, graph: StubGraph) -> tuple[list, str]:
    """Derive the same whole-run key parts the wrapper uses (no key_fields)."""
    from memory_reuse.integrations.langgraph import _derive_graph_key, _resolve_graph_id

    return _derive_graph_key(state, None, _resolve_graph_id(graph, None))
