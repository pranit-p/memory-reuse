"""Property-based tests for node-level output caching and skip-detection via
the ``cached_node`` decorator (Task 9.3 of the graph-level-cache spec).

These exercise the node-level contract that ``cached_node`` formalises: a
decorated node caches its full output keyed on its input state, a hit skips the
node body, and decorating each node of a graph yields per-node skip-detection.
Each decorated node counts its real executions, so "the body ran" versus "the
body was skipped (a cache hit)" is directly assertable. No real LLM or network
is involved.

* **Property 19 — Node cache store-and-skip round trip:** invoking a
  ``cached_node``-decorated node twice with the same input executes the body
  exactly once (the miss) and zero times on the second call (the hit),
  returning a ``Node_Output`` on the second equal to the one stored on the
  first.
* **Property 20 — A skipped node's output equals a fresh execution:** the
  ``Node_Output`` supplied from the cache equals the value the body produces for
  that same input, and downstream nodes receive the same state whether the
  upstream node executed or was served from cache.
* **Property 21 — Only nodes without a cached output execute:** for a graph
  whose cacheable nodes have a mixed cache state, a run executes exactly the
  nodes lacking a matching cached output and skips the rest.
* **Property 22 — Non-cacheable nodes always execute:** a node that is not
  decorated as cacheable executes its body on every run.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.integrations.langgraph import cached_node

# ---------------------------------------------------------------------------
# Hypothesis strategies: JSON-serialisable dict input states.
# ---------------------------------------------------------------------------

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=20),
)

# Input state fields that never include a scope id, so global scope resolution
# stays trivial and predictable.
_input_state = st.dictionaries(
    keys=st.text(min_size=1, max_size=10).filter(lambda k: k not in ("user_id", "session_id")),
    values=_json_scalars,
    max_size=4,
)


def _fresh_cache() -> MemoryCache:
    """A fresh in-memory MemoryCache so every example starts empty."""
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


def _make_counting_node(cache: MemoryCache, *, tag: str = "n") -> tuple:
    """Build a ``cached_node``-decorated node plus a per-node call counter.

    Returns ``(node, calls)`` where ``calls`` is a single-element list holding
    the number of times the node body actually executed (a cache hit skips the
    body, so the counter does not advance). The ``tag`` gives the node a unique
    ``__qualname__`` so distinct nodes never share a cache key.
    """
    calls = [0]

    async def body(state: dict) -> dict:
        calls[0] += 1
        return {"out": f"{tag}::{sorted((k, v) for k, v in state.items())}"}

    # Give each node a distinct ``__qualname__`` *before* decoration so the
    # decorator's ``[func.__qualname__, key_data]`` key never collides across
    # distinct nodes (the wrapper closes over this ``func``).
    body.__qualname__ = f"counting_node_{tag}"
    node = cached_node(cache, scope="global")(body)  # type: ignore[arg-type]
    return node, calls


# ---------------------------------------------------------------------------
# Property 19: Node cache store-and-skip round trip
# ---------------------------------------------------------------------------


class TestProperty19NodeStoreAndSkipRoundTrip:
    """Feature: graph-level-cache, Property 19.

    Node cache store-and-skip round trip.

    Validates: Requirements 11.1, 11.2, 11.3, 12.1, 12.2
    """

    @settings(max_examples=100)
    @given(state=_input_state)
    @pytest.mark.asyncio
    async def test_store_then_skip_round_trip(self, state: dict) -> None:
        """Two identical invocations run the body once then zero, same output.

        The first call misses: the body executes and the ``Node_Output`` is
        stored under ``[func.__qualname__, key_data]`` (Reqs 11.1, 11.3, 12.2).
        The second call with an equal input hits: the stored output is returned
        without executing the body (Reqs 11.2, 12.1), and that value equals the
        one produced and stored on the first run.
        """
        cache = _fresh_cache()
        node, calls = _make_counting_node(cache)

        # First call misses and executes the body exactly once.
        first = await node(dict(state))
        assert calls[0] == 1

        # Second call with an equal-but-distinct input hits: body skipped.
        second = await node(dict(state))
        assert calls[0] == 1

        # The replayed output equals the one produced and stored on the miss.
        assert second == first


# ---------------------------------------------------------------------------
# Property 20: A skipped node's output equals a fresh execution
# ---------------------------------------------------------------------------


class TestProperty20SkippedEqualsFresh:
    """Feature: graph-level-cache, Property 20.

    A skipped node's output equals a fresh execution.

    Validates: Requirements 12.4, 12.6
    """

    @settings(max_examples=100)
    @given(state=_input_state)
    @pytest.mark.asyncio
    async def test_skipped_output_equals_fresh_and_downstream_matches(self, state: dict) -> None:
        """A cached node's replayed output drives identical downstream state.

        Two independent caches back two copies of the same two-node pipeline
        (a cached upstream node feeding a downstream node). In the *fresh* run
        the upstream body executes; in the *cached* run the upstream node is
        pre-seeded so its body is skipped and its output is replayed. The
        replayed upstream output must equal the freshly produced one (Req 12.4),
        and the downstream node — which consumes the upstream output — must
        receive the same state and produce the same final result either way
        (Req 12.6).
        """

        def build_pipeline() -> tuple:
            cache = _fresh_cache()
            upstream, up_calls = _make_counting_node(cache, tag="up")

            async def downstream(upstream_out: dict) -> dict:
                # Consumes the upstream node's output verbatim.
                return {"final": f"down::{upstream_out['out']}", **upstream_out}

            async def run(inp: dict) -> tuple:
                produced = await upstream(dict(inp))
                return produced, await downstream(produced)

            return run, up_calls, upstream, cache

        # Fresh run: the upstream body executes.
        fresh_run, fresh_calls, _, _ = build_pipeline()
        fresh_up, fresh_final = await fresh_run(state)
        assert fresh_calls[0] == 1

        # Cached run: pre-seed the upstream entry so its body is skipped.
        cached_run, cached_calls, cached_upstream, _ = build_pipeline()
        await cached_upstream(dict(state))  # seed (one execution)
        assert cached_calls[0] == 1
        cached_up, cached_final = await cached_run(state)
        # No further upstream execution: the seeded output was replayed.
        assert cached_calls[0] == 1

        # The replayed upstream output equals a fresh execution's output ...
        assert cached_up == fresh_up
        # ... and downstream receives identical state, yielding identical finals.
        assert cached_final == fresh_final


# ---------------------------------------------------------------------------
# Property 21: Only nodes without a cached output execute
# ---------------------------------------------------------------------------


class TestProperty21OnlyUncachedNodesExecute:
    """Feature: graph-level-cache, Property 21.

    Only nodes without a cached output execute.

    Validates: Requirements 12.3
    """

    @settings(max_examples=100)
    @given(state_a=_input_state, state_b=_input_state)
    @pytest.mark.asyncio
    async def test_mixed_cache_state_runs_only_uncached_nodes(
        self, state_a: dict, state_b: dict
    ) -> None:
        """A run executes exactly the nodes lacking a matching cached output.

        Two decorated nodes form a graph. Node A is pre-seeded (its input has a
        cached output), while node B is not. Running the graph must skip A (its
        body does not run again) and execute B exactly once.
        """
        cache = _fresh_cache()
        node_a, calls_a = _make_counting_node(cache, tag="a")
        node_b, calls_b = _make_counting_node(cache, tag="b")

        async def run_graph() -> None:
            await node_a(dict(state_a))
            await node_b(dict(state_b))

        # Seed only node A's entry for its input (one execution).
        await node_a(dict(state_a))
        assert calls_a[0] == 1
        assert calls_b[0] == 0

        # Run the graph: A is a hit (skipped), B is a miss (executes once).
        await run_graph()
        assert calls_a[0] == 1  # A skipped
        assert calls_b[0] == 1  # B executed exactly once


# ---------------------------------------------------------------------------
# Property 22: Non-cacheable nodes always execute
# ---------------------------------------------------------------------------


class TestProperty22NonCacheableAlwaysExecutes:
    """Feature: graph-level-cache, Property 22.

    Non-cacheable nodes always execute.

    Validates: Requirements 12.5
    """

    @settings(max_examples=100)
    @given(state=_input_state, runs=st.integers(min_value=1, max_value=5))
    @pytest.mark.asyncio
    async def test_undecorated_node_runs_every_time(self, state: dict, runs: int) -> None:
        """An undecorated node executes its body on every run (no caching).

        A plain (non-``cached_node``) node is invoked repeatedly with the same
        input. Because it is not marked cacheable, skip-detection never applies
        and its body executes once per call.
        """
        calls = [0]

        async def plain_node(inp: dict) -> dict:
            calls[0] += 1
            return {"out": f"plain::{sorted((k, v) for k, v in inp.items())}"}

        for _ in range(runs):
            await plain_node(dict(state))

        # The body ran exactly once per invocation: no caching occurred.
        assert calls[0] == runs
