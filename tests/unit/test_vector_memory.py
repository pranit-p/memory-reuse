"""Unit and property-based tests for :class:`InMemoryVectorIndex`.

Example-based tests cover add/search behaviour, top-k ordering, namespace
isolation, TTL filtering on read, LRU eviction, and provider/model mismatch.

The property-based tests (hypothesis) exercise the invariants the design calls
out for this component:

* **Property 1 — Scope non-leakage:** a search in namespace ``N`` never returns
  a record stored under any other namespace.
* **Property 3 — Best-match selection:** the top result's score is greater than
  or equal to every other record's score in the namespace.
* **Property 4 — Provider consistency:** a search/add whose ``provider_model``
  differs from the namespace's records raises ``ProviderMismatchError``.
* **Property 5 — Expiry safety:** an expired record is never returned.
* **Property 8 — Determinism of storage key:** re-adding an identical query
  under the same provider+model updates rather than duplicates the record.
"""

from __future__ import annotations

import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from memory_reuse._utils import cosine_similarity, deserialize_value, hash_value, serialize_value
from memory_reuse.exceptions import ProviderMismatchError
from memory_reuse.vector import InMemoryVectorIndex, VectorRecord

PROVIDER = "fake:test-model"


def _record(
    vector: list[float],
    value: object,
    *,
    provider_model: str = PROVIDER,
    expires_at: float | None = None,
) -> VectorRecord:
    """Build a :class:`VectorRecord` with a serialised value."""
    return VectorRecord(
        vector=vector,
        value=serialize_value(value),
        provider_model=provider_model,
        expires_at=expires_at,
    )


def _rid(query: str, provider_model: str = PROVIDER) -> str:
    """Deterministic record id following the documented convention."""
    return hash_value([provider_model, query])


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_max_vectors_raises(self) -> None:
        with pytest.raises(ValueError, match="max_vectors_per_namespace"):
            InMemoryVectorIndex(max_vectors_per_namespace=0)


class TestAddAndSearch:
    @pytest.mark.asyncio
    async def test_add_then_search_returns_value(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("global", _rid("q"), _record([1.0, 0.0], {"answer": 42}))

        matches = await index.search("global", [1.0, 0.0], PROVIDER, top_k=1)

        assert len(matches) == 1
        assert matches[0].score == pytest.approx(1.0)
        assert deserialize_value(matches[0].value) == {"answer": 42}

    @pytest.mark.asyncio
    async def test_search_empty_namespace_returns_empty(self) -> None:
        index = InMemoryVectorIndex()
        assert await index.search("global", [1.0, 0.0], PROVIDER) == []

    @pytest.mark.asyncio
    async def test_search_top_k_zero_returns_empty(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("global", _rid("q"), _record([1.0, 0.0], "v"))
        assert await index.search("global", [1.0, 0.0], PROVIDER, top_k=0) == []

    @pytest.mark.asyncio
    async def test_top_k_ordering_is_descending_by_score(self) -> None:
        index = InMemoryVectorIndex()
        # Query [1, 0]; closest is [1, 0] (1.0), then [1, 1] (~0.85), then [-1, 0] (0.0).
        await index.add("global", _rid("near"), _record([1.0, 0.0], "near"))
        await index.add("global", _rid("mid"), _record([1.0, 1.0], "mid"))
        await index.add("global", _rid("far"), _record([-1.0, 0.0], "far"))

        matches = await index.search("global", [1.0, 0.0], PROVIDER, top_k=3)

        assert [deserialize_value(m.value) for m in matches] == ["near", "mid", "far"]
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_top_k_limits_returned_results(self) -> None:
        index = InMemoryVectorIndex()
        for i in range(5):
            await index.add("global", _rid(f"q{i}"), _record([float(i), 1.0], f"v{i}"))

        matches = await index.search("global", [1.0, 1.0], PROVIDER, top_k=2)
        assert len(matches) == 2


class TestNamespaceIsolation:
    @pytest.mark.asyncio
    async def test_search_never_crosses_namespaces(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("user:alice", _rid("q"), _record([1.0, 0.0], "alice-secret"))
        await index.add("user:bob", _rid("q"), _record([1.0, 0.0], "bob-secret"))
        await index.add("global", _rid("q"), _record([1.0, 0.0], "global-value"))

        alice = await index.search("user:alice", [1.0, 0.0], PROVIDER, top_k=10)
        assert [deserialize_value(m.value) for m in alice] == ["alice-secret"]

        bob = await index.search("user:bob", [1.0, 0.0], PROVIDER, top_k=10)
        assert [deserialize_value(m.value) for m in bob] == ["bob-secret"]

    @pytest.mark.asyncio
    async def test_unknown_namespace_returns_empty(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("user:alice", _rid("q"), _record([1.0], "v"))
        assert await index.search("user:charlie", [1.0], PROVIDER) == []


class TestTTLFiltering:
    @pytest.mark.asyncio
    async def test_expired_record_is_filtered_on_read(self) -> None:
        index = InMemoryVectorIndex()
        await index.add(
            "global",
            _rid("expiring"),
            _record([1.0, 0.0], "gone", expires_at=time.monotonic() - 1.0),
        )
        assert await index.search("global", [1.0, 0.0], PROVIDER) == []

    @pytest.mark.asyncio
    async def test_non_expired_record_is_returned(self) -> None:
        index = InMemoryVectorIndex()
        await index.add(
            "global",
            _rid("live"),
            _record([1.0, 0.0], "here", expires_at=time.monotonic() + 100.0),
        )
        matches = await index.search("global", [1.0, 0.0], PROVIDER)
        assert deserialize_value(matches[0].value) == "here"

    @pytest.mark.asyncio
    async def test_expired_record_is_dropped_from_store(self) -> None:
        index = InMemoryVectorIndex()
        await index.add(
            "global",
            _rid("expiring"),
            _record([1.0], "gone", expires_at=time.monotonic() - 1.0),
        )
        await index.search("global", [1.0], PROVIDER)
        assert index.namespace_size("global") == 0

    @pytest.mark.asyncio
    async def test_none_expiry_never_expires(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("global", _rid("q"), _record([1.0], "forever", expires_at=None))
        matches = await index.search("global", [1.0], PROVIDER)
        assert deserialize_value(matches[0].value) == "forever"


class TestLRUEviction:
    @pytest.mark.asyncio
    async def test_eviction_at_capacity(self) -> None:
        index = InMemoryVectorIndex(max_vectors_per_namespace=3)
        for i in range(3):
            await index.add("global", _rid(f"q{i}"), _record([float(i)], f"v{i}"))
        # Adding a 4th evicts the LRU (q0).
        await index.add("global", _rid("q3"), _record([3.0], "v3"))

        assert index.namespace_size("global") == 3
        matches = await index.search("global", [0.0], PROVIDER, top_k=10)
        values = {deserialize_value(m.value) for m in matches}
        assert "v0" not in values
        assert {"v1", "v2", "v3"} == values

    @pytest.mark.asyncio
    async def test_search_refreshes_lru_ordering(self) -> None:
        index = InMemoryVectorIndex(max_vectors_per_namespace=3)
        await index.add("global", _rid("q0"), _record([1.0], "v0"))
        await index.add("global", _rid("q1"), _record([1.0], "v1"))
        await index.add("global", _rid("q2"), _record([1.0], "v2"))

        # Searching touches all records; then re-add q0 to make it MRU again.
        await index.search("global", [1.0], PROVIDER, top_k=1)
        await index.add("global", _rid("q0"), _record([1.0], "v0-updated"))

        # Add a new record -> LRU (q1) is evicted, q0 survives.
        await index.add("global", _rid("q3"), _record([1.0], "v3"))
        matches = await index.search("global", [1.0], PROVIDER, top_k=10)
        values = {deserialize_value(m.value) for m in matches}
        assert "v1" not in values
        assert "v0-updated" in values

    @pytest.mark.asyncio
    async def test_eviction_is_per_namespace(self) -> None:
        index = InMemoryVectorIndex(max_vectors_per_namespace=2)
        await index.add("user:a", _rid("q0"), _record([1.0], "a0"))
        await index.add("user:a", _rid("q1"), _record([1.0], "a1"))
        await index.add("user:b", _rid("q0"), _record([1.0], "b0"))
        # user:a at capacity but user:b unaffected.
        assert index.namespace_size("user:a") == 2
        assert index.namespace_size("user:b") == 1


class TestProviderMismatch:
    @pytest.mark.asyncio
    async def test_add_with_mismatched_provider_raises(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("global", _rid("q0"), _record([1.0], "v0", provider_model="openai:a"))
        with pytest.raises(ProviderMismatchError):
            await index.add(
                "global",
                _rid("q1"),
                _record([1.0], "v1", provider_model="local:b"),
            )

    @pytest.mark.asyncio
    async def test_search_with_mismatched_provider_raises(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("global", _rid("q0"), _record([1.0], "v0", provider_model="openai:a"))
        with pytest.raises(ProviderMismatchError):
            await index.search("global", [1.0], "local:b")

    @pytest.mark.asyncio
    async def test_overwrite_same_record_id_same_provider_ok(self) -> None:
        index = InMemoryVectorIndex()
        rid = _rid("q")
        await index.add("global", rid, _record([1.0], "v0"))
        await index.add("global", rid, _record([1.0], "v1"))
        assert index.namespace_size("global") == 1

    @pytest.mark.asyncio
    async def test_mismatch_ignores_expired_records(self) -> None:
        index = InMemoryVectorIndex()
        # Only record is expired; its provider identity should not block a new one.
        await index.add(
            "global",
            _rid("old"),
            _record([1.0], "old", provider_model="openai:a", expires_at=time.monotonic() - 1.0),
        )
        await index.add(
            "global",
            _rid("new"),
            _record([1.0], "new", provider_model="local:b"),
        )
        matches = await index.search("global", [1.0], "local:b")
        assert deserialize_value(matches[0].value) == "new"


class TestDeleteAndFlush:
    @pytest.mark.asyncio
    async def test_delete_namespace(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("user:a", _rid("q"), _record([1.0], "v"))
        await index.delete_namespace("user:a")
        assert await index.search("user:a", [1.0], PROVIDER) == []

    @pytest.mark.asyncio
    async def test_delete_unknown_namespace_no_error(self) -> None:
        index = InMemoryVectorIndex()
        await index.delete_namespace("nope")  # should not raise

    @pytest.mark.asyncio
    async def test_flush_clears_all_namespaces(self) -> None:
        index = InMemoryVectorIndex()
        await index.add("user:a", _rid("q"), _record([1.0], "a"))
        await index.add("global", _rid("q"), _record([1.0], "g"))
        await index.flush()
        assert await index.search("user:a", [1.0], PROVIDER) == []
        assert await index.search("global", [1.0], PROVIDER) == []


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

# Dimension used across property tests; kept small for speed.
_DIM = 4

# Non-zero float vectors: bounded, finite, and not all-zero so cosine is defined.
_vectors = st.lists(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    min_size=_DIM,
    max_size=_DIM,
).filter(lambda v: any(abs(x) > 1e-6 for x in v))

_namespaces = st.sampled_from(["global", "user:alice", "user:bob", "session:s1"])


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=st.lists(
        st.tuples(_namespaces, st.text(min_size=1, max_size=12), _vectors),
        min_size=1,
        max_size=12,
        unique_by=lambda e: (e[0], e[1]),
    ),
    search_ns=_namespaces,
    query=_vectors,
)
@pytest.mark.asyncio
async def test_property_scope_non_leakage(
    entries: list[tuple[str, str, list[float]]],
    search_ns: str,
    query: list[float],
) -> None:
    """Property 1: a search never returns a record from another namespace.

    **Validates: Requirements 5.1, 5.2, 5.4**
    """
    index = InMemoryVectorIndex()
    values_in_search_ns: set[str] = set()
    for namespace, text, vector in entries:
        value = f"{namespace}|{text}"
        await index.add(namespace, _rid(text), _record(vector, value))
        if namespace == search_ns:
            values_in_search_ns.add(value)

    matches = await index.search(search_ns, query, PROVIDER, top_k=len(entries))
    returned = {deserialize_value(m.value) for m in matches}
    assert returned <= values_in_search_ns


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    vectors=st.lists(_vectors, min_size=1, max_size=10),
    query=_vectors,
)
@pytest.mark.asyncio
async def test_property_best_match_selection(
    vectors: list[list[float]],
    query: list[float],
) -> None:
    """Property 3: the top result scores >= every other record in the namespace.

    Also confirms the returned list is sorted by descending score.

    **Validates: Requirements 1.5**
    """
    index = InMemoryVectorIndex(max_vectors_per_namespace=len(vectors) + 1)
    for i, vector in enumerate(vectors):
        await index.add("global", _rid(f"q{i}"), _record(vector, i))

    matches = await index.search("global", query, PROVIDER, top_k=len(vectors))

    # Best score equals the max cosine over all stored vectors.
    expected_scores = sorted((cosine_similarity(query, v) for v in vectors), reverse=True)
    actual_scores = [m.score for m in matches]
    assert actual_scores == pytest.approx(expected_scores)
    # Descending order.
    assert all(a >= b for a, b in zip(actual_scores, actual_scores[1:], strict=False))


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    first=_vectors,
    second=_vectors,
    provider_a=st.sampled_from(["openai:a", "local:b", "litellm:c"]),
    provider_b=st.sampled_from(["openai:a", "local:b", "litellm:c"]),
)
@pytest.mark.asyncio
async def test_property_provider_consistency(
    first: list[float],
    second: list[float],
    provider_a: str,
    provider_b: str,
) -> None:
    """Property 4: mismatched provider_model is refused on add and search.

    **Validates: Requirements 3.7, 3.8**
    """
    index = InMemoryVectorIndex()
    await index.add("global", _rid("q0"), _record(first, "v0", provider_model=provider_a))

    if provider_a == provider_b:
        # Same provider: both operations succeed.
        await index.add("global", _rid("q1"), _record(second, "v1", provider_model=provider_b))
        matches = await index.search("global", second, provider_b, top_k=2)
        assert len(matches) == 2
    else:
        with pytest.raises(ProviderMismatchError):
            await index.add("global", _rid("q1"), _record(second, "v1", provider_model=provider_b))
        with pytest.raises(ProviderMismatchError):
            await index.search("global", second, provider_b)


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=st.lists(
        st.tuples(st.text(min_size=1, max_size=12), _vectors, st.booleans()),
        min_size=1,
        max_size=10,
        unique_by=lambda e: e[0],
    ),
    query=_vectors,
)
@pytest.mark.asyncio
async def test_property_expiry_safety(
    entries: list[tuple[str, list[float], bool]],
    query: list[float],
) -> None:
    """Property 5: expired records are never returned as matches.

    **Validates: Requirements 6.5, 6.7**
    """
    index = InMemoryVectorIndex(max_vectors_per_namespace=len(entries) + 1)
    now = time.monotonic()
    live_values: set[str] = set()
    for text, vector, expired in entries:
        value = f"v|{text}"
        expires_at = (now - 10.0) if expired else (now + 1000.0)
        await index.add("global", _rid(text), _record(vector, value, expires_at=expires_at))
        if not expired:
            live_values.add(value)

    matches = await index.search("global", query, PROVIDER, top_k=len(entries))
    returned = {deserialize_value(m.value) for m in matches}
    assert returned <= live_values


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    text=st.text(min_size=1, max_size=20),
    first_vector=_vectors,
    second_vector=_vectors,
    first_value=st.integers(),
    second_value=st.integers(),
)
@pytest.mark.asyncio
async def test_property_determinism_of_storage_key(
    text: str,
    first_vector: list[float],
    second_vector: list[float],
    first_value: int,
    second_value: int,
) -> None:
    """Property 8: re-adding the same query updates rather than duplicates.

    **Validates: Requirements 1.4**
    """
    index = InMemoryVectorIndex()
    rid = _rid(text)
    await index.add("global", rid, _record(first_vector, first_value))
    await index.add("global", rid, _record(second_vector, second_value))

    # Deterministic record id means a single record, holding the latest value.
    assert index.namespace_size("global") == 1
    matches = await index.search("global", second_vector, PROVIDER, top_k=1)
    assert len(matches) == 1
    assert deserialize_value(matches[0].value) == second_value
