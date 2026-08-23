"""Unit and property-based tests for :class:`SemanticCache`.

Example-based tests cover the core lookup/store behaviour called out for this
component: a hit at or above the threshold, a miss below it, a per-call
threshold override, the embedding-failure path (recorded as error + miss, never
raised), and the shared scope-violation guard.

The property-based tests exercise the invariants the design attaches to the
semantic cache:

* **Property 2 — Threshold monotonicity:** when a value is returned the best
  match scored ``>= effective_threshold``; when ``None`` is returned no stored
  record scored ``>= effective_threshold``.
* **Property 3 — Best-match selection:** a returned hit is the single
  highest-scoring record in the namespace.
* **Property 5 — Expiry safety:** an expired record is never returned as a hit.

A deterministic ``FakeEmbedder`` keeps the whole suite offline and fast — no
real model or API is ever contacted.
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from memory_reuse._utils import cosine_similarity
from memory_reuse.cache.semantic import SemanticCache
from memory_reuse.config import CacheConfig
from memory_reuse.embeddings.base import EmbeddingProvider
from memory_reuse.exceptions import ScopeViolationError
from memory_reuse.stats import StatsTracker
from memory_reuse.vector import InMemoryVectorIndex


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedding provider for tests.

    Produces a fixed-dimension unit-ish vector derived from a stable hash of
    the text, so identical text always yields the identical vector without any
    external dependency.
    """

    def __init__(self, model: str = "fake-model", dimension: int = 8) -> None:
        self._model = model
        self._dimension = dimension

    @property
    def identity(self) -> str:
        return f"fake:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]


class ControlledEmbedder(EmbeddingProvider):
    """Embedder that returns caller-supplied vectors per text.

    Lets a test pin the exact vectors (and therefore exact cosine scores) so
    threshold behaviour is fully deterministic.
    """

    def __init__(self, vectors: dict[str, list[float]], model: str = "ctrl") -> None:
        self._vectors = vectors
        self._model = model

    @property
    def identity(self) -> str:
        return f"fake:{self._model}"

    @property
    def dimension(self) -> int:
        return len(next(iter(self._vectors.values())))

    async def embed(self, text: str) -> list[float]:
        return list(self._vectors[text])


class FailingEmbedder(EmbeddingProvider):
    """Embedder whose :meth:`embed` always raises, to exercise the error path."""

    @property
    def identity(self) -> str:
        return "fake:failing"

    @property
    def dimension(self) -> int:
        return 4

    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("boom: embedding backend unavailable")


def _make_cache(
    embedder: EmbeddingProvider | None = None,
    *,
    threshold: float = 0.95,
    ttl: int | None = 3600,
) -> tuple[SemanticCache, StatsTracker, InMemoryVectorIndex]:
    """Build a SemanticCache wired to an in-memory vector index."""
    index = InMemoryVectorIndex()
    config = CacheConfig(similarity_threshold=threshold, default_ttl=ttl)
    stats = StatsTracker()
    embedder = embedder or FakeEmbedder()
    cache = SemanticCache(index=index, embedder=embedder, config=config, stats=stats)
    return cache, stats, index


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------


class TestHitMiss:
    @pytest.mark.asyncio
    async def test_identical_query_hits(self) -> None:
        cache, stats, _ = _make_cache()
        await cache.set("what is 2+2?", {"answer": 4}, scope="global", scope_id=None)

        result = await cache.get("what is 2+2?", scope="global", scope_id=None)

        assert result == {"answer": 4}
        assert stats.get_stats().semantic_hits == 1
        assert stats.get_stats().hits == 1

    @pytest.mark.asyncio
    async def test_empty_namespace_is_miss(self) -> None:
        cache, stats, _ = _make_cache()
        result = await cache.get("anything", scope="global", scope_id=None)
        assert result is None
        assert stats.get_stats().misses == 1
        assert stats.get_stats().semantic_hits == 0

    @pytest.mark.asyncio
    async def test_hit_at_exact_threshold(self) -> None:
        # Score of query vs stored is exactly 1.0 (identical vectors) — a hit
        # when threshold is 1.0 (>= comparison).
        embedder = ControlledEmbedder({"q": [1.0, 0.0], "stored": [1.0, 0.0]})
        cache, stats, _ = _make_cache(embedder, threshold=1.0)
        await cache.set("stored", "value", scope="global", scope_id=None)

        result = await cache.get("q", scope="global", scope_id=None)

        assert result == "value"
        assert stats.get_stats().semantic_hits == 1

    @pytest.mark.asyncio
    async def test_miss_below_threshold(self) -> None:
        # Orthogonal vectors score 0.5 (normalised cosine); below a 0.95
        # threshold this is a miss.
        embedder = ControlledEmbedder({"q": [1.0, 0.0], "stored": [0.0, 1.0]})
        cache, stats, _ = _make_cache(embedder, threshold=0.95)
        await cache.set("stored", "value", scope="global", scope_id=None)

        result = await cache.get("q", scope="global", scope_id=None)

        assert result is None
        assert stats.get_stats().misses == 1
        assert stats.get_stats().semantic_hits == 0

    @pytest.mark.asyncio
    async def test_best_match_of_several_is_returned(self) -> None:
        embedder = ControlledEmbedder(
            {
                "q": [1.0, 0.0],
                "near": [1.0, 0.01],  # ~1.0 score
                "far": [0.0, 1.0],  # 0.5 score
            }
        )
        cache, _, _ = _make_cache(embedder, threshold=0.9)
        await cache.set("near", "near-value", scope="global", scope_id=None)
        await cache.set("far", "far-value", scope="global", scope_id=None)

        result = await cache.get("q", scope="global", scope_id=None)
        assert result == "near-value"


class TestThresholdOverride:
    @pytest.mark.asyncio
    async def test_per_call_threshold_allows_hit(self) -> None:
        # Orthogonal vectors score 0.5. Config threshold 0.95 would miss, but a
        # per-call override of 0.4 turns it into a hit.
        embedder = ControlledEmbedder({"q": [1.0, 0.0], "stored": [0.0, 1.0]})
        cache, stats, _ = _make_cache(embedder, threshold=0.95)
        await cache.set("stored", "value", scope="global", scope_id=None)

        assert await cache.get("q", scope="global", scope_id=None) is None
        result = await cache.get("q", scope="global", scope_id=None, threshold=0.4)
        assert result == "value"
        assert stats.get_stats().semantic_hits == 1

    @pytest.mark.asyncio
    async def test_per_call_threshold_forces_miss(self) -> None:
        # Identical vectors score 1.0; a per-call threshold above 1.0 is
        # impossible to reach, forcing a miss even for an identical query.
        embedder = ControlledEmbedder({"q": [1.0, 0.0], "stored": [1.0, 0.0]})
        cache, stats, _ = _make_cache(embedder, threshold=0.5)
        await cache.set("stored", "value", scope="global", scope_id=None)

        result = await cache.get("q", scope="global", scope_id=None, threshold=1.0000001)
        assert result is None
        assert stats.get_stats().misses == 1


class TestEmbeddingFailure:
    @pytest.mark.asyncio
    async def test_get_embedding_failure_records_error_and_miss(self) -> None:
        cache, stats, _ = _make_cache(FailingEmbedder())
        result = await cache.get("anything", scope="global", scope_id=None)

        assert result is None
        snapshot = stats.get_stats()
        assert snapshot.errors == 1
        assert snapshot.misses == 1
        assert snapshot.semantic_hits == 0

    @pytest.mark.asyncio
    async def test_get_embedding_failure_does_not_raise(self) -> None:
        cache, _, _ = _make_cache(FailingEmbedder())
        # Should return None rather than propagating RuntimeError.
        assert await cache.get("anything", scope="global", scope_id=None) is None

    @pytest.mark.asyncio
    async def test_set_embedding_failure_skips_store(self) -> None:
        cache, stats, index = _make_cache(FailingEmbedder())
        await cache.set("q", "v", scope="global", scope_id=None)

        assert index.namespace_size("global") == 0
        assert stats.get_stats().errors == 1

    @pytest.mark.asyncio
    async def test_set_reuses_precomputed_embedding_without_embedder(self) -> None:
        # Even with a failing embedder, a precomputed embedding is stored.
        cache, _, index = _make_cache(FailingEmbedder())
        await cache.set(
            "q",
            "v",
            scope="global",
            scope_id=None,
            precomputed_embedding=[1.0, 0.0, 0.0, 0.0],
        )
        assert index.namespace_size("global") == 1


class TestScopeIsolation:
    @pytest.mark.asyncio
    async def test_user_scope_requires_scope_id_on_get(self) -> None:
        cache, _, _ = _make_cache()
        with pytest.raises(ScopeViolationError):
            await cache.get("q", scope="user", scope_id=None)

    @pytest.mark.asyncio
    async def test_session_scope_requires_scope_id_on_set(self) -> None:
        cache, _, _ = _make_cache()
        with pytest.raises(ScopeViolationError):
            await cache.set("q", "v", scope="session", scope_id=None)

    @pytest.mark.asyncio
    async def test_user_a_cannot_see_user_b_entry(self) -> None:
        cache, _, _ = _make_cache()
        await cache.set("q", "alice-secret", scope="user", scope_id="alice")

        result = await cache.get("q", scope="user", scope_id="bob")
        assert result is None

    @pytest.mark.asyncio
    async def test_global_and_user_scopes_isolated(self) -> None:
        cache, _, _ = _make_cache()
        await cache.set("q", "global-value", scope="global", scope_id=None)
        result = await cache.get("q", scope="user", scope_id="alice")
        assert result is None


class TestRoundTripAndReuse:
    @pytest.mark.asyncio
    async def test_round_trip_integrity(self) -> None:
        cache, _, _ = _make_cache()
        payload = {"nested": [1, 2, 3], "text": "hello"}
        await cache.set("q", payload, scope="global", scope_id=None)
        assert await cache.get("q", scope="global", scope_id=None) == payload

    @pytest.mark.asyncio
    async def test_restore_same_query_updates_not_duplicates(self) -> None:
        cache, _, index = _make_cache()
        await cache.set("q", "first", scope="global", scope_id=None)
        await cache.set("q", "second", scope="global", scope_id=None)

        assert index.namespace_size("global") == 1
        assert await cache.get("q", scope="global", scope_id=None) == "second"


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

_DIM = 4

_vectors = st.lists(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    min_size=_DIM,
    max_size=_DIM,
).filter(lambda v: any(abs(x) > 1e-6 for x in v))


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=st.lists(
        st.tuples(st.text(min_size=1, max_size=12), _vectors),
        min_size=1,
        max_size=8,
        unique_by=lambda e: e[0],
    ),
    query=_vectors,
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
@pytest.mark.asyncio
async def test_property_threshold_monotonicity(
    entries: list[tuple[str, list[float]]],
    query: list[float],
    threshold: float,
) -> None:
    """Property 2: hit implies best score >= threshold; miss implies none did.

    **Validates: Requirements 1.2, 1.3**
    """
    vectors = dict(entries)
    vectors["__query__"] = query
    embedder = ControlledEmbedder(vectors)
    cache, _, _ = _make_cache(embedder, threshold=threshold)

    for text, _ in entries:
        await cache.set(text, f"v|{text}", scope="global", scope_id=None)

    best = max(cosine_similarity(query, vec) for _, vec in entries)
    result = await cache.get("__query__", scope="global", scope_id=None)

    if result is None:
        assert best < threshold
    else:
        assert best >= threshold


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=st.lists(
        st.tuples(st.text(min_size=1, max_size=12), _vectors),
        min_size=1,
        max_size=8,
        unique_by=lambda e: e[0],
    ),
    query=_vectors,
)
@pytest.mark.asyncio
async def test_property_best_match_selection(
    entries: list[tuple[str, list[float]]],
    query: list[float],
) -> None:
    """Property 3: a returned hit is the highest-scoring record in the namespace.

    **Validates: Requirements 1.5**
    """
    vectors = dict(entries)
    vectors["__query__"] = query
    embedder = ControlledEmbedder(vectors)
    # Threshold 0.0 so any non-empty namespace yields a hit.
    cache, _, _ = _make_cache(embedder, threshold=0.0)

    for text, _ in entries:
        await cache.set(text, f"v|{text}", scope="global", scope_id=None)

    result = await cache.get("__query__", scope="global", scope_id=None)
    assert result is not None

    # The returned value must correspond to a record whose score equals the max.
    best_score = max(cosine_similarity(query, vec) for _, vec in entries)
    winner_texts = {
        f"v|{text}"
        for text, vec in entries
        if cosine_similarity(query, vec) == pytest.approx(best_score)
    }
    assert result in winner_texts


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(query=_vectors)
@pytest.mark.asyncio
async def test_property_expiry_safety(query: list[float]) -> None:
    """Property 5: an expired record is never returned as a hit.

    **Validates: Requirements 6.5, 6.7**
    """
    embedder = ControlledEmbedder({"stored": query, "__query__": query})
    cache, stats, _ = _make_cache(embedder, threshold=0.0, ttl=1)
    # Store with an already-elapsed TTL by writing directly via a precomputed
    # embedding and an expired record is enforced by the negative TTL path:
    await cache.set("stored", "value", scope="global", scope_id=None, ttl=-1)

    result = await cache.get("__query__", scope="global", scope_id=None)
    assert result is None
    assert stats.get_stats().semantic_hits == 0


# ---------------------------------------------------------------------------
# Answer extraction (extract_answer flag)
# ---------------------------------------------------------------------------


def _make_extract_cache(
    embedder: EmbeddingProvider,
    *,
    threshold: float = 0.0,
    extract_min_similarity: float = 0.5,
) -> tuple[SemanticCache, StatsTracker, InMemoryVectorIndex]:
    """Build a SemanticCache with answer extraction enabled."""
    index = InMemoryVectorIndex()
    config = CacheConfig(
        similarity_threshold=threshold,
        extract_answer=True,
        extract_min_similarity=extract_min_similarity,
    )
    stats = StatsTracker()
    cache = SemanticCache(index=index, embedder=embedder, config=config, stats=stats)
    return cache, stats, index


class TestAnswerExtraction:
    @pytest.mark.asyncio
    async def test_returns_best_matching_sentence(self) -> None:
        # The stored answer has two sentences; the query vector matches the
        # second sentence's vector exactly, so extraction returns only it.
        answer = "Python is a language. It was created by Guido van Rossum."
        vectors = {
            "store-q": [1.0, 0.0],
            "who made it": [0.0, 1.0],
            "Python is a language.": [1.0, 0.0],
            "It was created by Guido van Rossum.": [0.0, 1.0],
        }
        embedder = ControlledEmbedder(vectors)
        cache, _, _ = _make_extract_cache(embedder, extract_min_similarity=0.9)
        await cache.set("store-q", answer, scope="global", scope_id=None)

        result = await cache.get("who made it", scope="global", scope_id=None)
        assert result == "It was created by Guido van Rossum."

    @pytest.mark.asyncio
    async def test_falls_back_to_full_answer_when_no_sentence_confident(self) -> None:
        # Query vector is orthogonal to both sentences (score 0.5), below a
        # high extract_min_similarity, so the full answer is returned.
        answer = "Alpha sentence here. Beta sentence there."
        vectors = {
            "store-q": [1.0, 0.0],
            "unrelated": [1.0, 0.0],
            "Alpha sentence here.": [0.0, 1.0],
            "Beta sentence there.": [0.0, 1.0],
        }
        embedder = ControlledEmbedder(vectors)
        cache, _, _ = _make_extract_cache(embedder, extract_min_similarity=0.99)
        await cache.set("store-q", answer, scope="global", scope_id=None)

        result = await cache.get("unrelated", scope="global", scope_id=None)
        assert result == answer

    @pytest.mark.asyncio
    async def test_non_string_value_is_untouched(self) -> None:
        payload = {"answer": 4}
        vectors = {"store-q": [1.0, 0.0], "reworded": [1.0, 0.0]}
        embedder = ControlledEmbedder(vectors)
        cache, _, _ = _make_extract_cache(embedder)
        await cache.set("store-q", payload, scope="global", scope_id=None)

        result = await cache.get("reworded", scope="global", scope_id=None)
        assert result == payload

    @pytest.mark.asyncio
    async def test_single_sentence_answer_is_untouched(self) -> None:
        answer = "Just one sentence with no split"
        vectors = {"store-q": [1.0, 0.0], "reworded": [1.0, 0.0]}
        embedder = ControlledEmbedder(vectors)
        cache, _, _ = _make_extract_cache(embedder)
        await cache.set("store-q", answer, scope="global", scope_id=None)

        result = await cache.get("reworded", scope="global", scope_id=None)
        assert result == answer

    @pytest.mark.asyncio
    async def test_disabled_by_default_returns_full_answer(self) -> None:
        # Without extract_answer, the whole stored answer comes back verbatim.
        answer = "First sentence. Second sentence."
        vectors = {
            "store-q": [1.0, 0.0],
            "reworded": [1.0, 0.0],
            "First sentence.": [1.0, 0.0],
            "Second sentence.": [0.0, 1.0],
        }
        embedder = ControlledEmbedder(vectors)
        cache, _, _ = _make_cache(embedder, threshold=0.0)
        await cache.set("store-q", answer, scope="global", scope_id=None)

        result = await cache.get("reworded", scope="global", scope_id=None)
        assert result == answer
