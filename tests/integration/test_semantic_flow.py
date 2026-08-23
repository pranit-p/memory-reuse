"""Integration tests for the combined exact + semantic lookup/store flow.

These exercise :meth:`MemoryCache.lookup` and :meth:`MemoryCache.store` wired
end-to-end with the in-memory backend, the in-memory vector index, and a
deterministic ``FakeEmbedder`` so the whole suite stays offline and fast.

They cover the behaviour called out for this component and the two invariants
the design attaches to the combined flow:

* **Property 6 — Exact-first cost bound:** an exact hit never invokes the
  embedding provider (asserted via a spy).
* **Property 7 — Disabled equals Phase 1:** with ``semantic_enabled=False`` or
  ``exact_only=True`` the observable behaviour matches the exact cache directly.
"""

from __future__ import annotations

import hashlib

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.embeddings.base import EmbeddingProvider


class SpyEmbedder(EmbeddingProvider):
    """Deterministic embedder that counts how many times ``embed`` is called.

    Identical text always yields the identical vector (a stable hash of the
    text), so semantic matches are predictable, while ``embed_calls`` lets a
    test assert whether an embedding was computed at all.
    """

    def __init__(self, model: str = "spy", dimension: int = 8) -> None:
        self._model = model
        self._dimension = dimension
        self.embed_calls = 0

    @property
    def identity(self) -> str:
        return f"fake:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]


def _semantic_cache(embedder: EmbeddingProvider, **overrides: object) -> MemoryCache:
    """Build a semantic-enabled MemoryCache with an injected embedder.

    The embedder is patched onto the already-built ``SemanticCache`` so the test
    controls exactly which vectors are produced without needing a real provider.
    """
    params: dict[str, object] = {
        "backend": "memory",
        "semantic_enabled": True,
        "embedding_provider": "local",
        "similarity_threshold": 0.95,
    }
    params.update(overrides)
    config = CacheConfig(**params)  # type: ignore[arg-type]
    cache = MemoryCache(config)
    assert cache.semantic is not None
    cache.semantic._embedder = embedder  # type: ignore[attr-defined]
    return cache


class TestExactFirst:
    async def test_exact_hit_short_circuits_embedder(self) -> None:
        """Property 6: an exact hit must not compute an embedding."""
        spy = SpyEmbedder()
        cache = _semantic_cache(spy)

        await cache.store(["k"], "what is 2+2?", 4, scope="global", scope_id=None)
        embed_after_store = spy.embed_calls

        result = await cache.lookup(["k"], "what is 2+2?", scope="global", scope_id=None)

        assert result == 4
        # The lookup itself computed no embedding beyond what store needed.
        assert spy.embed_calls == embed_after_store
        assert cache.stats.exact_hits == 1
        assert cache.stats.semantic_hits == 0

    async def test_exact_miss_then_semantic_hit(self) -> None:
        """An exact miss falls back to a semantic match on reworded input."""
        spy = SpyEmbedder()
        cache = _semantic_cache(spy, similarity_threshold=0.0)

        # Store under one key/text, look up with a different key (exact miss)
        # but same query text so the semantic vector matches.
        await cache.store(["k1"], "how many moons does mars have", 2, scope="global", scope_id=None)

        result = await cache.lookup(
            ["k2"], "how many moons does mars have", scope="global", scope_id=None
        )

        assert result == 2
        assert cache.stats.semantic_hits == 1
        assert cache.stats.exact_hits == 0

    async def test_semantic_miss_returns_none(self) -> None:
        spy = SpyEmbedder()
        cache = _semantic_cache(spy, similarity_threshold=1.0)

        await cache.store(["k1"], "alpha text", 1, scope="global", scope_id=None)
        result = await cache.lookup(
            ["k2"], "completely different beta text", scope="global", scope_id=None
        )

        assert result is None
        assert cache.stats.misses >= 1


class TestStoreExactOnSemanticHit:
    async def test_semantic_hit_promotes_to_exact(self) -> None:
        """Req 7.4: a semantic hit writes an exact entry for the new key."""
        spy = SpyEmbedder()
        cache = _semantic_cache(spy, similarity_threshold=0.0)
        await cache.store(["k1"], "same query", "value", scope="global", scope_id=None)

        # First lookup with a new key is a semantic hit and promotes to exact.
        assert await cache.lookup(["k2"], "same query", scope="global", scope_id=None) == "value"

        calls_before = spy.embed_calls
        # Second lookup on the promoted key is now an exact hit — no embedding.
        assert await cache.lookup(["k2"], "same query", scope="global", scope_id=None) == "value"
        assert spy.embed_calls == calls_before
        assert cache.stats.exact_hits == 1

    async def test_promotion_disabled_leaves_exact_empty(self) -> None:
        spy = SpyEmbedder()
        cache = _semantic_cache(spy, similarity_threshold=0.0, store_exact_on_semantic_hit=False)
        await cache.store(["k1"], "same query", "value", scope="global", scope_id=None)

        assert await cache.lookup(["k2"], "same query", scope="global", scope_id=None) == "value"
        # Without promotion the second lookup is still a semantic hit, not exact.
        assert await cache.lookup(["k2"], "same query", scope="global", scope_id=None) == "value"
        assert cache.stats.exact_hits == 0
        assert cache.stats.semantic_hits == 2


class TestExactOnly:
    async def test_exact_only_bypasses_semantic(self) -> None:
        """Req 9.4: exact_only never consults the semantic cache."""
        spy = SpyEmbedder()
        cache = _semantic_cache(spy, similarity_threshold=0.0)
        await cache.store(["k1"], "shared query", 7, scope="global", scope_id=None)
        calls_before = spy.embed_calls

        result = await cache.lookup(
            ["k2"], "shared query", scope="global", scope_id=None, exact_only=True
        )

        assert result is None
        assert spy.embed_calls == calls_before  # no embedding computed
        assert cache.stats.semantic_hits == 0

    async def test_exact_only_store_skips_semantic(self) -> None:
        spy = SpyEmbedder()
        cache = _semantic_cache(spy, similarity_threshold=0.0)
        await cache.store(["k1"], "shared query", 7, scope="global", scope_id=None, exact_only=True)
        # A later semantic lookup on a different key must miss — nothing stored.
        assert await cache.lookup(["k2"], "shared query", scope="global", scope_id=None) is None


class TestDisabledEqualsPhase1:
    async def test_disabled_lookup_matches_exact_cache(self) -> None:
        """Property 7: with semantic disabled, lookup/store == exact cache."""
        cache = MemoryCache(CacheConfig(backend="memory"))
        assert cache.semantic is None

        await cache.store(["k"], "irrelevant text", "v", scope="global", scope_id=None)

        via_combined = await cache.lookup(["k"], "irrelevant text", scope="global", scope_id=None)
        via_exact = await cache.exact.get(["k"], scope="global", scope_id=None)
        assert via_combined == via_exact == "v"

    async def test_disabled_semantic_lookup_is_miss(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory"))
        await cache.store(["k1"], "same query", "v", scope="global", scope_id=None)
        # Different key, same text — semantic would hit, but it's disabled.
        assert await cache.lookup(["k2"], "same query", scope="global", scope_id=None) is None

    async def test_no_semantic_deps_imported_when_disabled(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory"))
        assert cache.semantic is None


class TestPerCallThreshold:
    async def test_per_call_threshold_allows_hit(self) -> None:
        spy = SpyEmbedder()
        # Config threshold is high (0.99); orthogonal-ish texts won't hit there.
        cache = _semantic_cache(spy, similarity_threshold=0.99)
        await cache.store(["k1"], "query one", "value", scope="global", scope_id=None)

        # A generous per-call threshold turns the fuzzy match into a hit.
        result = await cache.lookup(
            ["k2"], "query two", scope="global", scope_id=None, threshold=0.0
        )
        assert result == "value"
        assert cache.stats.semantic_hits == 1


class TestScopeParity:
    async def test_user_scope_isolated_in_combined_flow(self) -> None:
        spy = SpyEmbedder()
        cache = _semantic_cache(spy, similarity_threshold=0.0)
        await cache.store(["k"], "shared", "alice-value", scope="user", scope_id="alice")

        # Bob must not see Alice's semantic entry.
        assert await cache.lookup(["k2"], "shared", scope="user", scope_id="bob") is None

    async def test_scope_violation_raises(self) -> None:
        from memory_reuse.exceptions import ScopeViolationError

        cache = MemoryCache(CacheConfig(backend="memory"))
        with pytest.raises(ScopeViolationError):
            await cache.lookup(["k"], "q", scope="user", scope_id=None)
