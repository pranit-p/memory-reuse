"""Example tests for version-aware cache identity (Phase 6).

Covers the config guard (Req 3.6), the semantic-layer isolation half of Req 3.5
(the exact/tool halves are property-tested), and the byte-identical-when-unset
guarantee at the MemoryCache level (Req 3.3, 8.2).
"""

from __future__ import annotations

import pytest

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.exceptions import ConfigurationError
from tests.conftest import StubEmbedder, make_semantic_cache


def test_non_string_cache_version_rejected() -> None:
    """A non-string cache_version raises ConfigurationError (Req 3.6)."""
    with pytest.raises(ConfigurationError):
        CacheConfig(cache_version=123)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        CacheConfig(cache_version=["v1"])  # type: ignore[arg-type]


def test_none_and_str_cache_version_accepted() -> None:
    """None (default) and a str cache_version both construct cleanly (Req 3.1)."""
    assert CacheConfig().cache_version is None
    assert CacheConfig(cache_version="v5").cache_version == "v5"


def test_default_config_keys_are_byte_identical_to_unversioned() -> None:
    """A default config derives the pre-Phase-6 key exactly (Req 3.3, 8.2)."""
    cache = MemoryCache(CacheConfig(backend="memory"))
    # The private key builder is what every layer uses; with cache_version=None
    # it must equal build_cache_key called without a version.
    from memory_reuse._utils import build_cache_key

    built = cache.exact._build_key(["a", "b"], "global", None)  # type: ignore[attr-defined]
    expected = build_cache_key("memreuse", "global", None, "a", "b")
    assert built == expected


class TestSemanticVersionIsolation:
    """A version bump isolates semantic search too (Req 3.5)."""

    @pytest.mark.asyncio
    async def test_version_b_does_not_match_version_a_vector_on_shared_index(self) -> None:
        """A vector stored under version A is never returned under version B.

        Both semantic caches share **one** vector index (as they would over a
        shared Redis vector store), and one injected embedder maps the query to a
        single fixed vector — so *without* version scoping the version-B lookup
        would score a perfect semantic match on version A's stored vector.
        Version-scoped namespaces must place them in different namespaces so the
        version-B search never sees version A's entry.
        """
        embedder = StubEmbedder()
        cache_a = make_semantic_cache(embedder, cache_version="v1")
        cache_b = make_semantic_cache(embedder, cache_version="v2")
        # Share the same underlying vector index and embedder across both.
        cache_b.semantic._index = cache_a.semantic._index  # type: ignore[union-attr]
        cache_b.semantic._embedder = embedder  # type: ignore[union-attr]

        await cache_a.semantic.set(  # type: ignore[union-attr]
            "how do I reset my password?",
            "Go to settings > security.",
            scope="global",
            scope_id=None,
        )
        # Same-version lookup on the shared index hits.
        assert (
            await cache_a.semantic.get(  # type: ignore[union-attr]
                "how do I reset my password?", scope="global", scope_id=None
            )
            == "Go to settings > security."
        )
        # Different-version lookup on the *same* index must miss (namespace v2).
        assert (
            await cache_b.semantic.get(  # type: ignore[union-attr]
                "how do I reset my password?", scope="global", scope_id=None
            )
            is None
        )
