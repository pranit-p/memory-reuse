"""Similarity-based semantic cache for LLM call results.

:class:`SemanticCache` is the Phase 2 counterpart to
:class:`~memory_reuse.cache.exact.ExactCache`.  Where the exact cache keys
entries by a hash of their inputs, the semantic cache embeds the query text and
returns a stored result when a new query is *close enough* (by cosine
similarity) to a previously stored one.  This lets reworded but equivalent
requests reuse a cached answer instead of triggering a fresh LLM call.

It mirrors the shape of :class:`ExactCache` — an injected storage component
(here a :class:`~memory_reuse.vector.base.VectorIndex`), a
:class:`~memory_reuse.config.CacheConfig`, and a shared
:class:`~memory_reuse.stats.StatsTracker` — plus an injected
:class:`~memory_reuse.embeddings.base.EmbeddingProvider` that turns text into
vectors.

Scope isolation reuses the same :func:`memory_reuse._utils.check_scope` guard as
the exact cache, so :class:`~memory_reuse.exceptions.ScopeViolationError`
behaviour is identical, and the query's scope is mapped to a vector namespace via
:func:`memory_reuse._utils.build_namespace` so a search never crosses scope
boundaries.
"""

from __future__ import annotations

import logging
from typing import Any

from memory_reuse._utils import (
    build_namespace,
    check_scope,
    cosine_similarity,
    deserialize_value,
    hash_value,
    serialize_value,
    split_sentences,
)
from memory_reuse.config import CacheConfig
from memory_reuse.embeddings.base import EmbeddingProvider
from memory_reuse.stats import StatsTracker
from memory_reuse.vector.base import VectorIndex, VectorRecord

logger = logging.getLogger(__name__)


class SemanticCache:
    """Similarity-based cache backed by a :class:`VectorIndex`.

    On :meth:`get`, the query text is embedded, the scope's namespace is
    searched for the single closest stored vector, and the associated value is
    returned when its similarity score is at or above the effective threshold.
    On :meth:`set`, the query text is embedded (or a precomputed embedding is
    reused) and stored alongside the value in the scope's namespace.

    Args:
        index: The vector index used to store and search embeddings.
        embedder: The embedding provider that turns query text into vectors.
        config: Cache configuration. Supplies the default
            ``similarity_threshold`` and ``default_ttl``.
        stats: Statistics tracker shared with the parent
            :class:`~memory_reuse.core.MemoryCache`.

    Example::

        cache = SemanticCache(index, embedder, config, stats)
        await cache.set("What is 128 times 47?", 6016, scope="global",
                        scope_id=None)
        # A reworded but equivalent query hits the cache:
        result = await cache.get("What is 128 multiplied by 47?",
                                 scope="global", scope_id=None)
    """

    def __init__(
        self,
        index: VectorIndex,
        embedder: EmbeddingProvider,
        config: CacheConfig,
        stats: StatsTracker,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._config = config
        self._stats = stats

    async def get(
        self,
        query_text: str,
        scope: str,
        scope_id: str | None,
        threshold: float | None = None,
    ) -> Any | None:
        """Look up a cached value by semantic similarity to ``query_text``.

        Embeds the query, searches the scope's namespace for the closest stored
        vector, and returns its value when the best match's similarity score is
        at or above the effective threshold.

        Args:
            query_text: The natural-language query to match semantically.
            scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
            scope_id: User ID or session ID. Required for non-global scopes.
            threshold: Optional per-call similarity threshold overriding
                :attr:`~memory_reuse.config.CacheConfig.similarity_threshold`.
                Must lie in ``[0.0, 1.0]``.

        Returns:
            The cached value of the best match at or above the effective
            threshold, or ``None`` on a miss.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but none
                is provided.
        """
        check_scope(scope, scope_id)
        namespace = build_namespace(scope, scope_id)
        effective_threshold = self._effective_threshold(threshold)

        try:
            embedding = await self._embedder.embed(query_text)
        except Exception:
            # An embedding failure turns what would be a miss into a recorded
            # error + miss, never a raised exception (Req 8.4).
            self._stats.record_error()
            self._stats.record_miss()
            logger.debug("SemanticCache: embedding failed on GET, treating as miss")
            return None

        try:
            matches = await self._index.search(
                namespace, embedding, self._embedder.identity, top_k=1
            )
        except Exception:
            self._stats.record_error()
            self._stats.record_miss()
            logger.debug("SemanticCache: index error on SEARCH, treating as miss")
            return None

        if not matches or matches[0].score < effective_threshold:
            self._stats.record_miss()
            logger.debug("SemanticCache: MISS scope=%s", scope)
            return None

        self._stats.record_semantic_hit()
        logger.debug("SemanticCache: HIT scope=%s score=%.4f", scope, matches[0].score)
        value = deserialize_value(matches[0].value)

        if self._config.extract_answer:
            value = await self._maybe_extract_answer(value, embedding)
        return value

    async def set(
        self,
        query_text: str,
        value: Any,
        scope: str,
        scope_id: str | None,
        ttl: int | None = None,
        precomputed_embedding: list[float] | None = None,
    ) -> None:
        """Store a query embedding and its value in the scope's namespace.

        Args:
            query_text: The natural-language query whose embedding is stored.
            value: The value to cache. Must be JSON-serialisable.
            scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
            scope_id: User or session identifier for non-global scopes.
            ttl: Time-to-live in seconds. Falls back to
                :attr:`~memory_reuse.config.CacheConfig.default_ttl` when
                ``None``.
            precomputed_embedding: An embedding for ``query_text`` computed
                earlier in the same lookup cycle, reused to avoid embedding the
                same text twice. When ``None`` the query is embedded here.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but none
                is provided.
        """
        check_scope(scope, scope_id)
        namespace = build_namespace(scope, scope_id)

        if precomputed_embedding is not None:
            embedding = precomputed_embedding
        else:
            try:
                embedding = await self._embedder.embed(query_text)
            except Exception:
                self._stats.record_error()
                logger.debug("SemanticCache: embedding failed on SET, skipping store")
                return

        effective_ttl = ttl if ttl is not None else self._config.default_ttl
        provider_model = self._embedder.identity
        record_id = hash_value([provider_model, query_text])
        record = VectorRecord(
            vector=embedding,
            value=serialize_value(value),
            provider_model=provider_model,
            expires_at=self._index.expiry_for_ttl(effective_ttl),
        )

        try:
            await self._index.add(namespace, record_id, record)
            logger.debug("SemanticCache: SET scope=%s", scope)
        except Exception:
            self._stats.record_error()
            logger.debug("SemanticCache: index error on ADD")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_threshold(self, threshold: float | None) -> float:
        """Return the similarity threshold to apply for a lookup.

        Args:
            threshold: An optional per-call override.

        Returns:
            ``threshold`` when provided, otherwise the configured
            :attr:`~memory_reuse.config.CacheConfig.similarity_threshold`.
        """
        if threshold is None:
            return self._config.similarity_threshold
        return threshold

    async def _maybe_extract_answer(self, value: Any, query_embedding: list[float]) -> Any:
        """Narrow a string answer to the sentence(s) best matching the query.

        This is a purely *extractive*, embedding-based step (no LLM, no
        generation): the stored answer is split into sentences, each is embedded
        with the same provider, and the single sentence whose embedding is
        closest to ``query_embedding`` is returned — but only if it clears
        :attr:`~memory_reuse.config.CacheConfig.extract_min_similarity`.
        Otherwise the full answer is returned unchanged, so extraction never
        yields an empty or fabricated result.

        Extraction only applies to plain ``str`` values with more than one
        sentence; any other value (dict, number, single sentence) is returned
        untouched. Any failure falls back to the full value.

        Args:
            value: The cached value from the semantic hit.
            query_embedding: The already-computed embedding of the query.

        Returns:
            The best-matching sentence, or the original ``value`` when
            extraction does not apply or no sentence is confident enough.
        """
        if not isinstance(value, str):
            return value

        sentences = split_sentences(value)
        if len(sentences) <= 1:
            # Nothing to narrow.
            return value

        try:
            sentence_vectors = await self._embedder.embed_batch(sentences)
        except Exception:
            # Best-effort: on any embedding failure, return the full answer.
            self._stats.record_error()
            logger.debug("SemanticCache: extraction embedding failed, returning full answer")
            return value

        best_sentence = value
        best_score = self._config.extract_min_similarity
        found = False
        for sentence, vector in zip(sentences, sentence_vectors, strict=False):
            score = cosine_similarity(query_embedding, vector)
            if score >= best_score:
                best_score = score
                best_sentence = sentence
                found = True

        if found:
            logger.debug("SemanticCache: extracted answer sentence (score=%.4f)", best_score)
            return best_sentence
        logger.debug("SemanticCache: no sentence cleared extract_min_similarity; full answer")
        return value
