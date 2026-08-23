"""Configuration dataclass for the memory-reuse."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class CacheConfig:
    """Configuration for the MemoryCache.

    All values can be overridden via environment variables when using
    :meth:`MemoryCache.from_env`.

    Attributes:
        backend: Storage backend to use. ``"memory"`` requires no extra
            dependencies; ``"redis"`` requires ``pip install memory-reuse[redis]``.
        redis_url: Connection URL for the Redis backend. Should be supplied
            via the ``MEMORY_REUSE_REDIS_URL`` environment variable rather
            than hardcoded.
        default_ttl: Default time-to-live in seconds for cached entries.
            ``None`` means entries never expire.
        default_scope: Default cache scope applied when no explicit scope is
            passed to cache operations.
        key_prefix: String prepended to every cache key for namespacing.
        max_key_size: Maximum allowed cache-key length in bytes. Requests
            that would produce a longer key raise ``ValueError``.
        enable_stats: When ``True``, hit/miss/error counts are tracked and
            accessible via :attr:`MemoryCache.stats`.
        semantic_enabled: When ``True``, the semantic (similarity-based) cache
            is enabled in addition to the exact cache. Off by default so
            existing behaviour is unchanged; requires ``embedding_provider``.
        similarity_threshold: Minimum cosine similarity (normalised to
            ``[0.0, 1.0]``) at which two requests are treated as the same and a
            cached result is reused. Higher values favour precision over recall.
        embedding_provider: Which embedding provider to use for semantic
            lookups. ``None`` disables provider selection; required when
            ``semantic_enabled`` is ``True``.
        embedding_model: Optional model name passed to the embedding provider.
            ``None`` lets the provider pick its default model.
        max_vectors_per_namespace: Maximum number of stored vectors per scope
            namespace before LRU eviction applies.
        store_exact_on_semantic_hit: When ``True``, a semantic hit also writes
            an exact-match entry so the next identical request hits the faster
            exact path.
        extract_answer: When ``True`` (and the cached value is a string), a
            semantic hit returns only the sentence(s) of the stored answer that
            best match the query, rather than the whole answer. This is a purely
            *extractive*, embedding-based narrowing — no LLM call and no
            generation — so it can only return text already present in the
            stored answer. Off by default; the core cache returns values
            verbatim.
        extract_min_similarity: Minimum normalised cosine similarity
            (``[0.0, 1.0]``) a sentence must reach against the query for
            :attr:`extract_answer` to return just that sentence. When no
            sentence clears this bar the full stored answer is returned, so
            extraction never yields an empty result.

    Example::

        config = CacheConfig(
            backend="redis",
            redis_url=os.environ["REDIS_URL"],
            default_ttl=600,
            default_scope="user",
        )
    """

    backend: Literal["memory", "redis"] = "memory"
    redis_url: str | None = None
    default_ttl: int | None = 3600
    default_scope: Literal["global", "user", "session"] = "global"
    key_prefix: str = "memreuse"
    max_key_size: int = 512
    enable_stats: bool = True
    semantic_enabled: bool = False
    similarity_threshold: float = 0.95
    embedding_provider: Literal["openai", "local", "litellm"] | None = None
    embedding_model: str | None = None
    max_vectors_per_namespace: int = 10_000
    store_exact_on_semantic_hit: bool = True
    extract_answer: bool = False
    extract_min_similarity: float = 0.5

    def __post_init__(self) -> None:
        """Validate the configuration after initialisation."""
        if self.default_ttl is not None and self.default_ttl <= 0:
            from memory_reuse.exceptions import InvalidTTLError

            raise InvalidTTLError(
                f"default_ttl must be a positive integer or None, got {self.default_ttl}"
            )
        if self.max_key_size <= 0:
            raise ValueError(f"max_key_size must be positive, got {self.max_key_size}")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            from memory_reuse.exceptions import ConfigurationError

            raise ConfigurationError(
                "similarity_threshold must be within [0.0, 1.0], "
                f"got {self.similarity_threshold}"
            )
        if not 0.0 <= self.extract_min_similarity <= 1.0:
            from memory_reuse.exceptions import ConfigurationError

            raise ConfigurationError(
                "extract_min_similarity must be within [0.0, 1.0], "
                f"got {self.extract_min_similarity}"
            )
        if self.semantic_enabled and self.embedding_provider is None:
            from memory_reuse.exceptions import ConfigurationError

            raise ConfigurationError(
                "semantic_enabled=True requires an embedding_provider "
                "(one of 'openai', 'local', 'litellm')"
            )

    @classmethod
    def from_env(cls) -> CacheConfig:
        """Create a :class:`CacheConfig` from ``MEMORY_REUSE_*`` environment variables.

        Recognised variables:

        * ``MEMORY_REUSE_BACKEND`` — ``"memory"`` or ``"redis"``
        * ``MEMORY_REUSE_REDIS_URL`` — Redis connection URL
        * ``MEMORY_REUSE_DEFAULT_TTL`` — integer seconds or ``"none"``
        * ``MEMORY_REUSE_DEFAULT_SCOPE`` — ``"global"``, ``"user"``, or ``"session"``
        * ``MEMORY_REUSE_KEY_PREFIX`` — string prefix for all keys
        * ``MEMORY_REUSE_ENABLE_STATS`` — ``"true"`` / ``"false"``
        * ``MEMORY_REUSE_SEMANTIC_ENABLED`` — ``"true"`` / ``"false"``
        * ``MEMORY_REUSE_SIMILARITY_THRESHOLD`` — float in ``[0.0, 1.0]``
        * ``MEMORY_REUSE_EMBEDDING_PROVIDER`` — ``"openai"``, ``"local"``, or ``"litellm"``
        * ``MEMORY_REUSE_EMBEDDING_MODEL`` — embedding model name

        Returns:
            A :class:`CacheConfig` populated from the environment.
        """
        raw_ttl = os.environ.get("MEMORY_REUSE_DEFAULT_TTL")
        if raw_ttl is None:
            default_ttl: int | None = 3600
        elif raw_ttl.lower() == "none":
            default_ttl = None
        else:
            default_ttl = int(raw_ttl)

        raw_stats = os.environ.get("MEMORY_REUSE_ENABLE_STATS", "true")
        enable_stats = raw_stats.lower() not in {"false", "0", "no"}

        backend = os.environ.get("MEMORY_REUSE_BACKEND", "memory")
        scope = os.environ.get("MEMORY_REUSE_DEFAULT_SCOPE", "global")

        raw_semantic = os.environ.get("MEMORY_REUSE_SEMANTIC_ENABLED", "false")
        semantic_enabled = raw_semantic.lower() in {"true", "1", "yes"}

        raw_threshold = os.environ.get("MEMORY_REUSE_SIMILARITY_THRESHOLD")
        similarity_threshold = 0.95 if raw_threshold is None else float(raw_threshold)

        embedding_provider = os.environ.get("MEMORY_REUSE_EMBEDDING_PROVIDER")

        return cls(
            backend=backend,  # type: ignore[arg-type]
            redis_url=os.environ.get("MEMORY_REUSE_REDIS_URL"),
            default_ttl=default_ttl,
            default_scope=scope,  # type: ignore[arg-type]
            key_prefix=os.environ.get("MEMORY_REUSE_KEY_PREFIX", "memreuse"),
            enable_stats=enable_stats,
            semantic_enabled=semantic_enabled,
            similarity_threshold=similarity_threshold,
            embedding_provider=embedding_provider,  # type: ignore[arg-type]
            embedding_model=os.environ.get("MEMORY_REUSE_EMBEDDING_MODEL"),
        )
