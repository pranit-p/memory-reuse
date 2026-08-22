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

    def __post_init__(self) -> None:
        """Validate the configuration after initialisation."""
        if self.default_ttl is not None and self.default_ttl <= 0:
            from memory_reuse.exceptions import InvalidTTLError

            raise InvalidTTLError(
                f"default_ttl must be a positive integer or None, got {self.default_ttl}"
            )
        if self.max_key_size <= 0:
            raise ValueError(f"max_key_size must be positive, got {self.max_key_size}")

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

        return cls(
            backend=backend,  # type: ignore[arg-type]
            redis_url=os.environ.get("MEMORY_REUSE_REDIS_URL"),
            default_ttl=default_ttl,
            default_scope=scope,  # type: ignore[arg-type]
            key_prefix=os.environ.get("MEMORY_REUSE_KEY_PREFIX", "memreuse"),
            enable_stats=enable_stats,
        )
