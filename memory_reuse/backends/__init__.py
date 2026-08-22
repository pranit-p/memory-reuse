"""Cache backend implementations for memory-reuse.

Available backends:

* :class:`~memory_reuse.backends.memory.InMemoryBackend` — zero-dependency,
  in-process storage with LRU eviction and TTL support.
* :class:`~memory_reuse.backends.redis.RedisBackend` — Redis-backed storage
  (requires ``pip install memory-reuse[redis]``).
"""

from memory_reuse.backends.base import AbstractBackend
from memory_reuse.backends.memory import InMemoryBackend

__all__ = ["AbstractBackend", "InMemoryBackend"]

try:
    from memory_reuse.backends.redis import RedisBackend  # noqa: F401

    __all__.append("RedisBackend")
except ImportError:
    pass
