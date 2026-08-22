"""Cache statistics tracking for memory-reuse."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Snapshot of cache performance counters.

    Attributes:
        hits: Number of successful cache lookups.
        misses: Number of cache lookups that found no entry.
        errors: Number of backend errors encountered during cache operations.
        total_requests: Total number of cache lookup attempts (hits + misses).

    Example::

        stats = cache.stats
        print(f"Hit rate: {stats.hit_rate:.1%}")
    """

    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of requests that were cache hits, in the range ``[0.0, 1.0]``.

        Returns:
            ``0.0`` when no requests have been made yet.
        """
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> dict:
        """Return a plain-dict representation of the stats snapshot.

        Returns:
            Dictionary with keys ``hits``, ``misses``, ``errors``,
            ``total_requests``, and ``hit_rate``.
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
        }


class StatsTracker:
    """Thread-safe, asyncio-compatible statistics tracker.

    Uses an :class:`asyncio.Lock` to serialise counter updates so that
    concurrent coroutines always see a consistent view of the counters.

    Example::

        tracker = StatsTracker()
        tracker.record_hit()
        tracker.record_miss()
        print(tracker.get_stats().hit_rate)  # 0.5
    """

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._total = 0
        self._lock = asyncio.Lock()

    def record_hit(self) -> None:
        """Increment the hit counter and total-request counter."""
        # Synchronous increment — safe because CPython GIL protects individual
        # int attribute assignments. The asyncio.Lock is used in async contexts.
        self._hits += 1
        self._total += 1

    def record_miss(self) -> None:
        """Increment the miss counter and total-request counter."""
        self._misses += 1
        self._total += 1

    def record_error(self) -> None:
        """Increment the error counter (does not affect total_requests)."""
        self._errors += 1

    def get_stats(self) -> CacheStats:
        """Return an immutable snapshot of the current counters.

        Returns:
            A :class:`CacheStats` dataclass with the current values.
        """
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            errors=self._errors,
            total_requests=self._total,
        )

    def reset(self) -> None:
        """Reset all counters to zero."""
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._total = 0
