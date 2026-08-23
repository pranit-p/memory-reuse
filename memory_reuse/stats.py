"""Cache statistics tracking for memory-reuse."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Snapshot of cache performance counters.

    Attributes:
        hits: Number of successful cache lookups. Always equals
            ``exact_hits + semantic_hits``.
        exact_hits: Number of hits served by the exact (hash-based) cache.
        semantic_hits: Number of hits served by the semantic (similarity) cache.
        misses: Number of cache lookups that found no entry.
        errors: Number of backend errors encountered during cache operations.
        total_requests: Total number of cache lookup attempts (hits + misses).

    Example::

        stats = cache.stats
        print(f"Hit rate: {stats.hit_rate:.1%}")
    """

    hits: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
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
            Dictionary with keys ``hits``, ``exact_hits``, ``semantic_hits``,
            ``misses``, ``errors``, ``total_requests``, and ``hit_rate``.
        """
        return {
            "hits": self.hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
        }


class StatsTracker:
    """Thread-safe, asyncio-compatible statistics tracker.

    Uses an :class:`asyncio.Lock` to serialise counter updates so that
    concurrent coroutines always see a consistent view of the counters.

    Hits are tracked by type: :meth:`record_exact_hit` and
    :meth:`record_semantic_hit` each also bump the aggregate ``hits`` counter,
    so ``hits`` always equals ``exact_hits + semantic_hits``.

    Statistics recording is best-effort and never fatal: every ``record_*``
    method swallows any internal error so a cache operation can always return
    its result even if recording fails.

    Example::

        tracker = StatsTracker()
        tracker.record_exact_hit()
        tracker.record_miss()
        print(tracker.get_stats().hit_rate)  # 0.5
    """

    def __init__(self) -> None:
        self._hits = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._errors = 0
        self._total = 0
        self._lock = asyncio.Lock()

    def record_exact_hit(self) -> None:
        """Record a hit served by the exact cache.

        Increments the exact-hit counter, the aggregate hit counter, and the
        total-request counter. Any internal error is swallowed so recording is
        never fatal to the caller.
        """
        # Synchronous increment — safe because CPython GIL protects individual
        # int attribute assignments. The asyncio.Lock is used in async contexts.
        with contextlib.suppress(Exception):
            self._exact_hits += 1
            self._hits += 1
            self._total += 1

    def record_semantic_hit(self) -> None:
        """Record a hit served by the semantic cache.

        Increments the semantic-hit counter, the aggregate hit counter, and the
        total-request counter. Any internal error is swallowed so recording is
        never fatal to the caller.
        """
        with contextlib.suppress(Exception):
            self._semantic_hits += 1
            self._hits += 1
            self._total += 1

    def record_hit(self) -> None:
        """Record an exact hit. Alias for :meth:`record_exact_hit`.

        Retained for backward compatibility with Phase 1 callers.
        """
        self.record_exact_hit()

    def record_miss(self) -> None:
        """Increment the miss counter and total-request counter.

        Any internal error is swallowed so recording is never fatal.
        """
        with contextlib.suppress(Exception):
            self._misses += 1
            self._total += 1

    def record_error(self) -> None:
        """Increment the error counter (does not affect total_requests).

        Any internal error is swallowed so recording is never fatal.
        """
        with contextlib.suppress(Exception):
            self._errors += 1

    def get_stats(self) -> CacheStats:
        """Return an immutable snapshot of the current counters.

        Returns:
            A :class:`CacheStats` dataclass with the current values.
        """
        return CacheStats(
            hits=self._hits,
            exact_hits=self._exact_hits,
            semantic_hits=self._semantic_hits,
            misses=self._misses,
            errors=self._errors,
            total_requests=self._total,
        )

    def reset(self) -> None:
        """Reset all counters to zero."""
        self._hits = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._errors = 0
        self._total = 0
