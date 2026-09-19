"""Single-flight coalescing of concurrent cache misses (Phase 6, Req 1, 2).

When several concurrent callers miss the same cache key at once, executing the
expensive underlying operation once and sharing its result — instead of every
caller paying — is a pure reliability and cost win. :class:`SingleFlight`
provides that coalescing.

In-process coordination uses an :class:`asyncio.Future` registry keyed by cache
key: the first caller for a key runs the compute; concurrent callers await the
same Future and receive the one result. Different keys proceed independently.

An optional distributed variant (enabled with ``distributed=True`` and a Redis
backend) extends coordination across processes with a per-key lock. It is
strictly fail-open: any lock failure or timeout degrades to running the compute
rather than blocking or raising, consistent with the library's posture that a
cache-coordination fault must never become an application failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class SingleFlight:
    """Coalesces concurrent misses for the same key so compute runs once.

    Args:
        backend: The storage backend. Used only by the optional distributed
            lock; the in-process path never touches it.
        distributed: When ``True``, coordinate across processes with a per-key
            Redis lock in addition to the in-process registry. Requires a
            backend that exposes the Redis client operations; falls open on any
            lock failure.
        lock_ttl: Expiry (seconds) applied to the distributed lock so a crashed
            holder cannot deadlock other processes (Req 2.4).
        lock_timeout: Maximum time (seconds) to wait for the distributed lock
            before falling open and computing anyway (Req 2.3).

    Example::

        sf = SingleFlight(backend)
        value = await sf.run(
            key,
            compute=lambda: call_llm(prompt),
            load=lambda: cache.exact.get(parts, scope="global", scope_id=None),
            store=lambda v: cache.exact.set(parts, v, scope="global", scope_id=None),
        )
    """

    def __init__(
        self,
        backend: Any,
        *,
        distributed: bool = False,
        lock_ttl: float = 30.0,
        lock_timeout: float = 10.0,
    ) -> None:
        self._backend = backend
        self._distributed = distributed
        self._lock_ttl = lock_ttl
        self._lock_timeout = lock_timeout
        # Per-process registry of in-flight computations, keyed by cache key.
        self._in_flight: dict[str, asyncio.Future[Any]] = {}

    async def run(
        self,
        key: str,
        *,
        compute: Callable[[], Awaitable[Any]],
        load: Callable[[], Awaitable[Any]],
        store: Callable[[Any], Awaitable[None]],
    ) -> Any:
        """Return the value for ``key``, computing it at most once across callers.

        The flow is: try ``load()`` first (a cache read) — on a hit return it
        without touching the registry (Req 1.3). On a miss, coalesce: the first
        caller for the key runs ``compute()``, ``store()``s the result, and
        resolves a shared Future; concurrent callers await that Future and share
        the result (Req 1.2, 1.4). On failure the exception propagates to all
        waiters, nothing is stored, and the key is cleared so a later call
        retries (Req 1.5, 1.6).

        Args:
            key: The cache key identifying the computation.
            compute: Async no-arg callable producing the value on a miss.
            load: Async no-arg callable returning the cached value or ``None``.
            store: Async callable persisting a computed value under the key.

        Returns:
            The cached value on a hit, or the freshly computed value on a miss.
        """
        # Fast path: an existing stored value never recomputes (Req 1.3).
        existing = await load()
        if existing is not None:
            return existing

        # Attach to an in-flight computation for this key if one exists, so
        # concurrent callers share the single result (Req 1.2, 1.7).
        in_flight = self._in_flight.get(key)
        if in_flight is not None:
            return await in_flight

        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._in_flight[key] = future

        try:
            result = await self._compute_once(key, compute=compute, load=load, store=store)
        except Exception as exc:
            # Propagate to every waiter; store nothing (Req 1.5).
            if not future.done():
                future.set_exception(exc)
            raise
        else:
            if not future.done():
                future.set_result(result)
            return result
        finally:
            # Always clear the key so a later call retries / re-reads (Req 1.6).
            self._in_flight.pop(key, None)

    async def _compute_once(
        self,
        key: str,
        *,
        compute: Callable[[], Awaitable[Any]],
        load: Callable[[], Awaitable[Any]],
        store: Callable[[Any], Awaitable[None]],
    ) -> Any:
        """Run the compute once for a key, optionally under a distributed lock.

        Without the distributed lock this simply computes and stores. With it,
        coordination extends across processes; any lock fault falls open to a
        plain compute (Req 2.3).
        """
        if self._distributed:
            return await self._compute_distributed(key, compute=compute, load=load, store=store)
        result = await compute()
        await store(result)
        return result

    async def _compute_distributed(
        self,
        key: str,
        *,
        compute: Callable[[], Awaitable[Any]],
        load: Callable[[], Awaitable[Any]],
        store: Callable[[Any], Awaitable[None]],
    ) -> Any:
        """Compute under a best-effort cross-process Redis lock (Req 2).

        Attempts to acquire a per-key lock before computing. On acquire it
        computes, stores, and releases (Req 2.1, 2.4). If the lock is held by
        another process, it polls until either the value appears (the holder
        stored it — Req 2.2) or the wait times out, at which point it falls open
        and computes anyway rather than blocking indefinitely (Req 2.3). Any lock
        fault degrades to a plain compute, never raising from the primitive.
        """
        # Fall open if the backend does not expose the lock operations — the
        # in-memory backend, for example, has no distributed lock.
        acquire = getattr(self._backend, "acquire_lock", None)
        release = getattr(self._backend, "release_lock", None)
        if acquire is None or release is None:
            result = await compute()
            await store(result)
            return result

        lock_key = self._lock_key(key)
        token = self._new_lock_token()
        deadline = self._now() + self._lock_timeout

        while True:
            try:
                acquired = await acquire(lock_key, token, int(self._lock_ttl * 1000))
            except Exception:
                # Lock backend fault → fall open (Req 2.3).
                acquired = True
                release = None  # nothing to release

            if acquired:
                try:
                    result = await compute()
                    await store(result)
                    return result
                finally:
                    if release is not None:
                        await release(lock_key, token)

            # Another process holds the lock: it may have stored the value.
            existing = await load()
            if existing is not None:
                return existing  # reuse the holder's result (Req 2.2)

            if self._now() >= deadline:
                # Timed out waiting: fall open and compute (Req 2.3).
                result = await compute()
                await store(result)
                return result

            await self._sleep(0.05)

    # ------------------------------------------------------------------
    # Distributed-lock helpers (used by the Task 4.1 Redis path)
    # ------------------------------------------------------------------

    @staticmethod
    def _new_lock_token() -> str:
        """Return a unique token identifying this holder of a distributed lock."""
        return secrets.token_hex(16)

    @staticmethod
    def _lock_key(key: str) -> str:
        """Derive the distributed-lock key for a cache key."""
        return f"memreuse:lock:{key}"

    @staticmethod
    def _now() -> float:
        """Monotonic clock reference for lock-wait timeouts."""
        return time.monotonic()

    @staticmethod
    async def _sleep(seconds: float) -> None:
        """Awaitable sleep used while polling for a distributed lock."""
        with contextlib.suppress(Exception):
            await asyncio.sleep(seconds)
