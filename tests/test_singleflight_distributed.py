"""Example tests for the distributed single-flight lock (Phase 6, Req 2).

These exercise the cross-process coordination path against an in-process fake
backend exposing the same ``acquire_lock`` / ``release_lock`` surface the Redis
backend provides, so the behaviour is deterministic and offline (no Redis
server). Covered:

* a holder computes and stores; a second caller reads the stored value rather
  than recomputing (Req 2.2);
* a failed/timed-out lock falls open to a compute rather than blocking (Req 2.3);
* ``distributed_lock=True`` without a Redis backend raises at construction
  (Req 2.6).
"""

from __future__ import annotations

import pytest

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.exceptions import ConfigurationError
from memory_reuse.execution import SingleFlight


class _LockableBackend(InMemoryBackend):
    """An in-memory backend that also exposes a simple distributed-lock surface.

    Models the Redis lock semantics (``SET NX`` + delete-if-owner) in-process so
    the distributed path can be tested without a Redis server. A ``fail_acquire``
    toggle simulates a lock backend that never grants the lock, exercising the
    fall-open path.
    """

    def __init__(self) -> None:
        super().__init__()
        self._locks: dict[str, str] = {}
        self.fail_acquire = False
        self.acquire_calls = 0

    async def acquire_lock(self, key: str, token: str, ttl_ms: int) -> bool:
        self.acquire_calls += 1
        if self.fail_acquire:
            return False
        if key in self._locks:
            return False
        self._locks[key] = token
        return True

    async def release_lock(self, key: str, token: str) -> None:
        if self._locks.get(key) == token:
            del self._locks[key]


@pytest.mark.asyncio
async def test_holder_stores_and_second_caller_reads_it() -> None:
    """A second caller reads the holder's stored value instead of recomputing (Req 2.2)."""
    backend = _LockableBackend()
    sf = SingleFlight(backend, distributed=True)
    calls = {"n": 0}

    async def compute() -> bytes:
        calls["n"] += 1
        return b"stored-by-holder"

    store: dict[str, bytes] = {}

    async def load() -> bytes | None:
        return store.get("k")

    async def do_store(v: bytes) -> None:
        store["k"] = v

    first = await sf.run("k", compute=compute, load=load, store=do_store)
    assert first == b"stored-by-holder"
    assert calls["n"] == 1

    # Second run: the value already exists, so compute is never called again.
    second = await sf.run("k", compute=compute, load=load, store=do_store)
    assert second == b"stored-by-holder"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_lock_denied_falls_open_to_compute() -> None:
    """When the lock can't be acquired and no value appears, compute runs (Req 2.3)."""
    backend = _LockableBackend()
    backend.fail_acquire = True  # the lock is never granted
    sf = SingleFlight(backend, distributed=True, lock_timeout=0.1)
    calls = {"n": 0}

    async def compute() -> bytes:
        calls["n"] += 1
        return b"computed-fallopen"

    async def load() -> bytes | None:
        return None  # holder never stores anything

    async def do_store(v: bytes) -> None:
        pass

    result = await sf.run("k", compute=compute, load=load, store=do_store)
    assert result == b"computed-fallopen"
    # Fell open rather than blocking forever.
    assert calls["n"] == 1


def test_distributed_lock_without_redis_raises() -> None:
    """distributed_lock=True on a non-Redis backend raises ConfigurationError (Req 2.6)."""
    with pytest.raises(ConfigurationError):
        MemoryCache(CacheConfig(backend="memory", single_flight=True, distributed_lock=True))
