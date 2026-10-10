"""Regression tests for namespace-safe Redis clearing (Fix #3).

``RedisBackend.flush()`` must delete only this cache's own keys (those under the
configured ``key_prefix``), never unrelated application keys or another cache
instance's data, and never ``FLUSHDB``. These tests drive the flush logic with a
small in-process fake Redis client (no server needed) so the key-selection
behaviour is asserted deterministically; the unrestricted ``flush_database`` is
verified to still exist for callers that explicitly want it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from memory_reuse.backends.redis import RedisBackend


class _FakeRedis:
    """Minimal in-process async stand-in for the redis.asyncio client.

    Supports exactly the operations RedisBackend.flush / flush_database use:
    ``scan_iter`` (glob match), ``unlink`` / ``delete``, and ``flushdb``.
    """

    def __init__(self, keys: dict[bytes, bytes]) -> None:
        self.store = dict(keys)
        self.flushdb_called = False

    async def scan_iter(self, match: str, count: int = 100) -> AsyncIterator[bytes]:
        # Translate the ``prefix:*`` glob into a startswith check.
        assert match.endswith("*")
        prefix = match[:-1].encode("utf-8")
        for key in list(self.store):
            if key.startswith(prefix):
                yield key

    async def unlink(self, *keys: bytes) -> int:
        n = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                n += 1
        return n

    async def delete(self, *keys: bytes) -> int:
        return await self.unlink(*keys)

    async def flushdb(self) -> None:
        self.flushdb_called = True
        self.store.clear()


def _backend_with(fake: _FakeRedis, *, key_prefix: str | None) -> RedisBackend:
    backend = RedisBackend(url="redis://unused", key_prefix=key_prefix)
    # Inject the fake client so no real connection is attempted.
    backend._client = fake  # type: ignore[assignment]
    return backend


def _seed() -> dict[bytes, bytes]:
    return {
        b"memreuse:global:aaa": b"1",
        b"memreuse:user:alice:bbb": b"2",
        b"other-app:session:xyz": b"3",  # unrelated application key
        b"another-cache:global:zzz": b"4",  # a different cache instance's key
    }


class TestNamespaceSafeFlush:
    async def test_flush_deletes_only_prefixed_keys(self) -> None:
        fake = _FakeRedis(_seed())
        backend = _backend_with(fake, key_prefix="memreuse")

        await backend.flush()

        remaining = set(fake.store)
        assert remaining == {b"other-app:session:xyz", b"another-cache:global:zzz"}
        assert not fake.flushdb_called, "flush must not call FLUSHDB"

    async def test_flush_does_not_touch_other_cache_instance(self) -> None:
        fake = _FakeRedis(_seed())
        backend = _backend_with(fake, key_prefix="another-cache")

        await backend.flush()

        # Only the ``another-cache`` namespace was cleared.
        assert b"another-cache:global:zzz" not in fake.store
        assert b"memreuse:global:aaa" in fake.store
        assert b"other-app:session:xyz" in fake.store

    async def test_flush_without_prefix_is_noop_not_flushdb(self) -> None:
        fake = _FakeRedis(_seed())
        backend = _backend_with(fake, key_prefix=None)

        await backend.flush()

        # Refuses to guess a scope: nothing deleted, FLUSHDB never called.
        assert set(fake.store) == set(_seed())
        assert not fake.flushdb_called


class TestExplicitFullFlush:
    async def test_flush_database_calls_flushdb(self) -> None:
        fake = _FakeRedis(_seed())
        backend = _backend_with(fake, key_prefix="memreuse")

        await backend.flush_database()

        assert fake.flushdb_called
        assert fake.store == {}
