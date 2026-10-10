"""Integration tests for namespace-safe Redis clearing (Fix #3).

These require a running Redis server and are skipped automatically when one is
not reachable, so the default offline run stays green. They assert on a real
server that ``RedisBackend.flush()`` removes only keys under the configured
``key_prefix`` — never unrelated application keys — and that TTL, delete
(invalidation), and the explicit full ``flush_database`` still behave correctly.

Set ``MEMORY_REUSE_REDIS_URL`` to point at a Redis instance (defaults to
``redis://localhost:6379/15`` — database 15 to avoid clobbering real data).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from memory_reuse.backends.redis import RedisBackend

REDIS_URL = os.environ.get("MEMORY_REUSE_REDIS_URL", "redis://localhost:6379/15")


def _redis_available() -> bool:
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return False

    async def _ping() -> bool:
        client = aioredis.Redis.from_url(REDIS_URL, decode_responses=False)
        try:
            return bool(await client.ping())
        finally:
            await client.aclose()

    try:
        return asyncio.run(_ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis server not available at MEMORY_REUSE_REDIS_URL",
)


@pytest.fixture
async def backend():
    be = RedisBackend(url=REDIS_URL, key_prefix="memreuse_itest")
    await be.flush()
    try:
        yield be
    finally:
        await be.flush()
        await be.close()


@pytest.mark.asyncio
async def test_flush_removes_only_prefixed_keys(backend: RedisBackend) -> None:
    """flush() clears this cache's namespace but leaves unrelated keys intact."""
    await backend.set("memreuse_itest:global:a", b"1", ttl=300)
    await backend.set("memreuse_itest:user:alice:b", b"2", ttl=300)

    # An unrelated key written directly, outside the cache's prefix.
    client = await backend._get_client()
    await client.set("unrelated_itest_key", b"keep-me")

    try:
        await backend.flush()

        assert await backend.get("memreuse_itest:global:a") is None
        assert await backend.get("memreuse_itest:user:alice:b") is None
        # The unrelated key survives — flush never ran FLUSHDB.
        assert await client.get("unrelated_itest_key") == b"keep-me"
    finally:
        await client.delete("unrelated_itest_key")


@pytest.mark.asyncio
async def test_ttl_expiry_and_invalidation(backend: RedisBackend) -> None:
    """TTL expiry and explicit delete both still work (preserved behaviour)."""
    await backend.set("memreuse_itest:global:ttl", b"v", ttl=1)
    assert await backend.get("memreuse_itest:global:ttl") == b"v"

    await backend.set("memreuse_itest:global:del", b"v", ttl=300)
    await backend.delete("memreuse_itest:global:del")
    assert await backend.get("memreuse_itest:global:del") is None

    await asyncio.sleep(1.2)
    assert await backend.get("memreuse_itest:global:ttl") is None


@pytest.mark.asyncio
async def test_flush_database_clears_everything(backend: RedisBackend) -> None:
    """flush_database() is the explicit, unrestricted FLUSHDB."""
    await backend.set("memreuse_itest:global:x", b"1", ttl=300)
    client = await backend._get_client()
    await client.set("another_itest_key", b"2")

    await backend.flush_database()

    assert await backend.get("memreuse_itest:global:x") is None
    assert await client.get("another_itest_key") is None
