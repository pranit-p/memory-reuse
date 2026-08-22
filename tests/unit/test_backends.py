"""Unit tests for cache backends."""

from __future__ import annotations

import pytest

from memory_reuse.backends.memory import InMemoryBackend


class TestInMemoryBackend:
    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        backend = InMemoryBackend()
        await backend.set("key1", b"value1")
        assert await backend.get("key1") == b"value1"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self) -> None:
        backend = InMemoryBackend()
        assert await backend.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        backend = InMemoryBackend()
        await backend.set("key", b"data")
        await backend.delete("key")
        assert await backend.get("key") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_no_error(self) -> None:
        backend = InMemoryBackend()
        await backend.delete("does_not_exist")  # Should not raise

    @pytest.mark.asyncio
    async def test_exists_true(self) -> None:
        backend = InMemoryBackend()
        await backend.set("k", b"v")
        assert await backend.exists("k") is True

    @pytest.mark.asyncio
    async def test_exists_false(self) -> None:
        backend = InMemoryBackend()
        assert await backend.exists("missing") is False

    @pytest.mark.asyncio
    async def test_flush(self) -> None:
        backend = InMemoryBackend()
        await backend.set("a", b"1")
        await backend.set("b", b"2")
        await backend.flush()
        assert await backend.get("a") is None
        assert await backend.get("b") is None
        assert backend.size == 0

    @pytest.mark.asyncio
    async def test_ping_returns_true(self) -> None:
        backend = InMemoryBackend()
        assert await backend.ping() is True

    @pytest.mark.asyncio
    async def test_ttl_expiry(self) -> None:
        backend = InMemoryBackend()
        # Set with a very short TTL (use internal clock trick)
        await backend.set("expiring", b"val", ttl=1)
        assert await backend.get("expiring") == b"val"
        # Manually expire by advancing the stored timestamp
        entry = backend._store["expiring"]
        entry.expires_at = 0.0  # Unix epoch — already in the past
        assert await backend.get("expiring") is None

    @pytest.mark.asyncio
    async def test_ttl_expiry_exists(self) -> None:
        backend = InMemoryBackend()
        await backend.set("exp2", b"val", ttl=1)
        entry = backend._store["exp2"]
        entry.expires_at = 0.0
        assert await backend.exists("exp2") is False

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        backend = InMemoryBackend(max_entries=3)
        await backend.set("a", b"1")
        await backend.set("b", b"2")
        await backend.set("c", b"3")
        # Adding a 4th entry evicts "a" (LRU)
        await backend.set("d", b"4")
        assert await backend.get("a") is None
        assert await backend.get("b") == b"2"
        assert await backend.get("c") == b"3"
        assert await backend.get("d") == b"4"
        assert backend.size == 3

    @pytest.mark.asyncio
    async def test_lru_ordering_on_access(self) -> None:
        """Accessing 'a' should protect it from eviction."""
        backend = InMemoryBackend(max_entries=3)
        await backend.set("a", b"1")
        await backend.set("b", b"2")
        await backend.set("c", b"3")
        # Access "a" to make it recently used
        await backend.get("a")
        # Add "d" — now "b" is LRU
        await backend.set("d", b"4")
        assert await backend.get("b") is None
        assert await backend.get("a") == b"1"

    @pytest.mark.asyncio
    async def test_overwrite_updates_value(self) -> None:
        backend = InMemoryBackend()
        await backend.set("k", b"old")
        await backend.set("k", b"new")
        assert await backend.get("k") == b"new"

    @pytest.mark.asyncio
    async def test_invalid_max_entries_raises(self) -> None:
        with pytest.raises(ValueError):
            InMemoryBackend(max_entries=0)

    @pytest.mark.asyncio
    async def test_no_expiry(self) -> None:
        backend = InMemoryBackend()
        await backend.set("forever", b"val", ttl=None)
        assert await backend.get("forever") == b"val"
