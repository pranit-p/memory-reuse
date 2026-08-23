"""Integration tests for :class:`RedisVectorIndex`.

These tests require a running Redis server. They are skipped automatically when
one is not reachable, so the default offline test run stays green.

Set ``MEMORY_REUSE_REDIS_URL`` to point at a Redis instance (defaults to
``redis://localhost:6379/15`` — database 15 is used to avoid clobbering real
data). The suite runs against whatever Redis is available: a Redis Stack build
exercises the native ``FT.SEARCH`` KNN path, while a plain Redis exercises the
in-process cosine fallback. Both must satisfy the same contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from collections.abc import AsyncIterator

import pytest

from memory_reuse._utils import deserialize_value, hash_value, serialize_value
from memory_reuse.exceptions import BackendConnectionError, ProviderMismatchError
from memory_reuse.vector import RedisVectorIndex, VectorRecord

REDIS_URL = os.environ.get("MEMORY_REUSE_REDIS_URL", "redis://localhost:6379/15")
PROVIDER = "fake:test-model"


def _redis_available() -> bool:
    """Return ``True`` when a Redis server answers PING at ``REDIS_URL``."""
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


def _record(
    vector: list[float],
    value: object,
    *,
    provider_model: str = PROVIDER,
    expires_at: float | None = None,
) -> VectorRecord:
    """Build a :class:`VectorRecord` with a serialised value."""
    return VectorRecord(
        vector=vector,
        value=serialize_value(value),
        provider_model=provider_model,
        expires_at=expires_at,
    )


def _rid(query: str, provider_model: str = PROVIDER) -> str:
    """Deterministic record id following the documented convention."""
    return hash_value([provider_model, query])


@pytest.fixture
async def index() -> AsyncIterator[RedisVectorIndex]:
    """A RedisVectorIndex backed by a flushed test namespace prefix."""
    idx = RedisVectorIndex(url=REDIS_URL)
    await idx.flush()
    try:
        yield idx
    finally:
        await idx.flush()
        await idx.close()


class TestAddAndSearch:
    @pytest.mark.asyncio
    async def test_add_then_search_returns_value(self, index: RedisVectorIndex) -> None:
        await index.add("global", _rid("q"), _record([1.0, 0.0], {"answer": 42}))

        matches = await index.search("global", [1.0, 0.0], PROVIDER, top_k=1)

        assert len(matches) == 1
        assert matches[0].score == pytest.approx(1.0, abs=1e-4)
        assert deserialize_value(matches[0].value) == {"answer": 42}

    @pytest.mark.asyncio
    async def test_search_empty_namespace_returns_empty(self, index: RedisVectorIndex) -> None:
        assert await index.search("global", [1.0, 0.0], PROVIDER) == []

    @pytest.mark.asyncio
    async def test_search_top_k_zero_returns_empty(self, index: RedisVectorIndex) -> None:
        await index.add("global", _rid("q"), _record([1.0, 0.0], "v"))
        assert await index.search("global", [1.0, 0.0], PROVIDER, top_k=0) == []

    @pytest.mark.asyncio
    async def test_top_k_ordering_is_descending_by_score(self, index: RedisVectorIndex) -> None:
        await index.add("global", _rid("near"), _record([1.0, 0.0], "near"))
        await index.add("global", _rid("mid"), _record([1.0, 1.0], "mid"))
        await index.add("global", _rid("far"), _record([-1.0, 0.0], "far"))

        matches = await index.search("global", [1.0, 0.0], PROVIDER, top_k=3)

        assert [deserialize_value(m.value) for m in matches] == ["near", "mid", "far"]
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_top_k_limits_returned_results(self, index: RedisVectorIndex) -> None:
        for i in range(5):
            await index.add("global", _rid(f"q{i}"), _record([float(i), 1.0], f"v{i}"))

        matches = await index.search("global", [1.0, 1.0], PROVIDER, top_k=2)
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_overwrite_same_record_id_updates(self, index: RedisVectorIndex) -> None:
        rid = _rid("q")
        await index.add("global", rid, _record([1.0, 0.0], "v0"))
        await index.add("global", rid, _record([1.0, 0.0], "v1"))

        matches = await index.search("global", [1.0, 0.0], PROVIDER, top_k=5)
        assert [deserialize_value(m.value) for m in matches] == ["v1"]


class TestNamespaceIsolation:
    @pytest.mark.asyncio
    async def test_search_never_crosses_namespaces(self, index: RedisVectorIndex) -> None:
        await index.add("user:alice", _rid("q"), _record([1.0, 0.0], "alice-secret"))
        await index.add("user:bob", _rid("q"), _record([1.0, 0.0], "bob-secret"))
        await index.add("global", _rid("q"), _record([1.0, 0.0], "global-value"))

        alice = await index.search("user:alice", [1.0, 0.0], PROVIDER, top_k=10)
        assert [deserialize_value(m.value) for m in alice] == ["alice-secret"]

        bob = await index.search("user:bob", [1.0, 0.0], PROVIDER, top_k=10)
        assert [deserialize_value(m.value) for m in bob] == ["bob-secret"]

    @pytest.mark.asyncio
    async def test_unknown_namespace_returns_empty(self, index: RedisVectorIndex) -> None:
        await index.add("user:alice", _rid("q"), _record([1.0], "v"))
        assert await index.search("user:charlie", [1.0], PROVIDER) == []


class TestTTLFiltering:
    @pytest.mark.asyncio
    async def test_expired_record_is_filtered_on_read(self, index: RedisVectorIndex) -> None:
        # Already-expired records are simply not written.
        await index.add(
            "global",
            _rid("expiring"),
            _record([1.0, 0.0], "gone", expires_at=time.time() - 1.0),
        )
        assert await index.search("global", [1.0, 0.0], PROVIDER) == []

    @pytest.mark.asyncio
    async def test_non_expired_record_is_returned(self, index: RedisVectorIndex) -> None:
        await index.add(
            "global",
            _rid("live"),
            _record([1.0, 0.0], "here", expires_at=time.time() + 100.0),
        )
        matches = await index.search("global", [1.0, 0.0], PROVIDER)
        assert deserialize_value(matches[0].value) == "here"

    @pytest.mark.asyncio
    async def test_ttl_is_set_on_write(self, index: RedisVectorIndex) -> None:
        rid = _rid("live")
        await index.add("global", rid, _record([1.0], "v", expires_at=time.time() + 100.0))
        client = await index._get_client()
        ttl = await client.ttl(RedisVectorIndex._record_key("global", rid))
        assert 0 < ttl <= 100


class TestProviderMismatch:
    @pytest.mark.asyncio
    async def test_add_with_mismatched_provider_raises(self, index: RedisVectorIndex) -> None:
        await index.add("global", _rid("q0"), _record([1.0], "v0", provider_model="openai:a"))
        with pytest.raises(ProviderMismatchError):
            await index.add("global", _rid("q1"), _record([1.0], "v1", provider_model="local:b"))

    @pytest.mark.asyncio
    async def test_search_with_mismatched_provider_raises(self, index: RedisVectorIndex) -> None:
        await index.add("global", _rid("q0"), _record([1.0], "v0", provider_model="openai:a"))
        with pytest.raises(ProviderMismatchError):
            await index.search("global", [1.0], "local:b")

    @pytest.mark.asyncio
    async def test_same_provider_ok(self, index: RedisVectorIndex) -> None:
        await index.add("global", _rid("q0"), _record([1.0, 0.0], "v0"))
        await index.add("global", _rid("q1"), _record([0.0, 1.0], "v1"))
        matches = await index.search("global", [1.0, 0.0], PROVIDER, top_k=2)
        assert len(matches) == 2


class TestDeleteAndFlush:
    @pytest.mark.asyncio
    async def test_delete_namespace(self, index: RedisVectorIndex) -> None:
        await index.add("user:a", _rid("q"), _record([1.0], "v"))
        await index.delete_namespace("user:a")
        assert await index.search("user:a", [1.0], PROVIDER) == []

    @pytest.mark.asyncio
    async def test_delete_unknown_namespace_no_error(self, index: RedisVectorIndex) -> None:
        await index.delete_namespace("nope")  # should not raise

    @pytest.mark.asyncio
    async def test_flush_clears_all_namespaces(self, index: RedisVectorIndex) -> None:
        await index.add("user:a", _rid("q"), _record([1.0], "a"))
        await index.add("global", _rid("q"), _record([1.0], "g"))
        await index.flush()
        assert await index.search("user:a", [1.0], PROVIDER) == []
        assert await index.search("global", [1.0], PROVIDER) == []


class TestFallbackCandidateLimit:
    @pytest.mark.asyncio
    async def test_fallback_refuses_above_candidate_limit(self, index: RedisVectorIndex) -> None:
        # Force the in-process fallback path regardless of the server build.
        index._search_available = False
        small = RedisVectorIndex(url=REDIS_URL, max_scan_candidates=2)
        small._search_available = False
        await small.flush()
        try:
            for i in range(3):
                await small.add("global", _rid(f"q{i}"), _record([float(i)], f"v{i}"))
            with pytest.raises(BackendConnectionError, match="candidate limit"):
                await small.search("global", [1.0], PROVIDER)
        finally:
            await small.flush()
            await small.close()


# ---------------------------------------------------------------------------
# Native FT.SEARCH KNN path
# ---------------------------------------------------------------------------
#
# RediSearch only indexes database 0, so the native path is exercised against a
# db-0 URL with a per-run unique index name. All vectors share one dimension
# because a single FT index has a fixed vector width. These tests skip unless a
# Redis Stack server (search module) is reachable on db 0.

_NATIVE_URL = os.environ.get("MEMORY_REUSE_REDIS_URL_DB0", "redis://localhost:6379/0")
_NATIVE_DIM = 4


def _native_search_available() -> bool:
    """Return ``True`` when a Redis Stack search module is reachable on db 0."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return False

    async def _probe() -> bool:
        client = aioredis.Redis.from_url(_NATIVE_URL, decode_responses=False)
        try:
            modules = await client.execute_command("MODULE", "LIST")
            for entry in modules:
                if isinstance(entry, dict):
                    name = entry.get(b"name") or entry.get("name")
                    if name and str(name).lower().find("search") != -1:
                        return True
            return False
        finally:
            await client.aclose()

    try:
        return asyncio.run(_probe())
    except Exception:
        return False


def _vec(*values: float) -> list[float]:
    """Pad/trim to the fixed native dimension so one FT index fits all vectors."""
    out = list(values)[:_NATIVE_DIM]
    out.extend([0.0] * (_NATIVE_DIM - len(out)))
    return out


@pytest.mark.skipif(
    not _native_search_available(),
    reason="Redis Stack search module not available on db 0",
)
class TestNativeKnnPath:
    @pytest.fixture
    async def native_index(self) -> AsyncIterator[RedisVectorIndex]:
        idx = RedisVectorIndex(url=_NATIVE_URL, index_name=f"memreuse_test_{uuid.uuid4().hex}")
        await idx.flush()
        try:
            yield idx
        finally:
            await idx.flush()
            client = await idx._get_client()
            with contextlib.suppress(Exception):
                await client.ft(idx._index_name).dropindex()
            await idx.close()

    @pytest.mark.asyncio
    async def test_uses_native_path(self, native_index: RedisVectorIndex) -> None:
        # Trigger index creation and confirm the native path is selected.
        await native_index.add("global", _rid("q"), _record(_vec(1.0, 0.0), {"answer": 42}))
        client = await native_index._get_client()
        assert await native_index._search_module_available(client) is True
        assert native_index._search_available is True
        assert native_index._index_ready is True

    @pytest.mark.asyncio
    async def test_add_then_search_returns_value(self, native_index: RedisVectorIndex) -> None:
        await native_index.add("global", _rid("q"), _record(_vec(1.0, 0.0), {"answer": 42}))
        matches = await native_index.search("global", _vec(1.0, 0.0), PROVIDER, top_k=1)
        assert len(matches) == 1
        assert matches[0].score == pytest.approx(1.0, abs=1e-3)
        assert deserialize_value(matches[0].value) == {"answer": 42}

    @pytest.mark.asyncio
    async def test_best_match_ordering(self, native_index: RedisVectorIndex) -> None:
        await native_index.add("global", _rid("near"), _record(_vec(1.0, 0.0), "near"))
        await native_index.add("global", _rid("mid"), _record(_vec(1.0, 1.0), "mid"))
        await native_index.add("global", _rid("far"), _record(_vec(-1.0, 0.0), "far"))
        matches = await native_index.search("global", _vec(1.0, 0.0), PROVIDER, top_k=3)
        assert [deserialize_value(m.value) for m in matches] == ["near", "mid", "far"]
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, native_index: RedisVectorIndex) -> None:
        await native_index.add("user:alice", _rid("q"), _record(_vec(1.0, 0.0), "alice"))
        await native_index.add("user:bob", _rid("q"), _record(_vec(1.0, 0.0), "bob"))
        alice = await native_index.search("user:alice", _vec(1.0, 0.0), PROVIDER, top_k=10)
        assert [deserialize_value(m.value) for m in alice] == ["alice"]

    @pytest.mark.asyncio
    async def test_expired_record_filtered(self, native_index: RedisVectorIndex) -> None:
        await native_index.add(
            "global",
            _rid("live"),
            _record(_vec(1.0, 0.0), "here", expires_at=time.time() + 100.0),
        )
        matches = await native_index.search("global", _vec(1.0, 0.0), PROVIDER)
        assert deserialize_value(matches[0].value) == "here"

    @pytest.mark.asyncio
    async def test_provider_mismatch_raises(self, native_index: RedisVectorIndex) -> None:
        await native_index.add(
            "global", _rid("q0"), _record(_vec(1.0, 0.0), "v0", provider_model="openai:a")
        )
        with pytest.raises(ProviderMismatchError):
            await native_index.search("global", _vec(1.0, 0.0), "local:b")
