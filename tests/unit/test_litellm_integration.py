"""Unit tests for the LiteLLM integration.

LiteLLM itself is mocked so these tests run without a real API key or
network connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.integrations.litellm import (
    cached_litellm_completion,
    cached_litellm_embedding,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache() -> MemoryCache:
    """In-memory cache with a 1-hour default TTL."""
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


def _make_completion_response(content: str = "Paris") -> MagicMock:
    """Build a minimal mock litellm ModelResponse."""
    msg = MagicMock()
    msg.content = content

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    resp.model_dump.return_value = {
        "choices": [{"message": {"content": content, "role": "assistant"}}],
        "model": "gpt-4o-mini",
    }
    return resp


def _make_embedding_response(dims: int = 3) -> MagicMock:
    """Build a minimal mock litellm EmbeddingResponse."""
    item = {"embedding": [0.1] * dims, "index": 0}
    resp = MagicMock()
    resp.data = [item]
    resp.model_dump.return_value = {"data": [item], "model": "text-embedding-3-small"}
    return resp


# ---------------------------------------------------------------------------
# cached_litellm_completion
# ---------------------------------------------------------------------------


class TestCachedLitellmCompletion:
    async def test_miss_calls_litellm(self, cache: MemoryCache) -> None:
        """On cache miss the underlying LiteLLM call must be made exactly once."""
        mock_resp = _make_completion_response("Paris")

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            result = await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Capital of France?"}],
                ttl=300,
            )

        fake_litellm.acompletion.assert_awaited_once()
        assert result is mock_resp

    async def test_hit_does_not_call_litellm(self, cache: MemoryCache) -> None:
        """On cache hit LiteLLM must NOT be called a second time."""
        mock_resp = _make_completion_response("Berlin")

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            messages = [{"role": "user", "content": "Capital of Germany?"}]

            # First call — miss
            await cached_litellm_completion(cache, model="gpt-4o-mini", messages=messages, ttl=300)
            # Second call — hit
            await cached_litellm_completion(cache, model="gpt-4o-mini", messages=messages, ttl=300)

        # acompletion should have been awaited exactly once (first call only)
        assert fake_litellm.acompletion.await_count == 1

    async def test_different_messages_different_cache_entries(self, cache: MemoryCache) -> None:
        """Different messages must produce separate cache entries."""
        resp_a = _make_completion_response("Rome")
        resp_b = _make_completion_response("Tokyo")

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(side_effect=[resp_a, resp_b])
            mock_import.return_value = fake_litellm

            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Capital of Italy?"}],
                ttl=300,
            )
            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Capital of Japan?"}],
                ttl=300,
            )

        assert fake_litellm.acompletion.await_count == 2

    async def test_different_models_different_cache_entries(self, cache: MemoryCache) -> None:
        """Same prompt with different models must NOT share a cache entry."""
        resp_a = _make_completion_response("answer-a")
        resp_b = _make_completion_response("answer-b")
        messages = [{"role": "user", "content": "Hello"}]

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(side_effect=[resp_a, resp_b])
            mock_import.return_value = fake_litellm

            await cached_litellm_completion(cache, model="gpt-4o-mini", messages=messages, ttl=300)
            await cached_litellm_completion(
                cache, model="claude-3-haiku", messages=messages, ttl=300
            )

        assert fake_litellm.acompletion.await_count == 2

    async def test_stats_updated_on_hit_and_miss(self, cache: MemoryCache) -> None:
        """Cache stats must reflect hits and misses correctly."""
        mock_resp = _make_completion_response("answer")
        messages = [{"role": "user", "content": "Stat test"}]

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            await cached_litellm_completion(
                cache, model="gpt-4o-mini", messages=messages, ttl=300
            )  # miss
            await cached_litellm_completion(
                cache, model="gpt-4o-mini", messages=messages, ttl=300
            )  # hit

        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    async def test_user_scope_isolation(self, cache: MemoryCache) -> None:
        """User-scoped entries must be isolated — alice cannot see bob's cache."""
        resp_alice = _make_completion_response("alice-answer")
        resp_bob = _make_completion_response("bob-answer")
        messages = [{"role": "user", "content": "Same question"}]

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(side_effect=[resp_alice, resp_bob])
            mock_import.return_value = fake_litellm

            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=messages,
                scope="user",
                user_id="alice",
                ttl=300,
            )
            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=messages,
                scope="user",
                user_id="bob",
                ttl=300,
            )

        # Both alice and bob trigger separate LLM calls
        assert fake_litellm.acompletion.await_count == 2

    async def test_user_scope_hit_same_user(self, cache: MemoryCache) -> None:
        """Same user asking the same question twice must hit the cache."""
        mock_resp = _make_completion_response("cached")
        messages = [{"role": "user", "content": "Repeat question"}]

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=messages,
                scope="user",
                user_id="alice",
                ttl=300,
            )
            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=messages,
                scope="user",
                user_id="alice",
                ttl=300,
            )

        assert fake_litellm.acompletion.await_count == 1

    async def test_litellm_not_installed_raises(self, cache: MemoryCache) -> None:
        """ImportError raised with clear instructions when LiteLLM missing."""
        with (
            patch(
                "memory_reuse.integrations.litellm._assert_litellm_installed",
                side_effect=ImportError("LiteLLM is not installed"),
            ),
            pytest.raises(ImportError, match="LiteLLM is not installed"),
        ):
            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
            )

    async def test_context_user_id_used_as_fallback(self, cache: MemoryCache) -> None:
        """user_id from MemoryCache context is used when not passed explicitly."""
        cache.set_context(user_id="context-user")
        mock_resp = _make_completion_response("context-answer")

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            # No user_id kwarg — should fall back to cache context
            await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "context test"}],
                scope="user",
                ttl=300,
            )

        fake_litellm.acompletion.assert_awaited_once()


# ---------------------------------------------------------------------------
# cached_litellm_embedding
# ---------------------------------------------------------------------------


class TestCachedLitellmEmbedding:
    async def test_miss_calls_litellm(self, cache: MemoryCache) -> None:
        """On cache miss the embedding API is called exactly once."""
        mock_resp = _make_embedding_response()

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.aembedding = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            result = await cached_litellm_embedding(
                cache,
                model="text-embedding-3-small",
                input=["hello world"],
            )

        fake_litellm.aembedding.assert_awaited_once()
        assert result is mock_resp

    async def test_hit_does_not_call_litellm(self, cache: MemoryCache) -> None:
        """Second identical embedding call returns from cache."""
        mock_resp = _make_embedding_response()

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.aembedding = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            await cached_litellm_embedding(
                cache, model="text-embedding-3-small", input=["hello world"]
            )
            await cached_litellm_embedding(
                cache, model="text-embedding-3-small", input=["hello world"]
            )

        assert fake_litellm.aembedding.await_count == 1

    async def test_string_input_normalised_to_list(self, cache: MemoryCache) -> None:
        """String and list-of-one-string inputs must share the same cache entry."""
        mock_resp = _make_embedding_response()

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.aembedding = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            # String input
            await cached_litellm_embedding(
                cache, model="text-embedding-3-small", input="hello world"
            )
            # List input — same semantic content
            await cached_litellm_embedding(
                cache, model="text-embedding-3-small", input=["hello world"]
            )

        # Both calls resolve to the same cache key → only one LLM call
        assert fake_litellm.aembedding.await_count == 1

    async def test_different_inputs_different_cache_entries(self, cache: MemoryCache) -> None:
        """Different text inputs must not share cache entries."""
        resp_a = _make_embedding_response(3)
        resp_b = _make_embedding_response(3)

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.aembedding = AsyncMock(side_effect=[resp_a, resp_b])
            mock_import.return_value = fake_litellm

            await cached_litellm_embedding(cache, model="text-embedding-3-small", input=["text A"])
            await cached_litellm_embedding(cache, model="text-embedding-3-small", input=["text B"])

        assert fake_litellm.aembedding.await_count == 2
