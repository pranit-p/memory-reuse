"""Integration tests for semantic support in the integration decorators.

These exercise the ``semantic=True`` path of :func:`cached_node`,
:func:`cached_tool`, and :func:`cached_litellm_completion` end-to-end with the
in-memory backend, the in-memory vector index, and a deterministic
``FakeEmbedder`` (LiteLLM itself is mocked), so the whole suite stays offline
and fast.

They cover the behaviour called out for this component:

* a reworded call hits the semantic cache; and
* existing Phase 1 decorator calls (``semantic=False``) are unchanged.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.embeddings.base import EmbeddingProvider
from memory_reuse.integrations.langgraph import cached_node, cached_tool
from memory_reuse.integrations.litellm import cached_litellm_completion


class FakeEmbedder(EmbeddingProvider):
    """Embedder mapping specific texts to fixed vectors for predictable matches.

    Texts sharing an entry in ``vectors`` embed to the same vector (so they are
    a semantic match); any unknown text embeds to a distinct orthogonal-ish
    vector so it never accidentally matches.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors = vectors or {}

    @property
    def identity(self) -> str:
        return "fake:test"

    @property
    def dimension(self) -> int:
        return 3

    async def embed(self, text: str) -> list[float]:
        if text in self._vectors:
            return list(self._vectors[text])
        # Deterministic fallback: distinct from any registered vector.
        return [float(len(text) % 7), float(sum(map(ord, text)) % 11), 1.0]


def _semantic_cache(embedder: EmbeddingProvider, **overrides: object) -> MemoryCache:
    """Build a semantic-enabled MemoryCache with an injected embedder."""
    params: dict[str, object] = {
        "backend": "memory",
        "semantic_enabled": True,
        "embedding_provider": "local",
        "similarity_threshold": 0.95,
    }
    params.update(overrides)
    config = CacheConfig(**params)  # type: ignore[arg-type]
    cache = MemoryCache(config)
    assert cache.semantic is not None
    cache.semantic._embedder = embedder  # type: ignore[attr-defined]
    return cache


def _plain_cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


# ---------------------------------------------------------------------------
# cached_tool
# ---------------------------------------------------------------------------


class TestCachedToolSemantic:
    async def test_reworded_call_hits_semantic_cache(self) -> None:
        """Two different arg strings that embed alike reuse the cached result."""
        # Both argument renderings embed to the same vector.
        shared = [1.0, 0.0, 0.0]
        embedder = FakeEmbedder(
            {
                "{'question': 'what is 128 times 47'}": shared,
                "{'question': 'what is 128 multiplied by 47'}": shared,
            }
        )
        cache = _semantic_cache(embedder, similarity_threshold=0.9)
        call_count = 0

        @cached_tool(cache, scope="global", semantic=True)
        async def ask(question: str) -> int:
            nonlocal call_count
            call_count += 1
            return 6016

        first = await ask(question="what is 128 times 47")
        second = await ask(question="what is 128 multiplied by 47")

        assert first == second == 6016
        assert call_count == 1  # reworded call served from the semantic cache
        assert cache.stats.semantic_hits == 1

    async def test_exact_only_bypasses_semantic(self) -> None:
        """exact_only=True forces Phase 1 behaviour even with semantic enabled."""
        shared = [1.0, 0.0, 0.0]
        embedder = FakeEmbedder(
            {
                "{'question': 'a'}": shared,
                "{'question': 'b'}": shared,
            }
        )
        cache = _semantic_cache(embedder, similarity_threshold=0.0)
        call_count = 0

        @cached_tool(cache, scope="global", semantic=True, exact_only=True)
        async def ask(question: str) -> str:
            nonlocal call_count
            call_count += 1
            return "answer"

        await ask(question="a")
        await ask(question="b")

        assert call_count == 2  # never reused via semantic
        assert cache.stats.semantic_hits == 0


class TestCachedToolPhase1Unchanged:
    async def test_default_is_exact_only(self) -> None:
        """Without semantic=True the decorator uses the Phase 1 tool cache."""
        cache = _plain_cache()
        call_count = 0

        @cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result:{query}"

        assert await fetch("hello") == "result:hello"
        assert await fetch("hello") == "result:hello"
        assert call_count == 1  # exact hit
        assert cache.stats.exact_hits == 1


# ---------------------------------------------------------------------------
# cached_node
# ---------------------------------------------------------------------------


class TestCachedNodeSemantic:
    async def test_reworded_state_hits_semantic_cache(self) -> None:
        shared = [1.0, 0.0, 0.0]
        embedder = FakeEmbedder(
            {
                "{'query': 'summarise this'}": shared,
                "{'query': 'give me a summary'}": shared,
            }
        )
        cache = _semantic_cache(embedder, similarity_threshold=0.9)
        call_count = 0

        @cached_node(cache, scope="global", key_fields=["query"], semantic=True)
        async def summarise(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"summary": "done"}

        r1 = await summarise({"query": "summarise this"})
        r2 = await summarise({"query": "give me a summary"})

        assert r1 == r2 == {"summary": "done"}
        assert call_count == 1
        assert cache.stats.semantic_hits == 1


class TestCachedNodePhase1Unchanged:
    async def test_default_exact_behaviour(self) -> None:
        cache = _plain_cache()
        call_count = 0

        @cached_node(cache, scope="global")
        async def process(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"output": state["input"]}

        state = {"input": "hello"}
        assert await process(state) == {"output": "hello"}
        assert await process(state) == {"output": "hello"}
        assert call_count == 1
        assert cache.stats.exact_hits == 1


# ---------------------------------------------------------------------------
# cached_litellm_completion
# ---------------------------------------------------------------------------


def _make_completion_response(content: str = "6016") -> MagicMock:
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


class TestCachedLitellmCompletionSemantic:
    async def test_reworded_prompt_hits_semantic_cache(self) -> None:
        """A reworded prompt returns the stored answer without calling LiteLLM."""
        shared = [1.0, 0.0, 0.0]
        embedder = FakeEmbedder(
            {
                "What is 128 times 47?": shared,
                "What is 128 multiplied by 47?": shared,
            }
        )
        cache = _semantic_cache(embedder, similarity_threshold=0.9)
        mock_resp = _make_completion_response("6016")

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            # First prompt — miss, calls LiteLLM and stores the answer.
            first = await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What is 128 times 47?"}],
                semantic=True,
            )
            # Reworded prompt — semantic hit, LiteLLM not called again.
            second = await cached_litellm_completion(
                cache,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What is 128 multiplied by 47?"}],
                semantic=True,
            )

        assert fake_litellm.acompletion.await_count == 1
        assert first is mock_resp  # rich object on first call
        # The semantic hit returns the serialised dict form.
        assert second["choices"][0]["message"]["content"] == "6016"
        assert cache.stats.semantic_hits == 1


class TestCachedLitellmCompletionPhase1Unchanged:
    async def test_default_exact_behaviour(self) -> None:
        cache = _plain_cache()
        mock_resp = _make_completion_response("Berlin")
        messages = [{"role": "user", "content": "Capital of Germany?"}]

        with patch("memory_reuse.integrations.litellm._assert_litellm_installed") as mock_import:
            fake_litellm = MagicMock()
            fake_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = fake_litellm

            await cached_litellm_completion(cache, model="gpt-4o-mini", messages=messages)
            await cached_litellm_completion(cache, model="gpt-4o-mini", messages=messages)

        assert fake_litellm.acompletion.await_count == 1
        assert cache.stats.exact_hits == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
