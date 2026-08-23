"""Unit tests for the LiteLLM embedding provider.

These tests never make a real API call.  A stub ``litellm`` module exposing a
fake ``aembedding`` coroutine is injected into ``sys.modules`` so the lazy
import inside :class:`LiteLLMEmbedder` resolves to a deterministic fake, keeping
the suite offline and fast.  A separate test verifies the missing-dependency
path raises :class:`EmbeddingProviderError` with an install hint.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from memory_reuse.embeddings import EmbeddingProvider
from memory_reuse.embeddings.litellm import LiteLLMEmbedder
from memory_reuse.exceptions import EmbeddingProviderError


@dataclass
class _FakeEmbeddingResponse:
    """Deterministic stand-in for ``litellm.EmbeddingResponse``.

    LiteLLM returns items as dicts with an ``"embedding"`` key, mirrored here.
    """

    data: list[dict[str, object]]


def _vector(text: str) -> list[float]:
    """Deterministic 4-float vector derived from the first characters of text."""
    codes = [float(ord(c)) for c in text]
    return (codes + [0.0] * 4)[:4]


@pytest.fixture
def stub_litellm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``litellm`` module recording calls to ``aembedding``."""
    stub_module = types.ModuleType("litellm")
    calls: list[dict[str, object]] = []

    async def aembedding(*, model: str, input: str | list[str], **kwargs: object):
        calls.append({"model": model, "input": input, **kwargs})
        texts = [input] if isinstance(input, str) else list(input)
        return _FakeEmbeddingResponse(data=[{"embedding": _vector(t)} for t in texts])

    stub_module.aembedding = aembedding  # type: ignore[attr-defined]
    stub_module.calls = calls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", stub_module)
    return stub_module


class TestLiteLLMEmbedderConstruction:
    def test_is_embedding_provider(self) -> None:
        assert isinstance(LiteLLMEmbedder(), EmbeddingProvider)

    def test_default_model_identity(self) -> None:
        assert LiteLLMEmbedder().identity == "litellm:text-embedding-3-small"

    def test_custom_model_identity(self) -> None:
        embedder = LiteLLMEmbedder(model="bedrock/amazon.titan-embed-text-v2")
        assert embedder.identity == "litellm:bedrock/amazon.titan-embed-text-v2"

    def test_dimension_unknown_before_first_embed(self) -> None:
        embedder = LiteLLMEmbedder()
        with pytest.raises(EmbeddingProviderError):
            _ = embedder.dimension

    def test_construction_does_not_import_litellm(self, stub_litellm: types.ModuleType) -> None:
        # Constructing is cheap and makes no calls.
        LiteLLMEmbedder(model="bedrock/titan")
        assert stub_litellm.calls == []  # type: ignore[attr-defined]


class TestLiteLLMEmbedderWithStub:
    @pytest.mark.asyncio
    async def test_embed_returns_expected_vector(self, stub_litellm: types.ModuleType) -> None:
        embedder = LiteLLMEmbedder()
        vector = await embedder.embed("ab")
        assert vector == [float(ord("a")), float(ord("b")), 0.0, 0.0]
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_passes_model_and_input(self, stub_litellm: types.ModuleType) -> None:
        embedder = LiteLLMEmbedder(model="bedrock/amazon.titan-embed-text-v2")
        await embedder.embed("hello")
        assert stub_litellm.calls == [  # type: ignore[attr-defined]
            {"model": "bedrock/amazon.titan-embed-text-v2", "input": "hello"}
        ]

    @pytest.mark.asyncio
    async def test_dimension_discovered_after_embed(self, stub_litellm: types.ModuleType) -> None:
        embedder = LiteLLMEmbedder()
        await embedder.embed("abc")
        assert embedder.dimension == 4

    @pytest.mark.asyncio
    async def test_embed_batch(self, stub_litellm: types.ModuleType) -> None:
        embedder = LiteLLMEmbedder()
        batch = await embedder.embed_batch(["a", "b"])
        assert batch == [
            [float(ord("a")), 0.0, 0.0, 0.0],
            [float(ord("b")), 0.0, 0.0, 0.0],
        ]

    @pytest.mark.asyncio
    async def test_embed_batch_single_api_call(self, stub_litellm: types.ModuleType) -> None:
        embedder = LiteLLMEmbedder()
        await embedder.embed_batch(["a", "b", "c"])
        assert len(stub_litellm.calls) == 1  # type: ignore[attr-defined]
        assert stub_litellm.calls[0]["input"] == ["a", "b", "c"]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_embed_batch_empty_makes_no_call(self, stub_litellm: types.ModuleType) -> None:
        embedder = LiteLLMEmbedder()
        assert await embedder.embed_batch([]) == []
        assert stub_litellm.calls == []  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_dimension_discovered_after_batch(self, stub_litellm: types.ModuleType) -> None:
        embedder = LiteLLMEmbedder()
        await embedder.embed_batch(["ab"])
        assert embedder.dimension == 4


class TestLiteLLMEmbedderMissingDependency:
    def test_missing_dependency_raises_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure the import fails even if the real package is installed.
        monkeypatch.setitem(sys.modules, "litellm", None)
        embedder = LiteLLMEmbedder()
        with pytest.raises(EmbeddingProviderError) as exc_info:
            embedder._get_litellm()
        message = str(exc_info.value)
        assert "litellm" in message
        assert "memory-reuse[semantic]" in message

    @pytest.mark.asyncio
    async def test_embed_raises_when_dependency_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "litellm", None)
        embedder = LiteLLMEmbedder()
        with pytest.raises(EmbeddingProviderError):
            await embedder.embed("hello")
