"""Unit tests for the OpenAI embedding provider.

These tests never make a real API call.  A stub ``openai`` module exposing a
fake ``AsyncOpenAI`` client is injected into ``sys.modules`` so the lazy import
inside :class:`OpenAIEmbedder` resolves to a deterministic fake, keeping the
suite offline and fast.  A separate test verifies the missing-dependency path
raises :class:`EmbeddingProviderError` with an install hint.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from memory_reuse.embeddings import EmbeddingProvider
from memory_reuse.embeddings.openai import OpenAIEmbedder
from memory_reuse.exceptions import EmbeddingProviderError


@dataclass
class _FakeEmbeddingItem:
    embedding: list[float]
    index: int


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddings:
    """Deterministic stand-in for ``client.embeddings``."""

    def __init__(self, client: _FakeAsyncOpenAI) -> None:
        self._client = client

    async def create(self, *, model: str, input: str | list[str]) -> _FakeEmbeddingResponse:
        self._client.calls.append({"model": model, "input": input})
        texts = [input] if isinstance(input, str) else list(input)
        items = [_FakeEmbeddingItem(self._vector(t), i) for i, t in enumerate(texts)]
        return _FakeEmbeddingResponse(data=items)

    @staticmethod
    def _vector(text: str) -> list[float]:
        codes = [float(ord(c)) for c in text]
        return (codes + [0.0] * 4)[:4]


class _FakeAsyncOpenAI:
    """Deterministic stand-in for ``openai.AsyncOpenAI``."""

    last_api_key: str | None = None

    def __init__(self, api_key: str | None = None) -> None:
        type(self).last_api_key = api_key
        self.calls: list[dict[str, object]] = []
        self.embeddings = _FakeEmbeddings(self)


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAsyncOpenAI]:
    """Inject a fake ``openai`` module for the lazy import."""
    _FakeAsyncOpenAI.last_api_key = None
    stub_module = types.ModuleType("openai")
    stub_module.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", stub_module)
    return _FakeAsyncOpenAI


class TestOpenAIEmbedderConstruction:
    def test_is_embedding_provider(self) -> None:
        assert isinstance(OpenAIEmbedder(), EmbeddingProvider)

    def test_default_model_identity(self) -> None:
        assert OpenAIEmbedder().identity == "openai:text-embedding-3-small"

    def test_custom_model_identity(self) -> None:
        assert OpenAIEmbedder(model="text-embedding-3-large").identity == (
            "openai:text-embedding-3-large"
        )

    def test_default_model_dimension(self) -> None:
        assert OpenAIEmbedder().dimension == 1536

    def test_large_model_dimension(self) -> None:
        assert OpenAIEmbedder(model="text-embedding-3-large").dimension == 3072

    def test_unknown_model_falls_back_to_default_dimension(self) -> None:
        assert OpenAIEmbedder(model="mystery-model").dimension == 1536

    def test_construction_does_not_create_client(self, stub_openai: type[_FakeAsyncOpenAI]) -> None:
        OpenAIEmbedder(api_key="sk-test")
        # Lazy: the client is not created until first use.
        assert stub_openai.last_api_key is None


class TestOpenAIEmbedderWithStub:
    @pytest.mark.asyncio
    async def test_embed_returns_expected_vector(self, stub_openai: type[_FakeAsyncOpenAI]) -> None:
        embedder = OpenAIEmbedder()
        vector = await embedder.embed("ab")
        assert vector == [float(ord("a")), float(ord("b")), 0.0, 0.0]
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_passes_model_and_input(self, stub_openai: type[_FakeAsyncOpenAI]) -> None:
        embedder = OpenAIEmbedder(model="text-embedding-3-large")
        await embedder.embed("hello")
        client = embedder._get_client()
        assert client.calls == [{"model": "text-embedding-3-large", "input": "hello"}]

    @pytest.mark.asyncio
    async def test_api_key_forwarded_to_client(self, stub_openai: type[_FakeAsyncOpenAI]) -> None:
        embedder = OpenAIEmbedder(api_key="sk-secret")
        await embedder.embed("x")
        assert stub_openai.last_api_key == "sk-secret"

    @pytest.mark.asyncio
    async def test_client_created_once(self, stub_openai: type[_FakeAsyncOpenAI]) -> None:
        embedder = OpenAIEmbedder()
        first = embedder._get_client()
        second = embedder._get_client()
        assert first is second

    @pytest.mark.asyncio
    async def test_embed_batch(self, stub_openai: type[_FakeAsyncOpenAI]) -> None:
        embedder = OpenAIEmbedder()
        batch = await embedder.embed_batch(["a", "b"])
        assert batch == [
            [float(ord("a")), 0.0, 0.0, 0.0],
            [float(ord("b")), 0.0, 0.0, 0.0],
        ]

    @pytest.mark.asyncio
    async def test_embed_batch_single_api_call(self, stub_openai: type[_FakeAsyncOpenAI]) -> None:
        embedder = OpenAIEmbedder()
        await embedder.embed_batch(["a", "b", "c"])
        client = embedder._get_client()
        assert len(client.calls) == 1
        assert client.calls[0]["input"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order_when_shuffled(
        self, stub_openai: type[_FakeAsyncOpenAI]
    ) -> None:
        embedder = OpenAIEmbedder()

        # Force the fake to return items out of order to prove we sort by index.
        original_create = embedder._get_client().embeddings.create

        async def shuffled_create(*, model: str, input: object) -> _FakeEmbeddingResponse:
            response = await original_create(model=model, input=input)
            response.data = list(reversed(response.data))
            return response

        embedder._get_client().embeddings.create = shuffled_create  # type: ignore[method-assign]
        batch = await embedder.embed_batch(["a", "b"])
        assert batch == [
            [float(ord("a")), 0.0, 0.0, 0.0],
            [float(ord("b")), 0.0, 0.0, 0.0],
        ]

    @pytest.mark.asyncio
    async def test_embed_batch_empty_does_not_create_client(
        self, stub_openai: type[_FakeAsyncOpenAI]
    ) -> None:
        embedder = OpenAIEmbedder()
        assert await embedder.embed_batch([]) == []
        assert stub_openai.last_api_key is None


class TestOpenAIEmbedderMissingDependency:
    def test_missing_dependency_raises_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure the import fails even if the real package is installed.
        monkeypatch.setitem(sys.modules, "openai", None)
        embedder = OpenAIEmbedder()
        with pytest.raises(EmbeddingProviderError) as exc_info:
            embedder._get_client()
        message = str(exc_info.value)
        assert "openai" in message
        assert "memory-reuse[semantic]" in message

    @pytest.mark.asyncio
    async def test_embed_raises_when_dependency_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)
        embedder = OpenAIEmbedder()
        with pytest.raises(EmbeddingProviderError):
            await embedder.embed("hello")
