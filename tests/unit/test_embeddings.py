"""Unit tests for the embedding-provider interface and factory.

These tests use a deterministic ``FakeEmbedder`` test double so the suite stays
offline and fast — no real model or API is ever contacted.  The fake validates
the :class:`EmbeddingProvider` interface contract, while the factory tests
verify dispatch by ``embedding_provider`` value.
"""

from __future__ import annotations

import hashlib
import sys
import types

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.embeddings import EmbeddingProvider, create_embedder
from memory_reuse.exceptions import ConfigurationError


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedding provider for tests.

    Produces a fixed-dimension vector derived from a stable hash of the text,
    so identical text always yields the identical vector without any external
    dependency.
    """

    def __init__(self, model: str = "fake-model", dimension: int = 8) -> None:
        self._model = model
        self._dimension = dimension

    @property
    def identity(self) -> str:
        return f"fake:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Map bytes into [0, 1) floats, one per dimension.
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]


class TestEmbeddingProviderContract:
    def test_identity_is_stable_string(self) -> None:
        embedder = FakeEmbedder(model="m1")
        assert embedder.identity == "fake:m1"
        assert embedder.identity == embedder.identity

    def test_dimension_matches_config(self) -> None:
        embedder = FakeEmbedder(dimension=16)
        assert embedder.dimension == 16

    @pytest.mark.asyncio
    async def test_embed_returns_vector_of_declared_dimension(self) -> None:
        embedder = FakeEmbedder(dimension=8)
        vector = await embedder.embed("hello world")
        assert isinstance(vector, list)
        assert len(vector) == embedder.dimension
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_is_deterministic(self) -> None:
        embedder = FakeEmbedder()
        first = await embedder.embed("same text")
        second = await embedder.embed("same text")
        assert first == second

    @pytest.mark.asyncio
    async def test_different_text_produces_different_vector(self) -> None:
        embedder = FakeEmbedder()
        assert await embedder.embed("alpha") != await embedder.embed("beta")

    @pytest.mark.asyncio
    async def test_embed_batch_default_matches_embed(self) -> None:
        embedder = FakeEmbedder()
        texts = ["one", "two", "three"]
        batch = await embedder.embed_batch(texts)
        assert batch == [await embedder.embed(t) for t in texts]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        embedder = FakeEmbedder()
        assert await embedder.embed_batch([]) == []


def _install_stub_provider(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    captured: dict[str, str | None],
) -> type[FakeEmbedder]:
    """Register a stub provider module in ``sys.modules`` for factory dispatch.

    The concrete provider modules are implemented in a later task; injecting a
    stub module keeps this test focused on the factory's dispatch logic without
    depending on those implementations or their optional dependencies.
    """

    class _StubEmbedder(FakeEmbedder):
        def __init__(self, model: str | None = None) -> None:
            captured["model"] = model
            super().__init__(model=model or "stub-default")

    stub_module = types.ModuleType(f"memory_reuse.embeddings.{module_name}")
    setattr(stub_module, class_name, _StubEmbedder)
    monkeypatch.setitem(sys.modules, f"memory_reuse.embeddings.{module_name}", stub_module)
    return _StubEmbedder


class TestCreateEmbedderFactory:
    def test_none_provider_raises_configuration_error(self) -> None:
        config = CacheConfig(embedding_provider=None)
        with pytest.raises(ConfigurationError):
            create_embedder(config)

    def test_dispatches_to_openai_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str | None] = {}
        stub_cls = _install_stub_provider(monkeypatch, "openai", "OpenAIEmbedder", captured)

        config = CacheConfig(embedding_provider="openai", embedding_model="text-embedding-3-small")
        embedder = create_embedder(config)

        assert isinstance(embedder, stub_cls)
        assert captured["model"] == "text-embedding-3-small"

    def test_dispatches_to_local_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str | None] = {}
        stub_cls = _install_stub_provider(monkeypatch, "local", "LocalEmbedder", captured)

        config = CacheConfig(embedding_provider="local")
        embedder = create_embedder(config)

        assert isinstance(embedder, stub_cls)
        assert captured["model"] is None

    def test_dispatches_to_litellm_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str | None] = {}
        stub_cls = _install_stub_provider(monkeypatch, "litellm", "LiteLLMEmbedder", captured)

        config = CacheConfig(embedding_provider="litellm", embedding_model="bedrock/titan")
        embedder = create_embedder(config)

        assert isinstance(embedder, stub_cls)
        assert captured["model"] == "bedrock/titan"
