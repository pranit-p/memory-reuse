"""Unit tests for the local sentence-transformers embedding provider.

These tests never download a real model.  A stub ``sentence_transformers``
module is injected into ``sys.modules`` so the lazy import inside
:class:`LocalEmbedder` resolves to a deterministic fake, keeping the suite
offline and fast.  A separate test verifies the missing-dependency path raises
:class:`EmbeddingProviderError` with an install hint.
"""

from __future__ import annotations

import sys
import types

import pytest

from memory_reuse.embeddings import EmbeddingProvider
from memory_reuse.embeddings.local import LocalEmbedder
from memory_reuse.exceptions import EmbeddingProviderError


class _FakeSentenceTransformer:
    """Deterministic stand-in for ``sentence_transformers.SentenceTransformer``.

    Records the model name it was constructed with and produces stable vectors
    from the character codes of the input text, so no real model is loaded.
    """

    last_model_name: str | None = None
    dimension = 4

    def __init__(self, model_name: str) -> None:
        type(self).last_model_name = model_name
        self._model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, text: str | list[str], **kwargs: object) -> object:
        # Accept and ignore extra kwargs (e.g. show_progress_bar) the way the
        # real SentenceTransformer.encode does.
        if isinstance(text, list):
            return [self._vector(t) for t in text]
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        codes = [float(ord(c)) for c in text]
        # Pad or truncate to a fixed dimension.
        codes = (codes + [0.0] * self.dimension)[: self.dimension]
        return codes


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    """Inject a fake ``sentence_transformers`` module for the lazy import."""
    _FakeSentenceTransformer.last_model_name = None
    stub_module = types.ModuleType("sentence_transformers")
    stub_module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub_module)
    return _FakeSentenceTransformer


class TestLocalEmbedderConstruction:
    def test_is_embedding_provider(self) -> None:
        assert isinstance(LocalEmbedder(), EmbeddingProvider)

    def test_default_model_identity(self) -> None:
        embedder = LocalEmbedder()
        assert embedder.identity == "local:all-MiniLM-L6-v2"

    def test_custom_model_identity(self) -> None:
        embedder = LocalEmbedder(model="my-model")
        assert embedder.identity == "local:my-model"

    def test_construction_does_not_load_model(
        self, stub_sentence_transformers: type[_FakeSentenceTransformer]
    ) -> None:
        LocalEmbedder(model="lazy-model")
        # Lazy: the model is not loaded until first use.
        assert stub_sentence_transformers.last_model_name is None


class TestLocalEmbedderWithStub:
    def test_dimension_loads_model(
        self, stub_sentence_transformers: type[_FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder(model="dim-model")
        assert embedder.dimension == _FakeSentenceTransformer.dimension
        assert stub_sentence_transformers.last_model_name == "dim-model"

    @pytest.mark.asyncio
    async def test_embed_returns_expected_vector(
        self, stub_sentence_transformers: type[_FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder()
        vector = await embedder.embed("ab")
        assert vector == [float(ord("a")), float(ord("b")), 0.0, 0.0]
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_is_deterministic(
        self, stub_sentence_transformers: type[_FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder()
        assert await embedder.embed("same") == await embedder.embed("same")

    @pytest.mark.asyncio
    async def test_embed_batch(
        self, stub_sentence_transformers: type[_FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder()
        batch = await embedder.embed_batch(["a", "b"])
        assert batch == [
            [float(ord("a")), 0.0, 0.0, 0.0],
            [float(ord("b")), 0.0, 0.0, 0.0],
        ]

    @pytest.mark.asyncio
    async def test_embed_batch_empty_does_not_load_model(
        self, stub_sentence_transformers: type[_FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder()
        assert await embedder.embed_batch([]) == []
        assert stub_sentence_transformers.last_model_name is None

    def test_model_loaded_once(
        self, stub_sentence_transformers: type[_FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder(model="cache-model")
        first = embedder._load_model()
        second = embedder._load_model()
        assert first is second


class TestLocalEmbedderMissingDependency:
    def test_missing_dependency_raises_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure the import fails even if the real package is installed.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        embedder = LocalEmbedder()
        with pytest.raises(EmbeddingProviderError) as exc_info:
            embedder._load_model()
        message = str(exc_info.value)
        assert "sentence-transformers" in message
        assert "memory-reuse[semantic-local]" in message

    @pytest.mark.asyncio
    async def test_embed_raises_when_dependency_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        embedder = LocalEmbedder()
        with pytest.raises(EmbeddingProviderError):
            await embedder.embed("hello")
