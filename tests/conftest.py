"""Shared pytest fixtures for memory-reuse tests."""

from __future__ import annotations

import hashlib

import pytest

from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.exact import ExactCache
from memory_reuse.cache.tool import ToolCache
from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.embeddings.base import EmbeddingProvider
from memory_reuse.stats import StatsTracker


@pytest.fixture
def config() -> CacheConfig:
    """Default CacheConfig for tests (in-memory, stats enabled)."""
    return CacheConfig(backend="memory", default_ttl=3600, enable_stats=True)


@pytest.fixture
def backend() -> InMemoryBackend:
    """A fresh InMemoryBackend for each test."""
    return InMemoryBackend()


@pytest.fixture
def stats() -> StatsTracker:
    """A fresh StatsTracker for each test."""
    return StatsTracker()


@pytest.fixture
def exact_cache(backend: InMemoryBackend, config: CacheConfig, stats: StatsTracker) -> ExactCache:
    """ExactCache wired to an in-memory backend."""
    return ExactCache(backend=backend, config=config, stats=stats)


@pytest.fixture
def tool_cache(backend: InMemoryBackend, config: CacheConfig, stats: StatsTracker) -> ToolCache:
    """ToolCache wired to an in-memory backend."""
    return ToolCache(backend=backend, config=config, stats=stats)


@pytest.fixture
def memory_cache() -> MemoryCache:
    """A fully configured MemoryCache using the in-memory backend."""
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


# ---------------------------------------------------------------------------
# Phase 3 (graph-level cache) test scaffolding
# ---------------------------------------------------------------------------


class StubGraph:
    """Deterministic stub compiled graph with a node-execution counter.

    Computes ``Final_Result`` from the input state and increments
    ``node_calls`` on every real execution, so "zero nodes run" is directly
    assertable. Exposes both ``invoke`` and ``ainvoke``.
    """

    def __init__(self, name: str | None = None) -> None:
        self.node_calls = 0
        if name is not None:
            self.name = name

    def _run(self, state: object) -> object:
        self.node_calls += 1
        if isinstance(state, dict):
            return {**state, "Final_Result": f"result::{sorted(state.items())}"}
        return {"Final_Result": f"result::{state}"}

    def invoke(self, state: object, *args: object, **kwargs: object) -> object:
        return self._run(state)

    async def ainvoke(self, state: object, *args: object, **kwargs: object) -> object:
        return self._run(state)


class RaisingGraph:
    """Stub graph whose nodes always raise, to test error propagation."""

    def __init__(self) -> None:
        self.node_calls = 0

    def invoke(self, state: object, *args: object, **kwargs: object) -> object:
        self.node_calls += 1
        raise RuntimeError("boom")

    async def ainvoke(self, state: object, *args: object, **kwargs: object) -> object:
        self.node_calls += 1
        raise RuntimeError("boom")


class StubEmbedder(EmbeddingProvider):
    """Deterministic embedding provider with an embed-call counter.

    Identical text always maps to the identical vector (a stable hash of the
    text), so semantic matches are predictable, while ``embed_calls`` lets a
    test assert whether an embedding was computed at all. Used by the Phase 3
    graph-level cache property tests so "no embedding on an exact hit / an
    exact-only wrapper" is directly assertable.
    """

    def __init__(self, model: str = "stub", dimension: int = 8) -> None:
        self._model = model
        self._dimension = dimension
        self.embed_calls = 0

    @property
    def identity(self) -> str:
        return f"stub:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]


def make_semantic_cache(embedder: EmbeddingProvider, **overrides: object) -> MemoryCache:
    """Build a semantic-enabled MemoryCache with an injected embedder.

    The embedder is patched onto the already-built ``SemanticCache`` so tests
    control exactly which vectors are produced without needing a real provider.
    """
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
