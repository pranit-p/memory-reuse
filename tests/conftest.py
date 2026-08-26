"""Shared pytest fixtures for memory-reuse tests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import pytest
from hypothesis import strategies as st

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


# ---------------------------------------------------------------------------
# Phase 4 (integrations + backend) test scaffolding
# ---------------------------------------------------------------------------
#
# Everything below is deterministic and fully offline: no real network, no LLM,
# and no AWS calls. It provides three reusable pieces for the Phase 4 property
# and example tests:
#
#   1. An in-process fake AgentCore service + client (dict-backed) that two
#      ``AgentCoreBackend`` instances can share, so "a value set via one is read
#      via the other" (cross-microVM sharing) is directly assertable, along with
#      a toggle to simulate the service being unreachable.
#   2. Stub Strands/CrewAI tool bodies (sync + async) with a call counter,
#      reusing the Phase 2/3 ``InMemoryBackend`` and the embed-call-counting
#      ``StubEmbedder`` above, so "zero embeddings on an exact hit" and "the tool
#      body ran / did not run" stay directly assertable.
#   3. Hypothesis strategies for tool arguments and for value byte-blobs across
#      the whole 0..1_048_576-byte range the AgentCore backend must round-trip.


class FakeAgentCoreConnectionError(Exception):
    """Raised by the fake AgentCore client when the service is unreachable.

    The real ``AgentCoreBackend`` translates transport-level failures into
    :exc:`~memory_reuse.exceptions.BackendConnectionError`; this stand-in gives
    the tests a concrete, deterministic error type to raise from the fake client
    when :attr:`FakeAgentCoreService.reachable` is ``False``.
    """


class FakeAgentCoreService:
    """A dict-backed, in-process stand-in for the managed AgentCore store.

    Deterministic and offline. Instances are the *shared* storage a set of
    :class:`FakeAgentCoreClient` objects talk to, so two backends constructed
    against the same service observe each other's writes — modelling the
    cross-microVM sharing behaviour (Req 8.2) without any AWS call.

    Attributes:
        store: The raw item map, keyed by the transport item id.
        reachable: When ``False`` every client operation raises
            :class:`FakeAgentCoreConnectionError`, modelling an unreachable
            service (Req 8.6, 8.7).
    """

    def __init__(self) -> None:
        self.store: dict[str, dict[str, object]] = {}
        self.reachable: bool = True

    def reset(self) -> None:
        """Clear all items and mark the service reachable again."""
        self.store.clear()
        self.reachable = True


class FakeAgentCoreClient:
    """A deterministic, offline stand-in for a boto3 AgentCore client.

    Every method delegates to a shared :class:`FakeAgentCoreService`; two clients
    sharing one service therefore share state. Each call first checks
    ``service.reachable`` and raises :class:`FakeAgentCoreConnectionError` when
    the service is down, so the backend under test can map that to
    :exc:`~memory_reuse.exceptions.BackendConnectionError` and return ``False``
    from ``ping`` (Req 8.6, 8.7).

    The method surface is intentionally small and transport-shaped
    (``put_item`` / ``get_item`` / ``delete_item`` / ``scan`` / ``ping``): the
    backend adapts these to the ``AbstractBackend`` contract. Values are stored
    exactly as handed in so any 0..1_048_576-byte payload round-trips unchanged
    (Req 8.5).

    Args:
        service: The shared backing store this client reads and writes.
    """

    def __init__(self, service: FakeAgentCoreService) -> None:
        self._service = service

    def _check_reachable(self) -> None:
        if not self._service.reachable:
            raise FakeAgentCoreConnectionError("fake AgentCore service is unreachable")

    def put_item(self, item_id: str, item: dict[str, object]) -> None:
        """Store ``item`` (an opaque record) under ``item_id``."""
        self._check_reachable()
        self._service.store[item_id] = dict(item)

    def get_item(self, item_id: str) -> dict[str, object] | None:
        """Return the record stored under ``item_id`` or ``None`` when absent."""
        self._check_reachable()
        item = self._service.store.get(item_id)
        return dict(item) if item is not None else None

    def delete_item(self, item_id: str) -> None:
        """Remove ``item_id`` if present; a no-op otherwise."""
        self._check_reachable()
        self._service.store.pop(item_id, None)

    def scan(self, prefix: str = "") -> list[str]:
        """Return every stored item id beginning with ``prefix``."""
        self._check_reachable()
        return [item_id for item_id in self._service.store if item_id.startswith(prefix)]

    def ping(self) -> bool:
        """Return the service's current reachability without raising."""
        return self._service.reachable


@pytest.fixture
def fake_agentcore_service() -> FakeAgentCoreService:
    """A fresh, reachable in-process fake AgentCore service per test."""
    return FakeAgentCoreService()


@pytest.fixture
def fake_agentcore_client(fake_agentcore_service: FakeAgentCoreService) -> FakeAgentCoreClient:
    """A client bound to the per-test :func:`fake_agentcore_service`."""
    return FakeAgentCoreClient(fake_agentcore_service)


# ---------------------------------------------------------------------------
# Stub Strands / CrewAI tools (reuse the in-memory backend + fake embedder)
# ---------------------------------------------------------------------------


class StubTool:
    """A deterministic tool body with a call counter, for the decorator tests.

    Wraps a pure function of its arguments and increments :attr:`calls` on every
    real execution, so "the decorated tool returned a stored value without
    running the body" (a cache hit) versus "the body re-ran" (a miss / TTL
    expiry) is directly assertable across the Strands and CrewAI wrappers.
    Exposes both a sync (:meth:`run`) and an async (:meth:`arun`) callable over
    the same counter.

    Args:
        result_fn: Maps the bound arguments to the tool's return value. Defaults
            to echoing a stable string form of the arguments.
    """

    def __init__(self, result_fn: Callable[..., object] | None = None) -> None:
        self.calls = 0
        self._result_fn = result_fn if result_fn is not None else self._default_result

    @staticmethod
    def _default_result(**kwargs: object) -> str:
        return f"result::{sorted(kwargs.items(), key=lambda kv: kv[0])}"

    def run(self, **kwargs: object) -> object:
        """Synchronous tool body; increments the call counter."""
        self.calls += 1
        return self._result_fn(**kwargs)

    async def arun(self, **kwargs: object) -> object:
        """Asynchronous tool body; increments the call counter."""
        self.calls += 1
        return self._result_fn(**kwargs)


class RaisingTool:
    """A stub tool whose body always raises, for error-propagation tests.

    Increments :attr:`calls` before raising so a test can assert the body ran
    exactly once and that nothing was stored for that cache key.
    """

    def __init__(self) -> None:
        self.calls = 0

    def run(self, **kwargs: object) -> object:
        self.calls += 1
        raise RuntimeError("tool boom")

    async def arun(self, **kwargs: object) -> object:
        self.calls += 1
        raise RuntimeError("tool boom")


@pytest.fixture
def stub_tool() -> StubTool:
    """A fresh deterministic :class:`StubTool` per test."""
    return StubTool()


@pytest.fixture
def raising_tool() -> RaisingTool:
    """A fresh always-raising :class:`RaisingTool` per test."""
    return RaisingTool()


@pytest.fixture
def counting_embedder() -> StubEmbedder:
    """An embed-call-counting :class:`StubEmbedder` for the decorator tests."""
    return StubEmbedder()


def make_tool_cache(embedder: EmbeddingProvider | None = None, **overrides: object) -> MemoryCache:
    """Build a MemoryCache over the in-memory backend for the decorator tests.

    Reuses the Phase 2/3 building blocks: an :class:`InMemoryBackend` via
    ``backend="memory"`` and, when ``embedder`` is supplied, a semantic-enabled
    cache with that embed-call-counting embedder injected so "no embedding on an
    exact hit / an exact-only wrapper" stays assertable.

    Args:
        embedder: Optional embed-call-counting provider. When given, the cache is
            built semantic-enabled and the embedder is injected onto the built
            :class:`~memory_reuse.cache.semantic.SemanticCache`.
        **overrides: Extra :class:`~memory_reuse.config.CacheConfig` overrides.

    Returns:
        A configured :class:`~memory_reuse.core.MemoryCache`.
    """
    if embedder is not None:
        return make_semantic_cache(embedder, **overrides)
    params: dict[str, object] = {"backend": "memory", "default_ttl": 3600, "enable_stats": True}
    params.update(overrides)
    return MemoryCache(CacheConfig(**params))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hypothesis strategies: tool arguments and value byte-blobs
# ---------------------------------------------------------------------------

# The largest value the AgentCore backend must round-trip unchanged (Req 8.5).
MAX_VALUE_BYTES = 1_048_576

# JSON-serialisable scalars usable as tool argument values. Kept aligned with
# the graph-cache property strategies so the stored results stay serialisable.
tool_arg_values = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)

# A mapping of keyword arguments a decorated stub tool can be called with. Keys
# are valid Python identifiers so they can be bound to ``**kwargs``.
tool_arguments = st.dictionaries(
    keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8),
    values=tool_arg_values,
    max_size=5,
)


def value_blobs(max_size: int = MAX_VALUE_BYTES) -> st.SearchStrategy[bytes]:
    """Strategy for value byte-blobs spanning the 0..``max_size``-byte range.

    Covers the empty blob and blobs up to 1 MiB so the AgentCore byte round-trip
    property (Req 8.5) exercises the whole supported size range. The upper bound
    is generated sparsely by Hypothesis, so most examples stay small and fast
    while the boundary is still reachable.

    Args:
        max_size: Inclusive upper bound on blob length in bytes. Defaults to
            :data:`MAX_VALUE_BYTES` (1_048_576).

    Returns:
        A Hypothesis strategy producing ``bytes`` of length 0..``max_size``.
    """
    return st.binary(min_size=0, max_size=max_size)
