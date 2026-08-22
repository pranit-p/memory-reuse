"""Shared pytest fixtures for memory-reuse tests."""

from __future__ import annotations

import pytest

from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.exact import ExactCache
from memory_reuse.cache.tool import ToolCache
from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
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
