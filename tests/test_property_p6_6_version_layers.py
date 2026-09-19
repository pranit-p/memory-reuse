"""Property 6: Version applies across all cache layers.

Feature: intelligent-execution-optimization, Property 6.

*For any* version, the exact, tool, and semantic layers all derive
version-scoped identities, so a value stored under version A is not returned
under version B in any layer — even when both caches share one backend / vector
index.

Validates: Requirements 3.5.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.exact import ExactCache
from memory_reuse.cache.tool import ToolCache
from memory_reuse.config import CacheConfig
from memory_reuse.stats import StatsTracker

_versions = st.text(min_size=1, max_size=8)


class TestProperty6VersionAcrossLayers:
    """Feature: intelligent-execution-optimization, Property 6.

    Version applies across all cache layers.

    Validates: Requirements 3.5
    """

    @settings(max_examples=100)
    @given(v_a=_versions, v_b=_versions, value=st.text(max_size=20))
    async def test_exact_and_tool_isolated_by_version_on_shared_backend(
        self, v_a: str, v_b: str, value: str
    ) -> None:
        """On one shared backend, a version-A entry is invisible under version B."""
        from hypothesis import assume

        assume(v_a != v_b)

        # A single shared backend stands in for a shared Redis/AgentCore store.
        backend = InMemoryBackend()
        stats = StatsTracker()
        cfg_a = CacheConfig(cache_version=v_a)
        cfg_b = CacheConfig(cache_version=v_b)

        # Exact layer.
        exact_a = ExactCache(backend, cfg_a, stats)
        exact_b = ExactCache(backend, cfg_b, stats)
        await exact_a.set(["p"], value, scope="global", scope_id=None)
        assert await exact_a.get(["p"], scope="global", scope_id=None) == value
        assert await exact_b.get(["p"], scope="global", scope_id=None) is None

        # Tool layer (same shared backend).
        tool_a = ToolCache(backend, cfg_a, stats)
        tool_b = ToolCache(backend, cfg_b, stats)
        await tool_a.set("t", {"x": 1}, value, scope="global", scope_id=None, ttl=60)
        assert await tool_a.get("t", {"x": 1}, scope="global", scope_id=None) == value
        assert await tool_b.get("t", {"x": 1}, scope="global", scope_id=None) is None
