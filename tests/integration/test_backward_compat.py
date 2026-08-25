"""Backward-compatibility suite for Phase 3 (graph-level cache).

Task 10.4: confirm the existing Phase 1/2 suites pass unchanged and that the
``cached_node`` / ``cached_tool`` decorators keep the same signature and caching
semantics they had before the Phase 3 additions.

These are example tests (not property-based). Requirements 10.1, 10.5, 14.2,
14.3, 14.5.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.exceptions import ScopeViolationError
from memory_reuse.integrations.langgraph import cached_node, cached_tool

# Repository root (…/memory-reuse), two levels up from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The Phase 1/2 suites whose behaviour must remain unchanged by Phase 3. These
# cover the exact/tool/semantic caches, the decorators, config, stats, and the
# LangGraph integration decorators — everything shipped before wrap_graph.
PHASE12_SUITES = [
    "tests/integration/test_langgraph_integration.py",
    "tests/unit/test_exact_cache.py",
    "tests/unit/test_tool_cache.py",
    "tests/unit/test_semantic_cache.py",
    "tests/unit/test_config.py",
    "tests/unit/test_stats.py",
]


def make_cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


class TestPhase12SuitePassesUnchanged:
    """Run the existing Phase 1/2 suites and assert they all pass."""

    def test_phase12_suites_pass(self) -> None:
        # Only run suites that actually exist, so the test stays robust if the
        # layout shifts, while still failing loudly if a suite regresses.
        existing = [s for s in PHASE12_SUITES if (REPO_ROOT / s).is_file()]
        assert existing, "expected at least one Phase 1/2 suite to exist"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", *existing, "-q", "-p", "no:cacheprovider"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "Phase 1/2 suites must pass unchanged.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


class TestDecoratorSignaturesUnchanged:
    """The Phase 1/2 decorator signatures must be byte-for-byte unchanged."""

    def test_cached_node_signature(self) -> None:
        params = inspect.signature(cached_node).parameters
        # Positional: the cache instance.
        assert list(params)[0] == "cache"
        # Keyword-only options and their defaults, exactly as in Phases 1/2.
        assert params["scope"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["scope"].default == "global"
        assert params["ttl"].default is None
        assert params["key_fields"].default is None
        assert params["semantic"].default is False
        assert params["exact_only"].default is False
        # No extra parameters were added to the decorator.
        assert set(params) == {
            "cache",
            "scope",
            "ttl",
            "key_fields",
            "semantic",
            "exact_only",
        }

    def test_cached_tool_signature(self) -> None:
        params = inspect.signature(cached_tool).parameters
        assert list(params)[0] == "cache"
        assert params["scope"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["scope"].default == "global"
        assert params["ttl"].default == 300
        assert params["semantic"].default is False
        assert params["exact_only"].default is False
        assert set(params) == {"cache", "scope", "ttl", "semantic", "exact_only"}

    def test_cache_config_fields_unchanged(self) -> None:
        # CacheConfig fields/defaults must be preserved (Req 10.3).
        cfg = CacheConfig()
        assert cfg.backend == "memory"
        assert cfg.default_ttl == 3600
        assert cfg.default_scope == "global"
        assert cfg.key_prefix == "memreuse"
        assert cfg.max_key_size == 512
        assert cfg.enable_stats is True
        assert cfg.semantic_enabled is False
        assert cfg.similarity_threshold == 0.95


class TestCachedNodeSemanticsUnchanged:
    """``cached_node`` caching behaviour is identical to Phases 1/2."""

    @pytest.mark.asyncio
    async def test_hit_skips_body_and_returns_stored_output(self) -> None:
        cache = make_cache()
        calls = 0

        @cached_node(cache, scope="global")
        async def node(state: dict) -> dict:
            nonlocal calls
            calls += 1
            return {"out": state["in"]}

        state = {"in": "x"}
        first = await node(state)
        second = await node(state)
        assert first == second == {"out": "x"}
        assert calls == 1  # second call served from cache, body skipped

    @pytest.mark.asyncio
    async def test_key_fields_ignores_non_key_fields(self) -> None:
        cache = make_cache()
        calls = 0

        @cached_node(cache, scope="global", key_fields=["messages"])
        async def node(state: dict) -> dict:
            nonlocal calls
            calls += 1
            return {"done": True}

        await node({"messages": ["hi"], "ephemeral": "a"})
        await node({"messages": ["hi"], "ephemeral": "b"})
        assert calls == 1  # differing non-key field still hits

    @pytest.mark.asyncio
    async def test_user_scope_missing_id_raises(self) -> None:
        cache = make_cache()

        @cached_node(cache, scope="user")
        async def node(state: dict) -> dict:
            return {}

        with pytest.raises(ScopeViolationError):
            await node({"messages": ["hi"]})


class TestCachedToolSemanticsUnchanged:
    """``cached_tool`` caching behaviour is identical to Phases 1/2."""

    @pytest.mark.asyncio
    async def test_hit_skips_body_and_returns_stored_result(self) -> None:
        cache = make_cache()
        calls = 0

        @cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> str:
            nonlocal calls
            calls += 1
            return f"r:{query}"

        first = await fetch("hi")
        second = await fetch("hi")
        assert first == second == "r:hi"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_distinct_args_miss_separately(self) -> None:
        cache = make_cache()
        calls = 0

        @cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> str:
            nonlocal calls
            calls += 1
            return query

        await fetch("a")
        await fetch("b")
        assert calls == 2

    @pytest.mark.asyncio
    async def test_stats_recorded_like_phase12(self) -> None:
        cache = make_cache()

        @cached_tool(cache, scope="global", ttl=300)
        async def fn(x: int) -> int:
            return x

        await fn(1)  # miss
        await fn(1)  # hit
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
