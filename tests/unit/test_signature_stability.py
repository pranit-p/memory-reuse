"""Backward-compatibility tests for public API signatures and config.

Task 10.1 (graph-level-cache): use :func:`inspect.signature` to pin the public
API surface so Phase 3 additions stay strictly additive.

* ``cached_node`` / ``cached_tool`` decorator signatures are unchanged
  (Reqs 10.2, 11.4, 14.1).
* ``CacheConfig`` fields and defaults are unchanged (Req 10.3).
* ``MemoryCache.invalidate_node`` exists as the node-level invalidation entry
  point (Req 13.1).
"""

from __future__ import annotations

import dataclasses
import inspect

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.integrations.langgraph import cached_node, cached_tool


class TestCachedNodeSignature:
    """``cached_node`` keeps its Phase 1/2 signature (Reqs 10.2, 11.4, 14.1)."""

    def test_parameter_names_and_kinds(self) -> None:
        params = inspect.signature(cached_node).parameters
        assert list(params) == [
            "cache",
            "scope",
            "ttl",
            "key_fields",
            "semantic",
            "exact_only",
        ]

        assert params["cache"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        # Everything after ``cache`` is keyword-only.
        for name in ("scope", "ttl", "key_fields", "semantic", "exact_only"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY

    def test_defaults_unchanged(self) -> None:
        params = inspect.signature(cached_node).parameters
        assert params["cache"].default is inspect.Parameter.empty
        assert params["scope"].default == "global"
        assert params["ttl"].default is None
        assert params["key_fields"].default is None
        assert params["semantic"].default is False
        assert params["exact_only"].default is False


class TestCachedToolSignature:
    """``cached_tool`` keeps its Phase 1/2 signature (Reqs 10.2, 14.1)."""

    def test_parameter_names_and_kinds(self) -> None:
        params = inspect.signature(cached_tool).parameters
        assert list(params) == [
            "cache",
            "scope",
            "ttl",
            "semantic",
            "exact_only",
        ]

        assert params["cache"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in ("scope", "ttl", "semantic", "exact_only"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY

    def test_defaults_unchanged(self) -> None:
        params = inspect.signature(cached_tool).parameters
        assert params["cache"].default is inspect.Parameter.empty
        assert params["scope"].default == "global"
        assert params["ttl"].default == 300
        assert params["semantic"].default is False
        assert params["exact_only"].default is False


class TestCacheConfigFields:
    """``CacheConfig`` fields and defaults are unchanged (Req 10.3)."""

    EXPECTED_DEFAULTS = {
        "backend": "memory",
        "redis_url": None,
        "default_ttl": 3600,
        "default_scope": "global",
        "key_prefix": "memreuse",
        "max_key_size": 512,
        "enable_stats": True,
        "semantic_enabled": False,
        "similarity_threshold": 0.95,
        "embedding_provider": None,
        "embedding_model": None,
        "max_vectors_per_namespace": 10_000,
        "store_exact_on_semantic_hit": True,
        "extract_answer": False,
        "extract_min_similarity": 0.5,
    }

    def test_field_names_and_order_unchanged(self) -> None:
        field_names = [f.name for f in dataclasses.fields(CacheConfig)]
        assert field_names == list(self.EXPECTED_DEFAULTS)

    def test_field_defaults_unchanged(self) -> None:
        config = CacheConfig()
        for name, expected in self.EXPECTED_DEFAULTS.items():
            assert getattr(config, name) == expected

    def test_init_signature_defaults_unchanged(self) -> None:
        params = inspect.signature(CacheConfig).parameters
        assert list(params) == list(self.EXPECTED_DEFAULTS)
        for name, expected in self.EXPECTED_DEFAULTS.items():
            assert params[name].default == expected


class TestInvalidateNodeExists:
    """``MemoryCache.invalidate_node`` exists (Req 13.1)."""

    def test_method_present_and_coroutine(self) -> None:
        assert hasattr(MemoryCache, "invalidate_node")
        assert inspect.iscoroutinefunction(MemoryCache.invalidate_node)

    def test_signature_shape(self) -> None:
        params = inspect.signature(MemoryCache.invalidate_node).parameters
        assert list(params) == [
            "self",
            "node",
            "state",
            "scope",
            "scope_id",
            "key_fields",
        ]
        assert params["state"].default is None
        assert params["scope"].default == "global"
        assert params["scope_id"].default is None
        assert params["key_fields"].default is None
        for name in ("scope", "scope_id", "key_fields"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
