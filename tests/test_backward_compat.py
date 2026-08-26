"""Backward-compatibility suite for the Phase 1-3 public API surface.

Task 8.2 (analytics-and-integrations): pin the observable Phase 1-3 contract so
the Phase 4 additions stay strictly additive. Where
``tests/unit/test_signature_stability.py`` already pins the ``cached_node`` /
``cached_tool`` / ``CacheConfig`` / ``invalidate_node`` signatures, this suite
covers the *broader* surface the earlier file does not:

* the complete top-level public-symbol set (``memory_reuse.__all__``) plus the
  ``cache`` and ``integrations`` sub-package exports, asserting no public symbol
  was removed or renamed (Req 11.6);
* ``inspect.signature`` pinning for ``MemoryCache``, ``ExactCache``,
  ``ToolCache``, ``SemanticCache``, and ``wrap_graph`` — the classes/methods the
  earlier file does not cover (Reqs 11.1, 11.2);
* the ``CacheStats`` field set and the ``hits == exact_hits + semantic_hits``
  invariant, as an example and as a property (Req 11.4);
* that with no Phase 4 opt-in a sequence of cache operations yields the same
  results, the same ``CacheStats`` snapshot, and the same raised error *types*
  as Phase 3 (Req 11.5).

Requirement 11.7 ("the Phase 1-3 suite passes unchanged against Phase 4 code")
is satisfied by the existing Phase 1-3 tests already running under this same
``pytest`` invocation; this file deliberately does not modify or duplicate them.

Validates: Requirements 11.1, 11.2, 11.4, 11.5, 11.6, 11.7
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import memory_reuse
from memory_reuse import (
    AgentMemoryError,
    BackendConnectionError,
    BackendNotAvailableError,
    CacheConfig,
    CacheStats,
    InvalidTTLError,
    MemoryCache,
    ScopeViolationError,
)
from memory_reuse.cache import ExactCache, SemanticCache, ToolCache
from memory_reuse.exceptions import (
    ConfigurationError,
    EmbeddingProviderError,
    ProviderMismatchError,
)
from memory_reuse.integrations.langgraph import cached_node, cached_tool
from memory_reuse.stats import CacheStats as CacheStatsFromStats


class TestPublicSymbolSurface:
    """No public symbol from the Phase 1-3 API was removed or renamed (Req 11.6)."""

    # The frozen top-level export set. Any removal/rename breaks this equality;
    # a new *additive* export must be appended here deliberately.
    EXPECTED_TOP_LEVEL_ALL = {
        "MemoryCache",
        "CacheConfig",
        "CacheStats",
        "AgentMemoryError",
        "BackendConnectionError",
        "BackendNotAvailableError",
        "ScopeViolationError",
        "InvalidTTLError",
    }

    def test_top_level_all_unchanged(self) -> None:
        """``memory_reuse.__all__`` still exports exactly the Phase 1-3 names."""
        assert set(memory_reuse.__all__) == self.EXPECTED_TOP_LEVEL_ALL

    def test_top_level_symbols_are_importable_and_bound(self) -> None:
        """Every advertised top-level symbol resolves to a live object."""
        for name in self.EXPECTED_TOP_LEVEL_ALL:
            assert hasattr(memory_reuse, name), f"missing public symbol: {name}"
            assert getattr(memory_reuse, name) is not None

    def test_cache_subpackage_exports_unchanged(self) -> None:
        """``memory_reuse.cache`` still exports the three cache classes."""
        import memory_reuse.cache as cache_pkg

        assert set(cache_pkg.__all__) == {"ExactCache", "SemanticCache", "ToolCache"}
        assert cache_pkg.ExactCache is ExactCache
        assert cache_pkg.SemanticCache is SemanticCache
        assert cache_pkg.ToolCache is ToolCache

    def test_integrations_exports_include_phase_1_3_decorators(self) -> None:
        """``memory_reuse.integrations`` still exports ``cached_node`` / ``cached_tool``.

        The set is asserted as a superset (not equality) because Phase 4 is
        allowed to *add* integration exports; it must not drop the existing
        ones (Req 11.6).
        """
        import memory_reuse.integrations as integrations_pkg

        assert {"cached_node", "cached_tool"}.issubset(set(integrations_pkg.__all__))
        assert integrations_pkg.cached_node is cached_node
        assert integrations_pkg.cached_tool is cached_tool

    def test_cachestats_is_the_same_class_everywhere(self) -> None:
        """The re-exported ``CacheStats`` is the one defined in ``stats``."""
        assert CacheStats is CacheStatsFromStats


class TestExceptionSymbols:
    """The Phase 1-3 exception types still exist and are exported (Req 11.6)."""

    def test_public_exceptions_are_exported_and_subclass_base(self) -> None:
        """The five re-exported exceptions descend from ``AgentMemoryError``."""
        for exc in (
            BackendConnectionError,
            BackendNotAvailableError,
            ScopeViolationError,
            InvalidTTLError,
        ):
            assert issubclass(exc, AgentMemoryError)

    def test_exceptions_module_symbols_present(self) -> None:
        """Every Phase 1-3 exception class still lives in ``memory_reuse.exceptions``."""
        for exc in (
            AgentMemoryError,
            BackendConnectionError,
            BackendNotAvailableError,
            ScopeViolationError,
            InvalidTTLError,
            EmbeddingProviderError,
            ProviderMismatchError,
            ConfigurationError,
        ):
            assert issubclass(exc, Exception)
            assert exc.__module__ == "memory_reuse.exceptions"


class TestMemoryCacheSignatures:
    """``MemoryCache`` retains its Phase 1-3 method surface (Reqs 11.1, 11.2)."""

    def test_public_methods_present(self) -> None:
        """The documented public methods/properties still exist on the class."""
        for name in (
            "from_env",
            "set_context",
            "clear_context",
            "reset_stats",
            "lookup",
            "store",
            "wrap_graph",
            "invalidate_node",
            "ping",
            "close",
            "flush",
        ):
            assert hasattr(MemoryCache, name), f"MemoryCache lost method: {name}"
        assert isinstance(inspect.getattr_static(MemoryCache, "stats"), property)

    def test_init_signature(self) -> None:
        params = inspect.signature(MemoryCache).parameters
        assert list(params) == ["config", "kwargs"]
        assert params["config"].default is None
        assert params["config"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD

    def test_lookup_signature(self) -> None:
        params = inspect.signature(MemoryCache.lookup).parameters
        assert list(params) == [
            "self",
            "key_parts",
            "query_text",
            "scope",
            "scope_id",
            "exact_only",
            "threshold",
        ]
        for name in ("scope", "scope_id", "exact_only", "threshold"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["exact_only"].default is False
        assert params["threshold"].default is None

    def test_store_signature(self) -> None:
        params = inspect.signature(MemoryCache.store).parameters
        assert list(params) == [
            "self",
            "key_parts",
            "query_text",
            "value",
            "scope",
            "scope_id",
            "ttl",
            "exact_only",
        ]
        for name in ("scope", "scope_id", "ttl", "exact_only"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["ttl"].default is None
        assert params["exact_only"].default is False

    def test_wrap_graph_signature(self) -> None:
        """``wrap_graph`` keeps its Phase 3 parameter names, order, and defaults."""
        params = inspect.signature(MemoryCache.wrap_graph).parameters
        assert list(params) == [
            "self",
            "graph",
            "semantic",
            "similarity_threshold",
            "ttl",
            "scope",
            "key_fields",
            "exact_only",
            "graph_id",
        ]
        assert params["graph"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in (
            "semantic",
            "similarity_threshold",
            "ttl",
            "scope",
            "key_fields",
            "exact_only",
            "graph_id",
        ):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["semantic"].default is False
        assert params["similarity_threshold"].default is None
        assert params["ttl"].default is None
        assert params["scope"].default is None
        assert params["key_fields"].default is None
        assert params["exact_only"].default is False
        assert params["graph_id"].default is None


class TestExactCacheSignatures:
    """``ExactCache`` retains its Phase 1 signatures (Reqs 11.1)."""

    def test_init_signature(self) -> None:
        params = inspect.signature(ExactCache).parameters
        assert list(params) == ["backend", "config", "stats"]

    def test_get_signature(self) -> None:
        params = inspect.signature(ExactCache.get).parameters
        assert list(params) == ["self", "key_parts", "scope", "scope_id"]

    def test_set_signature(self) -> None:
        params = inspect.signature(ExactCache.set).parameters
        assert list(params) == ["self", "key_parts", "value", "scope", "scope_id", "ttl"]
        assert params["ttl"].default is None

    def test_invalidate_signature(self) -> None:
        params = inspect.signature(ExactCache.invalidate).parameters
        assert list(params) == ["self", "key_parts", "scope", "scope_id"]


class TestToolCacheSignatures:
    """``ToolCache`` retains its Phase 1 signatures (Reqs 11.1)."""

    def test_init_signature(self) -> None:
        params = inspect.signature(ToolCache).parameters
        assert list(params) == ["backend", "config", "stats"]

    def test_get_signature(self) -> None:
        params = inspect.signature(ToolCache.get).parameters
        assert list(params) == ["self", "tool_name", "args", "scope", "scope_id"]

    def test_set_signature(self) -> None:
        params = inspect.signature(ToolCache.set).parameters
        assert list(params) == ["self", "tool_name", "args", "value", "scope", "scope_id", "ttl"]
        # ToolCache.set requires an explicit positional TTL (no default).
        assert params["ttl"].default is inspect.Parameter.empty


class TestSemanticCacheSignatures:
    """``SemanticCache`` retains its Phase 2 signatures (Reqs 11.1)."""

    def test_init_signature(self) -> None:
        params = inspect.signature(SemanticCache).parameters
        assert list(params) == ["index", "embedder", "config", "stats"]

    def test_get_signature(self) -> None:
        params = inspect.signature(SemanticCache.get).parameters
        assert list(params) == ["self", "query_text", "scope", "scope_id", "threshold"]
        assert params["threshold"].default is None

    def test_set_signature(self) -> None:
        params = inspect.signature(SemanticCache.set).parameters
        assert list(params) == [
            "self",
            "query_text",
            "value",
            "scope",
            "scope_id",
            "ttl",
            "precomputed_embedding",
        ]
        assert params["ttl"].default is None
        assert params["precomputed_embedding"].default is None


class TestCacheStatsFields:
    """``CacheStats`` fields and the ``hits`` invariant hold (Req 11.4)."""

    EXPECTED_FIELDS = {
        "hits": 0,
        "exact_hits": 0,
        "semantic_hits": 0,
        "misses": 0,
        "errors": 0,
        "total_requests": 0,
    }

    def test_field_names_order_and_defaults_unchanged(self) -> None:
        fields = {f.name: f.default for f in dataclasses.fields(CacheStats)}
        assert list(fields) == list(self.EXPECTED_FIELDS)
        assert fields == self.EXPECTED_FIELDS

    def test_hit_rate_property_and_to_dict_keys(self) -> None:
        """The derived ``hit_rate`` and the ``to_dict`` shape are unchanged."""
        stats = CacheStats()
        assert isinstance(inspect.getattr_static(CacheStats, "hit_rate"), property)
        assert stats.hit_rate == 0.0
        assert set(stats.to_dict()) == {
            "hits",
            "exact_hits",
            "semantic_hits",
            "misses",
            "errors",
            "total_requests",
            "hit_rate",
        }

    def test_hits_invariant_example(self) -> None:
        """The reported ``hits`` equals ``exact_hits + semantic_hits`` (Req 11.4)."""
        stats = CacheStats(hits=5, exact_hits=3, semantic_hits=2, misses=1, total_requests=6)
        assert stats.hits == stats.exact_hits + stats.semantic_hits


class TestCacheStatsInvariantProperty:
    """The ``hits == exact_hits + semantic_hits`` invariant is maintained by the tracker.

    Validates: Requirements 11.4
    """

    @settings(max_examples=100)
    @given(
        ops=st.lists(
            st.sampled_from(["exact_hit", "semantic_hit", "miss", "error"]),
            max_size=40,
        )
    )
    async def test_tracker_preserves_hits_invariant(self, ops: list[str]) -> None:
        """For any sequence of recorded events, ``hits == exact_hits + semantic_hits``.

        Drives the real :class:`~memory_reuse.stats.StatsTracker` through an
        arbitrary sequence of the four record operations and asserts the
        snapshot invariant holds throughout, alongside the derived counts
        (``total_requests`` counts hits+misses, and ``hit_rate`` stays in
        ``[0.0, 1.0]``).
        """
        from memory_reuse.stats import StatsTracker

        tracker = StatsTracker()
        exact = semantic = misses = errors = 0
        for op in ops:
            if op == "exact_hit":
                tracker.record_exact_hit()
                exact += 1
            elif op == "semantic_hit":
                tracker.record_semantic_hit()
                semantic += 1
            elif op == "miss":
                tracker.record_miss()
                misses += 1
            else:
                tracker.record_error()
                errors += 1

            snap = tracker.get_stats()
            # The core invariant (Req 11.4).
            assert snap.hits == snap.exact_hits + snap.semantic_hits
            # And the counters track the driven sequence exactly.
            assert snap.exact_hits == exact
            assert snap.semantic_hits == semantic
            assert snap.hits == exact + semantic
            assert snap.misses == misses
            assert snap.errors == errors
            assert snap.total_requests == exact + semantic + misses
            assert 0.0 <= snap.hit_rate <= 1.0


class TestNoPhase4OptInMatchesPhase3:
    """Without any Phase 4 opt-in, behaviour matches Phase 3 (Req 11.5)."""

    def _phase3_cache(self) -> MemoryCache:
        """A default, exact-only cache — the Phase 3 baseline (no Phase 4 fields set)."""
        return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))

    def test_default_config_leaves_phase4_fields_at_defaults(self) -> None:
        """A default ``CacheConfig`` has the additive Phase 4 fields at their no-op defaults."""
        config = CacheConfig()
        assert config.backend == "memory"
        assert config.agentcore_region is None
        assert config.agentcore_memory_id is None
        # No semantic opt-in -> the exact-only path is used.
        assert config.semantic_enabled is False

    async def test_exact_only_cache_has_no_semantic_component(self) -> None:
        """With no semantic opt-in, ``MemoryCache.semantic`` is ``None`` (Phase 1 behaviour)."""
        cache = self._phase3_cache()
        assert cache.semantic is None

    async def test_result_and_stats_sequence_matches_phase3(self) -> None:
        """A store/lookup/miss sequence yields the Phase 3 results and stats snapshot (Req 11.5)."""
        cache = self._phase3_cache()

        # Miss on an empty cache.
        assert await cache.lookup(["k"], "k", scope="global", scope_id=None) is None
        # Store then hit.
        await cache.store(["k"], "k", "v", scope="global", scope_id=None)
        assert await cache.lookup(["k"], "k", scope="global", scope_id=None) == "v"
        # A second, distinct key misses.
        assert await cache.lookup(["other"], "other", scope="global", scope_id=None) is None

        stats = cache.stats
        # One exact hit, no semantic hits, two misses, three lookups counted.
        assert stats.exact_hits == 1
        assert stats.semantic_hits == 0
        assert stats.hits == 1
        assert stats.misses == 2
        assert stats.errors == 0
        assert stats.total_requests == 3
        # The invariant still holds on a live cache.
        assert stats.hits == stats.exact_hits + stats.semantic_hits

    async def test_error_types_match_phase3(self) -> None:
        """Error *types* raised on misuse are unchanged from Phase 3 (Req 11.5)."""
        cache = self._phase3_cache()

        # A non-global scope without a scope_id still raises ScopeViolationError.
        with pytest.raises(ScopeViolationError):
            await cache.store(["k"], "k", "v", scope="user", scope_id=None)
        with pytest.raises(ScopeViolationError):
            await cache.lookup(["k"], "k", scope="user", scope_id=None)

        # A non-positive TTL on the tool cache still raises InvalidTTLError.
        with pytest.raises(InvalidTTLError):
            await cache.tool.set("t", {"q": 1}, "v", scope="global", scope_id=None, ttl=0)

    def test_unknown_backend_raises_configuration_error(self) -> None:
        """An unrecognised backend value still raises ``ConfigurationError`` (Req 11.5)."""
        with pytest.raises(ConfigurationError):
            CacheConfig(backend="does-not-exist")  # type: ignore[arg-type]
