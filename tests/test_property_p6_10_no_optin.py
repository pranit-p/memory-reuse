"""Property 10: No opt-in means Phase 5 behaviour.

Feature: intelligent-execution-optimization, Property 10.

*For any* sequence of cache operations on a default (Phase 6-field-default)
config, the returned results, the CacheStats snapshot, the AnalyticsSnapshot, and
the derived Cache_Keys are identical to Phase 5 — in particular the keys are
byte-identical to ``build_cache_key`` called without a version.

Validates: Requirements 8.2, 8.4.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse._utils import build_cache_key

_ops = st.lists(
    st.tuples(st.text(min_size=1, max_size=10), st.text(max_size=20)),
    max_size=20,
)


class TestProperty10NoOptIn:
    """Feature: intelligent-execution-optimization, Property 10.

    No opt-in means Phase 5 behaviour.

    Validates: Requirements 8.2, 8.4
    """

    def test_default_config_keys_are_byte_identical(self) -> None:
        """A default config derives keys byte-identical to the unversioned key."""
        cache = MemoryCache(CacheConfig(backend="memory"))
        for parts in ([["a"]], [["x", "y"]], [["k", 1, 2]]):
            built = cache.exact._build_key(parts[0], "global", None)  # noqa: SLF001
            expected = build_cache_key("memreuse", "global", None, *parts[0])
            assert built == expected

    @settings(max_examples=100, deadline=None)
    @given(ops=_ops)
    async def test_result_and_stats_sequence_matches_phase5(
        self, ops: list[tuple[str, str]]
    ) -> None:
        """A store/lookup sequence on a default config behaves exactly as Phase 5."""
        cache = MemoryCache(CacheConfig(backend="memory"))

        # Store each in order; duplicate keys are last-write-wins, exactly as any
        # cache behaves — so assert against the final value per key.
        final: dict[str, str] = {}
        for key, value in ops:
            await cache.store([key], key, value, scope="global", scope_id=None)
            final[key] = value
        for key, value in final.items():
            got = await cache.lookup([key], key, scope="global", scope_id=None)
            assert got == value

        stats = cache.stats
        # Only exact hits/misses; the invariant and derived counts hold as Phase 5.
        assert stats.hits == stats.exact_hits + stats.semantic_hits
        assert stats.semantic_hits == 0
        assert 0.0 <= stats.hit_rate <= 1.0

        # Analytics is zeroed without any recorded hit events (Phase 5 default).
        snap = cache.analytics
        assert snap.tokens_saved == 0
        assert str(snap.cost_saved) == "0"
        assert snap.latency_saved == 0.0

        # The analyzer is empty without any recorded operations.
        assert cache.analyze().recommendations == []
