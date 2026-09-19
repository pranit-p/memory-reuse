"""Property 5: Version changes the key, absence preserves it.

Feature: intelligent-execution-optimization, Property 5.

*For any* key parts, scope, and scope id: two distinct non-empty versions yield
different Cache_Keys; the same version yields the same Cache_Key; and
``version is None`` yields exactly the pre-Phase-6 key (a byte-for-byte match
against ``build_cache_key`` called without the argument).

Validates: Requirements 3.2, 3.3, 3.4.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from memory_reuse._utils import build_cache_key

_parts = st.lists(
    st.one_of(st.text(max_size=12), st.integers(min_value=-1000, max_value=1000)),
    min_size=1,
    max_size=4,
)
_versions = st.text(min_size=1, max_size=12)
# scope/scope_id pairs that satisfy build_cache_key's scope guard.
_scope_pairs = st.one_of(
    st.tuples(st.just("global"), st.none()),
    st.tuples(st.just("user"), st.text(min_size=1, max_size=8)),
    st.tuples(st.just("session"), st.text(min_size=1, max_size=8)),
)


class TestProperty5VersionIdentity:
    """Feature: intelligent-execution-optimization, Property 5.

    Version changes the key, absence preserves it.

    Validates: Requirements 3.2, 3.3, 3.4
    """

    @settings(max_examples=100)
    @given(parts=_parts, scope_pair=_scope_pairs, v1=_versions, v2=_versions)
    def test_version_key_relationships(
        self,
        parts: list,
        scope_pair: tuple[str, str | None],
        v1: str,
        v2: str,
    ) -> None:
        """Distinct versions differ; same version matches; None == pre-Phase-6."""
        scope, scope_id = scope_pair

        key_none = build_cache_key("memreuse", scope, scope_id, *parts)
        key_none_explicit = build_cache_key("memreuse", scope, scope_id, *parts, version=None)
        key_v1 = build_cache_key("memreuse", scope, scope_id, *parts, version=v1)
        key_v1_again = build_cache_key("memreuse", scope, scope_id, *parts, version=v1)

        # Req 3.3: version=None is byte-identical to omitting the argument.
        assert key_none == key_none_explicit
        # Req 3.4: same version + same inputs -> same key (shared-backend reuse).
        assert key_v1 == key_v1_again
        # A set version differs from the unversioned key.
        assert key_v1 != key_none

        # Req 3.2: two *distinct* versions yield distinct keys.
        assume(v1 != v2)
        key_v2 = build_cache_key("memreuse", scope, scope_id, *parts, version=v2)
        assert key_v1 != key_v2
