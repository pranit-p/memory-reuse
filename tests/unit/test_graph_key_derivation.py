"""Property-based tests for the whole-run key/query derivation helper.

These exercise the pure ``_derive_graph_key`` helper in
``memory_reuse.integrations.langgraph`` (Req 3). The helper is framework-free,
so these tests never touch LangGraph, a backend, or an embedder.

* **Property 6 — Distinct graph identifiers never collide:** for any initial
  input state, two wrappers with distinct graph identifiers never share a cache
  entry, while two wrappers with the same graph identifier do share entries for
  equal states.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.integrations.langgraph import _derive_graph_key

# ---------------------------------------------------------------------------
# Strategies for JSON-serialisable dict states plus non-dict states.
# ---------------------------------------------------------------------------

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text(max_size=20),
)

_dict_states = st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=_json_scalars,
    max_size=5,
)

_non_dict_states = st.one_of(
    st.text(max_size=20),
    st.integers(),
    st.lists(_json_scalars, max_size=5),
)

_states = st.one_of(_dict_states, _non_dict_states)

_graph_ids = st.text(min_size=1, max_size=15)


# ---------------------------------------------------------------------------
# Property 6: Distinct graph identifiers never collide
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(state=_states, ids=st.tuples(_graph_ids, _graph_ids))
def test_property_distinct_graph_ids_never_collide(state: Any, ids: tuple[str, str]) -> None:
    """Property 6: distinct graph identifiers never collide.

    Feature: graph-level-cache, Property 6: Distinct graph identifiers never
    collide.

    **Validates: Requirements 3.4**
    """
    id_a, id_b = ids

    key_a, _ = _derive_graph_key(state, None, id_a)
    key_b, _ = _derive_graph_key(state, None, id_b)

    if id_a == id_b:
        # Same identifier + equal state -> the wrappers share a cache entry.
        assert key_a == key_b
    else:
        # Distinct identifiers -> the key parts differ, so entries never
        # collide even for an identical initial input state.
        assert key_a != key_b
