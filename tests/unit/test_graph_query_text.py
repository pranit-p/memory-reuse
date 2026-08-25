"""Property-based test for whole-run semantic query-text derivation.

Exercises the pure ``_derive_graph_key`` helper in
``memory_reuse.integrations.langgraph`` (Req 3.3). The helper is framework-free,
so this test never touches LangGraph, a backend, or an embedder.

* **Property 5 — Semantic query text derives from the selected fields:** for any
  initial input state and optional ``key_fields``, the derived semantic query
  text equals the string form of the selected key data — the ``key_fields``
  subset when configured, otherwise the full state.
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

# ``key_fields`` may be unset (full state) or an arbitrary list of field names,
# including names absent from the state.
_key_fields = st.one_of(
    st.none(),
    st.lists(st.text(min_size=1, max_size=10), max_size=5),
)

_graph_ids = st.text(min_size=1, max_size=15)


def _selected_key_data(state: Any, key_fields: list[str] | None) -> Any:
    """Reference for the ``key_data`` the query text must stringify (design)."""
    if key_fields is not None and isinstance(state, dict):
        return {f: state.get(f) for f in key_fields}
    if isinstance(state, dict):
        return state
    return str(state)


# ---------------------------------------------------------------------------
# Property 5: Semantic query text derives from the selected fields
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(state=_states, key_fields=_key_fields, graph_id=_graph_ids)
def test_property_query_text_derives_from_selected_fields(
    state: Any, key_fields: list[str] | None, graph_id: str
) -> None:
    """Property 5: semantic query text derives from the selected fields.

    Feature: graph-level-cache, Property 5: Semantic query text derives from the
    selected fields.

    The derived ``query_text`` equals the string form of the selected key data:
    the ``key_fields`` subset for a dict state when configured, the full dict
    state otherwise, and ``str(state)`` for a non-dict state.

    **Validates: Requirements 3.3**
    """
    _, query_text = _derive_graph_key(state, key_fields, graph_id)

    assert query_text == str(_selected_key_data(state, key_fields))
