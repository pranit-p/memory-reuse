"""Property-based tests for the whole-run graph key derivation helper.

Exercises the pure ``_derive_graph_key`` helper in
``memory_reuse.integrations.langgraph`` (Task 3 of the graph-level-cache spec).
No LangGraph, LLM, or network is required: the helper is a pure function over
the initial input state, optional ``key_fields``, and ``graph_id``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.integrations.langgraph import _derive_graph_key

# ---------------------------------------------------------------------------
# Hypothesis strategies for initial input states
# ---------------------------------------------------------------------------

# JSON-serialisable scalar leaf values.
_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=20),
)

# JSON-serialisable dict states keyed by short field names.
_dict_states = st.dictionaries(
    keys=st.text(alphabet="abcdefghij", min_size=1, max_size=4),
    values=_leaves,
    max_size=6,
)

# Non-dict states (str / int / list), keyed by their string form.
_non_dict_states = st.one_of(
    st.text(max_size=20),
    st.integers(min_value=-1000, max_value=1000),
    st.lists(_leaves, max_size=6),
)

_GRAPH_ID = "graph::test"


def _key(state: object, key_fields: list[str] | None) -> list:
    return _derive_graph_key(state, key_fields, _GRAPH_ID)[0]


class TestProperty4CacheKeyFields:
    """Feature: graph-level-cache, Property 4.

    Cache key is determined by the selected input fields.

    Validates: Requirements 3.1, 3.2, 3.5, 11.5, 13.4
    """

    @settings(max_examples=100)
    @given(state_a=_dict_states, state_b=_dict_states)
    def test_full_state_key_matches_iff_states_equal(self, state_a: dict, state_b: dict) -> None:
        """With no key_fields, two dict states share a key iff they are equal."""
        key_a = _key(state_a, None)
        key_b = _key(state_b, None)
        assert (key_a == key_b) == (state_a == state_b)

    @settings(max_examples=100)
    @given(
        state=_dict_states,
        extra_field=st.text(alphabet="uvwxyz", min_size=1, max_size=4),
        extra_value=_leaves,
        key_fields=st.lists(
            st.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=4,
            unique=True,
        ),
    )
    def test_field_outside_key_fields_does_not_change_key(
        self,
        state: dict,
        extra_field: str,
        extra_value: object,
        key_fields: list[str],
    ) -> None:
        """Changing a field not in key_fields leaves the key unchanged."""
        # ``extra_field`` is drawn from a disjoint alphabet, so it is never in
        # key_fields; mutating it must not affect the derived key.
        assert extra_field not in key_fields
        mutated = {**state, extra_field: extra_value}
        assert _key(state, key_fields) == _key(mutated, key_fields)

    @settings(max_examples=100)
    @given(
        state=_dict_states,
        key_field=st.text(alphabet="abcdefghij", min_size=1, max_size=4),
        new_value=_leaves,
    )
    def test_changing_a_key_field_yields_distinct_key(
        self, state: dict, key_field: str, new_value: object
    ) -> None:
        """Changing a field in key_fields to a new value yields a distinct key."""
        key_fields = [key_field]
        # Ensure the value actually changes so the derived key must differ.
        original = state.get(key_field)
        mutated = {**state, key_field: new_value}
        if mutated.get(key_field) == original:
            # No observable change (same value) -> keys must be identical.
            assert _key(state, key_fields) == _key(mutated, key_fields)
        else:
            assert _key(state, key_fields) != _key(mutated, key_fields)

    @settings(max_examples=100)
    @given(state_a=_non_dict_states, state_b=_non_dict_states)
    def test_non_dict_states_keyed_by_string_form(self, state_a: object, state_b: object) -> None:
        """Non-dict states collide iff their string forms are equal."""
        key_a = _key(state_a, None)
        key_b = _key(state_b, None)
        assert (key_a == key_b) == (str(state_a) == str(state_b))
