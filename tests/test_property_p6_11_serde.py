"""Property 11: Configured (de)serialization round-trips faithfully.

Feature: intelligent-execution-optimization, Property 11.

*For any* value the configured serializer/deserializer pair supports, storing
then loading reproduces a value equal to the original; and *for any* JSON-native
value with no hook configured, the stored bytes are byte-identical to the
pre-Requirement-9 behaviour.

Validates: Requirements 9.1, 9.3, 9.4.
"""

from __future__ import annotations

import gzip
import json

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse._utils import deserialize_value, serialize_value

# JSON-native values (the pre-hook contract).
_json_values = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1000, max_value=1000),
        st.text(max_size=20),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=6), children, max_size=4),
    ),
    max_leaves=10,
)


class TestProperty11Serde:
    """Feature: intelligent-execution-optimization, Property 11.

    Configured (de)serialization round-trips faithfully.

    Validates: Requirements 9.1, 9.3, 9.4
    """

    @settings(max_examples=100)
    @given(value=_json_values)
    def test_no_hook_is_byte_identical_to_plain_json(self, value: object) -> None:
        """With no hook, the stored bytes equal the plain gzip/JSON encoding (Req 9.4)."""
        got = serialize_value(value)
        expected = gzip.compress(
            json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"),
            compresslevel=6,
        )
        assert got == expected
        # And the value round-trips unchanged.
        assert deserialize_value(got) == value

    @settings(max_examples=100)
    @given(value=_json_values)
    def test_configured_pair_round_trips(self, value: object) -> None:
        """A serializer/deserializer pair reproduces the original value (Req 9.1, 9.3).

        Uses a wrap/unwrap codec (tags the payload on store, strips it on load)
        so the pair genuinely transforms the stored representation yet still
        reproduces the original on load.
        """

        def serializer(v: object) -> object:
            return {"__wrapped__": v}

        def deserializer(v: object) -> object:
            if isinstance(v, dict) and set(v) == {"__wrapped__"}:
                return v["__wrapped__"]
            return v

        raw = serialize_value(value, serializer=serializer)
        loaded = deserialize_value(raw, deserializer=deserializer)
        assert loaded == value
