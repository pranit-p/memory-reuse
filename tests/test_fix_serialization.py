"""Regression tests for hardened serialization (Fix #5).

The cache stores values as gzip-compressed JSON. Previously an unsupported type
was silently coerced to its ``str()`` form, storing a lossy value that could
never round-trip. It now raises :class:`SerializationError` instead, so garbage
is never cached. A corrupt stored entry decodes to a safe miss rather than a
wrong value or a raised error, and no code path uses pickle/eval.
"""

from __future__ import annotations

import gzip

import pytest

from memory_reuse import CacheConfig, MemoryCache, SerializationError
from memory_reuse._utils import deserialize_value, serialize_value


class _Unsupported:
    """A type JSON cannot represent and no serializer converts."""


class TestSerializeRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [None, False, 0, "", [], {}, {"a": [1, 2, {"b": None}]}, [1, "two", 3.0, True]],
    )
    def test_json_native_round_trips(self, value: object) -> None:
        assert deserialize_value(serialize_value(value)) == value


class TestUnsupportedTypes:
    def test_unsupported_type_raises_serialization_error(self) -> None:
        with pytest.raises(SerializationError):
            serialize_value(_Unsupported())

    def test_unsupported_type_inside_container_raises(self) -> None:
        with pytest.raises(SerializationError):
            serialize_value({"ok": 1, "bad": _Unsupported()})

    def test_coerce_unsupported_opt_in_stringifies(self) -> None:
        """The explicit opt-out (used by wrap_graph) still degrades to a string."""
        raw = serialize_value(_Unsupported(), coerce_unsupported=True)
        restored = deserialize_value(raw)
        assert isinstance(restored, str)

    def test_custom_serializer_handles_type(self) -> None:
        raw = serialize_value(_Unsupported(), serializer=lambda _v: {"kind": "unsupported"})
        assert deserialize_value(raw) == {"kind": "unsupported"}


class TestUnserialisableValueNotStored:
    async def test_exact_set_skips_unserialisable_value(self) -> None:
        """An unserialisable value is not stored (recorded error), never garbage."""
        cache = MemoryCache(CacheConfig(backend="memory"))
        await cache.exact.set(["k"], _Unsupported(), scope="global", scope_id=None, ttl=300)
        # Nothing was written, so a subsequent lookup is a clean miss.
        assert await cache.exact.get(["k"], scope="global", scope_id=None) is None
        assert cache.stats.errors >= 1


class TestCorruptEntries:
    def test_corrupt_bytes_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            deserialize_value(b"this is not gzip-compressed json")

    def test_valid_gzip_invalid_json_raises(self) -> None:
        bad = gzip.compress(b"{not valid json")
        with pytest.raises(ValueError):
            deserialize_value(bad)

    def test_deserializer_failure_raises_value_error(self) -> None:
        raw = serialize_value({"a": 1})

        def boom(_obj: object) -> object:
            raise KeyError("missing field")

        with pytest.raises(ValueError):
            deserialize_value(raw, deserializer=boom)

    async def test_corrupt_exact_entry_is_a_safe_miss(self) -> None:
        """A corrupt entry in the backend surfaces as a miss, not a wrong value."""
        cache = MemoryCache(CacheConfig(backend="memory"))
        # Write raw garbage under the exact key the cache would use.
        key = cache.exact._build_key(["k"], "global", None)
        await cache._backend.set(key, b"corrupt-not-gzip", ttl=300)

        # Public get and internal lookup both treat it as a miss (no raise).
        assert await cache.exact.get(["k"], scope="global", scope_id=None) is None
        from memory_reuse._utils import MISS

        assert await cache.exact._get(["k"], "global", None) is MISS


class TestNoUnsafeDeserialization:
    def test_serialization_is_json_not_pickle(self) -> None:
        """The stored bytes are gzip+JSON (safe), decodable without any pickle."""
        raw = serialize_value({"hello": "world"})
        # gzip magic bytes; decompresses to UTF-8 JSON text.
        assert raw[:2] == b"\x1f\x8b"
        assert gzip.decompress(raw).decode("utf-8") == '{"hello": "world"}'
