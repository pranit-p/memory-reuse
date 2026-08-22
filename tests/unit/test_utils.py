"""Unit tests for memory_reuse._utils."""

from __future__ import annotations

import pytest

from memory_reuse._utils import (
    build_cache_key,
    deserialize_value,
    hash_value,
    sanitize_key,
    serialize_value,
)


class TestSanitizeKey:
    def test_alphanumeric_unchanged(self) -> None:
        assert sanitize_key("abc123") == "abc123"

    def test_dash_and_dot_unchanged(self) -> None:
        assert sanitize_key("my-key.v1") == "my-key.v1"

    def test_at_sign_replaced(self) -> None:
        assert sanitize_key("user@example.com") == "user_example.com"

    def test_spaces_replaced(self) -> None:
        assert sanitize_key("hello world") == "hello_world"

    def test_empty_string(self) -> None:
        assert sanitize_key("") == ""


class TestHashValue:
    def test_returns_32_chars(self) -> None:
        assert len(hash_value("hello")) == 32

    def test_deterministic(self) -> None:
        assert hash_value({"a": 1, "b": 2}) == hash_value({"b": 2, "a": 1})

    def test_different_inputs_different_hashes(self) -> None:
        assert hash_value("foo") != hash_value("bar")

    def test_list_input(self) -> None:
        h = hash_value([1, 2, 3])
        assert isinstance(h, str)
        assert len(h) == 32

    def test_none_input(self) -> None:
        h = hash_value(None)
        assert isinstance(h, str)


class TestBuildCacheKey:
    def test_global_scope(self) -> None:
        key = build_cache_key("agentmem", "global", None, "part1")
        assert key.startswith("agentmem:global:")
        # No scope_id segment for global
        parts = key.split(":")
        assert len(parts) == 3

    def test_user_scope(self) -> None:
        key = build_cache_key("agentmem", "user", "alice", "part1")
        assert "alice" in key
        parts = key.split(":")
        assert len(parts) == 4

    def test_session_scope(self) -> None:
        key = build_cache_key("agentmem", "session", "sess-abc", "part1")
        assert "sess-abc" in key

    def test_missing_scope_id_raises(self) -> None:
        with pytest.raises(ValueError, match="scope_id is required"):
            build_cache_key("agentmem", "user", None, "part1")

    def test_deterministic_with_dict_parts(self) -> None:
        k1 = build_cache_key("p", "global", None, {"a": 1, "b": 2})
        k2 = build_cache_key("p", "global", None, {"b": 2, "a": 1})
        assert k1 == k2

    def test_different_users_different_keys(self) -> None:
        k_alice = build_cache_key("p", "user", "alice", "data")
        k_bob = build_cache_key("p", "user", "bob", "data")
        assert k_alice != k_bob


class TestSerializeDeserialize:
    def test_roundtrip_dict(self) -> None:
        original = {"key": "value", "number": 42}
        assert deserialize_value(serialize_value(original)) == original

    def test_roundtrip_list(self) -> None:
        original = [1, 2, 3, "hello"]
        assert deserialize_value(serialize_value(original)) == original

    def test_roundtrip_string(self) -> None:
        assert deserialize_value(serialize_value("hello world")) == "hello world"

    def test_roundtrip_none(self) -> None:
        assert deserialize_value(serialize_value(None)) is None

    def test_compressed_smaller_than_raw(self) -> None:
        large_text = "a" * 10_000
        compressed = serialize_value(large_text)
        raw = large_text.encode("utf-8")
        assert len(compressed) < len(raw)

    def test_invalid_bytes_raises(self) -> None:
        with pytest.raises(ValueError):
            deserialize_value(b"not compressed data")
