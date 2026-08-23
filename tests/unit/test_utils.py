"""Unit tests for memory_reuse._utils."""

from __future__ import annotations

import pytest

from memory_reuse._utils import (
    build_cache_key,
    cosine_similarity,
    deserialize_value,
    hash_value,
    sanitize_key,
    serialize_value,
    split_sentences,
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


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, list(v)) == pytest.approx(1.0)

    def test_opposite_vectors_return_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors_return_half(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.5)

    def test_result_within_unit_range(self) -> None:
        score = cosine_similarity([3.0, -1.0, 2.0], [1.0, 4.0, -2.0])
        assert 0.0 <= score <= 1.0

    def test_scaled_vectors_are_identical(self) -> None:
        # Cosine ignores magnitude; a scaled copy is maximally similar.
        assert cosine_similarity([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_returns_neutral(self) -> None:
        assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == pytest.approx(0.5)

    def test_both_zero_vectors_return_neutral(self) -> None:
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.5)

    def test_empty_vectors_return_neutral(self) -> None:
        assert cosine_similarity([], []) == pytest.approx(0.5)


class TestSplitSentences:
    def test_empty_string_returns_empty(self) -> None:
        assert split_sentences("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert split_sentences("   \n  ") == []

    def test_single_sentence_no_boundary(self) -> None:
        assert split_sentences("just one sentence") == ["just one sentence"]

    def test_splits_on_period(self) -> None:
        assert split_sentences("First. Second.") == ["First.", "Second."]

    def test_splits_on_question_and_exclamation(self) -> None:
        assert split_sentences("Really? Yes! Ok.") == ["Really?", "Yes!", "Ok."]

    def test_splits_on_newlines(self) -> None:
        # Bulleted/line-separated answers split per line even without a period.
        assert split_sentences("- one\n- two\n- three") == ["- one", "- two", "- three"]

    def test_strips_and_drops_empty_fragments(self) -> None:
        assert split_sentences("A.   \n\n  B.") == ["A.", "B."]
