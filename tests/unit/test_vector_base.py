"""Unit tests for the vector-index interface and data models.

These tests exercise the :class:`VectorRecord` and :class:`VectorMatch` data
models and confirm that :class:`VectorIndex` behaves as an abstract base class
whose full method set must be implemented by a concrete subclass.  Concrete
implementations (in-memory, Redis) and their behavioural properties are covered
in their own tasks; this module focuses purely on the abstraction defined in
``memory_reuse/vector/base.py``.
"""

from __future__ import annotations

import inspect

import pytest

from memory_reuse._utils import hash_value
from memory_reuse.vector import VectorIndex, VectorMatch, VectorRecord
from memory_reuse.vector import base as vector_base


class TestVectorRecord:
    def test_stores_all_fields(self) -> None:
        record = VectorRecord(
            vector=[0.1, 0.2, 0.3],
            value=b"gzipped",
            provider_model="openai:text-embedding-3-small",
            expires_at=123.0,
        )
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.value == b"gzipped"
        assert record.provider_model == "openai:text-embedding-3-small"
        assert record.expires_at == 123.0

    def test_expires_at_may_be_none(self) -> None:
        record = VectorRecord(vector=[1.0], value=b"", provider_model="fake:m", expires_at=None)
        assert record.expires_at is None

    def test_equality_by_value(self) -> None:
        a = VectorRecord(vector=[1.0], value=b"v", provider_model="fake:m", expires_at=None)
        b = VectorRecord(vector=[1.0], value=b"v", provider_model="fake:m", expires_at=None)
        assert a == b


class TestVectorMatch:
    def test_stores_score_and_value(self) -> None:
        match = VectorMatch(score=0.97, value=b"payload")
        assert match.score == 0.97
        assert match.value == b"payload"

    def test_equality_by_value(self) -> None:
        assert VectorMatch(score=0.5, value=b"x") == VectorMatch(score=0.5, value=b"x")


class TestRecordIdConvention:
    """The documented record-id convention must be deterministic."""

    def test_record_id_is_deterministic_for_same_inputs(self) -> None:
        provider_model = "openai:text-embedding-3-small"
        query = "What is 128 times 47?"
        first = hash_value([provider_model, query])
        second = hash_value([provider_model, query])
        assert first == second

    def test_record_id_differs_by_provider_model(self) -> None:
        query = "same query text"
        a = hash_value(["openai:text-embedding-3-small", query])
        b = hash_value(["local:all-MiniLM-L6-v2", query])
        assert a != b

    def test_record_id_differs_by_query_text(self) -> None:
        provider_model = "fake:m"
        assert hash_value([provider_model, "alpha"]) != hash_value([provider_model, "beta"])


class TestVectorIndexAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            VectorIndex()  # type: ignore[abstract]

    def test_declares_expected_abstract_methods(self) -> None:
        assert VectorIndex.__abstractmethods__ == frozenset(
            {"add", "search", "delete_namespace", "flush"}
        )

    def test_abstract_async_methods_are_coroutines(self) -> None:
        for name in ("add", "search", "delete_namespace", "flush"):
            assert inspect.iscoroutinefunction(getattr(VectorIndex, name))

    def test_partial_implementation_still_abstract(self) -> None:
        class Partial(VectorIndex):
            async def add(self, namespace: str, record_id: str, record: VectorRecord) -> None:
                return None

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_full_implementation_can_be_used(self) -> None:
        class InMemory(VectorIndex):
            def __init__(self) -> None:
                self.store: dict[str, dict[str, VectorRecord]] = {}

            async def add(self, namespace: str, record_id: str, record: VectorRecord) -> None:
                self.store.setdefault(namespace, {})[record_id] = record

            async def search(
                self,
                namespace: str,
                query: list[float],
                provider_model: str,
                top_k: int = 1,
            ) -> list[VectorMatch]:
                records = self.store.get(namespace, {})
                return [VectorMatch(score=1.0, value=r.value) for r in records.values()][:top_k]

            async def delete_namespace(self, namespace: str) -> None:
                self.store.pop(namespace, None)

            async def flush(self) -> None:
                self.store.clear()

        index = InMemory()
        record = VectorRecord(
            vector=[1.0, 0.0], value=b"payload", provider_model="fake:m", expires_at=None
        )
        record_id = hash_value(["fake:m", "hello"])
        await index.add("user:alice", record_id, record)

        matches = await index.search("user:alice", [1.0, 0.0], "fake:m", top_k=1)
        assert matches == [VectorMatch(score=1.0, value=b"payload")]

        # Namespace isolation: a different namespace has no records.
        assert await index.search("user:bob", [1.0, 0.0], "fake:m") == []

        await index.delete_namespace("user:alice")
        assert await index.search("user:alice", [1.0, 0.0], "fake:m") == []

    def test_module_documents_namespace_and_record_id(self) -> None:
        doc = vector_base.__doc__ or ""
        assert "user:<user_id>" in doc
        assert "session:<session_id>" in doc
        assert "global" in doc
        assert "record_id = hash_value([provider_model, query_text])" in doc
