"""Vector-index implementations for the semantic cache.

This package defines the abstract :class:`VectorIndex` interface together with
the :class:`VectorRecord` and :class:`VectorMatch` data models used to store
embeddings and return similarity-search results.

A vector index partitions stored embeddings by *namespace* — ``"global"``,
``"user:<user_id>"``, or ``"session:<session_id>"`` — so that similarity search
respects the same scope isolation as the exact cache.  Within a namespace each
record is keyed by ``record_id = hash_value([provider_model, query_text])``.

Concrete implementations (in-memory and Redis-backed) live in sibling modules
and are added to this package's exports as they are introduced.
"""

from memory_reuse.vector.base import VectorIndex, VectorMatch, VectorRecord
from memory_reuse.vector.memory import InMemoryVectorIndex
from memory_reuse.vector.redis import RedisVectorIndex

__all__ = [
    "InMemoryVectorIndex",
    "RedisVectorIndex",
    "VectorIndex",
    "VectorMatch",
    "VectorRecord",
]
