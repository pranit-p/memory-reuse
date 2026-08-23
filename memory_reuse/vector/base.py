"""Abstract vector-index interface and data models for the semantic cache.

The semantic cache stores query embeddings and searches them by cosine
similarity.  This lives in a dedicated :class:`VectorIndex` abstraction rather
than on :class:`~memory_reuse.backends.base.AbstractBackend` because vector
search has a fundamentally different shape (search-by-similarity rather than
get-by-key), and keeping it separate leaves the Phase 1 backend interface
untouched.

Namespaces and record ids
--------------------------

Every stored vector belongs to a **namespace** that encodes its cache scope so
that a search never crosses scope boundaries:

* ``"global"`` — the global scope, shared by all callers.
* ``"user:<user_id>"`` — a per-user scope.
* ``"session:<session_id>"`` — a per-session scope.

A search only ever compares against records in the single namespace it is
given, which enforces the same global/user/session isolation as the exact
cache.

Within a namespace each record is keyed by a **record id** derived from the
embedding provider identity and the query text::

    record_id = hash_value([provider_model, query_text])

Because the id is deterministic, re-storing the same ``query_text`` under the
same ``provider_model`` overwrites the existing record rather than creating a
duplicate.

Provider consistency and expiry
-------------------------------

The ``provider_model`` carried by every record namespaces vectors by their
origin.  Implementations validate it on both :meth:`VectorIndex.add` and
:meth:`VectorIndex.search`: an operation whose ``provider_model`` differs from
the records already present in a namespace must raise
:class:`~memory_reuse.exceptions.ProviderMismatchError` rather than compare
vectors of incompatible dimensionality or origin.

Implementations must also filter out records whose ``expires_at`` has passed on
read, so an expired entry is never returned as a match even if a backend's
native expiry mechanism fails.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorRecord:
    """A stored embedding together with its cached result and metadata.

    Attributes:
        vector: The embedding vector produced by the embedding provider.
        value: The cached result, gzip-compressed JSON bytes as produced by
            :func:`memory_reuse._utils.serialize_value`.
        provider_model: The stable ``"provider:model"`` identity of the
            embedding provider that produced ``vector`` (for example
            ``"openai:text-embedding-3-small"``).  Used to namespace vectors by
            origin and reject mismatched comparisons.
        expires_at: The monotonic timestamp (as returned by
            :func:`time.monotonic`) after which this record is considered
            expired and must not be returned as a match, or ``None`` for a
            record that never expires.
    """

    vector: list[float]
    value: bytes
    provider_model: str
    expires_at: float | None


@dataclass
class VectorMatch:
    """A single similarity-search result.

    Attributes:
        score: Cosine similarity normalised to ``[0.0, 1.0]`` (see
            :func:`memory_reuse._utils.cosine_similarity`), where ``1.0`` means
            identical and ``0.0`` means opposite.
        value: The cached result of the matched record, gzip-compressed JSON
            bytes as produced by :func:`memory_reuse._utils.serialize_value`.
    """

    score: float
    value: bytes


class VectorIndex(ABC):
    """Interface that all vector indexes must implement.

    A vector index stores :class:`VectorRecord` entries partitioned by
    namespace and supports nearest-neighbour search by cosine similarity within
    a single namespace.

    Every method is a coroutine so that network-bound implementations (for
    example a Redis-backed index) can be awaited without blocking the event
    loop, while an in-process index simply returns immediately.

    Implementors must:

    * never let a :meth:`search` return records from a namespace other than the
      one requested (scope isolation);
    * validate ``provider_model`` consistency on :meth:`add` and
      :meth:`search`, raising
      :class:`~memory_reuse.exceptions.ProviderMismatchError` on mismatch;
    * filter out records whose ``expires_at`` has passed on read.
    """

    @abstractmethod
    async def add(self, namespace: str, record_id: str, record: VectorRecord) -> None:
        """Store or overwrite a record in a namespace.

        If a record already exists for ``record_id`` in ``namespace`` it is
        overwritten, so re-adding an identical query updates rather than
        duplicates the entry.

        Args:
            namespace: The scope namespace, one of ``"global"``,
                ``"user:<user_id>"``, or ``"session:<session_id>"``.
            record_id: The deterministic record id, typically
                ``hash_value([provider_model, query_text])``.
            record: The :class:`VectorRecord` to store.

        Raises:
            ProviderMismatchError: If ``record.provider_model`` differs from the
                ``provider_model`` of records already present in ``namespace``.
        """

    @abstractmethod
    async def search(
        self,
        namespace: str,
        query: list[float],
        provider_model: str,
        top_k: int = 1,
    ) -> list[VectorMatch]:
        """Search a namespace for the closest records by cosine similarity.

        Only records within ``namespace`` are considered, so a search never
        crosses scope boundaries.  Expired records are filtered out before
        scoring.

        Args:
            namespace: The scope namespace to search within.
            query: The query embedding vector.
            provider_model: The ``"provider:model"`` identity of the embedding
                that produced ``query``.  Must match the records stored in
                ``namespace``.
            top_k: The maximum number of matches to return, highest score
                first.

        Returns:
            Up to ``top_k`` :class:`VectorMatch` results ordered by descending
            similarity score.  An empty list when the namespace holds no
            (non-expired) records.

        Raises:
            ProviderMismatchError: If ``provider_model`` differs from the
                ``provider_model`` of records already present in ``namespace``.
        """

    @abstractmethod
    async def delete_namespace(self, namespace: str) -> None:
        """Remove every record stored under a namespace.

        Args:
            namespace: The scope namespace to clear. A no-op if the namespace
                holds no records.
        """

    @abstractmethod
    async def flush(self) -> None:
        """Delete all records managed by this index across every namespace.

        Use with caution in production — this removes every stored vector.
        """

    def expiry_for_ttl(self, ttl: int | None) -> float | None:
        """Compute the ``expires_at`` value for a record with a given TTL.

        Different index implementations interpret :attr:`VectorRecord.expires_at`
        against different clocks — the in-memory index uses
        :func:`time.monotonic` while the Redis index uses wall-clock
        :func:`time.time`.  Callers that only know a TTL (such as the semantic
        cache) delegate to this method so the produced ``expires_at`` matches
        whatever clock the concrete index reads on expiry.

        The default implementation uses :func:`time.monotonic`, matching the
        in-memory index; implementations that expire against a different clock
        must override it.

        Args:
            ttl: Time-to-live in seconds, or ``None`` for a record that never
                expires.

        Returns:
            The ``expires_at`` timestamp on this index's clock, or ``None`` when
            ``ttl`` is ``None``.
        """
        if ttl is None:
            return None
        return time.monotonic() + ttl
