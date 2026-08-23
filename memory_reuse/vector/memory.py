"""In-memory vector index with brute-force cosine search.

:class:`InMemoryVectorIndex` is the zero-dependency vector store used for local
development and testing.  It keeps a separate :class:`~collections.OrderedDict`
per namespace and performs a brute-force cosine-similarity scan on
:meth:`search`, returning the top-``k`` records by score.

It upholds every invariant documented on
:class:`~memory_reuse.vector.base.VectorIndex`:

* **Namespace isolation** — a search only ever scans the single namespace it is
  given, so a ``user:A`` lookup can never surface a ``user:B`` or ``global``
  record (Requirements 5.1, 5.2, 5.4).
* **Provider consistency** — every :meth:`add` and :meth:`search` validates the
  ``provider_model`` against the records already present in the namespace and
  raises :class:`~memory_reuse.exceptions.ProviderMismatchError` on a mismatch
  (Requirements 3.7, 3.8).
* **Expiry safety** — records whose ``expires_at`` has passed are filtered out
  on read (and dropped) so an expired entry is never returned as a match, even
  if a backend-native expiry mechanism would have failed (Requirements 6.5,
  6.7).
* **LRU eviction** — each namespace holds at most
  ``max_vectors_per_namespace`` records; adding beyond that evicts the
  least-recently-used entry (Requirement 11.4).

Deterministic ``record_id`` values (``hash_value([provider_model,
query_text])``) mean re-adding an identical query overwrites the existing record
rather than duplicating it (Requirement 1.4 / Property 8).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

from memory_reuse._utils import cosine_similarity
from memory_reuse.exceptions import ProviderMismatchError
from memory_reuse.vector.base import VectorIndex, VectorMatch, VectorRecord

logger = logging.getLogger(__name__)

_DEFAULT_MAX_VECTORS = 10_000


class InMemoryVectorIndex(VectorIndex):
    """Fully in-memory vector index — no external dependencies required.

    Features:

    * **Brute-force cosine search** — every non-expired record in a namespace is
      scored with :func:`memory_reuse._utils.cosine_similarity` and the top
      ``top_k`` are returned, highest score first.
    * **Namespace isolation** — records are partitioned by namespace; a search
      never crosses namespace boundaries.
    * **Provider consistency** — ``provider_model`` is validated on ``add`` and
      ``search``; a mismatch raises
      :class:`~memory_reuse.exceptions.ProviderMismatchError`.
    * **Expiry safety** — expired records are filtered (and removed) on read.
    * **LRU eviction** — when a namespace reaches ``max_vectors_per_namespace``
      the least-recently-used record is dropped to make room.
    * **Async-safe** — an :class:`asyncio.Lock` serialises all mutations.

    Args:
        max_vectors_per_namespace: Maximum number of records to hold per
            namespace before LRU eviction applies. Defaults to 10 000.

    Example::

        index = InMemoryVectorIndex(max_vectors_per_namespace=500)
        await index.add("user:alice", record_id, record)
        matches = await index.search("user:alice", query_vec, "fake:m", top_k=1)
    """

    def __init__(self, max_vectors_per_namespace: int = _DEFAULT_MAX_VECTORS) -> None:
        if max_vectors_per_namespace <= 0:
            raise ValueError(
                f"max_vectors_per_namespace must be positive, got {max_vectors_per_namespace}"
            )
        self._max_vectors = max_vectors_per_namespace
        # namespace -> (record_id -> VectorRecord), most-recently-used last.
        self._store: dict[str, OrderedDict[str, VectorRecord]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # VectorIndex implementation
    # ------------------------------------------------------------------

    async def add(self, namespace: str, record_id: str, record: VectorRecord) -> None:
        """Store or overwrite a record in a namespace.

        Args:
            namespace: The scope namespace, one of ``"global"``,
                ``"user:<user_id>"``, or ``"session:<session_id>"``.
            record_id: The deterministic record id, typically
                ``hash_value([provider_model, query_text])``.
            record: The :class:`VectorRecord` to store.

        Raises:
            ProviderMismatchError: If ``record.provider_model`` differs from the
                ``provider_model`` of the live (non-expired) records already
                present in ``namespace``.
        """
        async with self._lock:
            bucket = self._store.get(namespace)
            if bucket is None:
                bucket = OrderedDict()
                self._store[namespace] = bucket

            self._check_provider_consistency(
                bucket, record.provider_model, skip_record_id=record_id
            )

            if record_id in bucket:
                # Overwrite in place and mark most-recently-used.
                bucket[record_id] = record
                bucket.move_to_end(record_id)
            else:
                if len(bucket) >= self._max_vectors:
                    evicted_id, _ = bucket.popitem(last=False)
                    logger.debug("InMemoryVectorIndex: LRU eviction in namespace %r", namespace)
                    del evicted_id
                bucket[record_id] = record

    async def search(
        self,
        namespace: str,
        query: list[float],
        provider_model: str,
        top_k: int = 1,
    ) -> list[VectorMatch]:
        """Search a namespace for the closest records by cosine similarity.

        Only records in ``namespace`` are considered, so a search never crosses
        scope boundaries.  Expired records are filtered out (and removed) before
        scoring.

        Args:
            namespace: The scope namespace to search within.
            query: The query embedding vector.
            provider_model: The ``"provider:model"`` identity of the embedding
                that produced ``query``.
            top_k: Maximum number of matches to return, highest score first.

        Returns:
            Up to ``top_k`` :class:`VectorMatch` results ordered by descending
            similarity score, or an empty list when the namespace holds no
            live records.

        Raises:
            ProviderMismatchError: If ``provider_model`` differs from the
                ``provider_model`` of the live records present in ``namespace``.
        """
        if top_k <= 0:
            return []

        async with self._lock:
            bucket = self._store.get(namespace)
            if not bucket:
                return []

            self._purge_expired(bucket)
            if not bucket:
                return []

            self._check_provider_consistency(bucket, provider_model)

            # Snapshot the ids first: touching LRU order below mutates the
            # OrderedDict, which would break a live iteration over it.
            record_ids = list(bucket)
            scored: list[VectorMatch] = []
            for record_id in record_ids:
                record = bucket[record_id]
                score = cosine_similarity(query, record.vector)
                scored.append(VectorMatch(score=score, value=record.value))
                # Touch matched records so repeated hits stay warm under LRU.
                bucket.move_to_end(record_id)

            scored.sort(key=lambda match: match.score, reverse=True)
            return scored[:top_k]

    async def delete_namespace(self, namespace: str) -> None:
        """Remove every record stored under a namespace.

        Args:
            namespace: The scope namespace to clear. A no-op if the namespace
                holds no records.
        """
        async with self._lock:
            self._store.pop(namespace, None)

    async def flush(self) -> None:
        """Delete all records across every namespace."""
        async with self._lock:
            self._store.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_expired(record: VectorRecord, now: float) -> bool:
        """Return ``True`` if ``record`` has passed its ``expires_at``.

        Args:
            record: The record to inspect.
            now: The current monotonic time, as returned by
                :func:`time.monotonic`.

        Returns:
            ``True`` when the record has an expiry that is at or before ``now``.
        """
        if record.expires_at is None:
            return False
        return now >= record.expires_at

    def _purge_expired(self, bucket: OrderedDict[str, VectorRecord]) -> None:
        """Drop every expired record from ``bucket`` in place.

        Args:
            bucket: The namespace's record map to filter.
        """
        now = time.monotonic()
        expired = [rid for rid, record in bucket.items() if self._is_expired(record, now)]
        for rid in expired:
            del bucket[rid]

    def _check_provider_consistency(
        self,
        bucket: OrderedDict[str, VectorRecord],
        provider_model: str,
        skip_record_id: str | None = None,
    ) -> None:
        """Ensure ``provider_model`` matches the live records in ``bucket``.

        The namespace's provider identity is taken from its live (non-expired)
        records.  If any such record carries a different ``provider_model`` the
        operation is refused rather than comparing incompatible vectors.

        Args:
            bucket: The namespace's record map.
            provider_model: The provider identity of the incoming operation.
            skip_record_id: A record id to ignore when scanning — used by
                ``add`` so overwriting an existing record with the same id does
                not compare against itself.

        Raises:
            ProviderMismatchError: If a live record uses a different
                ``provider_model``.
        """
        now = time.monotonic()
        for record_id, record in bucket.items():
            if record_id == skip_record_id:
                continue
            if self._is_expired(record, now):
                continue
            if record.provider_model != provider_model:
                raise ProviderMismatchError(
                    "Embedding provider/model mismatch for this namespace: "
                    f"existing records use {record.provider_model!r} but the "
                    f"operation supplied {provider_model!r}. Incompatible "
                    "vectors are never compared."
                )
            # First live record establishes the namespace's identity.
            return

    def namespace_size(self, namespace: str) -> int:
        """Return the number of records currently held in ``namespace``.

        Includes records that are expired but not yet purged. Intended for
        tests and introspection.

        Args:
            namespace: The namespace to measure.

        Returns:
            The record count, or ``0`` if the namespace is unknown.
        """
        bucket = self._store.get(namespace)
        return len(bucket) if bucket is not None else 0
