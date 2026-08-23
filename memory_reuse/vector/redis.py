"""Redis-backed vector index for the semantic cache.

:class:`RedisVectorIndex` persists query embeddings as Redis hashes so that
semantic matches survive process restarts and are shared across workers.  Each
record is stored under a deterministic key::

    memreuse:vec:<namespace>:<record_id>

with hash fields ``vector`` (packed ``float32`` bytes), ``value`` (the
gzip-compressed cached result), ``provider_model`` (the ``provider:model``
identity), ``namespace`` (a tag used for scope-scoped search), and
``expires_at`` (a wall-clock expiry used purely as a read-time safeguard).

Search strategy
---------------

The index adapts to the Redis deployment it is pointed at:

* **Redis Stack (search module present).** When the ``FT.*`` search commands are
  available a per-index :func:`FT.CREATE` schema is defined over the record
  hashes with the namespace as a tag field, and :func:`FT.SEARCH` runs a KNN
  query scoped to a single namespace.
* **Plain Redis (no search module).** The index falls back to scanning the
  namespace's keys, loading the candidate vectors, and computing cosine
  similarity in process.  To avoid unbounded work this fallback refuses — with a
  clear :class:`~memory_reuse.exceptions.BackendConnectionError` — to search a
  namespace holding more than ``max_scan_candidates`` records.

Both paths uphold the :class:`~memory_reuse.vector.base.VectorIndex` contract:

* **Namespace isolation** — a search only ever considers records in the single
  namespace it is given (Requirements 5.1, 5.2, 5.4).
* **Provider consistency** — ``provider_model`` is validated on ``add`` and
  ``search`` against the records already present in the namespace; a mismatch
  raises :class:`~memory_reuse.exceptions.ProviderMismatchError`
  (Requirements 3.7, 3.8).
* **Expiry safety** — a Redis TTL is set on write, and expired records are also
  filtered on read so a stale entry is never returned even if the native expiry
  lagged (Requirements 6.5, 6.7).
"""

from __future__ import annotations

import logging
import struct
import time
from typing import TYPE_CHECKING, Any

from memory_reuse._utils import cosine_similarity
from memory_reuse.exceptions import (
    BackendConnectionError,
    BackendNotAvailableError,
    ProviderMismatchError,
)
from memory_reuse.vector.base import VectorIndex, VectorMatch, VectorRecord

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "memreuse:vec"
_MAX_CONNECTIONS = 20
_DEFAULT_MAX_SCAN_CANDIDATES = 10_000
_VECTOR_DTYPE = "f"  # 32-bit float, matching the FT VectorField FLOAT32 type.


def _pack_vector(vector: list[float]) -> bytes:
    """Pack a float vector into little-endian ``float32`` bytes.

    Args:
        vector: The embedding vector.

    Returns:
        The packed byte representation used both for hash storage and as the
        KNN query blob.
    """
    return struct.pack(f"<{len(vector)}{_VECTOR_DTYPE}", *vector)


def _unpack_vector(data: bytes) -> list[float]:
    """Unpack ``float32`` bytes produced by :func:`_pack_vector`.

    Args:
        data: Packed vector bytes.

    Returns:
        The embedding vector as a list of floats.
    """
    count = len(data) // struct.calcsize(_VECTOR_DTYPE)
    return list(struct.unpack(f"<{count}{_VECTOR_DTYPE}", data))


class RedisVectorIndex(VectorIndex):
    """Persistent vector index backed by Redis.

    Requires the optional ``redis`` extra::

        pip install memory-reuse[redis]

    The connection is established lazily on the first operation and, on Redis
    Stack, the search index is created on demand.  Connection errors are
    converted to :class:`~memory_reuse.exceptions.BackendConnectionError` so
    callers need not handle redis-specific exceptions.

    Args:
        url: Redis connection URL (e.g. ``redis://localhost:6379/0``). Prefer
            reading this from the ``MEMORY_REUSE_REDIS_URL`` environment
            variable rather than hardcoding it.
        index_name: Name of the Redis Stack search index created over the
            record hashes. Defaults to ``"memreuse_vec_idx"``.
        max_scan_candidates: Maximum number of records a namespace may hold for
            the in-process fallback (used only when the search module is
            absent). Searching a larger namespace raises
            :class:`~memory_reuse.exceptions.BackendConnectionError`. Defaults
            to 10 000.
        max_connections: Maximum size of the underlying connection pool.
            Defaults to 20.

    Example::

        import os
        index = RedisVectorIndex(url=os.environ["MEMORY_REUSE_REDIS_URL"])
        await index.add("user:alice", record_id, record)
        matches = await index.search("user:alice", query_vec, "openai:m", top_k=1)
    """

    def __init__(
        self,
        url: str,
        *,
        index_name: str = "memreuse_vec_idx",
        max_scan_candidates: int = _DEFAULT_MAX_SCAN_CANDIDATES,
        max_connections: int = _MAX_CONNECTIONS,
    ) -> None:
        if max_scan_candidates <= 0:
            raise ValueError(f"max_scan_candidates must be positive, got {max_scan_candidates}")
        self._url = url
        self._index_name = index_name
        self._max_scan_candidates = max_scan_candidates
        self._max_connections = max_connections
        self._client: aioredis.Redis | None = None  # type: ignore[type-arg]
        # None until probed; True when FT.* is available, False for plain Redis.
        self._search_available: bool | None = None
        self._index_ready = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _get_client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        """Return the Redis client, creating it on first call.

        Returns:
            An initialised ``redis.asyncio.Redis`` client.

        Raises:
            BackendNotAvailableError: If the ``redis`` package is not installed.
            BackendConnectionError: If the connection attempt fails.
        """
        if self._client is not None:
            return self._client

        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Redis vector index requires 'redis' package. "
                "Install it with: pip install memory-reuse[redis]"
            ) from exc

        try:
            pool = aioredis.ConnectionPool.from_url(
                self._url,
                max_connections=self._max_connections,
                decode_responses=False,
            )
            self._client = aioredis.Redis(connection_pool=pool)
            await self._client.ping()
        except Exception as exc:
            self._client = None
            # Deliberately not logging self._url to avoid leaking credentials.
            logger.error("RedisVectorIndex: failed to connect to Redis server")
            raise BackendConnectionError(
                "Could not connect to Redis. Check your connection URL and network."
            ) from exc

        return self._client

    async def _search_module_available(self, client: aioredis.Redis) -> bool:  # type: ignore[type-arg]
        """Detect whether the Redis Stack search module is loaded.

        The result is cached for the lifetime of the index.

        Args:
            client: The connected Redis client.

        Returns:
            ``True`` when the ``FT.*`` search commands are available.
        """
        if self._search_available is not None:
            return self._search_available

        available = False
        try:
            modules = await client.execute_command("MODULE", "LIST")
            names = {_module_name(entry) for entry in modules}
            names.discard("")
            available = "search" in names or "searchlight" in names
        except Exception:
            # MODULE LIST may be restricted; assume no search module.
            available = False

        self._search_available = available
        logger.debug("RedisVectorIndex: search module available=%s", available)
        return available

    async def _ensure_index(self, client: aioredis.Redis, dimension: int) -> bool:  # type: ignore[type-arg]
        """Create the Redis Stack search index if it does not yet exist.

        If index creation fails for a reason other than the index already
        existing (for example RediSearch refusing to index a non-zero database),
        native search is disabled for this instance so callers transparently
        fall back to in-process cosine scoring.

        Args:
            client: The connected Redis client.
            dimension: Embedding dimensionality for the vector field schema.

        Returns:
            ``True`` when a usable search index exists, ``False`` when native
            search is unavailable and the caller should use the fallback.
        """
        if self._index_ready:
            return True

        from redis.commands.search.field import TagField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        schema = [
            TagField("namespace"),
            TagField("provider_model"),
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dimension,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        ]
        definition = IndexDefinition(prefix=[f"{_KEY_PREFIX}:"], index_type=IndexType.HASH)
        try:
            await client.ft(self._index_name).create_index(schema, definition=definition)
        except Exception as exc:  # noqa: BLE001
            if "Index already exists" in str(exc):
                self._index_ready = True
                return True
            # Native search is not usable here (e.g. "Cannot create index on
            # db != 0"). Disable it and fall back to in-process scoring.
            logger.debug("RedisVectorIndex: native search unavailable: %s", exc)
            self._search_available = False
            return False
        self._index_ready = True
        return True

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_key(namespace: str, record_id: str) -> str:
        """Return the Redis hash key for a record."""
        return f"{_KEY_PREFIX}:{namespace}:{record_id}"

    @staticmethod
    def _namespace_pattern(namespace: str) -> str:
        """Return the SCAN match pattern for every record in a namespace."""
        return f"{_KEY_PREFIX}:{namespace}:*"

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
                ``provider_model`` of the live records already present in
                ``namespace``.
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        key = self._record_key(namespace, record_id)

        await self._check_provider_consistency(
            client, namespace, record.provider_model, skip_key=key
        )

        ttl_seconds = self._ttl_seconds(record.expires_at)
        # A record already expired need not be written at all.
        if record.expires_at is not None and ttl_seconds is not None and ttl_seconds <= 0:
            return

        mapping: dict[str, Any] = {
            "vector": _pack_vector(record.vector),
            "value": record.value,
            "provider_model": record.provider_model.encode("utf-8"),
            "namespace": namespace.encode("utf-8"),
            "expires_at": (
                b"" if record.expires_at is None else str(record.expires_at).encode("utf-8")
            ),
        }

        try:
            if await self._search_module_available(client):
                await self._ensure_index(client, len(record.vector))
            await client.hset(key, mapping=mapping)  # type: ignore[arg-type]
            if ttl_seconds is not None:
                await client.expire(key, ttl_seconds)
        except ProviderMismatchError:
            raise
        except Exception as exc:
            logger.error("RedisVectorIndex: ADD failed in namespace %r", namespace)
            raise BackendConnectionError("Redis vector ADD operation failed") from exc

    async def search(
        self,
        namespace: str,
        query: list[float],
        provider_model: str,
        top_k: int = 1,
    ) -> list[VectorMatch]:
        """Search a namespace for the closest records by cosine similarity.

        Args:
            namespace: The scope namespace to search within.
            query: The query embedding vector.
            provider_model: The ``"provider:model"`` identity of the embedding
                that produced ``query``.
            top_k: Maximum number of matches to return, highest score first.

        Returns:
            Up to ``top_k`` :class:`VectorMatch` results ordered by descending
            similarity score, or an empty list when the namespace holds no live
            records.

        Raises:
            ProviderMismatchError: If ``provider_model`` differs from the
                ``provider_model`` of the live records present in ``namespace``.
            BackendConnectionError: On Redis connectivity failure, or when the
                in-process fallback would exceed ``max_scan_candidates``.
        """
        if top_k <= 0:
            return []

        client = await self._get_client()

        await self._check_provider_consistency(client, namespace, provider_model)

        try:
            if await self._search_module_available(client):
                return await self._search_native(client, namespace, query, top_k)
            return await self._search_fallback(client, namespace, query, top_k)
        except (ProviderMismatchError, BackendConnectionError):
            raise
        except Exception as exc:
            logger.error("RedisVectorIndex: SEARCH failed in namespace %r", namespace)
            raise BackendConnectionError("Redis vector SEARCH operation failed") from exc

    async def delete_namespace(self, namespace: str) -> None:
        """Remove every record stored under a namespace.

        Args:
            namespace: The scope namespace to clear. A no-op if the namespace
                holds no records.

        Raises:
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        try:
            keys = [key async for key in client.scan_iter(match=self._namespace_pattern(namespace))]
            if keys:
                await client.delete(*keys)
        except Exception as exc:
            logger.error("RedisVectorIndex: DELETE namespace %r failed", namespace)
            raise BackendConnectionError("Redis vector DELETE operation failed") from exc

    async def flush(self) -> None:
        """Delete every record managed by this index across all namespaces.

        Only keys under the ``memreuse:vec:`` prefix are removed; unrelated keys
        in the same database are left untouched.

        Raises:
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        try:
            keys = [key async for key in client.scan_iter(match=f"{_KEY_PREFIX}:*")]
            if keys:
                await client.delete(*keys)
        except Exception as exc:
            logger.error("RedisVectorIndex: FLUSH failed")
            raise BackendConnectionError("Redis vector FLUSH operation failed") from exc

    async def close(self) -> None:
        """Close the connection pool gracefully.

        Call this during application shutdown to release Redis connections.
        """
        import contextlib

        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    async def _search_native(
        self,
        client: aioredis.Redis,  # type: ignore[type-arg]
        namespace: str,
        query: list[float],
        top_k: int,
    ) -> list[VectorMatch]:
        """Run a KNN search using the Redis Stack search module.

        Falls back to in-process cosine scoring when the search index cannot be
        created (for example on a non-zero Redis database).
        """
        from redis.commands.search.query import Query

        if not await self._ensure_index(client, len(query)):
            return await self._search_fallback(client, namespace, query, top_k)

        escaped = _escape_tag(namespace)
        # KNN over records whose namespace tag matches, ordered by vector
        # distance. Only text-safe fields (score, expires_at) are returned via
        # FT.SEARCH — the gzip-compressed ``value`` is fetched separately with
        # HGET because the search reply would UTF-8-mangle raw binary bytes.
        query_str = f"(@namespace:{{{escaped}}})=>[KNN {top_k} @vector $vec AS score]"
        knn = (
            Query(query_str)
            .sort_by("score")
            .return_fields("score", "expires_at")
            .dialect(2)
            .paging(0, top_k)
        )
        params: dict[str, str | int | float | bytes] = {"vec": _pack_vector(query)}

        result = await client.ft(self._index_name).search(knn, query_params=params)

        now = time.time()
        matches: list[VectorMatch] = []
        for doc in result.docs:
            expires_at = _parse_expires_at(getattr(doc, "expires_at", None))
            if expires_at is not None and now >= expires_at:
                continue
            # Redis reports COSINE *distance* in [0, 2]; cosine = 1 - distance.
            # Re-normalise to [0, 1] via (cos + 1) / 2 for consistency with the
            # rest of the library.
            distance = float(getattr(doc, "score", 0.0))
            cosine = max(-1.0, min(1.0, 1.0 - distance))
            score = (cosine + 1.0) / 2.0
            raw_value = await client.hget(doc.id, "value")  # type: ignore[misc]
            if raw_value is None:
                continue
            matches.append(VectorMatch(score=score, value=_to_bytes(raw_value)))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    async def _search_fallback(
        self,
        client: aioredis.Redis,  # type: ignore[type-arg]
        namespace: str,
        query: list[float],
        top_k: int,
    ) -> list[VectorMatch]:
        """Scan a namespace and compute cosine similarity in process.

        Refuses namespaces holding more than ``max_scan_candidates`` records.
        """
        keys = [key async for key in client.scan_iter(match=self._namespace_pattern(namespace))]
        if not keys:
            return []
        if len(keys) > self._max_scan_candidates:
            raise BackendConnectionError(
                "Redis vector search fallback refused: namespace "
                f"{namespace!r} holds {len(keys)} records, exceeding the "
                f"in-process candidate limit of {self._max_scan_candidates}. "
                "Enable the Redis Stack search module for native KNN, or raise "
                "max_scan_candidates if in-process scoring is acceptable."
            )

        now = time.time()
        scored: list[VectorMatch] = []
        for key in keys:
            data = await client.hgetall(key)  # type: ignore[misc]
            if not data:
                continue
            record = _decode_hash(data)
            if record is None:
                continue
            if record.expires_at is not None and now >= record.expires_at:
                continue
            score = cosine_similarity(query, record.vector)
            scored.append(VectorMatch(score=score, value=record.value))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def expiry_for_ttl(self, ttl: int | None) -> float | None:
        """Compute a wall-clock ``expires_at`` for a record with a given TTL.

        The Redis index stores and reads ``expires_at`` as a wall-clock
        timestamp (:func:`time.time`), so this overrides the monotonic default
        on :class:`~memory_reuse.vector.base.VectorIndex`.

        Args:
            ttl: Time-to-live in seconds, or ``None`` for no expiry.

        Returns:
            The wall-clock ``expires_at`` timestamp, or ``None`` when ``ttl`` is
            ``None``.
        """
        if ttl is None:
            return None
        return time.time() + ttl

    @staticmethod
    def _ttl_seconds(expires_at: float | None) -> int | None:
        """Convert a wall-clock ``expires_at`` into a Redis TTL in seconds.

        Args:
            expires_at: The wall-clock expiry timestamp, or ``None``.

        Returns:
            The remaining whole seconds until expiry (at least 1 for a live
            record), or ``None`` when the record never expires.
        """
        if expires_at is None:
            return None
        remaining = expires_at - time.time()
        if remaining <= 0:
            return 0
        return max(1, int(remaining))

    async def _check_provider_consistency(
        self,
        client: aioredis.Redis,  # type: ignore[type-arg]
        namespace: str,
        provider_model: str,
        skip_key: str | None = None,
    ) -> None:
        """Ensure ``provider_model`` matches the live records in ``namespace``.

        The namespace's provider identity is taken from its first live
        (non-expired) record. A different identity refuses the operation rather
        than comparing incompatible vectors.

        Args:
            client: The connected Redis client.
            namespace: The namespace to inspect.
            provider_model: The provider identity of the incoming operation.
            skip_key: A record key to ignore — used by ``add`` so overwriting an
                existing record does not compare against itself.

        Raises:
            ProviderMismatchError: If a live record uses a different
                ``provider_model``.
            BackendConnectionError: On Redis connectivity failure.
        """
        now = time.time()
        try:
            async for key in client.scan_iter(match=self._namespace_pattern(namespace)):
                if skip_key is not None and _to_str(key) == skip_key:
                    continue
                stored_pm = await client.hget(key, "provider_model")  # type: ignore[misc]
                if stored_pm is None:
                    continue
                expires_raw = await client.hget(key, "expires_at")  # type: ignore[misc]
                expires_at = _parse_expires_at(expires_raw)
                if expires_at is not None and now >= expires_at:
                    continue
                existing = _to_str(stored_pm)
                if existing != provider_model:
                    raise ProviderMismatchError(
                        "Embedding provider/model mismatch for this namespace: "
                        f"existing records use {existing!r} but the operation "
                        f"supplied {provider_model!r}. Incompatible vectors are "
                        "never compared."
                    )
                # First live record establishes the namespace identity.
                return
        except ProviderMismatchError:
            raise
        except Exception as exc:
            logger.error("RedisVectorIndex: provider check failed in namespace %r", namespace)
            raise BackendConnectionError("Redis vector provider check failed") from exc


def _module_name(entry: Any) -> str:
    """Extract the lowercased module name from a ``MODULE LIST`` entry.

    ``MODULE LIST`` may decode as a mapping (``{b"name": b"search", ...}``) or
    as a flat sequence (``[b"name", b"search", b"ver", 20000, ...]``) depending
    on the redis-py version and RESP protocol in use. Both shapes are handled.

    Args:
        entry: A single element of the ``MODULE LIST`` reply.

    Returns:
        The lowercased module name, or an empty string if none can be read.
    """
    if isinstance(entry, dict):
        for key, value in entry.items():
            if _to_str(key).lower() == "name":
                return _to_str(value).lower()
        return ""
    if isinstance(entry, (list, tuple)):
        # Look for the token following a "name" marker; fall back to index 1.
        for i in range(len(entry) - 1):
            if _to_str(entry[i]).lower() == "name":
                return _to_str(entry[i + 1]).lower()
        if len(entry) >= 2:
            return _to_str(entry[1]).lower()
    return ""


def _to_bytes(value: Any) -> bytes:
    """Coerce a Redis hash field value to bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _to_str(value: Any) -> str:
    """Coerce a Redis field value (bytes or str) to str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_expires_at(value: Any) -> float | None:
    """Parse a stored ``expires_at`` field into a float, or ``None``."""
    if value is None:
        return None
    text = _to_str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _escape_tag(value: str) -> str:
    """Escape characters that are special inside a search TAG filter."""
    special = ".,<>{}[]\"':;!@#$%^&*()-+=~| "
    out = []
    for ch in value:
        if ch in special:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _decode_hash(data: dict[Any, Any]) -> VectorRecord | None:
    """Rebuild a :class:`VectorRecord` from a raw Redis hash mapping."""
    decoded: dict[str, Any] = {_to_str(k): v for k, v in data.items()}
    packed = decoded.get("vector")
    value = decoded.get("value")
    provider_model = decoded.get("provider_model")
    if packed is None or value is None or provider_model is None:
        return None
    return VectorRecord(
        vector=_unpack_vector(_to_bytes(packed)),
        value=_to_bytes(value),
        provider_model=_to_str(provider_model),
        expires_at=_parse_expires_at(decoded.get("expires_at")),
    )
