"""AWS AgentCore cache backend (shared across microVMs).

This backend implements :class:`~memory_reuse.backends.base.AbstractBackend`
on top of the managed AWS AgentCore store, so a value cached in one AgentCore
microVM is served to requests handled by another.

The optional ``boto3`` / AgentCore dependency is imported lazily behind
:func:`_require_agentcore`; the module itself imports without ``boto3``
installed, mirroring the ``redis`` / ``semantic`` / ``langgraph`` pattern.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

from memory_reuse.backends.base import AbstractBackend
from memory_reuse.exceptions import BackendConnectionError, BackendNotAvailableError

logger = logging.getLogger(__name__)

# Maximum key length in bytes (Req 8.10).
_MAX_KEY_BYTES = 512

# Item-record field names on the transport.
_FIELD_VALUE = "value"
_FIELD_EXPIRES_AT = "expires_at"


def _require_agentcore() -> Any:
    """Import and return the ``boto3`` module, or raise a named error.

    Returns:
        The imported ``boto3`` module.

    Raises:
        BackendNotAvailableError: If ``boto3`` is not installed. The message
            names the packaging extra to install rather than surfacing a
            low-level ``ImportError`` traceback.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise BackendNotAvailableError(
            "AgentCore backend requires 'boto3'. "
            'Install it with: pip install "memory-reuse[agentcore]"'
        ) from exc
    return boto3


@dataclass
class AgentCoreSettings:
    """Resolved connection settings for the AgentCore backend.

    Attributes:
        region: The AWS region hosting the AgentCore store.
        memory_id: The AgentCore memory / store resource identifier.
        key_prefix: Optional namespace prepended to every managed item id so
            that ``flush`` only removes entries this backend manages. Defaults
            to ``"memory-reuse:"``.
    """

    region: str | None = None
    memory_id: str | None = None
    key_prefix: str = "memory-reuse:"


class AgentCoreBackend(AbstractBackend):
    """:class:`AbstractBackend` backed by AWS AgentCore Memory.

    Because storage is the managed AgentCore service, a value ``set`` in one
    execution environment is visible to a ``get`` in another (Req 8.2).

    Values are stored as raw bytes, base64-encoded on write and decoded on
    read, so any 0..1_048_576-byte payload round-trips unchanged (Req 8.5).
    A ``ttl`` greater than zero is recorded as an absolute expiry and filtered
    lazily on read; a missing or zero ``ttl`` persists until the entry is
    explicitly deleted or flushed (Req 8.3).

    Args:
        settings: Resolved connection settings (region, memory id, key prefix).
        client: An already-constructed AgentCore client. When ``None`` a real
            client is created lazily via :func:`_require_agentcore`. Tests inject
            an in-process fake client here.

    Example::

        backend = AgentCoreBackend(
            AgentCoreSettings(region="us-east-1", memory_id="mem-123")
        )
        await backend.set("key", b"value", ttl=300)
        data = await backend.get("key")
    """

    def __init__(
        self,
        settings: AgentCoreSettings,
        *,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._prefix = settings.key_prefix
        if client is not None:
            self._client: Any = client
        else:
            boto3 = _require_agentcore()
            self._client = boto3.client("bedrock-agentcore", region_name=settings.region)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _item_id(self, key: str) -> str:
        """Return the namespaced transport item id for a cache ``key``."""
        return f"{self._prefix}{key}"

    @staticmethod
    def _validate_key(key: str) -> None:
        """Reject empty or oversized keys before any network call (Req 8.10).

        Raises:
            ValueError: If ``key`` is empty or longer than 512 bytes.
        """
        if not key:
            raise ValueError("AgentCore key must not be empty")
        if len(key.encode("utf-8")) > _MAX_KEY_BYTES:
            raise ValueError(f"AgentCore key must be at most {_MAX_KEY_BYTES} bytes")

    @staticmethod
    def _is_expired(item: dict[str, Any]) -> bool:
        """Return ``True`` when ``item`` carries an elapsed absolute expiry."""
        expires_at = item.get(_FIELD_EXPIRES_AT)
        if expires_at is None:
            return False
        return time.time() >= float(expires_at)

    # ------------------------------------------------------------------
    # AbstractBackend implementation
    # ------------------------------------------------------------------

    async def get(self, key: str) -> bytes | None:
        """Return stored bytes for ``key`` or ``None`` on miss/expiry.

        Args:
            key: Cache key to look up.

        Returns:
            The stored bytes, or ``None`` if absent or expired.

        Raises:
            BackendConnectionError: If the AgentCore service is unreachable.
        """
        item_id = self._item_id(key)
        try:
            item = self._client.get_item(item_id)
            if item is None:
                return None
            if self._is_expired(item):
                # Opportunistically drop the elapsed entry (lazy expiry).
                self._client.delete_item(item_id)
                return None
            encoded = item[_FIELD_VALUE]
            return base64.b64decode(encoded)
        except (BackendConnectionError, ValueError):
            raise
        except Exception as exc:
            logger.error("AgentCoreBackend: GET failed")
            raise BackendConnectionError("AgentCore GET operation failed") from exc

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL.

        Args:
            key: Cache key.
            value: Raw bytes to store.
            ttl: Time-to-live in seconds. ``None`` or ``0`` means no expiry.

        Raises:
            ValueError: If ``key`` is empty or larger than 512 bytes.
            BackendConnectionError: If the AgentCore service is unreachable.
        """
        self._validate_key(key)

        expires_at: float | None = None
        if ttl is not None and ttl > 0:
            expires_at = time.time() + ttl

        item: dict[str, Any] = {
            _FIELD_VALUE: base64.b64encode(value).decode("ascii"),
            _FIELD_EXPIRES_AT: expires_at,
        }
        try:
            self._client.put_item(self._item_id(key), item)
        except Exception as exc:
            logger.error("AgentCoreBackend: SET failed")
            raise BackendConnectionError("AgentCore SET operation failed") from exc

    async def delete(self, key: str) -> None:
        """Remove the entry for ``key``; a no-op when the key is absent.

        Args:
            key: Cache key to remove.

        Raises:
            BackendConnectionError: If the AgentCore service is unreachable.
        """
        try:
            self._client.delete_item(self._item_id(key))
        except Exception as exc:
            logger.error("AgentCoreBackend: DELETE failed")
            raise BackendConnectionError("AgentCore DELETE operation failed") from exc

    async def exists(self, key: str) -> bool:
        """Return ``True`` only for an unexpired entry under ``key``.

        Args:
            key: Cache key to check.

        Returns:
            ``True`` if an unexpired entry exists, ``False`` otherwise.

        Raises:
            BackendConnectionError: If the AgentCore service is unreachable.
        """
        item_id = self._item_id(key)
        try:
            item = self._client.get_item(item_id)
            if item is None:
                return False
            if self._is_expired(item):
                self._client.delete_item(item_id)
                return False
            return True
        except BackendConnectionError:
            raise
        except Exception as exc:
            logger.error("AgentCoreBackend: EXISTS failed")
            raise BackendConnectionError("AgentCore EXISTS operation failed") from exc

    async def flush(self) -> None:
        """Remove every entry this backend manages (namespaced by prefix).

        Raises:
            BackendConnectionError: If the AgentCore service is unreachable.
        """
        try:
            for item_id in self._client.scan(self._prefix):
                self._client.delete_item(item_id)
        except Exception as exc:
            logger.error("AgentCoreBackend: FLUSH failed")
            raise BackendConnectionError("AgentCore FLUSH operation failed") from exc

    async def ping(self) -> bool:
        """Return ``True`` when the AgentCore service is reachable.

        Returns:
            ``True`` if reachable, ``False`` otherwise. Never raises.
        """
        try:
            return bool(self._client.ping())
        except Exception:
            return False
