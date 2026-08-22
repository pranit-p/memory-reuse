"""Internal utilities for key building, hashing, and serialisation.

These are implementation details — do not import from outside this package.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from typing import Any

# Characters allowed in cache key segments (alphanumeric, dash, underscore, dot)
_SAFE_KEY_RE = re.compile(r"[^a-zA-Z0-9_\-.]")


def sanitize_key(key: str) -> str:
    """Replace unsafe characters in a key segment with underscores.

    Args:
        key: Raw key segment string.

    Returns:
        A string where any character not matching ``[a-zA-Z0-9_\\-.]`` is
        replaced with ``_``.

    Example::

        >>> sanitize_key("user@example.com")
        'user_example.com'
    """
    return _SAFE_KEY_RE.sub("_", key)


def build_cache_key(prefix: str, scope: str, scope_id: str | None, *parts: Any) -> str:
    """Build a namespaced, deterministic cache key.

    The resulting key follows the pattern::

        <prefix>:<scope>:<scope_id>:<hash_of_parts>

    When ``scope`` is ``"global"`` the ``scope_id`` segment is omitted.

    Args:
        prefix: Top-level namespace (e.g. ``"agentmem"``).
        scope: One of ``"global"``, ``"user"``, or ``"session"``.
        scope_id: Identifier for the scope (user ID or session ID).
            Must be provided for non-global scopes.
        *parts: Arbitrary values that together identify the cached item.
            They are JSON-serialised and hashed.

    Returns:
        A colon-separated cache key string.

    Raises:
        ValueError: If a non-global scope is used without a ``scope_id``.
    """
    if scope != "global" and not scope_id:
        raise ValueError(f"scope_id is required for scope='{scope}'")

    parts_hash = hash_value(list(parts))
    safe_prefix = sanitize_key(prefix)
    safe_scope = sanitize_key(scope)

    if scope == "global":
        return f"{safe_prefix}:{safe_scope}:{parts_hash}"

    safe_scope_id = sanitize_key(scope_id)  # type: ignore[arg-type]
    return f"{safe_prefix}:{safe_scope}:{safe_scope_id}:{parts_hash}"


def hash_value(value: Any) -> str:
    """Produce a 32-character SHA-256 hex digest of a JSON-serialised value.

    Keys in mappings are sorted to ensure deterministic output regardless of
    insertion order.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        The first 32 hex characters of the SHA-256 digest.

    Raises:
        TypeError: If ``value`` cannot be JSON-serialised.
    """
    serialised = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return digest[:32]


def serialize_value(value: Any) -> bytes:
    """Serialise and gzip-compress a value for storage in the backend.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        gzip-compressed JSON bytes.

    Raises:
        TypeError: If ``value`` cannot be JSON-serialised.
    """
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return gzip.compress(raw, compresslevel=6)


def deserialize_value(data: bytes) -> Any:
    """Decompress and deserialise bytes produced by :func:`serialize_value`.

    Args:
        data: gzip-compressed JSON bytes.

    Returns:
        The original Python object.

    Raises:
        ValueError: If the data cannot be decompressed or parsed.
    """
    try:
        raw = gzip.decompress(data)
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to deserialise cached value: {exc}") from exc
