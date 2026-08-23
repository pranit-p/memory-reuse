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


def check_scope(scope: str, scope_id: str | None) -> None:
    """Raise :exc:`ScopeViolationError` when a scope requires an ID but none given.

    This is the single shared scope guard used by every cache layer
    (:class:`~memory_reuse.cache.exact.ExactCache`,
    :class:`~memory_reuse.cache.tool.ToolCache`, and
    :class:`~memory_reuse.cache.semantic.SemanticCache`) so that
    :exc:`~memory_reuse.exceptions.ScopeViolationError` behaviour is identical
    across them.

    Args:
        scope: Requested scope — ``"global"``, ``"user"``, or ``"session"``.
        scope_id: Provided scope identifier.

    Raises:
        ScopeViolationError: When ``scope`` is ``"user"`` or ``"session"`` and
            ``scope_id`` is ``None`` or empty.
    """
    if scope in ("user", "session") and not scope_id:
        from memory_reuse.exceptions import ScopeViolationError

        raise ScopeViolationError(
            f"scope='{scope}' requires a non-empty scope_id, but none was provided. "
            "Set user_id or session_id via MemoryCache.set_context() or pass it "
            "explicitly to avoid accidental cross-user cache sharing."
        )


def build_namespace(scope: str, scope_id: str | None) -> str:
    """Build the vector-index namespace string for a scope.

    The namespace encodes the cache scope so a similarity search never crosses
    scope boundaries:

    * ``"global"`` for the global scope,
    * ``"user:<scope_id>"`` for the user scope,
    * ``"session:<scope_id>"`` for the session scope.

    Args:
        scope: One of ``"global"``, ``"user"``, or ``"session"``.
        scope_id: The scope identifier. Required for non-global scopes.

    Returns:
        The namespace string.

    Raises:
        ScopeViolationError: If a non-global scope is used without a
            ``scope_id`` (delegated to :func:`check_scope`).
    """
    check_scope(scope, scope_id)
    if scope == "global":
        return "global"
    return f"{scope}:{scope_id}"


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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, normalised to ``[0.0, 1.0]``.

    The raw cosine similarity lies in ``[-1.0, 1.0]``; it is mapped to
    ``[0.0, 1.0]`` via ``(cos + 1) / 2`` so that identical vectors score
    ``1.0``, opposite vectors score ``0.0``, and orthogonal vectors score
    ``0.5``. This makes the configured similarity threshold intuitive.

    Uses ``numpy`` when it is importable for speed, otherwise falls back to a
    pure-Python implementation. Both paths produce equivalent results.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        A float in ``[0.0, 1.0]``. When either vector is all zeros the vectors
        have no defined orientation, so the neutral score ``0.5`` is returned.

    Raises:
        ValueError: If the two vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: len(a)={len(a)} != len(b)={len(b)}")

    if len(a) == 0:
        # No components to compare; treat as neutral.
        return 0.5

    try:
        import numpy as np  # noqa: PLC0415

        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.5
        cos = float(np.dot(va, vb) / (norm_a * norm_b))
    except ImportError:
        dot = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for x, y in zip(a, b, strict=True):
            dot += x * y
            norm_a_sq += x * x
            norm_b_sq += y * y
        if norm_a_sq == 0.0 or norm_b_sq == 0.0:
            return 0.5
        cos = dot / ((norm_a_sq**0.5) * (norm_b_sq**0.5))

    # Guard against floating-point drift outside [-1.0, 1.0].
    cos = max(-1.0, min(1.0, cos))
    return (cos + 1.0) / 2.0


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


# Sentence boundary: a period, question mark, or exclamation mark (optionally
# followed by a closing quote/bracket) then whitespace. Deliberately simple —
# good enough to narrow a cached answer, with no NLP dependency.
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])["\')\]]?\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks using a lightweight heuristic.

    This is intentionally dependency-free (no NLTK/spaCy). It splits on
    sentence-ending punctuation followed by whitespace, then also breaks on
    newlines so bulleted or line-separated answers yield separate chunks. Empty
    fragments are dropped.

    Args:
        text: The text to split.

    Returns:
        A list of non-empty, stripped sentence chunks. Returns an empty list
        for empty/whitespace-only input, and a single-element list when no
        boundary is found.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    # First split on hard line breaks (handles bullet/markdown-style answers),
    # then on sentence punctuation within each line.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for piece in _SENTENCE_BOUNDARY_RE.split(line):
            piece = piece.strip()
            if piece:
                chunks.append(piece)
    return chunks
