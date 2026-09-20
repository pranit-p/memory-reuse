"""Faithful (de)serialization of LangChain objects for the cache (Phase 6, Req 9).

By default the cache stores values as JSON with ``default=str``, which
stringifies non-JSON objects. LangChain message objects (``AIMessage`` etc.)
therefore do not survive a store/load round-trip: a cache *hit* would replay
strings while a *miss* returns message objects. This module provides an opt-in
codec — a ``(serializer, deserializer)`` pair — that converts LangChain objects
with LangChain's own lossless :func:`langchain_core.load.dumpd` /
:func:`langchain_core.load.load` so a hit returns the same object shapes as a
miss.

LangChain is an **optional** dependency: it is imported lazily inside
:func:`langchain_message_codec`, so importing this module (and the core package)
pulls in nothing. A missing dependency surfaces as a clear, named error naming
the extra to install.

Usage::

    from memory_reuse import CacheConfig, MemoryCache
    from memory_reuse.integrations.langchain_serde import langchain_message_codec

    serializer, deserializer = langchain_message_codec()
    cache = MemoryCache(CacheConfig(serializer=serializer, deserializer=deserializer))
    cached_graph = cache.wrap_graph(agent, scope="user", key_fields=["messages"])
"""

from __future__ import annotations

from typing import Any

from memory_reuse.exceptions import BackendNotAvailableError

# A marker key wrapping a LangChain-encoded payload so the deserializer knows to
# decode it back into a LangChain object. Chosen to be unlikely to collide with
# real data keys.
_LC_MARKER = "__lc_serialized__"


def _require_langchain() -> Any:
    """Return LangChain's ``dumpd`` / ``load`` helpers, or raise a named error.

    Returns:
        A ``(dumpd, load)`` tuple from ``langchain_core.load``.

    Raises:
        BackendNotAvailableError: If ``langchain-core`` is not installed, naming
            the extra to install (Req 9.8).
    """
    try:
        from langchain_core.load import dumpd, load  # noqa: PLC0415

        return dumpd, load
    except ImportError as exc:
        raise BackendNotAvailableError(
            "The LangChain serialization codec requires langchain-core. Install "
            'it with `pip install "memory-reuse[langchain]"`.'
        ) from exc


def langchain_message_codec() -> tuple[Any, Any]:
    """Build a ``(serializer, deserializer)`` pair for LangChain objects (Req 9.5, 9.6).

    The serializer walks a value and encodes any LangChain
    ``Serializable`` object (notably message objects, and lists of them such as a
    graph state's ``messages`` field) with :func:`langchain_core.load.dumpd`,
    wrapping each encoded payload under a marker key. All other JSON-native
    content passes through unchanged. The deserializer reverses this, decoding
    marked payloads with :func:`langchain_core.load.load` so a cache hit returns
    the same object types and contents as a miss.

    Returns:
        A ``(serializer, deserializer)`` tuple suitable for
        :attr:`CacheConfig.serializer` / :attr:`CacheConfig.deserializer`.

    Raises:
        BackendNotAvailableError: If ``langchain-core`` is not installed.
    """
    dumpd, load = _require_langchain()

    from langchain_core.load.serializable import Serializable  # noqa: PLC0415

    def _load(payload: Any) -> Any:
        """Reconstruct a LangChain object, tolerating un-loadable sub-objects.

        ``dumpd`` never fails: for a sub-object LangChain cannot serialise (for
        example a provider tool-call type embedded in a tool-calling
        ``AIMessage``'s ``additional_kwargs``) it emits a
        ``{"type": "not_implemented", ...}`` placeholder. ``load`` normally
        *refuses* such a payload with ``NotImplementedError``; passing
        ``ignore_unserializable_fields=True`` makes it **skip** those nodes and
        still rebuild the real message (its ``content`` / ``type`` etc.), which
        is what a cache hit needs. ``allowed_objects="messages"`` scopes
        reconstruction to chat-message types — the correct, safe default for
        untrusted cached content (and it silences the pending-deprecation
        warning about that default).
        """
        return load(
            payload,
            allowed_objects="messages",
            ignore_unserializable_fields=True,
        )

    def _encode(value: Any) -> Any:
        # Any LangChain Serializable (messages, and lists of them such as a
        # graph state's ``messages`` field) is wrapped as a marked, encoded
        # payload. ``dumpd`` always succeeds; the tolerant ``_load`` above
        # handles any embedded un-loadable sub-object on the way back, so a
        # tool-calling message still rebuilds as a real message on a cache hit.
        if isinstance(value, Serializable):
            return {_LC_MARKER: dumpd(value)}
        if isinstance(value, dict):
            return {k: _encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_encode(v) for v in value]
        return value

    def _decode(value: Any) -> Any:
        if isinstance(value, dict):
            if _LC_MARKER in value and len(value) == 1:
                # A cache fault must never become an app fault: on any residual
                # load failure, return the raw encoded payload rather than raise.
                try:
                    return _load(value[_LC_MARKER])
                except Exception:  # noqa: BLE001 — best-effort, never fatal
                    return value[_LC_MARKER]
            return {k: _decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_decode(v) for v in value]
        return value

    return _encode, _decode
