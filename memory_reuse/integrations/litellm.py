"""LiteLLM integration: drop-in cached wrappers for LiteLLM calls.

This module provides two thin wrappers:

* :func:`cached_litellm_completion` — wraps ``litellm.completion`` /
  ``litellm.acompletion`` with exact-match caching.
* :func:`cached_litellm_embedding` — wraps ``litellm.embedding`` /
  ``litellm.aembedding`` with exact-match caching.

Both functions are framework-agnostic — they work in plain Python scripts,
FastAPI handlers, LangGraph nodes, Strands tools, or any other async context.

LiteLLM is an **optional** dependency.  The wrappers import it lazily so that
``memory-reuse`` itself does not require LiteLLM to be installed.

Example::

    from memory_reuse import MemoryCache
    from memory_reuse.integrations.litellm import cached_litellm_completion

    cache = MemoryCache()

    # First call hits the LLM; subsequent identical calls return from cache.
    response = await cached_litellm_completion(
        cache,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        ttl=3600,
        scope="global",
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from memory_reuse.core import MemoryCache

logger = logging.getLogger(__name__)

_SCOPE_TYPE = Literal["global", "user", "session"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_litellm_installed() -> Any:
    """Return the ``litellm`` module, raising a clear error if not installed.

    Returns:
        The ``litellm`` module.

    Raises:
        ImportError: If LiteLLM is not installed.
    """
    try:
        import litellm  # type: ignore[import-untyped]

        return litellm
    except ImportError as exc:
        raise ImportError(
            "LiteLLM is not installed.  Install it with:\n\n"
            "    pip install litellm\n\n"
            "or add the extra:\n\n"
            "    pip install memory-reuse[litellm]"
        ) from exc


def _scope_id_from_kwargs(
    scope: str,
    user_id: str | None,
    session_id: str | None,
    cache: MemoryCache,
) -> str | None:
    """Resolve the scope ID from explicit args or MemoryCache context.

    Args:
        scope: Requested scope.
        user_id: Explicitly provided user identifier.
        session_id: Explicitly provided session identifier.
        cache: The :class:`~memory_reuse.core.MemoryCache` instance — used
            as a fallback when explicit IDs are not given.

    Returns:
        The scope ID string, or ``None`` for global scope.
    """
    if scope == "global":
        return None
    if scope == "user":
        return user_id or cache._context.get("user_id")
    if scope == "session":
        return session_id or cache._context.get("session_id")
    return None


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


async def cached_litellm_completion(
    cache: MemoryCache,
    *,
    model: str,
    messages: list[dict[str, str]],
    ttl: int = 3600,
    scope: _SCOPE_TYPE = "global",
    user_id: str | None = None,
    session_id: str | None = None,
    **litellm_kwargs: Any,
) -> Any:
    """Cached wrapper around ``litellm.acompletion``.

    On a cache hit the LLM is **not** called and the previously stored
    response object is returned immediately.  On a cache miss the response is
    fetched from LiteLLM, stored in the cache, and returned.

    The cache key is derived from ``model``, ``messages``, and any extra
    ``litellm_kwargs`` (e.g. ``temperature``, ``max_tokens``).  Changing any
    of these values produces a different cache key and triggers a fresh LLM
    call.

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance to use.
        model: LiteLLM model string, e.g. ``"gpt-4o-mini"``,
            ``"anthropic/claude-3-haiku"``, ``"bedrock/amazon.titan-text-v2"``.
        messages: Chat messages in OpenAI format.
        ttl: Time-to-live in seconds.  Defaults to 3600 (1 hour).
        scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
        user_id: User identifier for ``scope="user"``.  Falls back to the
            value set via :meth:`~memory_reuse.core.MemoryCache.set_context`.
        session_id: Session identifier for ``scope="session"``.  Falls back
            to context.
        **litellm_kwargs: Additional keyword arguments forwarded to
            ``litellm.acompletion`` (e.g. ``temperature``, ``max_tokens``,
            ``stream=False``).

    Returns:
        A ``litellm.ModelResponse`` object (identical shape to the OpenAI
        ``ChatCompletion`` response).

    Raises:
        ScopeViolationError: If ``scope`` requires an ID that cannot be found.
        ImportError: If LiteLLM is not installed.

    Example::

        response = await cached_litellm_completion(
            cache,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Explain caching in one line"}],
            ttl=600,
            scope="user",
            user_id="alice",
        )
        print(response.choices[0].message.content)
    """
    litellm = _assert_litellm_installed()

    scope_id = _scope_id_from_kwargs(scope, user_id, session_id, cache)

    # Build a stable, deterministic cache key.
    # We include litellm_kwargs so that different temperature/max_tokens
    # values produce different cached answers.
    key_parts = ["litellm.completion", model, messages, litellm_kwargs]

    cached_response = await cache.exact.get(key_parts, scope=scope, scope_id=scope_id)
    if cached_response is not None:
        logger.debug("cached_litellm_completion: HIT model=%s scope=%s", model, scope)
        return cached_response

    logger.debug(
        "cached_litellm_completion: MISS model=%s scope=%s — calling LiteLLM",
        model,
        scope,
    )

    # Always use the async variant for consistency across sync/async callers.
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        **litellm_kwargs,
    )

    # Store a plain dict so the response survives JSON serialisation in the
    # backend.  We reconstruct a dict rather than the ModelResponse object
    # because ModelResponse is not always directly JSON-serialisable.
    serialisable = response.model_dump() if hasattr(response, "model_dump") else dict(response)

    await cache.exact.set(key_parts, serialisable, scope=scope, scope_id=scope_id, ttl=ttl)
    logger.debug("cached_litellm_completion: stored response in cache")

    # Return the original rich response object on first call.
    return response


async def cached_litellm_embedding(
    cache: MemoryCache,
    *,
    model: str,
    input: list[str] | str,
    ttl: int = 86400,
    scope: _SCOPE_TYPE = "global",
    user_id: str | None = None,
    session_id: str | None = None,
    **litellm_kwargs: Any,
) -> Any:
    """Cached wrapper around ``litellm.aembedding``.

    Embedding calls are expensive and deterministic — the same text always
    produces the same vector.  Caching them with a long TTL (default 24 hours)
    is safe and eliminates significant cost for repeated RAG pipelines.

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance to use.
        model: LiteLLM embedding model string, e.g.
            ``"text-embedding-3-small"``, ``"bedrock/amazon.titan-embed-text-v2"``.
        input: A single string or list of strings to embed.
        ttl: Time-to-live in seconds.  Defaults to 86400 (24 hours) — safe
            because embeddings are deterministic.
        scope: Cache scope.
        user_id: User identifier for ``scope="user"``.
        session_id: Session identifier for ``scope="session"``.
        **litellm_kwargs: Additional keyword arguments forwarded to
            ``litellm.aembedding``.

    Returns:
        A ``litellm.EmbeddingResponse`` object.

    Raises:
        ScopeViolationError: If ``scope`` requires an ID that cannot be found.
        ImportError: If LiteLLM is not installed.

    Example::

        response = await cached_litellm_embedding(
            cache,
            model="text-embedding-3-small",
            input=["What is machine learning?", "Explain neural networks"],
        )
        vectors = [item["embedding"] for item in response.data]
    """
    litellm = _assert_litellm_installed()

    scope_id = _scope_id_from_kwargs(scope, user_id, session_id, cache)

    # Normalise input to a list so the cache key is stable regardless of
    # whether the caller passes a string or a list.
    normalised_input = [input] if isinstance(input, str) else list(input)
    key_parts = ["litellm.embedding", model, normalised_input, litellm_kwargs]

    cached_response = await cache.exact.get(key_parts, scope=scope, scope_id=scope_id)
    if cached_response is not None:
        logger.debug("cached_litellm_embedding: HIT model=%s scope=%s", model, scope)
        return cached_response

    logger.debug(
        "cached_litellm_embedding: MISS model=%s scope=%s — calling LiteLLM",
        model,
        scope,
    )

    response = await litellm.aembedding(
        model=model,
        input=normalised_input,
        **litellm_kwargs,
    )

    serialisable = response.model_dump() if hasattr(response, "model_dump") else dict(response)

    await cache.exact.set(key_parts, serialisable, scope=scope, scope_id=scope_id, ttl=ttl)
    logger.debug("cached_litellm_embedding: stored response in cache")

    return response
