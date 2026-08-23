"""Custom exceptions for the memory-reuse package."""


class AgentMemoryError(Exception):
    """Base exception for all memory-reuse errors.

    All exceptions raised by this library inherit from this class,
    making it easy to catch any library error with a single handler.
    """


class BackendConnectionError(AgentMemoryError):
    """Raised when the cache backend cannot be reached.

    Typically wraps connection failures from Redis or other network
    backends. Check your ``redis_url`` and network connectivity.
    """


class ScopeViolationError(AgentMemoryError):
    """Raised when user-scoped data would be cached under the global scope.

    This is a safety guard: if you call a cache operation with
    ``scope='user'`` but no ``user_id`` is available in the current
    context, this exception is raised rather than silently caching
    data that could be shared across users.
    """


class InvalidTTLError(AgentMemoryError):
    """Raised when a TTL value is not valid.

    TTL must be a positive integer (seconds) or ``None`` for no expiry.
    """


class BackendNotAvailableError(AgentMemoryError):
    """Raised when the requested backend is not importable or configured.

    For example, requesting the ``redis`` backend without the ``redis``
    package installed will raise this exception.
    """


class EmbeddingProviderError(AgentMemoryError):
    """Raised when an embedding provider cannot be used.

    Typically raised when the provider's optional dependency is not
    installed (naming the extra to install, e.g.
    ``pip install "memory-reuse[semantic]"``) or when the provider's
    backing model or API call fails.
    """


class ProviderMismatchError(AgentMemoryError):
    """Raised when embeddings from different providers or models are mixed.

    Vectors are namespaced by ``provider:model``. Comparing vectors from
    incompatible providers or models is refused rather than silently
    producing meaningless similarity scores.
    """


class ConfigurationError(AgentMemoryError):
    """Raised when a :class:`CacheConfig` value is invalid.

    For example, a ``similarity_threshold`` outside ``[0.0, 1.0]`` or
    enabling ``semantic_enabled`` without selecting an
    ``embedding_provider``.
    """
