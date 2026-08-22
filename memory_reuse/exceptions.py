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
