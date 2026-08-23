"""Embedding providers for the semantic cache.

This package defines the abstract :class:`EmbeddingProvider` interface and the
:func:`create_embedder` factory that resolves a concrete provider from a
:class:`~memory_reuse.config.CacheConfig`.

Concrete providers lazily import their heavy optional dependencies, so simply
importing this package pulls in no third-party libraries.  The factory selects
the implementation by :attr:`CacheConfig.embedding_provider`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory_reuse.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from memory_reuse.config import CacheConfig

__all__ = ["EmbeddingProvider", "create_embedder"]


def create_embedder(config: CacheConfig) -> EmbeddingProvider:
    """Create an :class:`EmbeddingProvider` from a cache configuration.

    The provider is selected by :attr:`CacheConfig.embedding_provider` and
    configured with :attr:`CacheConfig.embedding_model` (when set).  Concrete
    provider modules are imported lazily so their optional dependencies are
    only required when that provider is actually selected.

    Args:
        config: The cache configuration.  Its ``embedding_provider`` field
            selects the implementation.

    Returns:
        A concrete :class:`EmbeddingProvider` instance.

    Raises:
        ConfigurationError: If ``embedding_provider`` is ``None`` or is not one
            of the supported values.
    """
    provider = config.embedding_provider

    if provider is None:
        from memory_reuse.exceptions import ConfigurationError

        raise ConfigurationError(
            "embedding_provider is not set; select one of "
            "'openai', 'local', or 'litellm' to create an embedder."
        )

    if provider == "openai":
        from memory_reuse.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model=config.embedding_model)

    if provider == "local":
        from memory_reuse.embeddings.local import LocalEmbedder

        return LocalEmbedder(model=config.embedding_model)

    if provider == "litellm":
        from memory_reuse.embeddings.litellm import LiteLLMEmbedder

        return LiteLLMEmbedder(model=config.embedding_model)

    from memory_reuse.exceptions import ConfigurationError

    raise ConfigurationError(
        f"Unknown embedding_provider {provider!r}; "
        "expected one of 'openai', 'local', or 'litellm'."
    )
