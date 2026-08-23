"""Abstract embedding-provider interface for the semantic cache.

An :class:`EmbeddingProvider` turns text into fixed-length embedding vectors
that the semantic cache compares by cosine similarity.  Implementations wrap a
hosted API (OpenAI), a cloud gateway (LiteLLM), or a local model
(sentence-transformers), and lazily import their heavy optional dependency so
the core package keeps zero required runtime dependencies.

All methods are asynchronous so network-bound providers can be awaited without
blocking the event loop; a purely local provider simply returns immediately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface that all embedding providers must implement.

    The :attr:`identity` string namespaces stored vectors so embeddings from
    different providers or models are never compared, and :attr:`dimension`
    lets callers validate vector shape before a similarity search.

    Implementors should raise
    :class:`~memory_reuse.exceptions.EmbeddingProviderError` (with an install
    hint) when their optional dependency is missing, and should keep
    :attr:`identity` stable for a given provider+model pairing.
    """

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return the stable ``"provider:model"`` identity string.

        This value namespaces stored vectors so that embeddings produced by
        different providers or models are never compared against one another
        (for example ``"openai:text-embedding-3-small"``).

        Returns:
            The provider+model identity string.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the produced embedding vectors.

        Returns:
            The number of floats in each embedding vector.
        """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats of length
            :attr:`dimension`.
        """

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts.

        The default implementation calls :meth:`embed` sequentially.  Providers
        that support a native batch call should override this for efficiency.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same
            order as ``texts``.
        """
        return [await self.embed(text) for text in texts]
