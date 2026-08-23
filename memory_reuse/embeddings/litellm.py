"""LiteLLM embedding provider.

:class:`LiteLLMEmbedder` produces embeddings through
`LiteLLM <https://docs.litellm.ai/>`_, which exposes a single, uniform API over
100+ embedding models — including AWS Bedrock (for example
``bedrock/amazon.titan-embed-text-v2``), OpenAI, Cohere, and many others.  This
is the most flexible provider: any model LiteLLM supports can be used simply by
passing its model string.

``litellm`` is an **optional** dependency.  It is imported lazily the first time
an embedding is computed so that merely importing this module (for example via
the :func:`~memory_reuse.embeddings.create_embedder` factory) pulls in no heavy
libraries.  When the dependency is missing an
:class:`~memory_reuse.exceptions.EmbeddingProviderError` is raised naming the
extra to install.
"""

from __future__ import annotations

from typing import Any

from memory_reuse.embeddings.base import EmbeddingProvider

# LiteLLM has no single "default" embedding model — the model string is what
# selects the underlying provider (OpenAI, Bedrock, Cohere, ...).  We default to
# a widely available, low-cost OpenAI model so the provider is usable with only
# a model-agnostic configuration, matching the other providers' ergonomics.
_DEFAULT_MODEL = "text-embedding-3-small"


class LiteLLMEmbedder(EmbeddingProvider):
    """Embedding provider backed by LiteLLM's uniform embedding API.

    LiteLLM routes to whichever backend the ``model`` string names, so a single
    provider class covers AWS Bedrock, OpenAI, Cohere, and every other
    LiteLLM-supported embedding model.  The heavy ``litellm`` dependency is
    imported lazily on first use, so constructing a :class:`LiteLLMEmbedder` is
    cheap and importing this module has no import-time side effects.

    Because the embedding dimension depends on the underlying model (and is only
    known reliably after a call), :attr:`dimension` is discovered from the first
    embedding response and cached.

    Args:
        model: The LiteLLM model string, e.g. ``"text-embedding-3-small"`` or
            ``"bedrock/amazon.titan-embed-text-v2"``.  ``None`` selects a
            low-cost default (``"text-embedding-3-small"``).

    Raises:
        EmbeddingProviderError: On first use, if ``litellm`` is not installed.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or _DEFAULT_MODEL
        self._dimension: int | None = None

    @property
    def identity(self) -> str:
        """Return the stable ``"litellm:<model>"`` identity string.

        Returns:
            The provider+model identity, e.g.
            ``"litellm:bedrock/amazon.titan-embed-text-v2"``.
        """
        return f"litellm:{self._model_name}"

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the produced embedding vectors.

        LiteLLM abstracts over many models whose dimensionality differs and is
        not known without a call, so the dimension is discovered lazily from the
        first :meth:`embed` / :meth:`embed_batch` response and cached.  Calling
        this before any embedding has been produced raises
        :class:`EmbeddingProviderError`.

        Returns:
            The number of floats in each embedding vector.

        Raises:
            EmbeddingProviderError: If no embedding has been produced yet, so
                the dimension is not yet known.
        """
        if self._dimension is None:
            from memory_reuse.exceptions import EmbeddingProviderError

            raise EmbeddingProviderError(
                "LiteLLMEmbedder.dimension is unknown until the first embedding "
                "is produced. Call embed() or embed_batch() first."
            )
        return self._dimension

    def _get_litellm(self) -> Any:
        """Import and return the ``litellm`` module.

        Returns:
            The ``litellm`` module.

        Raises:
            EmbeddingProviderError: If ``litellm`` is not installed.
        """
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as exc:
            from memory_reuse.exceptions import EmbeddingProviderError

            raise EmbeddingProviderError(
                "litellm is not installed, which is required for the LiteLLM "
                "embedding provider. Install it with:\n\n"
                '    pip install "memory-reuse[semantic]"\n\n'
                "(the 'semantic' extra bundles the OpenAI and LiteLLM clients)."
            ) from exc

        return litellm

    @staticmethod
    def _extract_vector(item: Any) -> list[float]:
        """Extract an embedding vector from a single LiteLLM response item.

        LiteLLM response items behave like dicts with an ``"embedding"`` key,
        but may also expose the vector as an attribute depending on the version.

        Args:
            item: A single item from ``response.data``.

        Returns:
            The embedding vector as a list of floats.
        """
        try:
            vector = item["embedding"]
        except (TypeError, KeyError):
            vector = item.embedding
        return [float(value) for value in vector]

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats of length
            :attr:`dimension`.

        Raises:
            EmbeddingProviderError: If ``litellm`` is not installed.
        """
        litellm = self._get_litellm()
        response = await litellm.aembedding(model=self._model_name, input=text)
        vector = self._extract_vector(response.data[0])
        self._dimension = len(vector)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts in a single API call.

        LiteLLM's embedding API accepts a list of inputs, so this overrides the
        default sequential implementation to reduce the number of requests.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order
            as ``texts``.

        Raises:
            EmbeddingProviderError: If ``litellm`` is not installed.
        """
        if not texts:
            return []
        litellm = self._get_litellm()
        response = await litellm.aembedding(model=self._model_name, input=texts)
        vectors = [self._extract_vector(item) for item in response.data]
        if vectors:
            self._dimension = len(vectors[0])
        return vectors
