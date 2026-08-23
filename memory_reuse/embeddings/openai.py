"""OpenAI embedding provider.

:class:`OpenAIEmbedder` produces embeddings using OpenAI's hosted embeddings
API (for example the ``text-embedding-3-small`` model).  It is a good default
when a managed, high-quality embedding model is preferred over running a model
locally.

``openai`` is an **optional** dependency.  It is imported lazily the first time
the client is needed so that merely importing this module (for example via the
:func:`~memory_reuse.embeddings.create_embedder` factory) pulls in no heavy
libraries.  When the dependency is missing an
:class:`~memory_reuse.exceptions.EmbeddingProviderError` is raised naming the
extra to install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memory_reuse.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Default model chosen for its low cost, speed, and strong general-purpose
# quality — 1536-dimensional vectors from OpenAI's hosted API.
_DEFAULT_MODEL = "text-embedding-3-small"

# Known output dimensionality for OpenAI's embedding models.  Used to report
# :attr:`dimension` without a network call; unknown models fall back to the
# small model's dimension.
_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(EmbeddingProvider):
    """Embedding provider backed by OpenAI's hosted embeddings API.

    The ``openai`` dependency and the API client are created lazily on first
    use, so constructing an :class:`OpenAIEmbedder` is cheap and importing this
    module has no import-time side effects.

    Args:
        model: The OpenAI embedding model name.  ``None`` selects a small,
            low-cost default (``"text-embedding-3-small"``).
        api_key: An explicit API key.  ``None`` lets the ``openai`` client read
            it from the environment (``OPENAI_API_KEY``).

    Raises:
        EmbeddingProviderError: On first use, if ``openai`` is not installed.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model_name = model or _DEFAULT_MODEL
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None

    @property
    def identity(self) -> str:
        """Return the stable ``"openai:<model>"`` identity string.

        Returns:
            The provider+model identity, e.g. ``"openai:text-embedding-3-small"``.
        """
        return f"openai:{self._model_name}"

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the produced embedding vectors.

        The value is looked up from a table of known OpenAI embedding models,
        so no network call is made.  Unknown models fall back to the default
        model's dimension.

        Returns:
            The number of floats in each embedding vector.
        """
        return _MODEL_DIMENSIONS.get(self._model_name, _MODEL_DIMENSIONS[_DEFAULT_MODEL])

    def _get_client(self) -> Any:
        """Create and cache the async OpenAI client.

        Returns:
            The ``openai.AsyncOpenAI`` instance.

        Raises:
            EmbeddingProviderError: If ``openai`` is not installed.
        """
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            from memory_reuse.exceptions import EmbeddingProviderError

            raise EmbeddingProviderError(
                "openai is not installed, which is required for the OpenAI "
                "embedding provider. Install it with:\n\n"
                '    pip install "memory-reuse[semantic]"\n\n'
                "(the 'semantic' extra bundles the OpenAI and LiteLLM clients)."
            ) from exc

        self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats of length
            :attr:`dimension`.

        Raises:
            EmbeddingProviderError: If ``openai`` is not installed.
        """
        client = self._get_client()
        response = await client.embeddings.create(model=self._model_name, input=text)
        return [float(value) for value in response.data[0].embedding]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts in a single API call.

        The OpenAI embeddings API accepts a list of inputs, so this overrides
        the default sequential implementation to reduce the number of requests.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order
            as ``texts``.

        Raises:
            EmbeddingProviderError: If ``openai`` is not installed.
        """
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(model=self._model_name, input=texts)
        # The API returns items with an ``index`` field; sort defensively so
        # the returned order always matches the input order.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in ordered]
