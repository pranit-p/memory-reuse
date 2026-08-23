"""Local sentence-transformers embedding provider.

:class:`LocalEmbedder` produces embeddings with a
`sentence-transformers <https://www.sbert.net/>`_ model that runs entirely
on the local machine, requiring no network access at inference time.  This is
useful when privacy, cost, or latency rule out a hosted embedding API.

``sentence-transformers`` is an **optional** dependency.  It is imported
lazily the first time an embedding is computed so that merely importing this
module (for example via the :func:`~memory_reuse.embeddings.create_embedder`
factory) pulls in no heavy libraries.  When the dependency is missing an
:class:`~memory_reuse.exceptions.EmbeddingProviderError` is raised naming the
extra to install.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from memory_reuse.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Default model chosen for its small size, speed, and strong general-purpose
# quality — 384-dimensional vectors, no network access after the initial
# (cached) model download.
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Third-party loggers that emit INFO-level HTTP/model chatter (e.g. the
# Hugging Face cache-validation requests) each time a model loads. We raise
# only their *log level* to WARNING for a quieter default — we do NOT touch
# any offline env var, so first-time model downloads still work and the user's
# other Hugging Face usage is left untouched.
_NOISY_LOGGERS = ("huggingface_hub", "transformers", "sentence_transformers")


def _quiet_model_load_logging() -> None:
    """Raise noisy third-party loggers to WARNING (scoped, no behaviour change).

    This only adjusts logging verbosity. It intentionally does not set
    ``HF_HUB_OFFLINE`` or any other process-wide environment variable, because a
    library must not change global behaviour that affects the rest of the host
    application (and forcing offline mode would break the first, un-cached
    model download).
    """
    for name in _NOISY_LOGGERS:
        logger = logging.getLogger(name)
        if logger.level < logging.WARNING:
            logger.setLevel(logging.WARNING)


class LocalEmbedder(EmbeddingProvider):
    """Embedding provider backed by a local sentence-transformers model.

    The heavy ``sentence-transformers`` dependency and the model itself are
    loaded lazily on first use, so constructing a :class:`LocalEmbedder` is
    cheap and importing this module has no import-time side effects.

    Args:
        model: The sentence-transformers model name to load.  ``None`` selects
            a small, fast default (``"all-MiniLM-L6-v2"``).

    Raises:
        EmbeddingProviderError: On first use, if ``sentence-transformers`` is
            not installed.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or _DEFAULT_MODEL
        self._model: SentenceTransformer | None = None

    @property
    def identity(self) -> str:
        """Return the stable ``"local:<model>"`` identity string.

        Returns:
            The provider+model identity, e.g. ``"local:all-MiniLM-L6-v2"``.
        """
        return f"local:{self._model_name}"

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the produced embedding vectors.

        Loads the model on first access to query its true output dimension.

        Returns:
            The number of floats in each embedding vector.

        Raises:
            EmbeddingProviderError: If ``sentence-transformers`` is not
                installed.
        """
        model = self._load_model()
        dimension = model.get_sentence_embedding_dimension()
        return int(dimension)

    def _load_model(self) -> Any:
        """Load and cache the sentence-transformers model.

        Returns:
            The loaded ``SentenceTransformer`` instance.

        Raises:
            EmbeddingProviderError: If ``sentence-transformers`` is not
                installed.
        """
        if self._model is not None:
            return self._model

        _quiet_model_load_logging()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            from memory_reuse.exceptions import EmbeddingProviderError

            raise EmbeddingProviderError(
                "sentence-transformers is not installed, which is required for "
                "the local embedding provider. On a CPU-only machine install a "
                "CPU torch wheel first to avoid a large GPU download:\n\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                '    pip install "memory-reuse[semantic-local]"\n'
            ) from exc

        self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text.

        The synchronous, potentially CPU-bound model call is run in a worker
        thread so it does not block the event loop.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats of length
            :attr:`dimension`.

        Raises:
            EmbeddingProviderError: If ``sentence-transformers`` is not
                installed.
        """
        model = self._load_model()
        vector = await asyncio.to_thread(model.encode, text, show_progress_bar=False)
        return [float(value) for value in vector]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts in a single model call.

        sentence-transformers encodes batches efficiently, so this overrides
        the default sequential implementation.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order
            as ``texts``.

        Raises:
            EmbeddingProviderError: If ``sentence-transformers`` is not
                installed.
        """
        if not texts:
            return []
        model = self._load_model()
        vectors = await asyncio.to_thread(model.encode, texts, show_progress_bar=False)
        return [[float(value) for value in vector] for vector in vectors]
