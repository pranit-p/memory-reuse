"""Third-party integrations for memory-reuse.

Currently provides:

* :func:`~memory_reuse.integrations.langgraph.cached_node` — LangGraph node
  caching decorator.
* :func:`~memory_reuse.integrations.langgraph.cached_tool` — tool/function
  caching decorator (works with any framework).
* :func:`~memory_reuse.integrations.litellm.cached_litellm_completion` —
  cached wrapper for ``litellm.acompletion``.
* :func:`~memory_reuse.integrations.litellm.cached_litellm_embedding` —
  cached wrapper for ``litellm.aembedding``.
"""

from memory_reuse.integrations.langgraph import cached_node, cached_tool
from memory_reuse.integrations.litellm import (
    cached_litellm_completion,
    cached_litellm_embedding,
)

__all__ = [
    "cached_node",
    "cached_tool",
    "cached_litellm_completion",
    "cached_litellm_embedding",
]
