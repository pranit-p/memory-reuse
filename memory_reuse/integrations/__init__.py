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

Framework tool-caching decorators mirroring the LangGraph ``cached_tool`` are
also available for Strands Agents
(:func:`~memory_reuse.integrations.strands.cached_tool`) and CrewAI
(:func:`~memory_reuse.integrations.crewai.cached_tool`). They are imported from
their own submodules (``memory_reuse.integrations.strands`` /
``memory_reuse.integrations.crewai``) rather than re-exported here, so the
top-level ``cached_tool`` name keeps referring to the LangGraph decorator and
the optional Strands/CrewAI dependencies stay lazily imported.
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
