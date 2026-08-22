"""Cache layer implementations for memory-reuse.

* :class:`~memory_reuse.cache.exact.ExactCache` — hash-keyed cache for
  LLM responses and other deterministic lookups.
* :class:`~memory_reuse.cache.tool.ToolCache` — TTL-enforced cache for
  tool/function call results.
"""

from memory_reuse.cache.exact import ExactCache
from memory_reuse.cache.tool import ToolCache

__all__ = ["ExactCache", "ToolCache"]
