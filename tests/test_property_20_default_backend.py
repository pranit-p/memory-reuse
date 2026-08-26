"""Property 20: Default backend is unchanged.

Feature: analytics-and-integrations, Property 20.

*For any* combination of **non-backend** :class:`~memory_reuse.config.CacheConfig`
options (``default_ttl``, ``default_scope``, ``key_prefix``, ``max_key_size``,
``enable_stats`` and friends), a ``CacheConfig(...)`` constructed **without**
specifying ``backend`` always resolves to ``backend == "memory"``, and a
:class:`~memory_reuse.core.MemoryCache` built from it uses the in-memory backend.

This pins the backward-compatibility guarantee that adding the ``"agentcore"``
option left the default untouched (Req 9.5): none of the unrelated config knobs
can flip the default away from ``"memory"``, and the resolved cache always wires
up the in-memory backend. (The lazy-import guarantee — that the default path
imports neither ``boto3`` nor ``redis`` — is validated by the dedicated
lazy-import smoke test in its own isolated module state, not here, since global
``sys.modules`` is not isolated across a full-suite session.)

Validates: Requirements 9.5

The property runs fully offline: no network, LLM, or AWS call. Options are
generated within their valid ranges (mirroring
:meth:`CacheConfig.__post_init__`) so every generated config is constructible,
and ``max_examples=100`` keeps a broad sweep of combinations fast.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache


def _non_backend_options() -> st.SearchStrategy[dict[str, object]]:
    """Strategy for a dict of valid, **non-backend** ``CacheConfig`` kwargs.

    Every value stays inside the range ``CacheConfig.__post_init__`` accepts so
    the constructed config is always valid, and ``backend`` (plus the
    backend-specific ``redis_url`` / ``agentcore_*`` fields) is deliberately
    never included — that is exactly the axis the property holds fixed.
    """
    return st.fixed_dictionaries(
        {},
        optional={
            "default_ttl": st.one_of(st.none(), st.integers(min_value=1, max_value=100_000)),
            "default_scope": st.sampled_from(["global", "user", "session"]),
            "key_prefix": st.text(max_size=20),
            "max_key_size": st.integers(min_value=1, max_value=4096),
            "enable_stats": st.booleans(),
            "similarity_threshold": st.floats(min_value=0.0, max_value=1.0),
            "max_vectors_per_namespace": st.integers(min_value=1, max_value=100_000),
            "store_exact_on_semantic_hit": st.booleans(),
            "extract_answer": st.booleans(),
            "extract_min_similarity": st.floats(min_value=0.0, max_value=1.0),
        },
    )


class TestProperty20DefaultBackendUnchanged:
    """Feature: analytics-and-integrations, Property 20.

    Default backend is unchanged.

    Validates: Requirements 9.5
    """

    @settings(max_examples=100)
    @given(options=_non_backend_options())
    async def test_default_backend_is_memory(self, options: dict[str, object]) -> None:
        """A config built without ``backend`` resolves to ``"memory"``.

        For any combination of unrelated (non-backend) options, omitting
        ``backend`` leaves it at its default ``"memory"`` value, and the
        ``MemoryCache`` built from that config wires up an in-memory backend
        (Req 9.5) — never Redis or AgentCore.
        """
        config = CacheConfig(**options)  # type: ignore[arg-type]

        # The default is preserved regardless of the other options.
        assert config.backend == "memory"

        cache = MemoryCache(config)

        # The constructed backend is the in-memory one, not Redis/AgentCore.
        assert isinstance(cache._backend, InMemoryBackend)

    async def test_bare_default_config_uses_memory(self) -> None:
        """A no-argument ``CacheConfig()`` defaults to the in-memory backend.

        The concrete baseline example alongside the property: the documented
        default construction path resolves to ``"memory"`` (Req 9.5).
        """
        config = CacheConfig()

        assert config.backend == "memory"
        assert isinstance(MemoryCache(config)._backend, InMemoryBackend)
