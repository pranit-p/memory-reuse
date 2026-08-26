"""Property 19: Unknown backend value is rejected at configuration time.

Feature: analytics-and-integrations, Property 19.

*For any* backend string that is NOT one of the three supported values
(``"memory"``, ``"redis"``, ``"agentcore"``), constructing
``CacheConfig(backend=<bad>)`` raises ``ConfigurationError`` at configuration
time -- i.e. inside :meth:`CacheConfig.__post_init__`, before any cache is
constructed -- and the raised error's message names the offending value.

Validates: Requirements 9.7.

The validation lives at the top of :meth:`CacheConfig.__post_init__`, which
rejects any ``backend`` not in ``{"memory", "redis", "agentcore"}`` by raising
``ConfigurationError`` with a message that interpolates the bad value. Because
``__post_init__`` runs during dataclass construction, the rejection happens at
configuration time. Arbitrary strings are generated with hypothesis and the
three valid values are excluded via ``assume`` so every example is genuinely
unknown.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from memory_reuse.config import CacheConfig
from memory_reuse.exceptions import ConfigurationError

VALID_BACKENDS = {"memory", "redis", "agentcore"}


class TestProperty19UnknownBackend:
    """Feature: analytics-and-integrations, Property 19.

    Unknown backend value is rejected at configuration time.

    Validates: Requirements 9.7
    """

    @settings(max_examples=100)
    @given(bad_backend=st.text())
    def test_unknown_backend_rejected_at_configuration_time(self, bad_backend: str) -> None:
        """Any backend outside the three valid values raises ``ConfigurationError``.

        The error is raised during construction (``__post_init__``) and its
        message names the offending value.
        """
        assume(bad_backend not in VALID_BACKENDS)

        with pytest.raises(ConfigurationError) as exc_info:
            CacheConfig(backend=bad_backend)  # type: ignore[arg-type]

        # The rejection must name the offending value so the developer can
        # see exactly what was wrong.
        assert bad_backend in str(exc_info.value)
