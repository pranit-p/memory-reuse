"""Execution-coordination primitives (Phase 6).

This subpackage is **additive and opt-in** and uses only the Python standard
library for its in-process path. It currently provides :class:`SingleFlight`,
which coalesces concurrent cache misses for the same key so an expensive compute
runs once and its result is shared with every waiting caller.

An optional distributed variant coordinates across processes using the existing
Redis backend; that path imports Redis only through the backend it is handed, so
``import memory_reuse.execution`` pulls in no third-party dependency.
"""

from __future__ import annotations

from memory_reuse.execution.singleflight import SingleFlight

__all__ = ["SingleFlight"]
