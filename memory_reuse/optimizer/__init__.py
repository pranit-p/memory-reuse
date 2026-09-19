"""Effectiveness analyzer and cache recommendations (Phase 6).

This subpackage is **additive and opt-in** and uses only the Python standard
library. It records per-operation hit/miss observations and turns them into
actionable, advisory recommendations — "cache this tool", "raise this TTL",
"don't semantically cache this" — rather than raw metrics.

The pieces are:

* :class:`OperationRecord` — one recorded execution observation.
* :class:`OperationStats` — the per-operation aggregate.
* :class:`OperationTracker` — bounded, best-effort recorder.
* :class:`EffectivenessAnalyzer` / :class:`EffectivenessReport` — statistics plus
  recommendations.
* :class:`Recommendation` / :class:`OptimizerConfig` — the advisory suggestion
  model and the thresholds driving the rule engine.

Nothing here imports a third-party dependency, so ``import memory_reuse.optimizer``
stays free of optional dependencies.
"""

from __future__ import annotations

from memory_reuse.optimizer.analyzer import (
    EffectivenessAnalyzer,
    EffectivenessReport,
    OperationTracker,
)
from memory_reuse.optimizer.recommendations import (
    OptimizerConfig,
    Recommendation,
    recommend,
)
from memory_reuse.optimizer.records import OperationRecord, OperationStats

__all__ = [
    "OperationRecord",
    "OperationStats",
    "OperationTracker",
    "EffectivenessAnalyzer",
    "EffectivenessReport",
    "Recommendation",
    "OptimizerConfig",
    "recommend",
]
