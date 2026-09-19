"""Actionable cache recommendations from per-operation statistics (Phase 6, Req 6).

:class:`Recommendation` is one advisory suggestion — a target operation, a
recommended action, a human-readable reason, and a confidence. The
:func:`recommend` rule engine turns :class:`~memory_reuse.optimizer.records.OperationStats`
into zero or more recommendations using deterministic thresholds.

Recommendations are **advisory only**: producing or reading them never changes
cache behaviour. The "already cached" and "volatile" signals come from the
caller (flags on the recorded observation), because the analyzer cannot infer a
tool's data-freshness on its own — keeping recommendations grounded rather than
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory_reuse.optimizer.records import OperationStats


@dataclass(frozen=True)
class OptimizerConfig:
    """Thresholds controlling the recommendation rules.

    Attributes:
        min_calls: Minimum recorded calls before any recommendation is emitted
            for an operation (below this, data is deemed insufficient).
        high_repeat_ratio: Repeat-ratio at or above which an uncached, repetitive
            operation is recommended for caching.
        low_hit_rate: Hit-rate below which an already-cached operation is
            recommended for a TTL/key adjustment.
    """

    min_calls: int = 20
    high_repeat_ratio: float = 0.5
    low_hit_rate: float = 0.2


@dataclass(frozen=True)
class Recommendation:
    """An advisory cache recommendation for one operation (Req 6.1)."""

    operation: str
    action: str
    reason: str
    confidence: float

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for serialisation (Req 6.7)."""
        return {
            "operation": self.operation,
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
        }


def _clamp(value: float) -> float:
    """Clamp a confidence to ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, value))


def recommend(stats: OperationStats, config: OptimizerConfig) -> Recommendation | None:
    """Produce at most one recommendation for an operation's statistics (Req 6).

    Rules, in priority order:

    * **Insufficient data** (`calls < min_calls`) → no recommendation (Req 6.6).
    * **Volatile** operation → recommend against semantic caching (Req 6.4).
    * **Uncached + highly repetitive** → recommend enabling caching (Req 6.2).
    * **Cached + low hit rate** → recommend a TTL/key adjustment (Req 6.3).

    Args:
        stats: The per-operation aggregate.
        config: The threshold configuration.

    Returns:
        A :class:`Recommendation`, or ``None`` when no rule fires.
    """
    if stats.calls < config.min_calls:
        return None  # Req 6.6 — do not guess on thin data.

    # Req 6.4: volatile data — similarity does not guarantee equivalence.
    if stats.volatile:
        return Recommendation(
            operation=stats.operation,
            action="avoid_semantic",
            reason=(
                "Operation is flagged as returning rapidly-changing or "
                "non-deterministic data; semantic similarity does not guarantee "
                "result equivalence, so semantic caching is unsafe here."
            ),
            confidence=0.9,
        )

    # Req 6.2: not yet cached but inputs repeat often — caching would pay off.
    if not stats.cached and stats.repeat_ratio >= config.high_repeat_ratio:
        pct = round(stats.repeat_ratio * 100)
        # Confidence scales with how far the repeat ratio clears the threshold.
        confidence = _clamp(0.5 + (stats.repeat_ratio - config.high_repeat_ratio))
        return Recommendation(
            operation=stats.operation,
            action="enable_cache",
            reason=(
                f"{pct}% of calls repeat identical inputs over {stats.calls} "
                "calls; enabling caching would avoid the repeated work."
            ),
            confidence=confidence,
        )

    # Req 6.3: already cached but rarely hitting — adjust TTL or narrow the key.
    if stats.cached and stats.hit_rate < config.low_hit_rate:
        pct = round(stats.hit_rate * 100)
        confidence = _clamp(0.5 + (config.low_hit_rate - stats.hit_rate))
        return Recommendation(
            operation=stats.operation,
            action="increase_ttl",
            reason=(
                f"Hit rate is only {pct}% over {stats.calls} calls; consider "
                "increasing the TTL or narrowing the cache key so more calls hit."
            ),
            confidence=confidence,
        )

    return None
