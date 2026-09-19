"""OpenTelemetry exporter for cache analytics (Phase 5, Req 6).

:class:`OpenTelemetryExporter` publishes the analytics snapshot as OpenTelemetry
**observable gauges** — one instrument per metric — whose callbacks read the
snapshot at collection time, so exported values always reflect the current state
(Req 6.4).

``opentelemetry`` is an **optional** dependency, imported lazily inside
:func:`_require_opentelemetry` (called from the constructor). The module imports
without ``opentelemetry`` installed; construction raises
:class:`~memory_reuse.exceptions.BackendNotAvailableError` naming the extra when
it is missing (Req 6.7, 7).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from memory_reuse.exceptions import BackendNotAvailableError

if TYPE_CHECKING:
    from memory_reuse.analytics.snapshot import AnalyticsSnapshot

logger = logging.getLogger(__name__)

_INSTRUMENT_NAME = "memory_reuse"


def _require_opentelemetry() -> Any:
    """Return the ``opentelemetry.metrics`` module, raising a clear error if absent.

    Returns:
        The imported ``opentelemetry.metrics`` module.

    Raises:
        BackendNotAvailableError: If ``opentelemetry`` is not installed, naming
            the extra to install (Req 6.7).
    """
    try:
        from opentelemetry import metrics  # noqa: PLC0415

        return metrics
    except ImportError as exc:
        raise BackendNotAvailableError(
            "OpenTelemetry export requires opentelemetry. Install it with "
            '`pip install "memory-reuse[opentelemetry]"`.'
        ) from exc


class OpenTelemetryExporter:
    """Publishes cache analytics as OpenTelemetry observable gauges.

    One observable gauge is registered per metric (hit rate, tokens saved, cost
    saved, latency saved); each gauge's callback reads the current snapshot at
    collection time (Req 6.1, 6.4). On empty state every metric reports ``0``
    (Req 6.5), and a failure in one callback omits only that instrument's
    measurement (Req 6.6).

    Args:
        snapshot_provider: Callable returning the current
            :class:`~memory_reuse.analytics.AnalyticsSnapshot`.
        meter_provider: Optional OpenTelemetry ``MeterProvider`` to register
            instruments with. When ``None`` the globally configured provider is
            used (Req 6.2, 6.3).

    Raises:
        BackendNotAvailableError: If ``opentelemetry`` is not installed.

    Example::

        exporter = OpenTelemetryExporter(lambda: cache.analytics)
    """

    def __init__(
        self,
        snapshot_provider: Callable[[], AnalyticsSnapshot],
        *,
        meter_provider: Any | None = None,
    ) -> None:
        metrics = _require_opentelemetry()
        self._snapshot_provider = snapshot_provider

        provider = meter_provider if meter_provider is not None else metrics.get_meter_provider()
        meter = provider.get_meter(_INSTRUMENT_NAME)

        Observation = metrics.Observation

        def _observe(read: Callable[[AnalyticsSnapshot], float]) -> Callable[[Any], list[Any]]:
            """Build an observable-gauge callback that reads the snapshot safely."""

            def callback(_options: Any) -> list[Any]:
                try:
                    value = read(self._snapshot_provider())
                except Exception:  # noqa: BLE001 — omit only this instrument (Req 6.6)
                    logger.debug("OpenTelemetryExporter: instrument reading omitted")
                    return []
                return [Observation(value)]

            return callback

        self._instruments = [
            meter.create_observable_gauge(
                "memory_reuse.hit_rate",
                callbacks=[_observe(lambda s: float(s.hit_rate))],
                description="Cache hit rate in [0,1].",
            ),
            meter.create_observable_gauge(
                "memory_reuse.tokens_saved",
                callbacks=[_observe(lambda s: float(s.tokens_saved))],
                description="Total tokens saved by cache hits.",
            ),
            meter.create_observable_gauge(
                "memory_reuse.cost_saved",
                callbacks=[_observe(lambda s: float(s.cost_saved))],
                description="Total cost saved by cache hits.",
            ),
            meter.create_observable_gauge(
                "memory_reuse.latency_saved_seconds",
                callbacks=[_observe(lambda s: float(s.latency_saved))],
                description="Total latency saved (seconds).",
            ),
        ]
