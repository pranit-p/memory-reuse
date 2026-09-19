"""Lazy-import / core-import smoke tests for the Phase 5 analytics layer.

Covers the cross-cutting optionality requirements for Phase 5 (Req 7):

* a bare ``import memory_reuse`` and ``import memory_reuse.analytics`` succeed
  and pull in neither ``prometheus_client`` nor ``opentelemetry`` (Req 7.1, 7.2),
  verified in a fresh subprocess so ``sys.modules`` pollution from the rest of
  the session cannot mask a leak;
* the exporter modules import even with their dependency simulated absent, and
  the dependency is only demanded (as a named ``BackendNotAvailableError``) on
  first construction/use (Req 7.3, 7.4);
* the analytics tracker + snapshot work with both exporter deps absent
  (Req 7.5).

Fully offline: no network, no real exporter backend required.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from memory_reuse.exceptions import BackendNotAvailableError

# The optional Phase 5 exporter dependencies that must stay lazily imported.
_OPTIONAL_DEPS = ("prometheus_client", "opentelemetry")


class TestBareImportLoadsNoExporterDep:
    """Req 7.1, 7.2: bare imports pull in neither exporter dependency."""

    def test_core_and_analytics_import_load_no_exporter_dep(self) -> None:
        """A fresh interpreter importing the package + analytics loads neither dep."""
        script = (
            "import sys\n"
            "import memory_reuse\n"
            "import memory_reuse.analytics\n"
            "import memory_reuse.analytics.exporters\n"
            + "".join(
                f'assert "{dep}" not in sys.modules, "{dep} imported by bare import"\n'
                for dep in _OPTIONAL_DEPS
            )
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "bare import pulled in an exporter dependency:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


class TestExporterModulesImportWithoutDep:
    """Req 7.3: exporter modules import even with their dependency absent."""

    @pytest.mark.parametrize("dep", _OPTIONAL_DEPS)
    def test_reimport_exporter_modules_with_dep_absent(
        self, monkeypatch: pytest.MonkeyPatch, dep: str
    ) -> None:
        # A ``None`` entry in sys.modules makes ``import <dep>`` raise
        # ImportError, simulating the dependency being uninstalled.
        monkeypatch.setitem(sys.modules, dep, None)
        for name in (
            "memory_reuse.analytics.exporters.prometheus",
            "memory_reuse.analytics.exporters.opentelemetry",
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
            __import__(name)


class TestExporterDepRequiredOnUse:
    """Req 7.4: exporter guards raise a named error when the dep is absent."""

    def test_prometheus_guard_named_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "prometheus_client", None)
        from memory_reuse.analytics.exporters import prometheus as prom_mod

        with pytest.raises(BackendNotAvailableError) as exc_info:
            prom_mod._require_prometheus()
        assert "memory-reuse[prometheus]" in str(exc_info.value)

    def test_opentelemetry_guard_named_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "opentelemetry", None)
        from memory_reuse.analytics.exporters import opentelemetry as otel_mod

        with pytest.raises(BackendNotAvailableError) as exc_info:
            otel_mod._require_opentelemetry()
        assert "memory-reuse[opentelemetry]" in str(exc_info.value)


class TestAnalyticsWorksWithoutExporterDeps:
    """Req 7.5: tracking + snapshot function with both exporter deps absent."""

    def test_tracker_and_snapshot_work_with_deps_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for dep in _OPTIONAL_DEPS:
            monkeypatch.setitem(sys.modules, dep, None)

        from memory_reuse.analytics import (
            AnalyticsTracker,
            CacheHitEvent,
            PricingConfig,
        )
        from memory_reuse.stats import StatsTracker

        tracker = AnalyticsTracker(
            StatsTracker(),
            pricing=PricingConfig(input_token_price=1e-3, output_token_price=1e-3),
        )
        tracker.record_hit_event(CacheHitEvent(tokens_in=1000, tokens_out=1000, latency_saved=0.5))
        snap = tracker.snapshot()
        assert snap.tokens_saved == 2000
        assert snap.latency_saved == 0.5
        assert float(snap.cost_saved) == 2.0
