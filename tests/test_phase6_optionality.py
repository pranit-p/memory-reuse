"""Optionality / core-import smoke tests for Phase 6 (Req 7).

The in-process single-flight, version identity, observation tracker, and analyzer
are standard library only, so a bare ``import memory_reuse`` plus
``import memory_reuse.execution`` and ``import memory_reuse.optimizer`` must load
no third-party dependency. Verified in a fresh subprocess so ``sys.modules``
pollution from the rest of the session cannot mask a leak.
"""

from __future__ import annotations

import subprocess
import sys

# Third-party deps that Phase 6 must NOT pull in on a bare import.
_OPTIONAL_DEPS = ("redis", "prometheus_client", "opentelemetry", "boto3")


def test_phase6_packages_import_with_no_third_party_dep() -> None:
    """Bare core + execution + optimizer imports load none of the optional deps."""
    script = (
        "import sys\n"
        "import memory_reuse\n"
        "import memory_reuse.execution\n"
        "import memory_reuse.optimizer\n"
        + "".join(
            f'assert "{dep}" not in sys.modules, "{dep} imported by a bare import"\n'
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
        "a Phase 6 import pulled in an optional dependency:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
