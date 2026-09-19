"""Packaging tests for the Phase 5 exporter extras and CLI entry (Req 9).

Static, offline assertions over ``pyproject.toml`` confirming:

* the two Phase 5 exporter extras (``prometheus``, ``opentelemetry``) are
  declared, each a single lowercase word/hyphen token (Req 9.1),
* both are folded into the ``all`` extra (Req 9.2),
* neither leaks into the core required-dependency list (Req 9.3),
* each extra is independent of the other's dependency (Req 9.4),
* the ``memory-reuse`` console-script entry point is declared (Req 4).

Pure parse of ``pyproject.toml`` — no install, network, or subprocess.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on <3.11
    import tomli as tomllib

# The two Phase 5 exporter extras, mapped to the distribution each pulls in.
PHASE5_EXTRAS: dict[str, tuple[str, ...]] = {
    "prometheus": ("prometheus-client",),
    "opentelemetry": ("opentelemetry-api", "opentelemetry-sdk"),
}

_EXTRA_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("could not locate pyproject.toml above the test file")


def _load_pyproject() -> dict[str, Any]:
    with (_repo_root() / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _requirement_name(requirement: str) -> str:
    match = _REQ_NAME_RE.match(requirement)
    return match.group(1).lower() if match else ""


def _all_referenced_extras(all_requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in all_requirements:
        for bracketed in re.findall(r"\[([^\]]*)\]", requirement):
            names.update(n.strip() for n in bracketed.split(",") if n.strip())
    return names


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return _load_pyproject()


@pytest.fixture(scope="module")
def optional_deps(pyproject: dict[str, Any]) -> dict[str, list[str]]:
    return pyproject["project"]["optional-dependencies"]


class TestPhase5ExtrasExist:
    """Req 9.1: each Phase 5 exporter extra is declared in proper form."""

    def test_both_extras_declared(self, optional_deps: dict[str, list[str]]) -> None:
        for extra in PHASE5_EXTRAS:
            assert extra in optional_deps, f"missing Phase 5 extra: {extra!r}"

    @pytest.mark.parametrize("extra", sorted(PHASE5_EXTRAS))
    def test_extra_name_is_declaration_form(self, extra: str) -> None:
        assert _EXTRA_NAME_RE.match(extra), f"extra name not in declaration form: {extra!r}"

    def test_each_extra_declares_its_dependency(self, optional_deps: dict[str, list[str]]) -> None:
        for extra, dists in PHASE5_EXTRAS.items():
            names = {_requirement_name(req) for req in optional_deps[extra]}
            for dist in dists:
                assert dist in names, f"extra {extra!r} does not declare {dist!r}"


class TestAllFoldsInPhase5:
    """Req 9.2: the ``all`` extra installs both Phase 5 extras."""

    def test_all_references_each_phase5_extra(self, optional_deps: dict[str, list[str]]) -> None:
        referenced = _all_referenced_extras(optional_deps["all"])
        for extra in PHASE5_EXTRAS:
            assert extra in referenced, f"{extra!r} not folded into 'all': {sorted(referenced)}"


class TestCoreExcludesPhase5:
    """Req 9.3: no Phase 5 optional dependency leaks into core dependencies."""

    def test_core_dependencies_is_empty(self, pyproject: dict[str, Any]) -> None:
        assert pyproject["project"].get("dependencies", []) == []


class TestPhase5ExtrasIndependent:
    """Req 9.4: each Phase 5 extra pulls in only its own dependency."""

    def test_prometheus_does_not_require_otel(self, optional_deps: dict[str, list[str]]) -> None:
        names = {_requirement_name(req) for req in optional_deps["prometheus"]}
        assert not (names & {"opentelemetry-api", "opentelemetry-sdk"})

    def test_otel_does_not_require_prometheus(self, optional_deps: dict[str, list[str]]) -> None:
        names = {_requirement_name(req) for req in optional_deps["opentelemetry"]}
        assert "prometheus-client" not in names


class TestCliScriptEntry:
    """Req 4: the ``memory-reuse`` console script is declared."""

    def test_memory_reuse_script_declared(self, pyproject: dict[str, Any]) -> None:
        scripts = pyproject["project"].get("scripts", {})
        assert scripts.get("memory-reuse") == "memory_reuse.cli:main"
