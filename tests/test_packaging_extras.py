"""Packaging tests for the Phase 4 integration/backend extras.

Task 8.3 (Reqs 12.1, 12.2, 12.3, 12.4, 12.5, 12.6): static assertions over the
project's ``pyproject.toml`` confirming the three Phase 4 integration/backend
Packaging_Extras — ``strands``, ``crewai``, and ``agentcore`` — are declared
correctly, are folded into the ``all`` extra, are kept out of the core
required-dependency list, and are mutually independent.

Scope note: the full analytics-and-integrations spec declares five Phase 4
extras (adding ``prometheus`` and ``opentelemetry``), but those two are deferred
to Phase 5. This file therefore asserts only over the three
integration/backend extras this phase ships; it does *not* require the Phase 5
extras to be present, so it stays green both before and after Phase 5 lands.

Everything here is a pure, offline parse of ``pyproject.toml`` via ``tomllib``
(Python 3.11+ stdlib) — no install, network, or subprocess.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import tomllib
from hypothesis import given
from hypothesis import strategies as st

# The three Phase 4 integration/backend extras this phase ships, mapped to the
# single third-party distribution each one is expected to pull in. This mirrors
# the current ``[project.optional-dependencies]`` declarations and is the source
# of truth for the independence checks below.
PHASE4_EXTRAS: dict[str, str] = {
    "strands": "strands-agents",
    "crewai": "crewai",
    "agentcore": "boto3",
}

# The distribution names owned by *other* Phase 4 extras — used to prove one
# extra never drags in another's dependency (Req 12.4, 12.5).
PHASE4_DEP_NAMES: set[str] = set(PHASE4_EXTRAS.values())

# A single lowercase word/hyphen extra name, matching the existing declaration
# form (``redis``, ``semantic-local``, ``langgraph`` ...).
_EXTRA_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# Matches the leading distribution name of a PEP 508 requirement string, e.g.
# ``strands-agents>=0.1.0`` -> ``strands-agents``. We only need the name here,
# not the full specifier grammar.
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _repo_root() -> Path:
    """Locate the repo root (the dir holding ``pyproject.toml``) from this file.

    Walk upward from the test file rather than assuming a fixed depth, so the
    test keeps working regardless of the working directory pytest is run from.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("could not locate pyproject.toml above the test file")


def _load_pyproject() -> dict[str, Any]:
    with (_repo_root() / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _requirement_name(requirement: str) -> str:
    """Return the lowercased distribution name from a PEP 508 requirement.

    Distribution names are case-insensitive and treat ``-``/``_``/``.`` as
    equivalent; we normalise to lowercase so comparisons are robust.
    """
    match = _REQ_NAME_RE.match(requirement)
    if match is None:  # pragma: no cover - defensive; declarations are well-formed
        return ""
    return match.group(1).lower()


def _parse_all_referenced_extras(all_requirements: list[str]) -> set[str]:
    """Extract the extra names referenced by the ``all`` extra.

    The ``all`` extra self-references the project with a bracketed extras list,
    e.g. ``memory-reuse[redis,litellm,...]``; return the set of names inside the
    brackets across every such requirement.
    """
    names: set[str] = set()
    for requirement in all_requirements:
        for bracketed in re.findall(r"\[([^\]]*)\]", requirement):
            for name in bracketed.split(","):
                stripped = name.strip()
                if stripped:
                    names.add(stripped)
    return names


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return _load_pyproject()


@pytest.fixture(scope="module")
def optional_deps(pyproject: dict[str, Any]) -> dict[str, list[str]]:
    return pyproject["project"]["optional-dependencies"]


class TestPhase4ExtrasExist:
    """Req 12.1: each Phase 4 integration/backend extra is declared."""

    def test_all_three_extras_declared(self, optional_deps: dict[str, list[str]]) -> None:
        for extra in PHASE4_EXTRAS:
            assert extra in optional_deps, f"missing Phase 4 extra: {extra!r}"

    def test_extras_are_distinct(self, optional_deps: dict[str, list[str]]) -> None:
        # Three distinct keys, and each maps to a distinct dependency.
        assert len(PHASE4_EXTRAS) == 3
        assert len(set(PHASE4_EXTRAS.values())) == 3

    def test_each_extra_declares_its_expected_dependency(
        self, optional_deps: dict[str, list[str]]
    ) -> None:
        for extra, dist in PHASE4_EXTRAS.items():
            names = {_requirement_name(req) for req in optional_deps[extra]}
            assert (
                dist in names
            ), f"extra {extra!r} does not declare {dist!r}: {optional_deps[extra]}"


class TestExtraNamesMatchDeclarationForm:
    """Req 12.6: each extra name is a single lowercase word/hyphen token."""

    @pytest.mark.parametrize("extra", sorted(PHASE4_EXTRAS))
    def test_extra_name_is_lowercase_word_or_hyphen(self, extra: str) -> None:
        assert _EXTRA_NAME_RE.match(extra), f"extra name not in declaration form: {extra!r}"

    def test_names_match_existing_extra_form(self, optional_deps: dict[str, list[str]]) -> None:
        # Sanity: the Phase 4 names follow the same shape as the pre-existing
        # extras (``redis``, ``langgraph``, ``semantic-local`` ...), so every
        # declared extra key is a valid lowercase word/hyphen token.
        for extra in optional_deps:
            assert _EXTRA_NAME_RE.match(extra), f"unexpected extra name form: {extra!r}"


class TestAllExtraFoldsInPhase4:
    """Req 12.2: the ``all`` extra installs each Phase 4 extra."""

    def test_all_extra_exists(self, optional_deps: dict[str, list[str]]) -> None:
        assert "all" in optional_deps

    def test_all_references_each_phase4_extra(self, optional_deps: dict[str, list[str]]) -> None:
        referenced = _parse_all_referenced_extras(optional_deps["all"])
        for extra in PHASE4_EXTRAS:
            assert (
                extra in referenced
            ), f"{extra!r} not folded into 'all' extra; found: {sorted(referenced)}"

    def test_all_self_references_the_project(
        self, pyproject: dict[str, Any], optional_deps: dict[str, list[str]]
    ) -> None:
        project_name = pyproject["project"]["name"].lower()
        names = {_requirement_name(req) for req in optional_deps["all"]}
        assert project_name in names, f"'all' extra does not self-reference {project_name!r}"


class TestCoreDependenciesExcludePhase4:
    """Req 12.3: no Phase 4 optional dependency leaks into core dependencies."""

    def test_core_dependencies_contain_no_phase4_dep(self, pyproject: dict[str, Any]) -> None:
        core_deps = pyproject["project"].get("dependencies", [])
        core_names = {_requirement_name(req) for req in core_deps}
        leaked = core_names & PHASE4_DEP_NAMES
        assert not leaked, f"Phase 4 deps leaked into core dependencies: {sorted(leaked)}"

    def test_core_dependencies_is_empty(self, pyproject: dict[str, Any]) -> None:
        # The project currently ships a zero-dependency core; guard that so a
        # future accidental addition of any runtime dep is caught here.
        assert pyproject["project"].get("dependencies", []) == []


class TestExtrasAreIndependent:
    """Req 12.4, 12.5: installing one Phase 4 extra pulls in no other's dep."""

    @pytest.mark.parametrize("extra", sorted(PHASE4_EXTRAS))
    def test_extra_does_not_require_another_phase4_dep(
        self, extra: str, optional_deps: dict[str, list[str]]
    ) -> None:
        own_dep = PHASE4_EXTRAS[extra]
        other_deps = PHASE4_DEP_NAMES - {own_dep}
        declared = {_requirement_name(req) for req in optional_deps[extra]}
        intruders = declared & other_deps
        assert (
            not intruders
        ), f"extra {extra!r} pulls in another Phase 4 extra's dependency: {sorted(intruders)}"

    @pytest.mark.parametrize("extra", sorted(PHASE4_EXTRAS))
    def test_extra_requires_only_its_own_phase4_dep(
        self, extra: str, optional_deps: dict[str, list[str]]
    ) -> None:
        # Each of the three extras is a single, self-contained dependency: it
        # declares exactly its own distribution and does not self-reference the
        # project (i.e. it is a leaf, unlike ``all`` / ``semantic-local``).
        declared = {_requirement_name(req) for req in optional_deps[extra]}
        assert declared == {
            PHASE4_EXTRAS[extra]
        }, f"extra {extra!r} should declare only {PHASE4_EXTRAS[extra]!r}, got {sorted(declared)}"


class TestExtraIndependenceProperty:
    """Property-flavoured independence check over Phase 4 extra pairs.

    Validates: Requirements 12.4, 12.5

    For every ordered pair of distinct Phase 4 extras, the first extra's
    requirement list must not contain the second extra's dependency. This is the
    same invariant as ``TestExtrasAreIndependent`` expressed exhaustively over
    all pairs, so no extra can ever require another Phase 4 extra's dep.
    """

    @given(pair=st.sampled_from([(a, b) for a in PHASE4_EXTRAS for b in PHASE4_EXTRAS if a != b]))
    def test_no_cross_extra_dependency(self, pair: tuple[str, str]) -> None:
        first, second = pair
        optional_deps = _load_pyproject()["project"]["optional-dependencies"]
        declared = {_requirement_name(req) for req in optional_deps[first]}
        assert PHASE4_EXTRAS[second] not in declared, (
            f"extra {first!r} must not require {second!r}'s dependency "
            f"{PHASE4_EXTRAS[second]!r}"
        )
