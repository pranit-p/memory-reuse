#!/usr/bin/env python3
"""Release helper for memory-reuse.

Bumps the version in ``pyproject.toml`` and ``memory_reuse/__init__.py``,
promotes the ``[Unreleased]`` section of ``CHANGELOG.md`` to the new version
with today's date, and (optionally) commits and tags the release.

Usage:

    python scripts/release.py 0.2.0            # edit files only, review, then commit yourself
    python scripts/release.py 0.2.0 --commit   # also git commit + tag
    python scripts/release.py 0.2.0 --commit --push   # also push the tag

The script is deliberately conservative: without ``--commit`` it only edits
files so you can review the diff before anything is committed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

# Repo root is the parent of the scripts/ directory.
ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_FILE = ROOT / "memory_reuse" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"

# Semantic version: MAJOR.MINOR.PATCH with optional pre-release suffix.
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-.][0-9A-Za-z.]+)?$")


def _fail(message: str) -> None:
    """Print an error and exit with a non-zero status."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _read(path: Path) -> str:
    """Read a UTF-8 text file, failing clearly if it is missing."""
    if not path.exists():
        _fail(f"expected file not found: {path}")
    return path.read_text(encoding="utf-8")


def current_version() -> str:
    """Return the version currently declared in ``pyproject.toml``."""
    text = _read(PYPROJECT)
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not match:
        _fail("could not find the version line in pyproject.toml")
    return match.group(1)  # type: ignore[union-attr]


def bump_pyproject(new_version: str) -> None:
    """Update the ``version`` field in ``pyproject.toml``."""
    text = _read(PYPROJECT)
    updated, count = re.subn(
        r'(?m)^version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        text,
        count=1,
    )
    if count != 1:
        _fail("failed to update version in pyproject.toml")
    PYPROJECT.write_text(updated, encoding="utf-8")
    print(f"  updated pyproject.toml         → {new_version}")


def bump_init(new_version: str) -> None:
    """Update ``__version__`` in ``memory_reuse/__init__.py``."""
    text = _read(INIT_FILE)
    updated, count = re.subn(
        r'(?m)^__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_version}"',
        text,
        count=1,
    )
    if count != 1:
        _fail("failed to update __version__ in memory_reuse/__init__.py")
    INIT_FILE.write_text(updated, encoding="utf-8")
    print(f"  updated memory_reuse/__init__  → {new_version}")


def update_changelog(new_version: str) -> None:
    """Promote the ``[Unreleased]`` section to a dated version section.

    The content currently under ``[Unreleased]`` is moved into the new version
    heading, and a fresh empty ``[Unreleased]`` section is inserted above it so
    future changes have a place to accumulate. Comparison link footers are
    refreshed.
    """
    text = _read(CHANGELOG)
    today = _dt.date.today().isoformat()

    if "## [Unreleased]" not in text:
        _fail("CHANGELOG.md has no '## [Unreleased]' section to promote")

    # Insert a fresh, empty [Unreleased] section above a dated heading for the
    # new version. Whatever content was under [Unreleased] now falls under the
    # new version heading — which is exactly what we want at release time.
    new_heading = (
        "## [Unreleased]\n\n"
        "_Nothing yet._\n\n"
        "---\n\n"
        f"## [{new_version}] — {today}"
    )
    text = text.replace("## [Unreleased]", new_heading, 1)

    # Refresh link footers if present (best-effort; safe to skip if absent).
    text = re.sub(
        r"(?m)^\[Unreleased\]:.*$",
        f"[Unreleased]: https://github.com/pranit-p/memory-reuse/compare/"
        f"v{new_version}...HEAD",
        text,
    )
    # Add a link for the new version right after the Unreleased link line.
    version_link = (
        f"[{new_version}]: https://github.com/pranit-p/memory-reuse/releases/"
        f"tag/v{new_version}"
    )
    if f"[{new_version}]:" not in text:
        text = re.sub(
            r"(?m)^(\[Unreleased\]:.*)$",
            rf"\1\n{version_link}",
            text,
            count=1,
        )

    CHANGELOG.write_text(text, encoding="utf-8")
    print(f"  updated CHANGELOG.md           → [{new_version}] — {today}")


def run_git(args: list[str]) -> None:
    """Run a git command, streaming output and failing on error."""
    print(f"  $ git {' '.join(args)}")
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut a new memory-reuse release.")
    parser.add_argument("version", help="new version, e.g. 0.2.0")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="git commit the changes and create an annotated tag",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="push the commit and tag to origin (implies --commit)",
    )
    args = parser.parse_args()

    new_version = args.version.lstrip("v")
    if not _SEMVER_RE.match(new_version):
        _fail(f"'{new_version}' is not a valid semantic version (expected e.g. 0.2.0)")

    old_version = current_version()
    if new_version == old_version:
        _fail(f"new version {new_version} is the same as the current version")

    print(f"Releasing memory-reuse {old_version} → {new_version}\n")

    bump_pyproject(new_version)
    bump_init(new_version)
    update_changelog(new_version)

    do_commit = args.commit or args.push
    if not do_commit:
        print(
            "\nFiles updated. Review the diff, edit the CHANGELOG entry, then:\n"
            f"  git commit -am 'release: v{new_version}'\n"
            f"  git tag v{new_version}\n"
            f"  git push origin main --tags"
        )
        return

    print()
    run_git(["commit", "-am", f"release: v{new_version}"])
    run_git(["tag", f"v{new_version}"])

    if args.push:
        run_git(["push", "origin", "main", "--tags"])
        print(f"\nPushed v{new_version}. The publish workflow will build and upload to PyPI.")
    else:
        print(
            f"\nCommitted and tagged v{new_version}. When ready, push to trigger release:\n"
            f"  git push origin main --tags"
        )


if __name__ == "__main__":
    main()
