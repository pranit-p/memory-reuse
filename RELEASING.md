# Releasing

This project publishes to PyPI automatically via GitHub Actions using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API
tokens are stored as secrets.

## Release policy

- Merging a PR to `main` does **not** trigger a release. Changes accumulate on
  `main` freely.
- A release happens **only** when a maintainer pushes a version tag (`vX.Y.Z`).
- Release when meaningful changes have landed — a notable feature, fix, or a
  batch of merged PRs — not for every small commit.
- Follow [Semantic Versioning](https://semver.org/): patch for fixes, minor for
  backward-compatible features, major for breaking changes.

## One-time setup

1. Create the project on PyPI (done — first release uploaded manually).
2. On PyPI, go to the project's **Publishing** settings and add a trusted
   publisher:
   - Owner: `pranit-p`
   - Repository: `memory-reuse`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In the GitHub repo settings, create an environment named `pypi`.
4. (Optional but recommended) In the `pypi` environment settings, add yourself
   as a **required reviewer** so every publish pauses for a manual approval
   click before uploading to PyPI.

## Cutting a release

Use the release helper script — it bumps the version in both files, updates
the CHANGELOG, and (optionally) commits, tags, and pushes.

```bash
# 1. Edit files only, so you can review the diff first:
python scripts/release.py 0.2.0

# 2. Review the diff and tidy the CHANGELOG entry, then commit + tag + push:
git commit -am "release: v0.2.0"
git tag v0.2.0
git push origin main --tags
```

Or do it all in one step:

```bash
python scripts/release.py 0.2.0 --commit --push
```

The `publish.yml` workflow then runs tests, builds the sdist + wheel, checks the
metadata, and uploads to PyPI automatically.

### Manual alternative

If you prefer to edit by hand, bump the version in **two** places
(`pyproject.toml` → `version`, `memory_reuse/__init__.py` → `__version__`),
move `[Unreleased]` items into a dated section in `CHANGELOG.md`, then commit,
tag, and push as above.

## Manual build (for local testing)

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
# Optional: upload to TestPyPI first
# twine upload --repository testpypi dist/*
```
