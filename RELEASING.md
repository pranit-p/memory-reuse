# Releasing

This project publishes to PyPI automatically via GitHub Actions using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API
tokens are stored as secrets.

## One-time setup

1. Create the project on PyPI (or reserve the name with a first manual upload).
2. On PyPI, go to the project's **Publishing** settings and add a trusted
   publisher:
   - Owner: `pranit-p`
   - Repository: `memory-reuse`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In the GitHub repo settings, create an environment named `pypi`.

## Cutting a release

1. Bump the version in **two** places:
   - `pyproject.toml` → `[project] version`
   - `memory_reuse/__init__.py` → `__version__`
2. Move items from `[Unreleased]` to a new version section in `CHANGELOG.md`
   with today's date.
3. Commit and tag:
   ```bash
   git commit -am "release: v0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```
4. The `publish.yml` workflow builds the sdist + wheel, runs `twine check`, and
   uploads to PyPI.

## Manual build (for local testing)

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
# Optional: upload to TestPyPI first
# twine upload --repository testpypi dist/*
```
