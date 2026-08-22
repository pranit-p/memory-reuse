# Contributing to memory-reuse

Thanks for your interest in contributing. Here's how to get started.

---

## Development setup

```bash
# Clone the repo
git clone https://github.com/pranit-p/memory-reuse.git
cd memory-reuse

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Running tests

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=memory_reuse --cov-report=term-missing
```

Run only unit tests:

```bash
pytest tests/unit/
```

---

## Code style

We use [black](https://black.readthedocs.io/) for formatting and [ruff](https://docs.astral.sh/ruff/) for linting.

```bash
black memory_reuse tests
ruff check memory_reuse tests
```

All CI checks must pass before a PR can be merged.

---

## Guidelines

- **Docstrings**: every public class and function needs a Google-style docstring.
- **Type hints**: required everywhere — no `Any` unless genuinely unavoidable.
- **Tests**: new features and bug fixes must include tests. Aim to keep coverage above 90%.
- **No `print()`**: use the `logging` module. Never log actual cached values or secrets.
- **File length**: keep source files under 200 lines. Split if needed.
- **Dependencies**: do not add new runtime dependencies without a discussion issue first. The core package intentionally has zero mandatory dependencies.

---

## PR guidelines

1. Open an issue before starting significant work so we can align on the approach.
2. Branch off `main`, use a descriptive branch name like `feat/redis-cluster` or `fix/ttl-overflow`.
3. Keep commits focused. Use conventional commit messages (`feat:`, `fix:`, `docs:`, etc.).
4. Fill in the PR template — describe what changed, why, and how it was tested.
5. PRs require at least one approving review and a passing CI run.

---

## Project layout

```
memory_reuse/        ← library source
tests/
  unit/              ← fast, no I/O
  integration/       ← tests that combine multiple modules
examples/            ← runnable examples
```
