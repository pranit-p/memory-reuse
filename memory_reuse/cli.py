"""``memory-reuse`` command-line interface for cache analytics (Phase 5, Req 4).

The CLI renders a previously dumped analytics snapshot. It runs **out of
process** from the cache: a running application writes a snapshot file (via
:func:`dump_snapshot`), and the CLI reads that file and renders it. This keeps
the CLI a thin, standard-library-only viewer with no access to the live cache
and no optional dependency (Req 4.6).

Subcommands::

    memory-reuse stats    --snapshot <file.json>   # hit rate + hit/miss counters
    memory-reuse savings  --snapshot <file.json>   # tokens / cost / latency saved

The snapshot file is the JSON dict written by :func:`dump_snapshot`, which
combines :meth:`~memory_reuse.stats.CacheStats.to_dict` (for the ``stats`` view's
``hits`` / ``misses`` / ``total_requests``) with
:meth:`~memory_reuse.analytics.AnalyticsSnapshot.to_dict` (for the savings view).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_reuse.core import MemoryCache


def dump_snapshot(cache: MemoryCache, path: str | Path) -> None:
    """Write a cache's combined stats + analytics snapshot to a JSON file.

    The written dict merges the core :class:`~memory_reuse.stats.CacheStats`
    fields (``hits`` / ``misses`` / ``total_requests`` / …) with the
    :class:`~memory_reuse.analytics.AnalyticsSnapshot` fields (``tokens_saved`` /
    ``cost_saved`` / ``currency`` / ``latency_saved``). The ``hit_rate`` from the
    analytics snapshot is authoritative and overwrites the stats one (they are
    equal by construction).

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` to snapshot.
        path: Destination file path for the JSON snapshot.
    """
    combined: dict[str, Any] = {}
    combined.update(cache.stats.to_dict())
    combined.update(cache.analytics.to_dict())
    Path(path).write_text(json.dumps(combined, indent=2), encoding="utf-8")


def _load_snapshot(path: str) -> dict[str, Any]:
    """Read and parse a snapshot file, raising ``FileNotFoundError`` / ``ValueError``.

    Args:
        path: Path to the JSON snapshot file.

    Returns:
        The parsed snapshot dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is not a JSON object.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"could not read snapshot file '{path}': {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"snapshot file '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"snapshot file '{path}' must contain a JSON object")
    return data


def _render_stats(snap: dict[str, Any]) -> str:
    """Render the ``stats`` view from a snapshot dict."""
    hit_rate = float(snap.get("hit_rate", 0.0) or 0.0)
    hits = int(snap.get("hits", 0) or 0)
    misses = int(snap.get("misses", 0) or 0)
    total = int(snap.get("total_requests", 0) or 0)
    return (
        "Cache Statistics\n"
        "----------------\n"
        f"Hit rate:        {hit_rate:.1%}\n"
        f"Hits:            {hits}\n"
        f"Misses:          {misses}\n"
        f"Total requests:  {total}"
    )


def _render_savings(snap: dict[str, Any]) -> str:
    """Render the ``savings`` view from a snapshot dict."""
    hit_rate = float(snap.get("hit_rate", 0.0) or 0.0)
    tokens = int(snap.get("tokens_saved", 0) or 0)
    cost = snap.get("cost_saved", "0")
    currency = snap.get("currency", "USD")
    latency = float(snap.get("latency_saved", 0.0) or 0.0)
    return (
        "Cache Savings\n"
        "-------------\n"
        f"Hit rate:        {hit_rate:.1%}\n"
        f"Tokens saved:    {tokens}\n"
        f"Cost saved:      {cost} {currency}\n"
        f"Latency saved:   {latency:.3f}s"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``stats`` and ``savings`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="memory-reuse",
        description="View memory-reuse cache analytics from a dumped snapshot.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("stats", "Show hit rate and hit/miss counters."),
        ("savings", "Show tokens, cost, and latency saved."),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument(
            "--snapshot",
            required=True,
            metavar="FILE",
            help="Path to a JSON snapshot written by dump_snapshot().",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (Req 4).

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit status: ``0`` on success, non-zero on a usage or file error.
    """
    parser = _build_parser()
    # argparse exits with status 2 on an unknown/missing subcommand, printing a
    # usage message to stderr (Req 4.5) — no traceback.
    args = parser.parse_args(argv)

    try:
        snap = _load_snapshot(args.snapshot)
    except (FileNotFoundError, ValueError) as exc:
        print(f"memory-reuse: {exc}", file=sys.stderr)
        return 1

    if args.command == "stats":
        print(_render_stats(snap))
    else:  # "savings" — the only other registered subcommand
        print(_render_savings(snap))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
