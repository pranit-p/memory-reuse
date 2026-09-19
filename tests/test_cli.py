"""Example tests for the ``memory-reuse`` analytics CLI (Phase 5, Req 4).

Covers rendering the ``stats`` and ``savings`` views from a dumped snapshot,
zeroed rendering on an empty snapshot, and non-zero exit with a message (not a
traceback) on an unknown subcommand or an unreadable snapshot file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_reuse import CacheConfig, CacheHitEvent, MemoryCache, PricingConfig
from memory_reuse.cli import dump_snapshot, main


def _write_snapshot(tmp_path: Path, data: dict) -> str:
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_stats_renders_counters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`stats` renders hit rate and hit/miss/total counters (Req 4.1)."""
    path = _write_snapshot(
        tmp_path,
        {"hit_rate": 0.75, "hits": 3, "misses": 1, "total_requests": 4},
    )
    assert main(["stats", "--snapshot", path]) == 0
    out = capsys.readouterr().out
    assert "75.0%" in out
    assert "Hits:            3" in out
    assert "Misses:          1" in out
    assert "Total requests:  4" in out


def test_savings_renders_totals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`savings` renders tokens / cost (+currency) / latency saved (Req 4.2)."""
    path = _write_snapshot(
        tmp_path,
        {
            "hit_rate": 0.5,
            "tokens_saved": 1500,
            "cost_saved": "2.40",
            "currency": "USD",
            "latency_saved": 0.35,
        },
    )
    assert main(["savings", "--snapshot", path]) == 0
    out = capsys.readouterr().out
    assert "Tokens saved:    1500" in out
    assert "Cost saved:      2.40 USD" in out
    assert "Latency saved:   0.350s" in out


def test_empty_snapshot_renders_zeros(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An empty snapshot renders zeros, never an error (Req 4.3)."""
    path = _write_snapshot(tmp_path, {})
    assert main(["savings", "--snapshot", path]) == 0
    out = capsys.readouterr().out
    assert "Tokens saved:    0" in out
    assert "0.0%" in out


def test_unknown_subcommand_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown subcommand exits non-zero with a usage message (Req 4.5)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus", "--snapshot", "x.json"])
    assert excinfo.value.code != 0


def test_missing_file_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """A missing snapshot file exits non-zero with a named message, no traceback (Req 4.5)."""
    code = main(["stats", "--snapshot", "/nonexistent/does-not-exist.json"])
    assert code == 1
    err = capsys.readouterr().err
    assert "memory-reuse:" in err
    assert "does-not-exist.json" in err


def test_dump_snapshot_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`dump_snapshot` produces a file the CLI renders with real values (Req 4.4)."""
    cache = MemoryCache(
        CacheConfig(
            backend="memory",
            pricing=PricingConfig(input_token_price=1e-3, output_token_price=2e-3),
        )
    )
    cache.record_hit_event(CacheHitEvent(tokens_in=1000, tokens_out=500, latency_saved=0.4))

    path = tmp_path / "dump.json"
    dump_snapshot(cache, path)

    assert main(["savings", "--snapshot", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Tokens saved:    1500" in out
    # (1000 * 1e-3) + (500 * 2e-3) = 1.0 + 1.0 = 2.00
    assert "Cost saved:      2.00 USD" in out
