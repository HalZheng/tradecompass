"""M0：init 的目录、文件复制与「绝不覆盖」语义（SPEC §3.3）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tradecompass.bootstrap import initialize
from tradecompass.storage.db import SCHEMA_VERSION, connect, user_version

EXPECTED_STRATEGIES = [
    "b1_pullback.yaml",
    "base_rebalance.yaml",
    "legacy_cleanup.yaml",
    "volume_breakout.yaml",
]


def test_creates_expected_layout(home: Path) -> None:
    for name in ("config", "config/strategies", "data", "reports", "logs"):
        assert (home / name).is_dir(), name
    assert (home / ".env").is_file()
    assert (home / "data" / "tradecompass.sqlite").is_file()


def test_copies_every_template_and_strips_example_suffix(home: Path) -> None:
    config_dir = home / "config"
    for name in (
        "system.yaml",
        "universe.yaml",
        "risk.yaml",
        "market.yaml",
        "market_view.yaml",
        "backtest.yaml",
        "broker.yaml",
    ):
        assert (config_dir / name).is_file(), name
        assert not (config_dir / name.replace(".yaml", ".example.yaml")).exists()

    strategies = sorted(p.name for p in (config_dir / "strategies").glob("*.yaml"))
    assert strategies == EXPECTED_STRATEGIES


def test_never_overwrites_existing_files(home: Path) -> None:
    sentinel = "# 用户自己改过的内容\n"
    target = home / "config" / "risk.yaml"
    target.write_text(sentinel, encoding="utf-8")

    report = initialize(home)

    assert target.read_text(encoding="utf-8") == sentinel
    assert target in report.skipped
    assert target not in report.copied


def test_second_run_copies_nothing(home: Path) -> None:
    report = initialize(home)
    assert report.copied == []
    assert report.skipped
    assert report.db_created is False


def test_db_is_created_at_current_version(home: Path) -> None:
    conn = connect(home / "data" / "tradecompass.sqlite")
    try:
        assert user_version(conn) == SCHEMA_VERSION
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {"trades", "position_meta", "actions", "alerts", "backtest_runs"} <= names


def test_init_db_is_idempotent(home: Path) -> None:
    assert initialize(home).db_created is False
    conn: sqlite3.Connection = connect(home / "data" / "tradecompass.sqlite")
    try:
        conn.execute(
            "INSERT INTO runs (task, run_date, started_at, status) VALUES (?,?,?,?)",
            ("eod", "2026-10-21", "2026-10-21T16:00:00", "ok"),
        )
        conn.commit()
    finally:
        conn.close()

    initialize(home)

    conn = connect(home / "data" / "tradecompass.sqlite")
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        assert row["n"] == 1
    finally:
        conn.close()
