"""参数指纹与运行记录。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def params_hash(params: dict[str, object]) -> str:
    """参数指纹：用于区分同一策略的不同参数组合，也用于判断"是否回测过这套参数"。"""
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def record_run(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    fingerprint: str,
    sample: str,
    start: str,
    end: str,
    passed: bool,
    failed_items: str,
    metrics_json: str,
    report_path: str,
) -> None:
    conn.execute(
        """
        INSERT INTO backtest_runs
        (strategy_id, params_hash, sample, start, end, passed, failed_items,
         metrics_json, report_path, run_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            strategy_id,
            fingerprint,
            sample,
            start,
            end,
            1 if passed else 0,
            failed_items,
            metrics_json,
            report_path,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    cursor = conn.execute(
        """
        SELECT strategy_id, params_hash, sample, start, end, passed,
               failed_items, report_path, run_at
        FROM backtest_runs ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    )
    return list(cursor.fetchall())


def report_directory(base: Path) -> Path:
    directory = base / "backtest"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
