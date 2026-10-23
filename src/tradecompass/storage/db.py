"""SQLite 连接与 schema 迁移（SPEC §18）。

迁移版本号存在 ``PRAGMA user_version``：0 表示空库，非 0 表示已迁移到该版本。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


class StorageError(RuntimeError):
    """数据库不可用或版本不兼容。"""


def schema_sql() -> str:
    return (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")


def connect(db_path: Path) -> sqlite3.Connection:
    """打开连接；父目录不存在时自动创建。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def init_db(db_path: Path) -> bool:
    """把空库建到最新版本；返回是否新建了 schema。

    已有库不做破坏性改动；版本高于本程序支持的版本时报错，避免旧代码写坏新库。
    """
    conn = connect(db_path)
    try:
        version = user_version(conn)
        if version == 0:
            conn.executescript(schema_sql())
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            return True
        if version > SCHEMA_VERSION:
            raise StorageError(
                f"数据库版本 {version} 高于程序支持版本 {SCHEMA_VERSION}：{db_path}"
            )
        return False
    finally:
        conn.close()


def require_ready(db_path: Path) -> None:
    """doctor 用：确认库存在且版本兼容。"""
    if not db_path.is_file():
        raise StorageError(f"数据库不存在：{db_path}")
    conn = connect(db_path)
    try:
        version = user_version(conn)
        if version != SCHEMA_VERSION:
            raise StorageError(f"数据库版本为 {version}，期望 {SCHEMA_VERSION}：{db_path}")
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    missing = {"trades", "position_meta", "actions", "runs"} - names
    if missing:
        raise StorageError(f"数据库缺少表：{', '.join(sorted(missing))}")
