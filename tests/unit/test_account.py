"""账户层：交割单解析、持仓推导、体检分类（SPEC §11）。"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tradecompass.account.checkup import classify
from tradecompass.account.importer import (
    import_into,
    normalize_symbol,
    parse_statement,
    read_statement,
)
from tradecompass.account.ledger import derive, derive_ledger
from tradecompass.config.models import BrokerMapping

MAPPING = BrokerMapping.model_validate(
    {
        "columns": {
            "trade_date": ["成交日期"],
            "trade_time": ["成交时间"],
            "symbol": ["证券代码"],
            "name": ["证券名称"],
            "side_text": ["操作"],
            "quantity": ["成交数量"],
            "price": ["成交均价"],
            "amount": ["成交金额"],
            "fee_commission": ["手续费"],
            "fee_stamp": ["印花税"],
            "fee_transfer": ["过户费"],
            "trade_id": ["成交编号"],
        },
        "sides": {
            "buy": ["证券买入"],
            "sell": ["证券卖出"],
            "dividend": ["红利入账"],
            "bonus_share": ["红股入账"],
            "tax": ["股息红利税补缴"],
            "ignore": ["银证转入", "利息归本"],
        },
    }
)

CSV = """成交日期,成交时间,证券代码,证券名称,操作,成交数量,成交均价,成交金额,手续费,印花税,过户费,成交编号
2026-09-01,09:31:00,600000,测试股,证券买入,1000,10.00,10000.00,5.00,0.00,0.10,A001
2026-09-05,10:02:00,600000,测试股,证券买入,500,12.00,6000.00,5.00,0.00,0.06,A002
2026-09-10,14:20:00,600000,测试股,证券卖出,600,15.00,9000.00,5.00,4.50,0.09,A003
2026-09-15,00:00:00,600000,测试股,红利入账,0,0,120.00,0.00,0.00,0.00,A004
2026-09-20,09:30:00,,,银证转入,0,0,50000.00,0,0,0,A005
2026-09-21,09:30:00,000001,平安银行,融资买入,100,12.00,1200.00,5.00,0,0,A006
"""


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema = Path(__file__).resolve().parents[2] / "src" / "tradecompass" / "storage" / "schema.sql"
    connection.executescript(schema.read_text(encoding="utf-8"))
    return connection


def write_csv(tmp_path: Path) -> Path:
    path = tmp_path / "statement.csv"
    path.write_text(CSV, encoding="utf-8-sig")
    return path


# ------------------------------------------------------------------ 解析


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", "000001"), ("600000.0", "600000"), ("159501", "159501"), ("", None)],
)
def test_normalize_symbol(raw: str, expected: str | None) -> None:
    assert normalize_symbol(raw) == expected


def test_parse_counts_and_issues(tmp_path: Path) -> None:
    result = parse_statement(read_statement(write_csv(tmp_path)), MAPPING)

    assert len(result.trades) == 4  # 2 买 + 1 卖 + 1 红利
    assert result.ignored == 1  # 银证转入
    assert len(result.issues) == 1  # 融资买入未识别
    assert "融资买入" in result.issues[0].reason
    assert result.trades[0].symbol == "600000"
    assert result.trades[0].side == "buy"
    assert result.trades[0].fee_transfer == pytest.approx(0.10)


def test_parse_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"foo": [1]})
    with pytest.raises(Exception, match="缺少必要列"):
        parse_statement(frame, MAPPING)


def test_import_is_idempotent(tmp_path: Path, conn: sqlite3.Connection) -> None:
    result = parse_statement(read_statement(write_csv(tmp_path)), MAPPING)

    first = import_into(conn, result)
    second = import_into(conn, result)

    assert first[0] == 4 and first[1] == 0
    assert second[0] == 0 and second[1] == 4  # 重复导入不产生重复记录


def test_unrecognized_rows_are_persisted(tmp_path: Path, conn: sqlite3.Connection) -> None:
    result = parse_statement(read_statement(write_csv(tmp_path)), MAPPING)
    import_into(conn, result)

    rows = conn.execute("SELECT * FROM import_issues").fetchall()
    assert len(rows) == 1
    assert "融资买入" in rows[0]["reason"]


# ------------------------------------------------------------------ 持仓


def test_derive_positions_uses_diluted_cost(tmp_path: Path, conn: sqlite3.Connection) -> None:
    import_into(conn, parse_statement(read_statement(write_csv(tmp_path)), MAPPING))
    trades = conn.execute("SELECT * FROM trades ORDER BY trade_date, id").fetchall()

    holdings = derive(trades)
    holding = holdings["600000"]

    # 买入费用含过户费：5+0.10 与 5+0.06，合计 10.16；卖出费用 5+4.50+0.09 = 9.59
    # 摊薄成本 = (16000 + 10.16 − 9000 + 9.59 − 120) / 900
    assert holding.quantity == 900
    expected = (16000 + 10.16 - 9000 + 9.59 - 120) / 900
    assert holding.avg_cost_raw == pytest.approx(expected, rel=1e-6)
    assert holding.realized_pnl > 0


def test_derive_history_keeps_closed_symbols_and_last_sell_pnl(
    conn: sqlite3.Connection,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "成交日期": "2026-09-01",
                "成交时间": "09:31:00",
                "证券代码": "600000",
                "证券名称": "测试股",
                "操作": "证券买入",
                "成交数量": 100,
                "成交均价": 11.0,
                "成交金额": 1100.0,
                "手续费": 5.0,
                "印花税": 0.0,
                "过户费": 0.01,
                "成交编号": "L001",
            },
            {
                "成交日期": "2026-09-05",
                "成交时间": "10:00:00",
                "证券代码": "600000",
                "证券名称": "测试股",
                "操作": "证券卖出",
                "成交数量": 100,
                "成交均价": 10.0,
                "成交金额": 1000.0,
                "手续费": 5.0,
                "印花税": 0.5,
                "过户费": 0.01,
                "成交编号": "L002",
            },
        ]
    )
    import_into(conn, parse_statement(frame, MAPPING))

    holdings, histories = derive_ledger(conn.execute("SELECT * FROM trades").fetchall())

    assert holdings == {}
    history = histories["600000"]
    assert history.last_buy_date == date(2026, 9, 1)
    assert history.last_sell_date == date(2026, 9, 5)
    assert history.last_sell_pnl is not None
    assert history.last_sell_pnl < 0
    assert history.consecutive_losses == 1


def test_fully_closed_position_disappears(tmp_path: Path, conn: sqlite3.Connection) -> None:
    frame = pd.DataFrame(
        [
            {"d": "2026-09-01", "sym": "600000", "s": "证券买入", "q": 100, "p": 10.0},
            {"d": "2026-09-02", "sym": "600000", "s": "证券卖出", "q": 100, "p": 11.0},
        ]
    )
    frame = frame.rename(columns={"d": "成交日期", "sym": "证券代码", "s": "操作",
                                  "q": "成交数量", "p": "成交均价"})
    frame["成交金额"] = frame["成交数量"] * frame["成交均价"]
    result = parse_statement(frame, MAPPING)
    import_into(conn, result)

    holdings = derive(conn.execute("SELECT * FROM trades").fetchall())
    assert "600000" not in holdings


# ------------------------------------------------------------------ 体检分类


@pytest.mark.parametrize(
    ("weekly_ok", "pnl_pct", "expected_verdict"),
    [
        (True, 0.12, "趋势健康"),
        (True, -0.08, "趋势健康但浮亏"),
        (False, 0.05, "周线走坏"),
        (False, -0.20, "周线走坏且浮亏"),
        (None, 0.0, "缺数据"),
    ],
)
def test_classify(weekly_ok: bool | None, pnl_pct: float, expected_verdict: str) -> None:
    verdict, action = classify(weekly_ok, pnl_pct)
    assert verdict == expected_verdict
    assert action


def test_classify_marks_loss_as_alert() -> None:
    _, action = classify(False, -0.20)
    assert action.startswith("⚠")
