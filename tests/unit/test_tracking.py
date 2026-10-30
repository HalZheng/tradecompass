"""建议事后追踪：T+N 口径、条件单触发、执行核对与汇总。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tradecompass.config.loader import Paths
from tradecompass.data.calendar import default_calendar
from tradecompass.research import data as rdata
from tradecompass.research import tracking
from tradecompass.research.tracking import evaluate_plan, load_history, summarize

TARGET = date(2026, 3, 3)
DAYS = pd.bdate_range("2026-02-24", "2026-04-20")


def _frame(
    base: float = 10.0, step: float = 0.1, highs: dict[date, float] | None = None
) -> pd.DataFrame:
    close = base + np.arange(len(DAYS)) * step
    preclose = np.concatenate([[close[0]], close[:-1]])
    high = close.copy()
    for day, value in (highs or {}).items():
        high[DAYS.get_loc(pd.Timestamp(day))] = value
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": close,
            "close": close,
            "preclose": preclose,
        },
        index=DAYS,
    )


def _write(paths: Paths, code: str, frame: pd.DataFrame, *, index: bool = False) -> None:
    path = rdata.index_bars_path(paths, code) if index else rdata.stock_bars_path(paths, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _paths(
    tmp_path: Path, *, highs: dict[date, float] | None = None, step: float = 0.1
) -> Paths:
    paths = Paths.from_home(tmp_path)
    _write(paths, rdata.BENCHMARKS["hs300"], _frame(base=100.0, step=0.0), index=True)
    _write(paths, "sh.600000", _frame(step=step, highs=highs))
    _write(paths, "sh.600001", _frame(step=step, highs=highs))
    return paths


def _plan() -> dict[str, object]:
    return {
        "schema": 1,
        "as_of": "2026-03-02",
        "target_day": TARGET.isoformat(),
        "buys": [
            {"code": "600000", "name": "浦发银行", "group": "防御", "rank": 1,
             "close": 10.0, "limit": 10.2, "shares": 100, "amount": 1020.0},
        ],
        "holdings": [
            {"code": "600001", "name": "邯郸钢铁", "action": "清理：反弹分批",
             "price": 10.0, "ma10": 10.5, "ma20": 11.0, "quantity": 200},
            {"code": "600002", "name": "保留的票", "action": "保留",
             "price": 10.0, "ma10": 9.0, "ma20": 9.0, "quantity": 100},
        ],
    }


def test_evaluate_plan_fills_horizons_from_target_day_open(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    outcomes = evaluate_plan(_plan(), paths=paths)
    assert [item.side for item in outcomes] == ["buy", "sell"]  # 保留不追踪
    buy = outcomes[0]
    idx = int(DAYS.get_loc(pd.Timestamp(TARGET)))
    close = 10.0 + np.arange(len(DAYS)) * 0.1
    for horizon in (1, 5, 20):
        expected = close[idx + horizon] / close[idx] - 1.0
        assert buy.returns[horizon] is not None
        assert abs(buy.returns[horizon] - expected) < 1e-9
    # 基准恒定，超额等于自身收益
    assert abs(buy.excess[5] - buy.returns[5]) < 1e-12


def test_cleanup_reference_is_first_target_above_price() -> None:
    assert tracking._cleanup_reference({"price": 10.0, "ma10": 10.5, "ma20": 11.0}) == 10.5
    assert tracking._cleanup_reference({"price": 10.6, "ma10": 10.5, "ma20": 11.0}) == 11.0
    assert tracking._cleanup_reference({"price": 12.0, "ma10": 10.5, "ma20": 11.0}) is None
    assert tracking._cleanup_reference({"price": 10.0, "ma10": None, "ma20": None}) is None


def test_touched_uses_high_within_five_sessions(tmp_path: Path) -> None:
    paths = _paths(tmp_path, highs={date(2026, 3, 6): 10.6})
    sell = evaluate_plan(_plan(), paths=paths)[1]
    assert sell.reference == 10.5
    assert sell.touched is True

    flat = _paths(tmp_path / "flat", step=0.0)
    assert evaluate_plan(_plan(), paths=flat)[1].touched is False


def test_executed_matches_same_side_trade_in_window(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    trades = pd.DataFrame(
        {
            "date": ["2026-03-05", "2026-03-05", "2026-03-20"],
            "code": ["600001", "600000", "600001"],
            "side": ["sell", "sell", "buy"],
        }
    )
    outcomes = evaluate_plan(_plan(), paths=paths, trades=trades)
    assert outcomes[0].executed is False  # 买入建议，窗口内只有一笔卖出
    assert outcomes[1].executed is True  # 03-05 卖出 600001 命中


def test_executed_is_unknown_when_statement_is_behind(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    trades = pd.DataFrame({"date": ["2026-02-20"], "code": ["600001"], "side": ["sell"]})
    assert evaluate_plan(_plan(), paths=paths, trades=trades)[1].executed is None


def test_load_history_skips_unreadable_and_sorts(tmp_path: Path) -> None:
    directory = tmp_path / "plan"
    directory.mkdir()
    (directory / "2026-03-03.json").write_text(
        json.dumps({"schema": 1, "target_day": "2026-03-03"}), encoding="utf-8"
    )
    (directory / "2026-03-02.json").write_text(
        json.dumps({"schema": 1, "target_day": "2026-03-02"}), encoding="utf-8"
    )
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    (directory / "other.json").write_text(json.dumps({"schema": 9}), encoding="utf-8")
    plans = load_history(directory)
    assert [item["target_day"] for item in plans] == ["2026-03-02", "2026-03-03"]


def test_summarize_reports_hit_rate_and_empty_state(tmp_path: Path) -> None:
    assert "还没有可追踪的提醒" in summarize([])
    paths = _paths(tmp_path)
    outcomes = evaluate_plan(_plan(), paths=paths)
    text = summarize(outcomes)
    assert "T+5 跑赢沪深300" in text and "条数 1" in text
    assert "| 建议日 | 类别 | 代码 |" in text


def _write_etf(paths: Paths, code: str, frame: pd.DataFrame) -> None:
    path = rdata.etf_bars_path(paths, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _etf_plan() -> dict[str, object]:
    return {
        "schema": 1,
        "as_of": "2026-03-02",
        "target_day": TARGET.isoformat(),
        "rules": {"fingerprint": "testfp"},
        "buys": [],
        "holdings": [],
        "base_orders": [
            {"code": "510300", "shares": 100, "limit": 10.2, "amount": 1020.0},
            {"code": "518880", "shares": 0, "limit": 5.0, "amount": 0.0},
        ],
        "steady_orders": [
            {"code": "159201", "target": 4000.0, "gap": 1000.0, "lots": 100, "order": "买入"},
            {"code": "159501", "target": 3000.0, "gap": -500.0, "lots": -100, "order": "卖出"},
        ],
    }


def test_evaluate_plan_tracks_base_and_steady_etf_orders(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for code in ("sh.510300", "sh.518880", "sz.159201", "sz.159501"):
        _write_etf(paths, code, _frame(step=0.05))
    outcomes = evaluate_plan(_etf_plan(), paths=paths)
    assert [(item.category, item.side) for item in outcomes] == [
        ("base", "buy"),
        ("steady", "buy"),
        ("steady", "sell"),
    ]
    # 0 份的底仓建议不追踪
    assert all(item.code != "518880" for item in outcomes)
    text = summarize(outcomes)
    assert "ETF 建议（底仓/稳健/波段）" in text and "规则指纹：testfp" in text
    assert "条数 3（买入 2、减仓 1）" in text


def test_base_etf_execution_matches_trade(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for code in ("sh.510300", "sh.518880", "sz.159201", "sz.159501"):
        _write_etf(paths, code, _frame(step=0.05))
    trades = pd.DataFrame(
        {
            "date": [pd.Timestamp(TARGET), pd.Timestamp(DAYS[-1])],
            "code": ["510300", "510300"],
            "side": ["buy", "buy"],
        }
    )
    outcomes = evaluate_plan(_etf_plan(), paths=paths, trades=trades)
    base = [item for item in outcomes if item.category == "base"][0]
    assert base.executed is True


def test_etf_holding_uses_etf_bars_and_category(tmp_path: Path) -> None:
    """用户持仓里的行业 ETF 不在静态名单，也要按 ETF 读行情、标成 ETF。"""
    paths = _paths(tmp_path)
    _write_etf(paths, "sz.159636", _frame(step=0.02))
    plan = {
        "schema": 1,
        "target_day": TARGET.isoformat(),
        "holdings": [
            {
                "code": "159636",
                "action": "清理：反弹分批",
                "price": 10.0,
                "ma10": 10.5,
                "ma20": 11.0,
                "quantity": 1000,
            }
        ],
    }
    outcomes = evaluate_plan(plan, paths=paths)
    assert len(outcomes) == 1
    assert outcomes[0].category == "etf"
    assert outcomes[0].returns[5] is not None


def test_load_history_since_filters_for_oos(tmp_path: Path) -> None:
    directory = tmp_path / "plans"
    directory.mkdir()
    for day in ("2026-03-02", "2026-03-03"):
        (directory / f"{day}.json").write_text(
            json.dumps({"schema": 1, "target_day": day}), encoding="utf-8"
        )
    plans = load_history(directory, since=date(2026, 3, 3))
    assert [item["target_day"] for item in plans] == ["2026-03-03"]


def test_horizons_beyond_available_data_stay_unknown(tmp_path: Path) -> None:
    """行情只到 T+1 时，不能拿“现在的价”冒充 T+5/T+20 收益。"""
    paths = Paths.from_home(tmp_path)
    cut = "2026-03-05"
    _write(
        paths,
        rdata.BENCHMARKS["hs300"],
        _frame(base=100.0, step=0.0).loc[:cut],
        index=True,
    )
    _write(paths, "sh.600000", _frame().loc[:cut])
    _write(paths, "sh.600001", _frame().loc[:cut])
    outcomes = evaluate_plan(_plan(), paths=paths)
    assert outcomes[0].returns[1] is not None
    assert outcomes[0].returns[5] is None and outcomes[0].returns[20] is None
    assert outcomes[1].touched is None


def test_calendar_sessions_truncates_instead_of_raising() -> None:
    cal = default_calendar()
    start = date(2026, 10, 23)
    span = cal.sessions(start, 5)
    assert span[0] == start and len(span) == 5
    assert cal.sessions(start, 0) == []
    assert cal.sessions(date(2099, 1, 4), 3) == []
