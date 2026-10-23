"""交易行为体检与周报清理价位。"""

from __future__ import annotations

import pandas as pd

from tradecompass.account.behavior import analyze
from tradecompass.research.weekly_plan import HoldingView, _cleanup, apply_patience


def trades() -> pd.DataFrame:
    rows = [
        ("2026-01-05", "buy", "600000", "浦发", 1000, 10.0, -10005.0),
        ("2026-01-06", "buy", "600000", "浦发", 1000, 8.0, -8005.0),
        ("2026-02-10", "sell", "600000", "浦发", 1000, 9.5, 9490.0),
        ("2026-03-02", "buy", "518880", "黄金", 1000, 8.0, -8001.0),
        ("2026-03-02", "sell", "518880", "黄金", 1000, 8.1, 8099.0),
    ]
    return pd.DataFrame(
        [
            {"date": d, "side": s, "code": c, "name": n, "quantity": q, "price": p,
             "amount": a, "fee": 1.0, "stamp": 0.0, "transfer_fee": 0.0}
            for d, s, c, n, q, p, a in rows
        ]
    )


def test_average_cost_realized_pnl_and_habits() -> None:
    report = analyze(trades())
    assert report.trades == 5 and report.buys == 3 and report.sells == 2
    first = report.closed[0]
    # 均价 (10005+8005)/2000 = 9.005，卖 1000 得 9490
    assert abs(first.pnl - (9490.0 - 9005.0)) < 1e-6
    assert first.hold_days == 36
    assert report.same_day_round_trips == 1
    assert abs(report.realized - (485.0 + 98.0)) < 1e-6


def test_cleanup_targets_rebound_levels_above_price() -> None:
    action, detail = _cleanup(10.0, 10.5, 11.0)
    assert action == "清理：反弹分批" and "10.500" in detail and "11.000" in detail
    action, _ = _cleanup(12.0, 10.5, 11.0)
    assert action == "清理：已在均线上方"
    action, detail = _cleanup(10.6, 10.5, 11.0)
    assert action == "清理：反弹卖出" and "11.000" in detail


def _view(code: str, action: str, quantity: int = 1000) -> HoldingView:
    return HoldingView(code, code, "股票", quantity, 10.0, 12.0, 10_000.0, -0.17, "防御",
                       None, None, 10.5, 11.0, None, action, "x；10.500 卖一半，11.000 卖剩余")


def test_cleanup_patience_is_remembered_and_escalates() -> None:
    dates = pd.bdate_range("2026-10-01", periods=15)
    views = [_view("600000", "清理：反弹分批"), _view("601398", "保留")]
    fresh, state = apply_patience(views, dates, {})
    assert fresh[0].action == "清理：反弹分批"
    assert state == {"600000": dates[-1].date().isoformat()}

    started = dates[2].date().isoformat()
    escalated, state = apply_patience(views, dates, {"600000": started, "999999": started})
    assert escalated[0].action == "清理：到期卖一半"
    assert "500 股" in escalated[0].detail
    assert escalated[1].action == "保留"
    # 已不在清理里的标的从状态中移除
    assert state == {"600000": started}
