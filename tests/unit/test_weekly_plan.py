"""交易行为体检与周报清理价位。"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from tradecompass.account.behavior import analyze
from tradecompass.research.etfs import etf_by_code, etf_roles, strategy_of
from tradecompass.research.weekly_plan import (
    BuyIdea,
    HoldingView,
    WatchView,
    _cleanup,
    _etf_position_band,
    _etf_trend_strength,
    _strength,
    _top_up_shares,
    apply_patience,
    base_orders,
    etf_action,
    plan_payload,
    render_plan,
    replay_paths,
)


def test_etf_roles_and_actions() -> None:
    roles = etf_roles()
    assert roles["sh.510300"] == "底仓"
    assert roles["sz.159201"] == "稳健" and roles["sh.513910"] == "稳健"
    assert roles["sz.159992"] == "波段"
    base = {"510300": 0.4}
    assert etf_action("底仓", "510300", 4.0, 3.9, 3.8, base)[0] == "底仓：保留"
    assert etf_action("稳健", "159201", 0.9, 1.0, 1.0, base)[0] == "稳健：持有"
    assert etf_action("波段", "159992", 1.1, 1.0, 1.0, base)[0] == "波段：持有观察"
    action, detail = etf_action("波段", "159992", 0.9, 1.0, 1.0, base)
    assert action.startswith("清理") and "暂行" in detail


def test_etf_asset_class_drives_strategy_family() -> None:
    """ETF 先按资产类别（收益驱动）分，再推导策略族；不再靠手工 STEADY_EXTRA。"""
    infos = etf_by_code()
    assert infos["sh.510300"].asset_class == "equity_broad"
    assert strategy_of(infos["sh.510300"]) == "core_band"
    assert infos["sh.518880"].asset_class == "commodity"
    assert strategy_of(infos["sh.518880"]) == "hold"
    assert infos["sh.511090"].asset_class == "bond"
    assert strategy_of(infos["sh.511090"]) == "hold"
    assert infos["sh.512480"].asset_class == "sector"
    assert strategy_of(infos["sh.512480"]) == "swing_cleanup"
    assert infos["sh.513100"].asset_class == "overseas"
    assert strategy_of(infos["sh.513100"]) == "core_band"
    assert infos["sz.159636"].asset_class == "overseas_sector"
    assert strategy_of(infos["sz.159636"]) == "swing_cleanup"


def test_etf_position_band_high_low_and_hold_no_band() -> None:
    """高位减半、低位未止跌保留 60%；黄金/债券是 hold，不做仓位带。"""
    up = _etf_position_band(_etf_trend_bars(0.002, n=300), "core_band")
    assert up is not None
    pos, _stabilized, band, action = up
    assert pos >= 85 and band == 0.5 and "减到 50%" in action

    down = _etf_position_band(_etf_trend_bars(-0.002, n=300), "core_band")
    assert down is not None
    pos, _stabilized, band, action = down
    assert pos <= 30 and band == 0.6 and "等止跌" in action

    assert _etf_position_band(_etf_trend_bars(0.002, n=300), "hold") is None


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


def test_cleanup_without_moving_averages_never_claims_above() -> None:
    """没有 K 线时说“已在均线上方”是假话，会误导卖出方向。"""
    action, detail = _cleanup(10.0, None, None)
    assert action == "清理：缺数据"
    assert "缺 MA10/MA20" in detail and "均线上方" not in detail

    action, detail = etf_action("波段", "159992", 0.9, None, None, {})
    assert action == "清理：缺数据"
    assert "跌破 MA20" not in detail


def _view(
    code: str, action: str, quantity: int = 1000, score: float | None = None,
    strength: float | None = None,
) -> HoldingView:
    return HoldingView(code, code, "股票", quantity, 10.0, 12.0, 10_000.0, -0.17, "防御",
                       None, None, 10.5, 11.0, None, action, "x；10.500 卖一半，11.000 卖剩余",
                       score, strength=strength)


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


def test_missing_data_is_not_escalated_by_patience() -> None:
    """缺数据给不出价位，按“等太久”升级成“收盘前卖一半”是无依据的。"""
    dates = pd.bdate_range("2026-10-01", periods=15)
    views = [_view("600000", "清理：缺数据")]
    out, _ = apply_patience(views, dates, {"600000": dates[0].date().isoformat()})
    assert out[0].action == "清理：缺数据"


def _empty_plan(as_of: date) -> dict[str, object]:
    return {
        "as_of": as_of,
        "views": [],
        "ideas": [],
        "trend": [("防御", -0.028, 3)],
        "slots": {"防御": 3},
        "slot_value": 10_116.0,
        "stock_capital": 121_387.0,
        "cash": 18_288.0,
        "total_assets": 242_774.0,
        "sold_recently": [],
        "cleanup_state": {},
        "etf_stale": [],
        "base_orders": [],
    }


def test_report_dates_come_from_data_not_literals() -> None:
    as_of = date(2026, 3, 5)
    text = render_plan(
        _empty_plan(as_of),
        base_targets={"510300": 0.5},
        behavior_lines=["x"],
        market_line="m",
        notes=["n"],
        next_day=date(2026, 3, 6),
        statement_period=(date(2024, 1, 2), as_of),
    )
    assert "2026-03-06（下一交易日）" in text
    assert "对账单 03-05" in text
    assert "只用截至 03-05 的数据" in text
    assert "03-05 收盘" in text
    assert "对账单 2024-01-02~2026-03-05" in text
    assert "10-23" not in text


def _base_bars(price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [price]}, index=pd.DatetimeIndex([pd.Timestamp("2026-03-05")], name="date")
    )


def _base_view(code: str, value: float) -> HoldingView:
    return HoldingView(code, code, "ETF", 100, 1.0, 1.0, value, 0.0, "底仓", None, None,
                       1.0, 1.0, None, "底仓：保留", "")


def test_base_orders_split_by_gap_and_round_to_lots() -> None:
    bars = {
        "sh.510300": _base_bars(4.0),
        "sh.518880": _base_bars(8.0),
        "sh.511090": _base_bars(110.0),
    }
    orders = base_orders(
        {"510300": 0.5, "518880": 0.3, "511090": 0.2},
        total_assets=100_000,
        views=[],
        etf_bars=bars,
        cash_now=10_000,
    )
    by_code = {item.code: item for item in orders}
    assert by_code["510300"].gap == 25_000 and by_code["511090"].gap == 10_000
    assert by_code["510300"].limit == 4.08
    # 缺口最大的分到最多，且都是整手
    assert by_code["510300"].shares_now > by_code["518880"].shares_now > 0
    assert all(item.shares_now % 100 == 0 for item in orders)
    # 一手要 11,220 元的 511090 在这点预算下买不到
    assert by_code["511090"].shares_now == 0
    # 预算不能超，且零钱不该大面积闲置（>85% 用掉）
    spent = sum(item.amount_now for item in orders)
    assert spent <= 10_000 and spent > 8_500
    # 每回笼 1 万元只拿一半补底仓
    unit_spent = sum(item.amount_per_unit for item in orders)
    assert unit_spent <= 5_000 and unit_spent > 4_000
    assert by_code["511090"].units_to_lot is not None
    assert by_code["511090"].units_to_lot > 1


def test_base_orders_skip_when_already_overweight() -> None:
    orders = base_orders(
        {"510300": 0.5},
        total_assets=100_000,
        views=[_base_view("510300", 30_000.0)],
        etf_bars={"sh.510300": _base_bars(4.0)},
        cash_now=50_000,
    )
    assert orders[0].target == 25_000 and orders[0].gap == 0
    assert orders[0].shares_now == 0 and orders[0].shares_per_unit == 0


def test_base_orders_without_bars_have_no_limit() -> None:
    orders = base_orders(
        {"511090": 1.0}, total_assets=100_000, views=[], etf_bars={}, cash_now=10_000
    )
    assert orders[0].limit is None and orders[0].shares_now == 0
    assert orders[0].amount_now == 0.0


def test_base_orders_reallocate_around_missing_price() -> None:
    """一只底仓 ETF 没有行情时，它的缺口不能稀释另外两只能买到的预算。"""
    bars = {"sh.510300": _base_bars(4.0), "sh.518880": _base_bars(8.0)}
    orders = base_orders(
        {"510300": 0.5, "518880": 0.3, "511090": 0.2},
        total_assets=100_000,
        views=[],
        etf_bars=bars,
        cash_now=10_000,
    )
    by_code = {item.code: item for item in orders}
    assert by_code["511090"].limit is None and by_code["511090"].shares_now == 0
    assert by_code["511090"].share_amount == 0.0
    # 5,000 元“每回笼 1 万元”的预算在两只可买 ETF 之间重新归一
    assert round(by_code["510300"].share_amount + by_code["518880"].share_amount, 2) == 5_000
    assert sum(item.amount_now for item in orders) > 9_000


def test_report_shows_group_score_and_says_it_is_not_comparable_across_groups() -> None:
    """结论只有方向不够，报告要给出组内强度；同时必须写清跨组不可比，免得被当成统一评分。"""
    as_of = date(2026, 3, 5)
    plan = _empty_plan(as_of)
    plan["views"] = [
        _view("600000", "清理：反弹分批", score=0.43, strength=93.0),
        _view("510300", "底仓：保留"),
    ]
    plan["ideas"] = [
        BuyIdea("000157", "中联重科", "周期制造", 1, 5.89, 6.01, 1600, 9616.0,
                score=1.57, strength=67.0)
    ]
    text = render_plan(
        plan,
        base_targets={"510300": 0.5},
        behavior_lines=["x"],
        market_line="m",
        notes=["n"],
        next_day=date(2026, 3, 6),
        statement_period=(date(2024, 1, 2), as_of),
    )
    assert "组内分" in text and "强度" in text
    assert "+0.43" in text and "+1.57" in text
    assert "| +0.43 | 93 |" in text and "| +1.57 | 67 |" in text
    assert "跨组不可比" in text


def test_strength_maps_score_relative_to_line() -> None:
    """0–100 强度：刚好在门槛上 50，高/低一个标准差约 84/16；缺数据返回 None。"""
    assert _strength(0.0, 0.0) == 50.0
    assert round(_strength(1.0, 0.0)) == 84
    assert round(_strength(-1.0, 0.0)) == 16
    assert _strength(None, 0.0) is None
    assert _strength(0.0, None) is None


def _etf_trend_bars(step: float, n: int = 200) -> pd.DataFrame:
    close = np.cumprod(np.full(n, 1.0 + step))
    preclose = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"close": close, "preclose": preclose},
        index=pd.bdate_range("2025-01-01", periods=n),
    )


def test_etf_trend_strength_rewards_uptrend_and_penalises_downtrend() -> None:
    """波段 ETF 用趋势健康度：上涨高分、下跌低分；不足 120 日不给分。"""
    assert _etf_trend_strength(_etf_trend_bars(0.002)) > 80
    assert _etf_trend_strength(_etf_trend_bars(-0.002)) < 20
    assert _etf_trend_strength(_etf_trend_bars(0.002, n=50)) is None


def test_report_shows_etf_trend_strength_and_says_it_is_reference_only() -> None:
    """波段 ETF 不再是 “-”，但报告必须写明它只是趋势参考、不改暂行买卖规则。"""
    plan = _empty_plan(date(2026, 3, 5))
    plan["views"] = [
        HoldingView(
            "159755", "电池ETF", "ETF", 100, 1.0, 1.0, 100.0, 0.0, "ETF·波段",
            None, None, 1.0, 1.0, None, "清理：反弹分批",
            "趋势强度 15/100；x", strength=15.0,
        )
    ]
    text = render_plan(
        plan,
        base_targets={"510300": 0.5},
        behavior_lines=["x"],
        market_line="m",
        notes=["n"],
        next_day=date(2026, 3, 6),
        statement_period=(date(2024, 1, 2), date(2026, 3, 5)),
    )
    assert "| ETF·波段 | - | 15 |" in text
    assert "波段 ETF=趋势健康度" in text
    assert "只作清理参考" in text


def test_report_shows_etf_position_band_but_not_auto_orders() -> None:
    """仓位带只给建议：报告要显示位置/系数/动作，并明确不自动下单。"""
    plan = _empty_plan(date(2026, 3, 5))
    plan["views"] = [
        HoldingView(
            "510300", "沪深300ETF", "ETF", 100, 4.0, 3.8, 400.0, 0.05, "底仓",
            None, None, 3.9, 3.8, None, "底仓：保留", "x",
            asset_class="equity_broad", strategy="core_band",
            etf_position=85.0, etf_band=0.5, etf_band_action="高位（85）：减到 50% 仓",
        )
    ]
    text = render_plan(
        plan,
        base_targets={"510300": 0.5},
        behavior_lines=["x"],
        market_line="m",
        notes=["n"],
        next_day=date(2026, 3, 6),
        statement_period=(date(2024, 1, 2), date(2026, 3, 5)),
    )
    assert "ETF 仓位带" in text and "只建议，不自动下单" in text
    assert "核心+仓位带" in text and "高位（85）：减到 50% 仓" in text
    assert "低位止跌" in text


def test_top_up_shares_rounds_to_lots_and_never_exceeds_target() -> None:
    """加仓只是补到目标仓位：够了一股不加，买不起一手也不硬凑。"""
    assert _top_up_shares(5_000, 10_116, 10.0) == 500
    assert _top_up_shares(10_116, 10_116, 10.0) == 0
    assert _top_up_shares(12_000, 10_116, 10.0) == 0
    assert _top_up_shares(0, 1_000, 12.5) == 0
    assert _top_up_shares(0, 5_000, 12.5) == 400


def test_report_renders_top_up_watchlist_and_drivers() -> None:
    """加仓、自选池体检、因子驱动都要出现在报告里，且数字来自 plan 而不是写死。"""
    as_of = date(2026, 3, 5)
    plan = _empty_plan(as_of)
    plan["views"] = [
        HoldingView("600000", "浦发银行", "股票", 1000, 10.0, 8.0, 10_000.0, 0.25,
                    "防御", 3, 9, 10.5, 11.0, None, "加仓",
                    "防御 第 3 名（保留线 9）；目标仓位 10,116 元、现 10,000 元，补 500 股",
                    1.42, 500, "EP=1/PE_TTM +2.4、60日波动 +1.9"),
    ]
    plan["watch"] = [
        WatchView("600900", "长江电力", "防御", 36, 9, 0.49,
                  "ROE≈PB/PE +1.6；拖累 BP=1/PB -0.9", "不买"),
    ]
    text = render_plan(
        plan,
        base_targets={"510300": 0.5},
        behavior_lines=["x"],
        market_line="m",
        notes=["n"],
        next_day=date(2026, 3, 6),
        statement_period=(date(2024, 1, 2), as_of),
    )
    assert "**加仓**" in text and "补 500 股" in text
    assert "自选池体检" in text and "长江电力" in text and "不买" in text
    assert "EP=1/PE_TTM +2.4" in text and "ROE≈PB/PE +1.6" in text
    # 章节编号连续，不能因为插了一节而错位
    for number in ("一", "二", "三", "四", "五", "六", "七", "八", "九"):
        assert f"## {number}、" in text


def test_conclusion_says_how_many_are_actually_affordable() -> None:
    """“最多 3 只”和“实际只够 2 只”必须都写出来，否则看结论会以为能买三只。"""
    as_of = date(2026, 3, 5)
    plan = _empty_plan(as_of)
    plan["cash"] = 18_288.0
    plan["ideas"] = [
        BuyIdea("000157", "中联重科", "周期制造", 1, 5.89, 6.01, 1600, 9616.0, tag="买入"),
        BuyIdea("000623", "吉林敖东", "消费医药", 1, 16.89, 17.23, 500, 8615.0, tag="买入"),
        BuyIdea("601186", "中国铁建", "防御", 1, 6.01, 6.13, 1600, 9808.0, tag="候补"),
    ]
    text = render_plan(
        plan,
        base_targets={"510300": 0.5},
        behavior_lines=["x"],
        market_line="m",
        notes=["n"],
        next_day=date(2026, 3, 6),
        statement_period=(date(2024, 1, 2), as_of),
    )
    assert "实际只够 **2 只**" in text and "还差 1 只的钱" in text


def test_cleanup_summary_shows_resulting_portfolio() -> None:
    """清理不是“卖完就完了”，要说清卖完之后组合长什么样。"""
    as_of = date(2026, 3, 5)
    plan = _empty_plan(as_of)
    plan["views"] = [
        _view("600000", "清理：反弹分批"),
        _view("600004", "保留"),
    ]
    text = render_plan(
        plan,
        base_targets={"510300": 0.5},
        behavior_lines=["x"],
        market_line="m",
        notes=["n"],
        next_day=date(2026, 3, 6),
        statement_period=(date(2024, 1, 2), as_of),
    )
    assert "清理完成后" in text and "个股持仓会从 2 只降到 1 只" in text


def test_plan_payload_carries_score() -> None:
    plan = _empty_plan(date(2026, 3, 5))
    plan["views"] = [_view("600000", "清理：反弹分批", score=0.4321, strength=41.0)]
    plan["ideas"] = [
        BuyIdea("000157", "中联重科", "周期制造", 1, 5.89, 6.01, 1600, 9616.0,
                score=1.5678, strength=78.0)
    ]
    payload = plan_payload(
        plan,
        target_day=date(2026, 3, 6),
        rules_fingerprint="abc123",
        max_new_buys=3,
    )
    assert payload["holdings"][0]["score"] == 0.4321  # type: ignore[index]
    assert payload["holdings"][0]["strength"] == 41.0  # type: ignore[index]
    assert payload["buys"][0]["score"] == 1.5678  # type: ignore[index]
    assert payload["buys"][0]["strength"] == 78.0  # type: ignore[index]


def test_replay_paths_are_isolated_from_live_and_track(tmp_path) -> None:
    """历史复现写到 plan/replay/，不覆盖当天实盘报告，也不会被 track 当成实盘建议。"""
    md, payload = replay_paths(tmp_path, date(2026, 3, 6), date(2026, 3, 5))
    assert md == tmp_path / "replay" / "2026-03-06-asof-2026-03-05.md"
    assert payload == tmp_path / "replay" / "2026-03-06-asof-2026-03-05.json"


def test_plan_payload_is_serialisable_and_keeps_basis() -> None:
    plan = _empty_plan(date(2026, 3, 5))
    plan["views"] = [_view("600000", "清理：反弹分批")]
    plan["ideas"] = [BuyIdea("000157", "中联重科", "周期制造", 1, 5.89, 6.01, 1600, 9616.0)]
    payload = plan_payload(
        plan,
        target_day=date(2026, 3, 6),
        rules_fingerprint="abc123",
        max_new_buys=3,
        statement_period=(date(2024, 1, 2), date(2026, 3, 5)),
    )
    assert json.loads(json.dumps(payload)) == payload  # 可落盘
    assert payload["schema"] == 1
    assert payload["target_day"] == "2026-03-06"
    assert payload["statement_period"] == ["2024-01-02", "2026-03-05"]
    assert payload["rules"] == {"fingerprint": "abc123", "max_new_buys": 3}
    holding = payload["holdings"][0]  # type: ignore[index]
    assert holding["action"] == "清理：反弹分批" and holding["ma10"] == 10.5
    assert payload["buys"][0]["code"] == "000157"  # type: ignore[index]
