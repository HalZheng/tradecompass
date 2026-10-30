"""分板块滚动决策引擎：无未来函数、次日开盘成交、一手约束、额度约束。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradecompass.config.models import CostsConfig
from tradecompass.research.panel import Panel, build_panel
from tradecompass.research.rolling import (
    CleanupState,
    EngineState,
    Position,
    RollingEngine,
    SleeveRule,
    group_indices,
    half_lot,
    run_rolling,
    should_top_up,
    trend_tilt,
)
from tradecompass.research.sleeves import sleeve_of
from tradecompass.research.weekly_plan import build_plan

DAYS = pd.bdate_range("2020-01-01", periods=420)
RULES = (
    SleeveRule("防御", ("大金融",), ("vol60", "ep"), period=20, buffer=2.0, min_hold=5),
    SleeveRule("科技成长", ("科技成长",), ("rev20", "ivol20"), period=10, buffer=1.5, min_hold=5),
)


def stock(seed: int, shock_from: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, len(DAYS))
    shock = rng.normal(-0.01, 0.05, len(DAYS))
    volume = rng.uniform(1e6, 2e6, len(DAYS))
    noise = rng.normal(0.0, 0.002, len(DAYS))
    if shock_from is not None:
        rets[shock_from:] = shock[shock_from:]
    close = 10.0 * np.cumprod(1.0 + rets)
    preclose = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": preclose * (1.0 + noise),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "preclose": preclose,
            "volume": volume,
            "amount": volume * close,
            "turn": volume / 1e8 * 100.0,
            "tradestatus": 1.0,
            "is_st": 0.0,
            "pe_ttm": 8.0 + seed % 7,
            "pb_mrq": 1.0,
            "ps_ttm": 1.0,
            "pcf_ttm": 1.0,
        },
        index=pd.DatetimeIndex(DAYS, name="date"),
    )


def universe(shock_from: int | None = None) -> tuple[Panel, dict[str, str]]:
    codes = [f"sh.{600000 + i}" for i in range(40)]
    bars = {c: stock(i, shock_from) for i, c in enumerate(codes)}
    members = pd.DataFrame({"date": DAYS[0], "code": codes, "index": "hs300"})
    industry = {
        c: ("J66货币金融服务" if i % 2 else "I65软件和信息技术服务业")
        for i, c in enumerate(codes)
    }
    return build_panel(bars, members), industry


def engine(
    panel: Panel,
    industry: dict[str, str],
    quota: pd.Series | None = None,
    *,
    cleanup: str = "immediate",
    cash_rate: float = 0.0,
) -> RollingEngine:
    return RollingEngine(
        panel,
        industry,
        CostsConfig(),
        rules=RULES,
        slots=6,
        quota=quota,
        cleanup=cleanup,
        cash_rate=cash_rate,
    )


def test_sleeve_mapping() -> None:
    assert sleeve_of("C39计算机、通信和其他电子设备制造业") == "科技成长"
    assert sleeve_of("J66货币金融服务") == "大金融"
    assert sleeve_of("C27医药制造业") == "医药"
    assert sleeve_of("C26化学原料和化学制品制造业") == "周期资源"
    assert sleeve_of(None) == "其他"


def test_default_cash_rate_is_zero() -> None:
    panel, industry = universe()
    assert RollingEngine(panel, industry, CostsConfig()).cash_rate == 0.0


def test_cash_rate_accrues_on_idle_cash() -> None:
    panel, industry = universe()
    quota = pd.Series(0.0, index=panel.dates)
    eng = RollingEngine(
        panel, industry, CostsConfig(), rules=RULES, slots=6, quota=quota, cash_rate=0.02
    )
    state = EngineState(cash=100_000.0)
    for i in range(244):
        eng.step(state, i, [])
    assert state.positions == {}
    assert abs(state.cash - 102_000.0) < 10.0


def test_orders_fill_next_open_in_lots() -> None:
    panel, industry = universe()
    start, end = DAYS[260].date(), DAYS[-1].date()
    result = run_rolling(engine(panel, industry), start, end, capital=100_000)
    buys = [t for t in result.trades if t.side == "buy"]
    assert buys
    assert buys[0].day == DAYS[261]
    assert all(t.shares % 100 == 0 and t.shares > 0 for t in buys)
    assert result.final_state.cash >= -1e-6
    assert len(result.equity) == 420 - 260


def test_future_data_does_not_change_past_decisions() -> None:
    cut = 330
    base_panel, industry = universe()
    shocked_panel, _ = universe(shock_from=cut + 1)
    start, end = DAYS[260].date(), DAYS[-1].date()
    base = run_rolling(engine(base_panel, industry), start, end, capital=100_000)
    shocked = run_rolling(engine(shocked_panel, industry), start, end, capital=100_000)

    def upto(trades: list) -> list[tuple]:
        return [(t.day, t.code, t.side, t.shares) for t in trades if t.day <= DAYS[cut]]

    assert upto(base.trades) == upto(shocked.trades)
    assert base.equity.loc[: DAYS[cut]].equals(shocked.equity.loc[: DAYS[cut]])


def test_default_cleanup_is_immediate() -> None:
    panel, industry = universe()
    assert RollingEngine(panel, industry, CostsConfig()).cleanup == "immediate"


def test_half_lot_rounds_to_board_lot() -> None:
    assert half_lot(1000) == 500
    assert half_lot(150) == 150  # 不足两手，全卖
    assert half_lot(250) == 100  # 取整到一手


def test_rebound_cleanup_does_not_look_ahead() -> None:
    cut = 330
    base_panel, industry = universe()
    shocked_panel, _ = universe(shock_from=cut + 1)
    start, end = DAYS[260].date(), DAYS[-1].date()
    base = run_rolling(
        engine(base_panel, industry, cleanup="rebound"), start, end, capital=100_000
    )
    shocked = run_rolling(
        engine(shocked_panel, industry, cleanup="rebound"), start, end, capital=100_000
    )

    def upto(trades: list) -> list[tuple]:
        return [(t.day, t.code, t.side, t.shares) for t in trades if t.day <= DAYS[cut]]

    assert upto(base.trades) == upto(shocked.trades)
    assert base.equity.loc[: DAYS[cut]].equals(shocked.equity.loc[: DAYS[cut]])


def test_rebound_cleanup_sells_at_ma_or_patience() -> None:
    panel, industry = universe()
    eng = engine(panel, industry, cleanup="rebound")
    j = eng.col["sh.600000"]
    candidates = np.flatnonzero(
        np.isfinite(eng.high[:, j])
        & np.isfinite(eng.ma10[:, j])
        & (eng.high[:, j] >= eng.ma10[:, j])
    )
    i = int(candidates[-1])
    code = "sh.600000"
    state = EngineState(
        cash=0.0,
        positions={
            code: Position(
                code,
                "防御",
                units=1000.0,
                entry_adj=float(eng.adj[i - 5, j]),
                entry_day=i - 5,
            )
        },
        cleanup={code: CleanupState(start=i - 5)},
    )
    trades: list = []
    eng.cleanup_exits(state, i, trades)
    assert trades and trades[0].reason.startswith("清理")
    assert state.cash > 0 or code not in state.positions


def test_rebound_cleanup_patience_sells_once() -> None:
    panel, industry = universe()
    eng = engine(panel, industry, cleanup="rebound")
    j = eng.col["sh.600000"]
    candidates = np.flatnonzero(
        np.isfinite(eng.high[:, j])
        & np.isfinite(eng.ma10[:, j])
        & np.isfinite(eng.ma20[:, j])
        & (eng.high[:, j] < eng.ma10[:, j])
        & (eng.high[:, j] < eng.ma20[:, j])
    )
    i = int(candidates[-1])
    code = "sh.600000"
    state = EngineState(
        cash=0.0,
        positions={
            code: Position(
                code,
                "防御",
                units=1000.0,
                entry_adj=float(eng.adj[i - 1, j]),
                entry_day=i - 10,
            )
        },
        cleanup={code: CleanupState(start=i - 10)},
    )
    trades: list = []
    eng.cleanup_exits(state, i, trades)
    assert len(trades) == 1 and trades[0].reason == "清理：到期卖半"
    # 余下继续等 MA20；不会第二天再卖一半
    eng.cleanup_exits(state, i, trades)
    assert len(trades) == 1


def test_panel_truncate_reproduces_history_without_lookahead() -> None:
    """``plan --as-of`` 的基础：截断到某天后，之前的分数与决策必须与全样本完全一致。"""
    panel, industry = universe()
    cut = DAYS[330]
    trimmed = panel.truncate(cut)
    assert trimmed.dates[-1] == cut
    assert trimmed.dates.equals(panel.dates[panel.dates <= cut])

    full_engine = engine(panel, industry)
    cut_engine = engine(trimmed, industry)
    for name in full_engine.rules:
        full_scores = full_engine.scores(
            full_engine.rules[name], int(panel.dates.searchsorted(cut))
        )
        cut_scores = cut_engine.scores(cut_engine.rules[name], len(trimmed.dates) - 1)
        pd.testing.assert_series_equal(full_scores, cut_scores, check_names=False)

    start = DAYS[260].date()
    full = run_rolling(full_engine, start, DAYS[-1].date(), capital=100_000)
    short = run_rolling(cut_engine, start, cut.date(), capital=100_000)

    def key(trade: object) -> tuple:
        return (trade.day, trade.code, trade.side, trade.shares)  # type: ignore[attr-defined]

    assert [key(t) for t in full.trades if t.day <= cut] == [key(t) for t in short.trades]
    assert full.equity.loc[:cut].equals(short.equity)


def test_zero_quota_never_buys() -> None:
    panel, industry = universe()
    quota = pd.Series(0.0, index=panel.dates)
    result = run_rolling(
        engine(panel, industry, quota), DAYS[260].date(), DAYS[-1].date(), capital=100_000
    )
    assert not [t for t in result.trades if t.side == "buy"]


def test_outsider_is_scored_on_pool_distribution_without_distorting_it() -> None:
    """池外的自选/持仓要能打分，但不能改变成分股自己的分数（不重拟合截面分布）。"""
    codes = [f"sh.{600000 + i}" for i in range(40)]
    outsider = "sh.600999"
    bars = {c: stock(i) for i, c in enumerate(codes)}
    bars[outsider] = stock(99)
    members = pd.DataFrame({"date": DAYS[0], "code": codes, "index": "hs300"})
    industry = {
        c: ("J66货币金融服务" if i % 2 else "I65软件和信息技术服务业")
        for i, c in enumerate(codes)
    }
    industry[outsider] = "J66货币金融服务"
    panel = build_panel(bars, members)
    eng = RollingEngine(panel, industry, CostsConfig(), rules=RULES, slots=6)
    i = len(panel.dates) - 1
    rule = eng.rules["防御"]

    base = eng.scores(rule, i)
    with_extra = eng.scores(rule, i, extra=[outsider])
    assert outsider in with_extra.index and outsider not in base.index
    common = base.index.intersection(with_extra.index)
    assert len(common) == len(base)
    assert np.allclose(base.loc[common].to_numpy(), with_extra.loc[common].to_numpy())


def test_switch_tolerance_never_increases_selling() -> None:
    """换股容差是“更不愿意换”，所以卖出次数只会变少、不会变多。"""
    panel, industry = universe()
    start, end = DAYS[260].date(), DAYS[-1].date()
    base = run_rolling(engine(panel, industry), start, end, capital=100_000)
    patient = RollingEngine(
        panel, industry, CostsConfig(), rules=RULES, slots=6, switch_tolerance=0.3
    )
    slower = run_rolling(patient, start, end, capital=100_000)
    sells = [t for t in base.trades if t.side == "sell"]
    slower_sells = [t for t in slower.trades if t.side == "sell"]
    assert len(slower_sells) <= len(sells)
    assert RollingEngine(panel, industry, CostsConfig()).switch_tolerance == 0.0


def test_should_top_up_respects_mode_and_deep_loss() -> None:
    """加仓门槛：只在低于目标仓位时补；no_deep_loss 模式不许补深套票。"""
    assert not should_top_up(5_000, 0.10, "off")
    assert should_top_up(5_000, 0.10, "always")
    assert should_top_up(5_000, -0.35, "always")
    assert should_top_up(5_000, 0.05, "no_deep_loss")
    assert not should_top_up(5_000, -0.25, "no_deep_loss")
    assert not should_top_up(0.0, 0.05, "always")
    assert not should_top_up(-3_000, 0.05, "always")


def test_topup_is_off_by_default() -> None:
    """默认不加仓，回测里不应该凭空多出买单。"""
    panel, industry = universe()
    start, end = DAYS[260].date(), DAYS[-1].date()
    result = run_rolling(engine(panel, industry), start, end, capital=100_000)
    assert not [t for t in result.trades if "加仓" in t.reason]
    assert RollingEngine(panel, industry, CostsConfig()).topup == "off"


def test_trend_tilt_is_causal_and_shifts_slots() -> None:
    panel, industry = universe()
    eng = engine(panel, industry)
    idx = group_indices(eng)
    tilt = trend_tilt(idx, ma=60)
    assert set(tilt.stack().unique()) <= {0.7, 1.3}
    # 改后半段指数不应影响前半段倍数
    shifted = idx.copy()
    shifted.iloc[300:] *= 0.5
    assert trend_tilt(shifted, ma=60).iloc[:300].equals(tilt.iloc[:300])

    eng.set_tilt(pd.DataFrame({"防御": 1.5, "科技成长": 0.5}, index=panel.dates))
    slots = eng.rule_slots(300)
    assert slots["防御"] > slots["科技成长"]
    assert sum(slots.values()) == 6


def _plan_universe(*, outsider_st: bool = False) -> tuple[Panel, dict[str, str], str]:
    """40 只成分（防御/科技各 20）+ 一只“低波+极便宜”的池外自选。"""
    codes = [f"sh.{600000 + i}" for i in range(40)]
    outsider = "sh.600999"
    bars = {c: stock(i) for i, c in enumerate(codes)}
    calm = stock(99)
    close = 10.0 * np.cumprod(np.full(len(DAYS), 1.0004))
    preclose = np.concatenate([[close[0]], close[:-1]])
    calm["close"] = close
    calm["preclose"] = preclose
    calm["open"] = preclose
    calm["high"] = close * 1.01
    calm["low"] = close * 0.99
    calm["pe_ttm"] = 0.001
    calm["is_st"] = 1.0 if outsider_st else 0.0
    bars[outsider] = calm
    members = pd.DataFrame({"date": DAYS[0], "code": codes, "index": "hs300"})
    industry = {
        c: ("J66货币金融服务" if i % 2 else "I65软件和信息技术服务业")
        for i, c in enumerate(codes)
    }
    industry[outsider] = "J66货币金融服务"
    return build_panel(bars, members), industry, outsider


def _holding(bs_code: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"code": bs_code[3:], "name": "测试", "price": 10.0, "cost": 10.0,
          "quantity": 100, "value": 1_000.0}]
    )


def _holding_value(bs_code: str, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"code": bs_code[3:], "name": "测试", "price": 10.0, "cost": 10.0,
          "quantity": int(value / 10.0), "value": value}]
    )


def _plan(
    eng: RollingEngine,
    holders: list[pd.DataFrame],
    *,
    watchlist: list[str] = (),
    trades: pd.DataFrame | None = None,
    top_up: bool = False,
    max_new_buys: int = 0,
    cash: float = 1_000_000.0,
    last_rebalance: dict[str, str] | None = None,
    cleanup_since: dict[str, str] | None = None,
) -> dict[str, object]:
    empty_trades = pd.DataFrame({"date": [], "code": [], "side": []})
    frame = (
        pd.concat(holders)
        if holders
        else pd.DataFrame(
            columns=["code", "name", "price", "cost", "quantity", "value"]
        )
    )
    return build_plan(
        eng,
        frame,
        empty_trades if trades is None else trades,
        cash=cash,
        total_assets=100_000.0,
        base_targets={},
        etf_bars={},
        etf_names={},
        names={},
        max_new_buys=max_new_buys,
        watchlist=watchlist,
        top_up=top_up,
        last_rebalance=last_rebalance,
        cleanup_since=cleanup_since,
    )


def test_holding_keep_line_uses_pool_rank_not_merged_rank() -> None:
    """自选票可以把持仓的“合并名次”挤出保留线，但保留/清理必须按池内名次判断。"""
    panel, industry, outsider = _plan_universe()
    eng = engine(panel, industry)
    i = len(panel.dates) - 1
    rule = eng.rules["防御"]
    pool = eng.scores(rule, i)
    line = int(np.ceil(eng.rule_slots(i)["防御"] * rule.buffer))
    keeper = str(pool.index[line - 1])  # 正好在保留线上
    merged = eng.scores(rule, i, extra=[outsider])
    assert list(merged.index).index(keeper) + 1 > line  # 确实被自选挤出去了

    plan = _plan(eng, [_holding(keeper)], watchlist=[outsider[3:]])
    view = plan["views"][0]  # type: ignore[index]
    assert view.action == "保留"
    assert f"第 {line} 名" in view.detail
    assert round(view.strength) == 50  # 刚好在保留线上


def test_plan_holds_recent_buy_even_below_keep_line() -> None:
    """回测有 min_hold，实盘也必须遵守：刚买两天、名次掉线也不立刻清理。"""
    panel, industry, _ = _plan_universe()
    eng = engine(panel, industry)
    i = len(panel.dates) - 1
    rule = eng.rules["防御"]
    loser = str(eng.scores(rule, i).index[-1])
    trades = pd.DataFrame({"date": [DAYS[i - 2]], "code": [loser[3:]], "side": ["buy"]})

    held = _plan(eng, [_holding(loser)], trades=trades)["views"][0]  # type: ignore[index]
    assert held.action == "保留"
    assert "最短持有 5 日" in held.detail
    assert held.strength is not None and held.strength < 50  # 分不够，只是没到最短持有

    without_trade = _plan(eng, [_holding(loser)])["views"][0]  # type: ignore[index]
    assert without_trade.action.startswith("清理")


def test_min_hold_uses_position_entry_not_last_buy() -> None:
    """部分卖出不重置最短持有（引擎的 entry_day 也不重置）；清仓后再买才重置。"""
    panel, industry, _ = _plan_universe()
    eng = engine(panel, industry)
    i = len(panel.dates) - 1
    loser = str(eng.scores(eng.rules["防御"], i).index[-1])
    code = loser[3:]

    partial = pd.DataFrame(
        {
            "date": [DAYS[i - 60], DAYS[i - 20]],
            "code": [code, code],
            "side": ["buy", "sell"],
            "quantity": [200, 100],
        }
    )
    old_position = _plan(eng, [_holding(loser)], trades=partial)["views"][0]  # type: ignore[index]
    assert old_position.action.startswith("清理")

    round_trip = pd.DataFrame(
        {
            "date": [DAYS[i - 60], DAYS[i - 30], DAYS[i - 2]],
            "code": [code, code, code],
            "side": ["buy", "sell", "buy"],
            "quantity": [200, 200, 100],
        }
    )
    new_position = _plan(eng, [_holding(loser)], trades=round_trip)["views"][0]  # type: ignore[index]
    assert new_position.action == "保留"
    assert "最短持有 5 日" in new_position.detail


def test_plan_top_up_follows_backtest_keep_scope() -> None:
    """加仓与回测同口径：保留线内都能补；现金按排名先给前面的，排不上退回保留。"""
    panel, industry, _ = _plan_universe()
    eng = engine(panel, industry)
    i = len(panel.dates) - 1
    rule = eng.rules["防御"]
    pool = eng.scores(rule, i)
    slots = eng.rule_slots(i)["防御"]
    inside = str(pool.index[0])
    buffer_only = str(pool.index[slots])  # 第 slots+1 名，仍在保留线内

    views = _plan(eng, [_holding(inside), _holding(buffer_only)], top_up=True)["views"]
    by_code = {view.code: view for view in views}  # type: ignore[union-attr]
    assert by_code[inside[3:]].add_shares > 0
    # 保留线以内（缓冲区内）也算保留，回测的加仓范围就是 keep，不再额外要求前 slots 名
    assert by_code[buffer_only[3:]].add_shares > 0

    # 现金只够补一只时，按排名先给第 1 名
    tight = _plan(
        eng, [_holding(inside), _holding(buffer_only)], top_up=True, cash=3_000
    )["views"]
    tight_by = {view.code: view for view in tight}  # type: ignore[union-attr]
    assert tight_by[inside[3:]].add_shares > 0
    assert tight_by[buffer_only[3:]].add_shares == 0


def test_watchlist_outsider_is_buyable_unless_st() -> None:
    """池外自选可以进新买清单，但 ST（以及新股/停牌）必须被挡掉。"""
    panel, industry, outsider = _plan_universe()
    ideas = _plan(
        engine(panel, industry), [], watchlist=[outsider[3:]], max_new_buys=3
    )["ideas"]
    assert outsider[3:] in {idea.code for idea in ideas}  # type: ignore[union-attr]
    top = next(idea for idea in ideas if idea.code == outsider[3:])  # type: ignore[union-attr]
    assert top.strength is not None and top.strength > 50  # 明显高于买入线

    st_panel, st_industry, st_outsider = _plan_universe(outsider_st=True)
    st_ideas = _plan(
        engine(st_panel, st_industry), [], watchlist=[st_outsider[3:]], max_new_buys=3
    )["ideas"]
    assert st_outsider[3:] not in {idea.code for idea in st_ideas}  # type: ignore[union-attr]


def test_plan_only_rebalances_on_due_days() -> None:
    """plan 与回测同节奏：未到调仓日不清理，到期才重排。"""
    panel, industry, _ = _plan_universe()
    eng = engine(panel, industry)
    i = len(panel.dates) - 1
    loser = str(eng.scores(eng.rules["防御"], i).index[-1])

    recent = _plan(
        eng, [_holding(loser)], last_rebalance={"防御": DAYS[i].date().isoformat()}
    )["views"][0]  # type: ignore[index]
    assert recent.action == "保留" and "未到调仓日" in recent.detail

    overdue = _plan(
        eng, [_holding(loser)], last_rebalance={"防御": DAYS[i - 60].date().isoformat()}
    )["views"][0]  # type: ignore[index]
    assert overdue.action.startswith("清理")


def test_cleanup_persists_until_next_rebalance() -> None:
    """已进入清理的持仓，在非调仓日继续挂条件单，不会因为当天没重排就变回“保留”。"""
    panel, industry, _ = _plan_universe()
    eng = engine(panel, industry)
    i = len(panel.dates) - 1
    loser = str(eng.scores(eng.rules["防御"], i).index[-1])
    view = _plan(
        eng,
        [_holding(loser)],
        last_rebalance={"防御": DAYS[i].date().isoformat()},
        cleanup_since={loser[3:]: DAYS[i - 3].date().isoformat()},
    )["views"][0]  # type: ignore[index]
    assert view.action.startswith("清理") and "继续挂条件单" in view.detail


def test_plan_never_buys_when_stock_sleeve_is_over_cap() -> None:
    """个股超过 40% 上限时，新买额度必须是 0；不能一边超配一边加股票。"""
    panel, industry, _ = _plan_universe()
    eng = engine(panel, industry)
    i = len(panel.dates) - 1
    winner = str(eng.scores(eng.rules["防御"], i).index[0])  # 保留线内，不会卖出
    plan = _plan(eng, [_holding_value(winner, 50_000.0)], max_new_buys=3)
    assert plan["stock_cap_room"] == 0
    assert plan["stock_budget"] == 0
    assert plan["ideas"] == []
