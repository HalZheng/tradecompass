"""横截面研究：面板拼装、无未来函数、IC/分组评估与事件研究。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradecompass.indicators.core import tdx_sma
from tradecompass.research.evaluate import (
    FactorResult,
    evaluate_factor,
    period_returns,
    rebalance_dates,
)
from tradecompass.research.events import event_study, tdx_sma_wide, weekly_on_daily
from tradecompass.research.factors import FactorSpec
from tradecompass.research.panel import Panel, _with_valuation, build_panel
from tradecompass.research.portfolio import PortfolioResult, _fill, run_portfolio

DAYS = pd.bdate_range("2020-01-01", periods=320)


def stock(seed: int, n: int = len(DAYS), *, start: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = DAYS[start : start + n]
    close = 10.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, len(idx)))
    preclose = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(1e6, 2e6, len(idx))
    return pd.DataFrame(
        {
            "open": preclose * (1.0 + rng.normal(0.0, 0.003, len(idx))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "preclose": preclose,
            "volume": volume,
            "amount": volume * close,
            "turn": volume / 1e8 * 100.0,
            "tradestatus": 1.0,
            "is_st": 0.0,
            "pe_ttm": 10.0 + seed,
            "pb_mrq": 1.0,
            "ps_ttm": 1.0,
            "pcf_ttm": 1.0,
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )


def members(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS[0], "code": codes, "index": "hs300"})


def universe(n: int = 80) -> Panel:
    codes = [f"sh.{600000 + i}" for i in range(n)]
    return build_panel({code: stock(i) for i, code in enumerate(codes)}, members(codes))


def test_dividend_gap_is_not_a_loss() -> None:
    frame = stock(1)
    day = frame.index[100]
    # 除息：收盘从 10 → 9.5，但交易所前收同步调成 9.5，复权收益应为 0
    frame.loc[day, "preclose"] = frame.loc[day, "close"]
    panel = build_panel({"sh.600000": frame}, members(["sh.600000"]))
    assert panel.ret.loc[day, "sh.600000"] == 0.0
    prev = panel.adj_close.index[99]
    assert panel.adj_close.loc[day, "sh.600000"] == panel.adj_close.loc[prev, "sh.600000"]


def test_membership_starts_at_snapshot() -> None:
    codes = ["sh.600000", "sh.600001"]
    snap = pd.DataFrame(
        {"date": [DAYS[0], DAYS[50]], "code": ["sh.600000", "sh.600001"], "index": "hs300"}
    )
    panel = build_panel({c: stock(i) for i, c in enumerate(codes)}, snap)
    assert not panel.member.loc[DAYS[49], "sh.600001"]
    assert panel.member.loc[DAYS[50], "sh.600001"]
    assert not panel.member.loc[DAYS[60], "sh.600000"]


def test_next_day_limit_up_is_not_eligible() -> None:
    frame = stock(2)
    day = frame.index[200]
    frame.loc[day, "open"] = frame.loc[day, "preclose"] * 1.10
    panel = build_panel({"sh.600000": frame}, members(["sh.600000"]))
    eligible = panel.eligible()
    assert not eligible.loc[frame.index[199], "sh.600000"]
    assert eligible.loc[frame.index[198], "sh.600000"]


def test_new_listing_needs_120_days() -> None:
    codes = ["sh.600000", "sh.600001"]
    bars = {"sh.600000": stock(0), "sh.600001": stock(1, n=150, start=170)}
    panel = build_panel(bars, members(codes))
    eligible = panel.eligible()
    assert not eligible.loc[DAYS[170 + 100], "sh.600001"]
    assert eligible.loc[DAYS[170 + 125], "sh.600001"]


def test_period_returns_enter_on_next_open() -> None:
    panel = universe(5)
    rebalance = rebalance_dates(panel.dates, DAYS[0].date(), DAYS[-1].date())
    forward = period_returns(panel, rebalance)
    first, second = rebalance[0], rebalance[1]
    entry = panel.adj_open.index[panel.dates.searchsorted(first) + 1]
    exit_ = panel.adj_open.index[panel.dates.searchsorted(second) + 1]
    code = panel.codes[0]
    expected = panel.adj_open.loc[exit_, code] / panel.adj_open.loc[entry, code] - 1.0
    assert np.isclose(forward.loc[first, code], expected)
    assert all(d.month != (d + pd.offsets.BDay(1)).month for d in rebalance)


def _evaluate(panel: Panel, values: pd.DataFrame) -> FactorResult:
    rebalance = rebalance_dates(panel.dates, DAYS[0].date(), DAYS[-1].date())
    forward = period_returns(panel, rebalance)
    spec = FactorSpec("probe", "probe", "test", 1, lambda p: values)
    return evaluate_factor(spec, values, panel, rebalance, forward)


def test_perfect_foresight_factor_has_high_ic() -> None:
    panel = universe()
    rebalance = rebalance_dates(panel.dates, DAYS[0].date(), DAYS[-1].date())
    forward = period_returns(panel, rebalance)
    values = forward.reindex(panel.dates)
    result = _evaluate(panel, values)
    assert result.ic_mean > 0.95
    q = result.annual_excess()
    assert q[-1] > q[0]
    assert result.long_short > 0


def test_random_factor_has_no_ic() -> None:
    panel = universe()
    rng = np.random.default_rng(7)
    values = pd.DataFrame(
        rng.normal(size=panel.ret.shape), index=panel.dates, columns=panel.codes
    )
    result = _evaluate(panel, values)
    assert abs(result.ic_mean) < 0.1


def test_tdx_sma_wide_matches_core() -> None:
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 100, size=(60, 3))
    data[:5, 1] = np.nan
    wide = tdx_sma_wide(data, 3, 1)
    for column in range(3):
        expected = tdx_sma(pd.Series(data[:, column]), 3, 1).to_numpy()
        assert np.allclose(wide[:, column], expected, equal_nan=True)


def test_weekly_uses_only_completed_weeks() -> None:
    panel = universe(3)
    weekly = weekly_on_daily(panel)
    wednesday = pd.Timestamp("2020-06-10")
    last_friday = pd.Timestamp("2020-06-05")
    code = panel.codes[0]
    assert weekly.close.loc[wednesday, code] == panel.adj_close.loc[last_friday, code]
    friday = pd.Timestamp("2020-06-12")
    assert weekly.close.loc[friday, code] == panel.adj_close.loc[friday, code]


def test_event_study_detects_planted_edge() -> None:
    panel = universe()
    events = pd.DataFrame(False, index=panel.dates, columns=panel.codes)
    forward = panel.adj_open.shift(-21) / panel.adj_open.shift(-1) - 1.0
    best = forward.rank(axis=1, ascending=False) <= 3
    events = events | best
    result = event_study(panel, "probe", "probe", events, DAYS[150].date(), DAYS[280].date())
    mean, _, hit, t = result.stats[20]
    assert result.count > 0
    assert mean > 0.05
    assert hit > 0.9
    assert t > 3


def test_portfolio_with_foresight_beats_universe_after_costs() -> None:
    panel = universe()
    rebalance = rebalance_dates(panel.dates, DAYS[0].date(), DAYS[-1].date())
    forward = period_returns(panel, rebalance).reindex(panel.dates)
    spec = FactorSpec("probe", "probe", "test", 1, lambda p: forward)
    result = run_portfolio(panel, (spec,), DAYS[0].date(), DAYS[-1].date(), top=5)
    ours = PortfolioResult.stats(result.returns)
    base = PortfolioResult.stats(result.benchmarks["股票池等权"].dropna())
    assert ours["total"] > base["total"]
    assert (result.turnover > 0).all()
    assert all(len(names) == 5 for names in result.holdings.values())


def test_valuation_override_uses_single_source() -> None:
    bars = stock(4)
    valuation = pd.DataFrame(
        {"pct_chg": 0.0, "pe_ttm": 33.0, "pb_mrq": 3.0, "ps_ttm": 2.0, "pcf_ttm": 9.0},
        index=bars.index[10:],
    )
    merged = _with_valuation(bars, valuation)
    assert merged["pe_ttm"].iloc[20] == 33.0
    assert np.isnan(merged["pe_ttm"].iloc[0])
    assert merged["close"].equals(bars["close"])
    assert _with_valuation(bars, pd.DataFrame()) is bars


def test_industry_cap_skips_crowded_sector() -> None:
    industry = {"a": "bank", "b": "bank", "c": "bank", "d": "power"}
    assert _fill([], ["a", "b", "c", "d", "e"], 3, industry, 2) == ["a", "b", "d"]
    assert _fill(["c"], ["a", "b", "d"], 3, industry, 2) == ["c", "a", "d"]
    assert _fill([], ["a", "b", "c"], 3, industry, None) == ["a", "b", "c"]
