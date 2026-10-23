"""策略条件级测试（SPEC §19 M4）：A1–A3、B1–B2、X1–X5 各一组命中/不命中用例。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from tradecompass.backtest.types import PositionView, StrategyContext
from tradecompass.data.universe import board_of
from tradecompass.domain.models import Instrument
from tradecompass.strategies.b1_pullback import B1PullbackStrategy
from tradecompass.strategies.exits import (
    ExitParams,
    compute_initial_stop,
    exit_signal,
    stop_too_wide,
)
from tradecompass.strategies.volume_breakout import VolumeBreakoutStrategy

INSTRUMENT = Instrument(
    symbol="600000", market="SH", name="测试", kind="stock",
    board=board_of("600000", market="SH"),
)


def frame(values: dict[str, list[float]], start: str = "2026-01-05") -> pd.DataFrame:
    length = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.bdate_range(start, periods=length))


def make_ctx(
    *,
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    bars: pd.DataFrame | None = None,
    position: PositionView | None = None,
    day: date | None = None,
) -> StrategyContext:
    if bars is None:
        bars = pd.DataFrame(
            {
                "open": daily["close"],
                "high": daily["close"] + 0.1,
                "low": daily["close"] - 0.1,
                "close": daily["close"],
                "volume": [1_000_000.0] * len(daily),
            },
            index=daily.index,
        )
    return StrategyContext(
        date=day or pd.Timestamp(bars.index[-1]).date(),
        instrument=INSTRUMENT,
        bars=bars,
        daily=daily,
        weekly=weekly,
        position=position,
    )


def weekly_ok(**overrides: float) -> pd.DataFrame:
    """21 根周线（A1 要求至少 21 根才能算出 MA20w）。最后一根是判定依据。"""
    length = 21
    values: dict[str, list[float]] = {
        "close": [10.0] * (length - 1) + [11.0],
        "ma20w": [9.0] * length,
        "ma20w_up": [1.0] * length,
        "jw": [10.0] * length,
    }
    for key, value in overrides.items():
        values[key] = [value] * length
    return pd.DataFrame(values, index=pd.date_range("2025-08-01", periods=length, freq="W-FRI"))


def daily_ok(**overrides: float) -> pd.DataFrame:
    """默认：收盘 12 > 前日最高 11.1，且站上 MA5。"""
    values = {
        "close": [10.0, 11.0, 12.0],
        "ma5": [9.0, 9.0, 9.0],
        "atr14": [1.0, 1.0, 1.0],
        "llv_low_10": [11.4, 11.4, 11.4],
        "j": [50.0, 50.0, 50.0],
        "vol_ma20": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        "hhv_close_60_prev": [11.0, 11.0, 11.0],
    }
    values.update({k: [v, v, v] for k, v in overrides.items()})
    return pd.DataFrame(values, index=pd.bdate_range("2026-01-05", periods=3))


def b1(**params: object) -> B1PullbackStrategy:
    return B1PullbackStrategy(
        {"weekly_j_max": 20, "entry_mode": "breakout_high", **params}, ExitParams()
    )


# ------------------------------------------------------------------ 策略 A


def test_a1_rejects_when_weekly_close_below_ma20w() -> None:
    ctx = make_ctx(weekly=weekly_ok(close=8.0), daily=daily_ok())
    assert b1().entry(ctx) is None


def test_a1_rejects_when_weekly_ma20w_falling() -> None:
    ctx = make_ctx(weekly=weekly_ok(ma20w_up=0.0), daily=daily_ok())
    assert b1().entry(ctx) is None


def test_a2_rejects_when_weekly_j_too_high() -> None:
    ctx = make_ctx(weekly=weekly_ok(jw=35.0), daily=daily_ok())
    assert b1().entry(ctx) is None


def test_a2_accepts_when_weekly_j_low() -> None:
    ctx = make_ctx(weekly=weekly_ok(jw=5.0), daily=daily_ok())
    signal = b1().entry(ctx)
    assert signal is not None
    trigger, reason = signal
    assert trigger == pytest.approx(12.1)  # 确认 K 线最高价
    assert "周J" in reason


def test_a3_rejects_when_close_below_ma5() -> None:
    ctx = make_ctx(weekly=weekly_ok(jw=5.0), daily=daily_ok(ma5=13.0))
    assert b1().entry(ctx) is None


def test_a3_rejects_when_close_below_previous_high() -> None:
    daily = daily_ok()
    daily.loc[daily.index[1], "close"] = 12.5  # 前一日最高变成 12.6 > 当日 12.0
    ctx = make_ctx(weekly=weekly_ok(jw=5.0), daily=daily)
    assert b1().entry(ctx) is None


def test_a2_daily_timeframe_uses_recent_daily_j() -> None:
    ctx = make_ctx(weekly=weekly_ok(jw=60.0), daily=daily_ok(j=-5.0))
    assert b1(timeframe="daily", daily_j_max=0) .entry(ctx) is not None


def test_a_rejects_when_stop_too_wide() -> None:
    daily = daily_ok(atr14=8.0, llv_low_10=7.0)  # 12 − 2×8 = −4，前低 7 → 止损距离 42%
    ctx = make_ctx(weekly=weekly_ok(jw=5.0), daily=daily)
    assert b1().entry(ctx) is None


def test_next_open_mode_returns_none_trigger() -> None:
    ctx = make_ctx(weekly=weekly_ok(jw=5.0), daily=daily_ok())
    signal = b1(entry_mode="next_open").entry(ctx)
    assert signal is not None
    assert signal[0] is None


# ------------------------------------------------------------------ 策略 B


def vb(**params: object) -> VolumeBreakoutStrategy:
    return VolumeBreakoutStrategy({"lookback": 2, "vol_mult": 2.0, **params}, ExitParams())


def test_b1_rejects_when_close_not_new_high() -> None:
    daily = daily_ok(hhv_close_2_prev=12.5)
    ctx = make_ctx(weekly=weekly_ok(), daily=daily)
    assert vb().entry(ctx) is None


def test_b2_rejects_when_volume_not_enough() -> None:
    daily = daily_ok(hhv_close_2_prev=11.0)
    bars = pd.DataFrame(
        {
            "open": daily["close"], "high": daily["close"] + 0.1,
            "low": daily["close"] - 0.1, "close": daily["close"],
            "volume": [1_000_000.0, 1_000_000.0, 1_500_000.0],  # 1.5 倍 < 2 倍
        },
        index=daily.index,
    )
    ctx = make_ctx(weekly=weekly_ok(), daily=daily, bars=bars)
    assert vb().entry(ctx) is None


def test_b_accepts_breakout_with_volume() -> None:
    daily = daily_ok(hhv_close_2_prev=11.0)
    bars = pd.DataFrame(
        {
            "open": daily["close"], "high": daily["close"] + 0.1,
            "low": daily["close"] - 0.1, "close": daily["close"],
            "volume": [1_000_000.0, 1_000_000.0, 3_000_000.0],
        },
        index=daily.index,
    )
    ctx = make_ctx(weekly=weekly_ok(), daily=daily, bars=bars)
    signal = vb().entry(ctx)
    assert signal is not None
    assert signal[0] is None  # next_open
    assert "放量突破" in signal[1]


# --------------------------------------------------------------- 退出规则


def test_x1_stop_takes_nearer_of_atr_and_prev_low() -> None:
    params = ExitParams(atr_mult=2.0)
    # 入场 20，2ATR=16，前低 18 → 取较近（更高）的 18
    assert compute_initial_stop(20.0, 2.0, 18.0, params) == pytest.approx(18.0)
    # 前低 14 比 2ATR 更远 → 取 16
    assert compute_initial_stop(20.0, 2.0, 14.0, params) == pytest.approx(16.0)


def test_x2_flags_wide_stop() -> None:
    assert stop_too_wide(20.0, 16.0, 0.10) is True  # 20% > 10%
    assert stop_too_wide(20.0, 19.0, 0.10) is False


def position(entry: str, highest: float) -> PositionView:
    return PositionView(
        symbol="600000", shares=100, entry_price_raw=10.0,
        entry_date=date.fromisoformat(entry), current_stop_raw=9.0,
        highest_close_raw=highest, unrealized_pct=0.0,
    )


def test_x3_exits_below_ma10() -> None:
    daily = daily_ok(ma5=9.0)
    daily["ma10"] = [11.0, 11.0, 11.0]  # 收盘 12 > 11 不触发
    ctx = make_ctx(weekly=weekly_ok(), daily=daily, position=position("2026-01-05", 12.0))
    assert exit_signal(ctx, ExitParams()) is None

    daily["ma10"] = [13.0, 13.0, 13.0]  # 收盘 12 < 13 触发
    ctx = make_ctx(weekly=weekly_ok(), daily=daily, position=position("2026-01-05", 12.0))
    signal = exit_signal(ctx, ExitParams())
    assert signal is not None and "X3" in signal[1]


def test_x5_time_stop() -> None:
    daily = daily_ok()
    daily["ma10"] = [1.0, 1.0, 1.0]
    params = ExitParams(time_stop_enabled=True, time_stop_days=5, trail_ma=99)
    ctx = make_ctx(
        weekly=weekly_ok(), daily=daily,
        position=position("2026-01-05", 12.0),
        day=date(2026, 1, 5) + timedelta(days=6),
    )
    signal = exit_signal(ctx, params)
    assert signal is not None and "X5" in signal[1]
