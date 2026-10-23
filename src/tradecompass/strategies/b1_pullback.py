"""策略 A：周线 KDJ 低位回调（SPEC §9.7）。

- A1 周线趋势：完结周线收盘 > 周 MA20 且 MA20 向上
- A2 位置：完结周线 J 值 < 阈值（或近 3 日出现过日线 J 低点）
- A3 触发：日线收盘 > MA5 且 > 前一日最高
- 次日以突破确认 K 线最高价买入（``entry_mode: breakout_high``）或开盘买入

A1/A2 全部读 ``ctx.weekly``——引擎已保证其中只有**已完结**的周线。
"""

from __future__ import annotations

import pandas as pd

from tradecompass.backtest.types import StrategyContext
from tradecompass.strategies.exits import (
    ExitParams,
    compute_initial_stop,
    exit_signal,
    stop_too_wide,
)

STRATEGY_ID = "b1_pullback"


def _num(source: dict[str, object], key: str, default: float) -> float:
    value = source.get(key, default)
    return float(value) if isinstance(value, int | float) else default


class B1PullbackStrategy:
    id = STRATEGY_ID

    def __init__(self, params: dict[str, object], exits: ExitParams) -> None:
        self.params = params
        self.exits = exits
        self.timeframe = str(params.get("timeframe", "weekly"))
        self.weekly_j_max = _num(params, "weekly_j_max", 20.0)
        self.daily_j_max = _num(params, "daily_j_max", 0.0)
        self.entry_mode = str(params.get("entry_mode", "breakout_high"))

    # ------------------------------------------------------------------ 条件
    def _a1(self, ctx: StrategyContext) -> bool:
        weekly = ctx.weekly
        if len(weekly) < 21:
            return False
        last = weekly.iloc[-1]
        close = last["close"]
        ma_value = last["ma20w"]
        if pd.isna(close) or pd.isna(ma_value):
            return False
        return bool(close > ma_value and bool(last["ma20w_up"]))

    def _a2(self, ctx: StrategyContext) -> bool:
        if self.timeframe == "daily":
            j = ctx.daily["j"].tail(3).dropna()
            return bool((j < self.daily_j_max).any())
        if ctx.weekly.empty:
            return False
        jw = ctx.weekly["jw"].iloc[-1]
        return bool(not pd.isna(jw) and jw < self.weekly_j_max)

    def _a3(self, ctx: StrategyContext) -> bool:
        bars = ctx.bars
        if len(bars) < 2:
            return False
        close = float(bars["close"].iloc[-1])
        prev_high = float(bars["high"].iloc[-2])
        if "ma5" not in ctx.daily.columns:
            return False
        ma5 = ctx.daily["ma5"].iloc[-1]
        if pd.isna(ma5):
            return False
        return bool(close > float(ma5) and close > prev_high)

    # ---------------------------------------------------------------- 接口
    def entry(self, ctx: StrategyContext) -> tuple[float | None, str] | None:
        if not (self._a1(ctx) and self._a2(ctx) and self._a3(ctx)):
            return None
        bars = ctx.bars
        close = float(bars["close"].iloc[-1])
        atr = _last(ctx.daily, "atr14")
        prev_low = _last(ctx.daily, "llv_low_10")
        stop = compute_initial_stop(close, atr, prev_low, self.exits)
        if stop is None or stop_too_wide(close, stop, self.exits.max_stop_pct):
            return None
        jw = _last(ctx.weekly, "jw")
        reason = f"周J={jw:.1f} 站上MA5" if jw is not None else "周线低位站上MA5"
        if self.entry_mode == "next_open":
            return None, reason
        return float(bars["high"].iloc[-1]), reason

    def exit(self, ctx: StrategyContext) -> tuple[float | None, str] | None:
        return exit_signal(ctx, self.exits)

    def initial_stop(self, ctx: StrategyContext, entry_price_raw: float) -> float | None:
        atr = _last(ctx.daily, "atr14")
        prev_low = _last(ctx.daily, "llv_low_10")
        return compute_initial_stop(entry_price_raw, atr, prev_low, self.exits)


def _last(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = frame[column].iloc[-1]
    return None if pd.isna(value) else float(value)
