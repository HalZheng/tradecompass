"""策略 B：放量突破（SPEC §9.8）。

- B1 收盘创 N 日新高（不含当日）
- B2 成交量 > N 日均量 × vol_mult
- 次日开盘买入

与 A 的区别：A 赌回调，B 追突破，两者信号天然互斥，可同时启用。
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

STRATEGY_ID = "volume_breakout"


def _num(source: dict[str, object], key: str, default: float) -> float:
    value = source.get(key, default)
    return float(value) if isinstance(value, int | float) else default


class VolumeBreakoutStrategy:
    id = STRATEGY_ID

    def __init__(self, params: dict[str, object], exits: ExitParams) -> None:
        self.params = params
        self.exits = exits
        self.lookback = int(_num(params, "lookback", 60))
        self.vol_mult = _num(params, "vol_mult", 2.0)

    def _breakout_high_column(self) -> str:
        return f"hhv_close_{self.lookback}_prev"

    def entry(self, ctx: StrategyContext) -> tuple[float | None, str] | None:
        bars = ctx.bars
        if len(bars) < self.lookback + 1:
            return None
        close = float(bars["close"].iloc[-1])
        volume = float(bars["volume"].iloc[-1])

        column = self._breakout_high_column()
        if column in ctx.daily.columns:
            prior_high = ctx.daily[column].iloc[-1]
        else:  # 指标窗口与策略参数不一致时自行计算
            prior_high = bars["close"].iloc[-(self.lookback + 1) : -1].max()
        if pd.isna(prior_high) or close <= float(prior_high):
            return None

        vol_ma = ctx.daily["vol_ma20"].iloc[-1] if "vol_ma20" in ctx.daily.columns else None
        if vol_ma is None or pd.isna(vol_ma) or vol_ma <= 0:
            return None
        if volume <= float(vol_ma) * self.vol_mult:
            return None

        atr = _last(ctx.daily, "atr14")
        prev_low = _last(ctx.daily, "llv_low_10")
        stop = compute_initial_stop(close, atr, prev_low, self.exits)
        if stop is None or stop_too_wide(close, stop, self.exits.max_stop_pct):
            return None
        return None, f"放量突破 {self.lookback} 日新高（量比 {volume / float(vol_ma):.1f}）"

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
