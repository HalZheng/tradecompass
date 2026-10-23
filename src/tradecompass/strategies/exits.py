"""退出规则（SPEC §9.9 X1–X7）。

止损由引擎按 ``initial_stop`` 在盘中执行；移动止盈与时间止损在这里返回卖出信号。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradecompass.backtest.types import StrategyContext
from tradecompass.config.models import StopsConfig


@dataclass(frozen=True, slots=True)
class ExitParams:
    atr_mult: float = 2.0
    max_stop_pct: float = 0.10
    trail_ma: int = 10
    trail_atr_mult: float = 3.0
    time_stop_enabled: bool = True
    time_stop_days: int = 20
    partial_enabled: bool = False
    partial_r: float = 2.0
    #: 移动止盈的口径：``ma``＝跌破均线（X3），``atr``＝自最高回撤 N×ATR（X4）
    trail_mode: str = "ma"
    #: 仅当已有浮盈时才允许均线止盈。False 时一跌破就卖（容易把亏损单也扫出去）
    trail_require_profit: bool = False
    #: 浮盈门槛（小数）。0 表示只要不亏就允许止盈
    trail_profit_threshold: float = 0.0

    @classmethod
    def from_config(cls, exits: dict[str, object], stops: StopsConfig) -> ExitParams:
        time_stop = exits.get("time_stop") or {}
        partial = exits.get("partial") or {}
        if not isinstance(time_stop, dict):
            time_stop = {}
        if not isinstance(partial, dict):
            partial = {}

        def num(source: dict[str, object], key: str, default: float) -> float:
            value = source.get(key, default)
            return float(value) if isinstance(value, int | float) else default

        return cls(
            atr_mult=num(exits, "atr_mult", stops.atr_mult),
            max_stop_pct=num(exits, "max_stop_pct", stops.max_stop_pct),
            trail_ma=int(num(exits, "trail_ma", 10)),
            trail_atr_mult=num(exits, "trail_atr_mult", 3.0),
            time_stop_enabled=bool(time_stop.get("enabled", True)),
            time_stop_days=int(num(time_stop, "days", 20)),
            partial_enabled=bool(partial.get("enabled", False)),
            partial_r=num(partial, "r", 2.0),
            trail_mode=str(exits.get("trail_mode", "ma")),
            trail_require_profit=bool(exits.get("trail_require_profit", False)),
            trail_profit_threshold=num(exits, "trail_profit_threshold", 0.0),
        )


def compute_initial_stop(
    entry_price_raw: float,
    atr: float | None,
    prev_low: float | None,
    params: ExitParams,
) -> float | None:
    """X1：``入场价 − atr_mult×ATR`` 与「前低」取较近者（即价格更高者）。"""
    if atr is None or atr <= 0:
        return None
    by_atr = entry_price_raw - params.atr_mult * atr
    candidates = [by_atr]
    if prev_low is not None and 0 < prev_low < entry_price_raw:
        candidates.append(prev_low)
    return max(candidates)


def stop_too_wide(entry_price_raw: float, stop_raw: float, max_stop_pct: float) -> bool:
    """X2：止损距离过宽则放弃入场。"""
    if entry_price_raw <= 0:
        return True
    return (entry_price_raw - stop_raw) / entry_price_raw > max_stop_pct


def _value(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = frame[column].iloc[-1]
    return None if pd.isna(value) else float(value)


def exit_signal(ctx: StrategyContext, params: ExitParams) -> tuple[float | None, str] | None:
    """移动止盈与时间止损，按优先级返回第一条触发的规则。"""
    position = ctx.position
    if position is None:
        return None
    close = _value(ctx.bars, "close")
    if close is None:
        return None

    atr = _value(ctx.daily, "atr14") or 0.0

    if params.time_stop_enabled and (ctx.date - position.entry_date).days >= params.time_stop_days:
        return None, f"持有满 {params.time_stop_days} 天（X5）"

    if params.trail_mode == "ma":
        ma_column = f"ma{params.trail_ma}"
        trail_ma = _value(ctx.daily, ma_column)
        if trail_ma is not None and close < trail_ma:
            # 只在"已经有浮盈"时才用均线止盈：否则刚买就被小回撤扫出，
            # 亏损单永远走不到 ATR 止损，形成赢小亏大。
            allow = (
                not params.trail_require_profit
                or position.unrealized_pct >= params.trail_profit_threshold
            )
            if allow:
                return None, f"收盘 {close:.2f} 跌破 MA{params.trail_ma} {trail_ma:.2f}（X3）"

    if atr > 0 and position.highest_close_raw > 0:
        drawdown = position.highest_close_raw - close
        if drawdown > params.trail_atr_mult * atr:
            return None, (
                f"自最高 {position.highest_close_raw:.2f} 回撤 {drawdown:.2f}"
                f" 超过 {params.trail_atr_mult:g}×ATR（X4）"
            )
    return None
