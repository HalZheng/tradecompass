"""M3 用的测试策略：收盘站上 MA20 买入、跌破 MA20 卖出。

它**不是交易策略**，只是回测引擎的冒烟对象（SPEC §19 M3）：
规则足够简单，可以在合成数据上手算出期望结果。
"""

from __future__ import annotations

from tradecompass.backtest.types import StrategyContext

STRATEGY_ID = "ma20_cross"


class Ma20CrossStrategy:
    id = STRATEGY_ID

    def __init__(self, ma_period: int = 20) -> None:
        self.ma_period = ma_period
        self.column = f"ma{ma_period}"

    def _row(self, ctx: StrategyContext) -> tuple[float, float] | None:
        if ctx.daily.empty or self.column not in ctx.daily.columns:
            return None
        row = ctx.daily.iloc[-1]
        close = row["close"]
        ma_value = row[self.column]
        if close != close or ma_value != ma_value:  # NaN
            return None
        return float(close), float(ma_value)

    def entry(self, ctx: StrategyContext) -> tuple[float | None, str] | None:
        values = self._row(ctx)
        if values is None:
            return None
        close, ma_value = values
        if close > ma_value:
            return None, f"收盘 {close:.2f} 站上 MA{self.ma_period} {ma_value:.2f}"
        return None

    def exit(self, ctx: StrategyContext) -> tuple[float | None, str] | None:
        values = self._row(ctx)
        if values is None:
            return None
        close, ma_value = values
        if close < ma_value:
            return None, f"收盘 {close:.2f} 跌破 MA{self.ma_period} {ma_value:.2f}"
        return None

    def initial_stop(self, ctx: StrategyContext, entry_price_raw: float) -> float | None:
        return None
