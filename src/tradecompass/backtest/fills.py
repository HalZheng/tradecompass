"""成交模拟（SPEC §12.2）。

一张表对应一个订单：

| 订单 | 成交条件 | 成交价 |
|---|---|---|
| 开盘买入 | 非一字涨停，且开盘涨幅 ≤ max_gap_open | open × (1+slippage) |
| 条件买入 cond_ge | high ≥ trigger 且非一字涨停 | max(open, trigger) × (1+slippage) |
| 开盘卖出 | 非一字跌停 | open × (1−slippage) |
| 止损卖出 cond_le | low ≤ stop 且非一字跌停 | min(open, stop) × (1−slippage) |
| 条件卖出 cond_ge | high ≥ trigger | max(open, trigger) × (1−slippage) |

不可成交的卖出顺延到下一交易日；买入未触发即作废（由调用方决定是否重下一单）。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradecompass.backtest.types import Order
from tradecompass.data.universe import (
    is_one_word_limit_down,
    is_one_word_limit_up,
    limit_down_price,
    limit_up_price,
)


@dataclass(frozen=True, slots=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    prev_close: float


@dataclass(frozen=True, slots=True)
class FillOutcome:
    filled: bool
    price_raw: float | None = None
    #: 卖出被一字跌停挡住时置 True，调用方把订单顺延
    deferred: bool = False
    note: str = ""


def one_word_flags(bar: Bar, limit_pct: float) -> tuple[bool, bool]:
    """返回 (一字涨停, 一字跌停)。\"一字\"指四价都贴在涨跌停价上。"""
    if limit_pct <= 0 or bar.prev_close <= 0:
        return False, False
    up = is_one_word_limit_up(
        bar.open, bar.high, bar.low, bar.close, limit_up_price(bar.prev_close, limit_pct)
    )
    down = is_one_word_limit_down(
        bar.open, bar.high, bar.low, bar.close, limit_down_price(bar.prev_close, limit_pct)
    )
    return up, down


def simulate(
    order: Order,
    bar: Bar,
    *,
    limit_pct: float,
    slippage: float,
    max_gap_open: float,
) -> FillOutcome:
    up, down = one_word_flags(bar, limit_pct)

    if order.order_type == "open":
        if order.side == "buy":
            if up:
                return FillOutcome(False, note="一字涨停，买入作废")
            if bar.prev_close > 0 and bar.open / bar.prev_close - 1.0 > max_gap_open:
                return FillOutcome(False, note="开盘涨幅超限，放弃买入")
            return FillOutcome(True, price_raw=bar.open * (1.0 + slippage))
        if down:
            return FillOutcome(False, deferred=True, note="一字跌停，卖出顺延")
        return FillOutcome(True, price_raw=bar.open * (1.0 - slippage))

    trigger = order.price_raw
    if trigger is None:
        return FillOutcome(False, note="条件单缺少触发价")

    if order.order_type == "cond_ge":
        if bar.high < trigger:
            return FillOutcome(False, note="未触发")
        if order.side == "buy" and up:
            return FillOutcome(False, note="一字涨停，买入作废")
        price = max(bar.open, trigger)
        factor = 1.0 + slippage if order.side == "buy" else 1.0 - slippage
        return FillOutcome(True, price_raw=price * factor)

    # cond_le：只用于卖出止损/止盈
    if bar.low > trigger:
        return FillOutcome(False, note="未触发")
    if down:
        return FillOutcome(False, deferred=True, note="一字跌停，卖出顺延")
    return FillOutcome(True, price_raw=min(bar.open, trigger) * (1.0 - slippage))
