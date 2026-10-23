"""回测层的公共数据类型（SPEC §12.1）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol

import pandas as pd

from tradecompass.domain.models import Instrument, OrderType

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class Order:
    """收盘后生成、次日执行的订单。"""

    symbol: str
    side: Side
    quantity: int
    order_type: OrderType
    price_raw: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class Fill:
    date: date
    symbol: str
    side: Side
    quantity: int
    price_raw: float
    fee: float
    reason: str


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    entry_date: date
    entry_price_raw: float
    exit_date: date
    exit_price_raw: float
    quantity: int
    gross_pnl: float
    fees: float
    net_pnl: float
    r_multiple: float | None
    holding_days: int
    exit_reason: str


@dataclass(frozen=True, slots=True)
class EquityPoint:
    date: date
    equity: float
    cash: float
    position_value: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy_id: str
    start: date
    end: date
    initial_cash: float
    trades: tuple[Trade, ...]
    equity: tuple[EquityPoint, ...]
    fills: tuple[Fill, ...]
    fees_total: float
    costless_equity: tuple[EquityPoint, ...] = ()


@dataclass(frozen=True, slots=True)
class PositionView:
    """策略看到的持仓（只读）。"""

    symbol: str
    shares: int
    entry_price_raw: float
    entry_date: date
    current_stop_raw: float | None
    highest_close_raw: float
    unrealized_pct: float


@dataclass(frozen=True, slots=True)
class StrategyContext:
    date: date
    instrument: Instrument
    #: 前复权日线，截至当日（含）
    bars: pd.DataFrame
    #: 日线指标表，截至当日（含）
    daily: pd.DataFrame
    #: 周线指标表，**只含已完结的周**；当前未走完的那一周不会出现
    weekly: pd.DataFrame
    position: PositionView | None
    extra: dict[str, object] = field(default_factory=dict)


class Strategy(Protocol):
    """策略只需要回答两个问题：今天要不要买、要不要卖。"""

    id: str

    def entry(self, ctx: StrategyContext) -> tuple[float | None, str] | None:
        """返回 ``(触发价, 理由)``。

        触发价为 ``None`` 表示次日开盘买入；给出价格表示突破该价才买。
        """

    def exit(self, ctx: StrategyContext) -> tuple[float | None, str] | None:
        """返回 ``(触发价, 理由)``；``None`` 表示不卖。"""

    def initial_stop(self, ctx: StrategyContext, entry_price_raw: float) -> float | None:
        """给出初始止损价（不复权）。"""
