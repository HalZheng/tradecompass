"""领域数据类（SPEC §7）。

这里只放不依赖任何 I/O 的纯数据结构：策略、风控、回测、输出层共用同一套对象，
保证回测与实盘走的是同一份语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Market = Literal["SH", "SZ", "BJ"]
Kind = Literal["stock", "sector_etf", "base_etf", "index"]
Board = Literal["main", "gem", "star", "bj", "etf", "index"]
Category = Literal["base", "swing", "legacy"]
ExitReason = Literal["stop", "trail_ma", "trail_atr", "time", "cleanup", "rebalance"]
Priority = Literal["must", "optional", "watch"]
ActionKind = Literal[
    "buy", "sell", "stop", "rebalance_buy", "rebalance_sell", "cleanup_sell"
]
OrderType = Literal["open", "cond_ge", "cond_le"]
RegimeLevel = Literal["attack", "neutral", "defense"]


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    market: Market
    name: str
    kind: Kind
    board: Board
    sector_sw: str | None = None
    t0: bool = False
    is_st: bool = False


@dataclass(frozen=True, slots=True)
class EntrySignal:
    strategy_id: str
    params_hash: str
    symbol: str
    signal_date: date
    entry_mode: Literal["next_open", "breakout_high"]
    trigger_price_raw: float | None
    stop_price_raw: float
    evidence: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExitSignal:
    symbol: str
    signal_date: date
    reason: ExitReason
    trigger_price_raw: float | None
    quantity: int | None
    evidence: dict[str, float | str] = field(default_factory=dict)


@dataclass(slots=True)
class Lot:
    buy_date: date
    quantity: int
    price_raw: float
    fee: float


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int
    avg_cost_raw: float
    lots: list[Lot]
    category: Category
    strategy_id: str | None
    entry_date: date | None
    initial_stop_raw: float | None
    current_stop_raw: float | None
    highest_close_raw: float | None


@dataclass(frozen=True, slots=True)
class MarketRegime:
    date: date
    score: int
    components: dict[str, int | None]
    level: RegimeLevel
    index_bear: bool


@dataclass(frozen=True, slots=True)
class Action:
    for_date: date
    priority: Priority
    kind: ActionKind
    symbol: str
    name: str
    quantity: int
    order_type: OrderType
    price_raw: float | None
    stop_price_raw: float | None
    rule_refs: tuple[str, ...]
    blocked_by: tuple[str, ...]
    reason_text: str
