"""信号层数据结构。

这些对象只承载语义，不做 I/O。因子快照（``FactorSnapshot``）由调用方传入，
持仓与交易历史由账户层组装，信号层只负责判断和打分。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from tradecompass.config.loader import AppConfig


class SignalKind(StrEnum):
    B1 = "B1"
    B2 = "B2"
    S1 = "S1"
    S2 = "S2"


@dataclass(frozen=True, slots=True)
class SignalParams:
    """信号阈值与仓位约束。默认值先写死在代码里，跑两周后再谈调参。"""

    # 置信度门槛：默认按 regime 动态取值；这里的 None 表示不强制覆盖。
    min_confidence_buy: int | None = None
    min_confidence_b2: int | None = None
    min_confidence_sell: int | None = None
    watch_confidence: int = 50

    # B1：RISK_ON 放宽，NEUTRAL 收紧；RISK_OFF/PANIC 不新开仓
    b1_threshold_risk_on: int = 55
    b1_threshold_neutral: int = 65

    # B2：突破买点门槛略高于 B1
    b2_threshold_risk_on: int = 65
    b2_threshold_neutral: int = 70
    b2_volume_mult: float = 1.5

    # S1：弱市更敏感（阈值更低），强市更宽容
    s1_threshold_risk_on: int = 70
    s1_threshold_neutral: int = 60
    s1_threshold_risk_off: int = 50
    s1_threshold_panic: int = 45

    # 行业强弱对阈值的修正
    sector_strong_delta: int = -3
    sector_weak_delta: int = 5

    # S1 的早期亏损干预线：-5% 就允许趋势减仓，不等 -10% 的 S2
    early_loss_pct: float = -0.05
    deep_loss_pct: float = -0.08

    # S2
    s2_loss_pct: float = -0.10
    s2_volume_mult: float = 1.5
    s2_single_day_drop: float = -0.07

    # 仓位与历史
    max_positions: int = 5
    max_weight: float = 0.30
    risk_per_trade: float = 0.01
    swing_ratio: float = 0.5
    min_trade_amount: float = 5000.0
    cooldown_days: int = 10
    profit_sell_cooldown_days: int = 5
    min_hold_days_before_add: int = 3
    min_hold_days_before_s1: int = 3

    # 止损
    atr_mult: float = 2.0
    max_stop_pct: float = 0.10

    # 分级移动止损（只影响回测/持仓管理层，不改变买入信号）。
    # 默认关闭：D035 实测紧移动止损（+0.5R 保本）会把右尾砍掉，PF 0.79→0.63。
    # 研究时用 --breakeven-r / --lock-r / --trail-r 打开。
    breakeven_at_r: float = 999.0
    lock_half_r_at_r: float = 999.0
    trail_at_r: float = 999.0
    trail_atr_mult: float = 2.0

    # B2 入场微调（来自 D034 归因）
    b2_rsi_max: float = 80.0
    b2_extreme_volume: float = 3.0

    @classmethod
    def from_config(cls, cfg: AppConfig) -> SignalParams:
        """把 risk.yaml 里已经存在的风控参数接进来，避免两套口径。"""
        return cls(
            max_positions=cfg.risk.swing.max_positions,
            risk_per_trade=cfg.risk.swing.risk_per_trade,
            swing_ratio=cfg.risk.allocation.swing,
            min_trade_amount=cfg.risk.swing.min_trade_amount,
            cooldown_days=cfg.risk.swing.cooldown_days_after_stop,
            atr_mult=cfg.risk.stops.atr_mult,
            max_stop_pct=cfg.risk.stops.max_stop_pct,
            s2_loss_pct=-cfg.risk.stops.max_stop_pct,
        )

    def buy_threshold(
        self, kind: str, market_state: str, sector_strength: str
    ) -> int:
        """动态买入门槛：强市放宽、弱市收紧，行业强弱再做修正。"""
        override = self.min_confidence_buy if kind == "B1" else self.min_confidence_b2
        if override is not None:
            return max(0, min(100, override))
        if market_state == "RISK_ON":
            base = self.b1_threshold_risk_on if kind == "B1" else self.b2_threshold_risk_on
        elif market_state == "NEUTRAL":
            base = self.b1_threshold_neutral if kind == "B1" else self.b2_threshold_neutral
        else:
            return 100  # RISK_OFF / PANIC 不新开仓
        if sector_strength == "strong":
            base += self.sector_strong_delta
        elif sector_strength == "weak":
            base += self.sector_weak_delta
        return max(0, min(100, base))

    def sell_threshold(self, market_state: str, sector_strength: str) -> int:
        """动态卖出门槛：弱市降低门槛（更敏感），强市提高门槛（更宽容）。"""
        if self.min_confidence_sell is not None:
            return max(0, min(100, self.min_confidence_sell))
        base = {
            "RISK_ON": self.s1_threshold_risk_on,
            "NEUTRAL": self.s1_threshold_neutral,
            "RISK_OFF": self.s1_threshold_risk_off,
            "PANIC": self.s1_threshold_panic,
        }.get(market_state, self.s1_threshold_neutral)
        if sector_strength == "strong":
            base += 3
        elif sector_strength == "weak":
            base -= 5
        return max(0, min(100, base))


@dataclass(frozen=True, slots=True)
class PositionContext:
    """当前持仓的单票视图（由账户层或期初持仓快照生成）。"""

    symbol: str
    name: str | None
    quantity: int
    avg_cost_raw: float
    close_raw: float
    weight: float
    pnl_pct: float
    holding_days: int | None = None


@dataclass(slots=True)
class SymbolHistory:
    """单票的交易历史摘要，用于冷却期和连亏判断（回测中会原地更新）。"""

    symbol: str
    last_buy_date: date | None = None
    last_sell_date: date | None = None
    last_sell_pnl: float | None = None
    realized_pnl: float = 0.0
    consecutive_losses: int = 0


@dataclass(frozen=True, slots=True)
class PortfolioContext:
    """整个账户的上下文。``source`` 说明持仓来自交割单还是期初快照。"""

    equity: float
    cash: float
    holdings_count: int
    positions: Mapping[str, PositionContext] = field(default_factory=dict)
    history: Mapping[str, SymbolHistory] = field(default_factory=dict)
    source: str = "none"

    def position_of(self, symbol: str) -> PositionContext | None:
        return self.positions.get(symbol)

    def history_of(self, symbol: str) -> SymbolHistory | None:
        return self.history.get(symbol)


@dataclass(frozen=True, slots=True)
class Signal:
    """一条可展示、可落盘的信号。"""

    kind: SignalKind
    symbol: str
    name: str
    as_of: date
    confidence: int
    actionable: bool
    action: str
    reasons: tuple[str, ...]
    #: 当日生效的动态门槛；S2 为 0（风控信号不受阈值限制）
    threshold: int = 0
    warnings: tuple[str, ...] = ()
    trigger_price_raw: float | None = None
    stop_price_raw: float | None = None
    suggested_quantity: int | None = None
    position_note: str = ""
    metrics: Mapping[str, float | None] = field(default_factory=dict)

    @property
    def priority(self) -> int:
        return {
            SignalKind.S2: 4,
            SignalKind.S1: 3,
            SignalKind.B2: 2,
            SignalKind.B1: 1,
        }[self.kind]
