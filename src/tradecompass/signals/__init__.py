"""信号层：把因子快照映射成 B1/B2/S1/S2 与置信度。"""

from tradecompass.signals.engine import evaluate, stop_levels
from tradecompass.signals.types import (
    PortfolioContext,
    PositionContext,
    Signal,
    SignalKind,
    SignalParams,
    SymbolHistory,
)

__all__ = [
    "PortfolioContext",
    "PositionContext",
    "Signal",
    "SignalKind",
    "SignalParams",
    "SymbolHistory",
    "evaluate",
    "stop_levels",
]
