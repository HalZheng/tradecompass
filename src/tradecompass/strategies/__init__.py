"""策略注册表：按 id 构造策略实例。"""

from __future__ import annotations

from typing import Protocol

from tradecompass.backtest.types import Strategy
from tradecompass.config.models import StopsConfig, StrategyConfig
from tradecompass.strategies.b1_pullback import B1PullbackStrategy
from tradecompass.strategies.exits import ExitParams
from tradecompass.strategies.ma20_cross import Ma20CrossStrategy
from tradecompass.strategies.volume_breakout import VolumeBreakoutStrategy

__all__ = ["AVAILABLE", "build", "ExitParams"]


class _Factory(Protocol):
    def __call__(self, params: dict[str, object], exits: ExitParams) -> Strategy: ...


AVAILABLE: dict[str, _Factory] = {
    "b1_pullback": B1PullbackStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
    "ma20_cross": lambda params, exits: Ma20CrossStrategy(),
}


def build(config: StrategyConfig, stops: StopsConfig) -> Strategy:
    """按配置构造策略；参数与退出规则都来自 ``config/strategies/*.yaml``。"""
    factory = AVAILABLE.get(config.id)
    if factory is None:
        raise KeyError(f"未知策略 {config.id}；可用：{sorted(AVAILABLE)}")
    return factory(config.params, ExitParams.from_config(config.exits, stops))
